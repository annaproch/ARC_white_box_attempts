"""v5 — the best-bet architecture from the 2026-07-12 gates. STRUCTURE FINAL,
CONSTANTS PROVISIONAL: every learned number lives in `v5.npz` (0-FLOP load), so
longer offline fits replace constants without touching a line of this file.

WHAT THE GATES SAID (STRATEGY.md 7):
  * The moment update reads ONLY the kappa3 SLICE (n^2), never the tensor.
    => the whole problem is "predict the kappa3(z)_ppq slice accurately".
       ~5% => beat the leaders. Propagation alone gives ~25-33%.
  * G5: a learned map (local state + star + PROPAGATED slice) -> true slice cut
    that to ~17% held-out, and BEAT an optimal per-layer gain (20.2%) -- i.e. the
    "shape error" E13 called untouchable IS learnable. This is the anchor.
  * G1c: kappa4 slices are LEARNABLE and worth ~16x in the moment update; a
    28.5%-wrong kappa4_iijj costs only 18% of the oracle ceiling.
  * The kappa3 TENSOR transport (3 mode products, n^4) is 305% of budget =>
    the carry MUST be factored. Lazy-CP: transport = R matvecs.
  * kappa4 BLOCKS in kappa3 generation need an n^3 object (kappa4_ppqr) we
    cannot afford -> NOT in v5. (Second lever, later.)

FLOP LEDGER (target, all measured by flopscope at export):
    (mu,Sigma) complete update          ~0.8%
    star generation (4 matmuls/layer)   ~1.6%
    lazy-CP carry (transport + births)  ~0.6%
    slice extraction (n^2 R)            ~0.1%
    anchor map (pointwise n^2)          ~0.2%
    kappa4 map (pointwise n^2)          ~0.2%
    ---------------------------------------- ~3.5% analytic
    QMC with the rest of the floor      ~6%
Fusion bar: closure b^2 < 1.02e-5 beats v4 (3.396e-7).

SAFETY: any guard trip falls back to calibrated-K2 (0.8%), so primary+fallback
stays under the free 10% floor. Missing npz => behaves exactly like v4.
"""

from __future__ import annotations

from pathlib import Path

import flopscope
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

V5_FILE = "v5.npz"
POINTS_FILE = "points.npz"

RANK = 64          # CP carry rank (R=128 was measured ~= exact; 64 is the cheap end)
BIRTHS = 16        # new CP directions per layer
ENVELOPE = 1e3     # guard: physics-violating magnitude (offline-fitted, generous)


class V5Params(flopscope.Module):
    """All learned constants. Fitted offline; refit freely without code changes.

    anchor_w1/w2 : pointwise map  (F -> H -> 1) predicting the TRUE kappa3 slice
                   from [t_i, t_j, rho_ij, prop_slice, star, layer_frac, 1]
    k4_w1/k4_w2  : same shape, predicting the kappa4_iijj slice (G1c)
    birth_p      : (n, BIRTHS) projector seeding new CP directions from the star
    gains        : (depth, 4) per-layer scalars (mu/cov correction trims)
    meta         : [rank, n_qmc, w_qmc, w_closure, anchor_on, k4_on]
    """

    def __init__(self) -> None:
        self.anchor_w1 = fnp.zeros((7, 8))
        self.anchor_w2 = fnp.zeros((8, 1))
        self.k4_w1 = fnp.zeros((7, 8))
        self.k4_w2 = fnp.zeros((8, 1))
        self.birth_p = fnp.zeros((1, 1))
        self.gains = fnp.zeros((1, 4))
        self.meta = fnp.zeros((6,))


class Points(flopscope.Module):
    def __init__(self) -> None:
        self.x = fnp.zeros((1, 1))


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._p = None            # V5Params or None
        self._points = None

    def setup(self, ctx) -> None:
        if getattr(ctx, "submission_dir", None) is None:
            return
        d = Path(ctx.submission_dir)
        if (d / POINTS_FILE).exists():
            self._points = Points.from_file(str(d / POINTS_FILE)).x
        if (d / V5_FILE).exists():
            self._p = V5Params.from_file(str(d / V5_FILE))

    # ------------------------------------------------------------------ #
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n, depth = mlp.width, mlp.depth
        rows = None
        if self._p is not None:
            rows = self._closure(mlp)          # None if a guard tripped
        if rows is None:
            rows = self._k2(mlp)               # free fallback (~0.8% of budget)

        # ---- QMC on the remaining floor, fused at the final layer only
        w_qmc = float(self._p.meta[2]) if self._p is not None else 1.0
        pts = self._points
        if pts is not None and pts.shape[1] == n and w_qmc > 1e-6:
            n_pts = int(float(self._p.meta[1])) if self._p is not None else 0
            per = depth * (2 * n * n + 2 * n)
            k = max(min(int(0.06 * budget / per), pts.shape[0], n_pts or 10 ** 9), 1)
            x = pts[:k]
            mc_last = None
            for w in mlp.weights:
                x = fnp.maximum(fnp.matmul(x, w), 0.0)
            mc_last = fnp.mean(x, axis=0)
            w_c = 1.0 - w_qmc
            rows[-1] = w_qmc * mc_last + w_c * rows[-1]
        return fnp.stack(rows, axis=0)

    # ------------------------------------------------------------------ #
    def _closure(self, mlp: MLP):
        """(mu, Sigma) + factored kappa3 carry + learned slice anchor + learned
        kappa4. Returns None if a guard trips (physics-violating state)."""
        p = self._p
        n, depth = mlp.width, mlp.depth
        anchor_on = float(p.meta[4]) > 0.5
        k4_on = float(p.meta[5]) > 0.5
        birth_p = p.birth_p
        if birth_p.shape[0] != n:
            return None

        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = fnp.eye(n, dtype=fnp.float32)
        V = fnp.zeros((n, BIRTHS), dtype=fnp.float32)     # CP directions
        c = fnp.zeros(BIRTHS, dtype=fnp.float32)          # CP coefficients
        rows = []

        for layer, w in enumerate(mlp.weights):
            # ---- linear step (EXACT) -------------------------------------
            mu_pre = fnp.matmul(mu, w)
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)   # symmetric-tag idiom
            var = fnp.maximum(fnp.diag(cov_pre), 1e-10)
            s = fnp.sqrt(var)
            t = mu_pre / s
            pdf = flops.stats.norm.pdf(t)
            cdf = flops.stats.norm.cdf(t)
            h1 = cdf                     # E[ReLU'] gain
            h2 = pdf / s                 # 2nd Hermite gain

            # ---- kappa3 slice from the CP carry:  (V.V) diag(c) V^T  -----
            # n^2 R -- the tensor is NEVER materialised (n^4 transport = 305% of B)
            prop = fnp.matmul((V * V) * c[None, :], fnp.transpose(V))

            # ---- star generation (the freshly-created kappa3) -------------
            # B = Sigma diag(h1) W ; slice = 2 (C^T B) + (D^T W).  matmul+reshape
            # ONLY -- einsum with repeated operands trips flopscope's symmetry tag.
            Bm = fnp.matmul(cov_pre * h1[None, :], w)
            Cm = h2[:, None] * w * Bm
            Dm = h2[:, None] * (Bm * Bm)
            star = 2.0 * fnp.matmul(fnp.transpose(Cm), Bm) + fnp.matmul(
                fnp.transpose(Dm), w)

            # ---- THE ANCHOR: learned correction of the slice (G5) ---------
            # features on the n^2 grid; the propagated slice is the key feature
            # (it carries the HISTORY that killed the local-only map, P1/E12).
            rho = cov_pre / fnp.outer(s, s)
            k3 = prop + star
            # NORMALISED features/targets (must match research/fit_v5.py EXACTLY,
            # or the offline fit will not transfer -- silent-failure class).
            nrm3 = fnp.outer(s * s, s)                 # s_i^2 s_j
            nrm4 = fnp.outer(s * s, s * s)             # s_i^2 s_j^2
            if anchor_on or k4_on:
                feats = fnp.stack([
                    fnp.broadcast_to(t[:, None], (n, n)),
                    fnp.broadcast_to(t[None, :], (n, n)),
                    rho, k3 / nrm3, star / nrm3,
                    fnp.full((n, n), (layer + 1.0) / depth, dtype=fnp.float32),
                    fnp.ones((n, n), dtype=fnp.float32),
                ], axis=-1).reshape(n * n, 7)
            if anchor_on:
                hid = fnp.tanh(fnp.matmul(feats, p.anchor_w1))
                k3 = k3 + fnp.matmul(hid, p.anchor_w2).reshape(n, n) * nrm3
            k4 = None
            if k4_on:
                hid4 = fnp.tanh(fnp.matmul(feats, p.k4_w1))
                k4 = fnp.matmul(hid4, p.k4_w2).reshape(n, n) * nrm4

            # ---- moment update with the corrected cumulant slices ---------
            g = p.gains[layer] if p.gains.shape[0] == depth else None
            k3d = fnp.diag(k3)
            mu = s * pdf + mu_pre * cdf + (k3d / 6.0) * (h2 * t)      # kappa3 mean corr
            ez2 = (mu_pre * mu_pre + var) * cdf + mu_pre * s * pdf + (k3d / 3.0) * h2
            if k4 is not None:
                ez2 = ez2 + fnp.diag(k4) / 24.0 * (-2.0 * t * pdf / var)
            var_post = fnp.maximum(ez2 - mu * mu, 1e-12)

            corr = fnp.outer(h2, h1) * k3
            cov = (fnp.outer(h1, h1) * cov_pre
                   + fnp.outer(h2, h2) * (cov_pre * cov_pre) * 0.5
                   + 0.5 * (corr + fnp.transpose(corr)))
            if k4 is not None:
                cov = cov + 0.25 * fnp.outer(h2, h2) * k4
            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(0.5 * (cov + fnp.transpose(cov)),
                                     symmetry=(0, 1))
            rows.append(mu)

            # ---- guard: physics-violating state -> bail to K2 -------------
            if not (bool(fnp.all(fnp.isfinite(mu)))
                    and float(fnp.max(fnp.abs(mu))) < ENVELOPE
                    and float(fnp.max(var_post)) < ENVELOPE):
                return None

            # ---- carry: transport the CP factors + seed new directions ----
            # transport is ONE (n,n)@(n,R) matmul per layer -- this is why the
            # carry is factored rather than a tensor.
            if layer + 1 < depth:
                wn = mlp.weights[layer + 1]
                V = fnp.matmul(fnp.transpose(wn), h1[:, None] * V)
                newV = fnp.matmul(fnp.matmul(fnp.transpose(wn), star), birth_p)
                V = fnp.concatenate([V, newV], axis=1)
                c = fnp.concatenate(
                    [c, fnp.ones(BIRTHS, dtype=fnp.float32)], axis=0)
                if V.shape[1] > RANK:                     # recompress
                    Q, Rm = fnp.linalg.qr(V)
                    U, S, _ = fnp.linalg.svd(Rm * c[None, :])
                    keep = min(RANK, U.shape[1])
                    V = fnp.matmul(Q, U[:, :keep] * S[:keep][None, :])
                    c = fnp.ones(keep, dtype=fnp.float32)
        return rows

    # ------------------------------------------------------------------ #
    def _k2(self, mlp: MLP):
        """Plain Gaussian K=2 propagation — the free fallback (~0.8% of B)."""
        n = mlp.width
        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = fnp.eye(n, dtype=fnp.float32)
        rows = []
        for w in mlp.weights:
            mu_pre = fnp.matmul(mu, w)
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            var = fnp.maximum(fnp.diag(cov_pre), 1e-10)
            s = fnp.sqrt(var)
            a = mu_pre / s
            pdf = flops.stats.norm.pdf(a)
            cdf = flops.stats.norm.cdf(a)
            mu = s * pdf + mu_pre * cdf
            ez2 = (mu_pre * mu_pre + var) * cdf + mu_pre * s * pdf
            var_post = fnp.maximum(ez2 - mu * mu, 1e-12)
            cov = (fnp.outer(cdf, cdf) * cov_pre
                   + fnp.outer(pdf / s, pdf / s) * (cov_pre * cov_pre) * 0.5)
            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(0.5 * (cov + fnp.transpose(cov)),
                                     symmetry=(0, 1))
            rows.append(mu)
        return rows
