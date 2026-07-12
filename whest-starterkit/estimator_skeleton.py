"""SK-4 walking skeleton — SHIPPABILITY VALIDATION ONLY, not a submission.

The lazy-CP kappa3/kappa4 closure op profile with CRUDE/RANDOM coefficients,
wired into the whestbench contract. Purpose: prove the target architecture's
cost, ops, and guard/fallback behaviour survive `whest validate` and a mini
run (0/100 failures, utilization in the ledger's range). MSE will be BAD by
design -- coefficients are placeholders. See SKELETON_PLAN.md gates.

Keep separate from estimator.py: v4 stays the live submission.

  uv run whest validate --estimator estimator_skeleton.py
  uv run whest run --estimator estimator_skeleton.py \
      --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
      --split mini --runner subprocess
"""

from __future__ import annotations

import numpy as np  # host-side only: stands in for shipped-npz tables (0-FLOP)
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

R_MAX = 128
BIRTHS = 12
RECOMPRESS_EVERY = 4
S_EXTRA = 8
PW_HIDDEN = 8
EPS = 1e-3
ENVELOPE = 1e3
SUPPORTED_DEPTH = 32   # skeleton knobs are sized for the phase-1 shape


def _tables(n: int, depth: int, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "pw_w1": rng.standard_normal((6, PW_HIDDEN)).astype(np.float32) * 0.1,
        "pw_w2": rng.standard_normal((PW_HIDDEN, 2)).astype(np.float32) * 0.1,
        "tab_x": np.linspace(-8, 8, 64).astype(np.float32),
        "tab_y": np.ones(64, dtype=np.float32),
        "fam_c": (rng.standard_normal((depth, S_EXTRA)) * 0.01).astype(np.float32),
        "birth_p": (rng.standard_normal((n, BIRTHS)) / np.sqrt(n)).astype(np.float32),
    }


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._tab = None

    def setup(self, ctx) -> None:
        # In the real submission these come from a shipped npz via
        # flopscope.Module.from_file (0 FLOPs). Host-side generation here
        # models the same cost.
        self._tab = _tables(256, SUPPORTED_DEPTH)

    # ------------------------------------------------------------------ #
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n, depth = mlp.width, mlp.depth
        if depth != SUPPORTED_DEPTH or self._tab is None or n != 256:
            return self._k2_all_layers(mlp)          # shape fallback

        rows, tripped = self._closure_all_layers(mlp)
        if tripped:
            return self._k2_all_layers(mlp)          # guard fallback (~0.8% of B)
        return rows

    # ------------------------------------------------------------------ #
    def _closure_all_layers(self, mlp: MLP):
        tab = self._tab
        n, depth = mlp.width, mlp.depth
        tab_x = fnp.asarray(tab["tab_x"])
        tab_y = fnp.asarray(tab["tab_y"])
        pw_w1 = fnp.asarray(tab["pw_w1"])
        pw_w2 = fnp.asarray(tab["pw_w2"])
        birth_p = fnp.asarray(tab["birth_p"])

        mu = fnp.zeros(n, dtype=fnp.float32)
        cov = fnp.eye(n, dtype=fnp.float32)
        V = fnp.zeros((n, BIRTHS), dtype=fnp.float32)
        c = fnp.zeros(BIRTHS, dtype=fnp.float32)
        rows = []

        for layer, w in enumerate(mlp.weights):
            mu_pre = fnp.matmul(mu, w)
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            var = fnp.maximum(fnp.diag(cov_pre), 1e-10)
            s = fnp.sqrt(var)
            t = mu_pre / s
            pdf = flops.stats.norm.pdf(t)
            cdf = flops.stats.norm.cdf(t)
            h1 = cdf * fnp.interp(t, tab_x, tab_y)
            h2 = pdf / s

            # lazy-CP transport + births
            V = fnp.matmul(fnp.transpose(w), h1[:, None] * V)
            Bm = fnp.matmul(cov_pre * h1[None, :], w)
            V = fnp.concatenate([V, fnp.matmul(Bm, birth_p)], axis=1)
            c = fnp.concatenate([c, fnp.full(BIRTHS, 1e-3, dtype=fnp.float32)], axis=0)
            if V.shape[1] > R_MAX and layer % RECOMPRESS_EVERY == 0:
                Q, Rm = fnp.linalg.qr(V)
                U, S, _ = fnp.linalg.svd(Rm * c[None, :])
                keep = min(R_MAX, U.shape[1])
                V = fnp.matmul(Q, U[:, :keep] * S[:keep][None, :])
                c = fnp.full(keep, 1.0, dtype=fnp.float32)

            # star slice: matmul+reshape only (einsum symmetry trap)
            Cm = h2[:, None] * w * Bm
            Dm = h2[:, None] * (Bm * Bm)
            star = 2.0 * fnp.matmul(fnp.transpose(Cm), Bm) + fnp.matmul(fnp.transpose(Dm), w)

            # extra families with shared bases (CSE)
            F1 = fnp.matmul(cov_pre * h1[None, :], w)
            F2 = fnp.matmul(cov_pre * h2[None, :], w)
            fam = fnp.zeros((n, n), dtype=fnp.float32)
            for k in range(S_EXTRA):
                ck = float(tab["fam_c"][layer, k])
                base = F2 if k % 2 else F1
                rhs = Bm if k % 3 == 0 else w
                fam = fam + ck * fnp.matmul(fnp.transpose(base), rhs)

            slice3 = fnp.matmul((V * V) * c[None, :], fnp.transpose(V)) + star + fam

            # pointwise stub heads (k3 correction + k4 slice)
            tp = fnp.broadcast_to(t[:, None], (n, n))
            tq = fnp.broadcast_to(t[None, :], (n, n))
            rho = cov_pre / fnp.outer(s, s)
            lf = fnp.full((n, n), (layer + 1.0) / depth, dtype=fnp.float32)
            feats = fnp.stack([tp, tq, rho, slice3, star, lf], axis=-1).reshape(n * n, 6)
            heads = fnp.matmul(fnp.tanh(fnp.matmul(feats, pw_w1)), pw_w2).reshape(n, n, 2)
            slice3 = slice3 + EPS * heads[:, :, 0]
            k4slice = EPS * heads[:, :, 1]

            # moment updates with shape-correct, EPS-scaled corrections
            dmu = EPS * fnp.sum(slice3 * rho, axis=1) / n
            mu = s * pdf + mu_pre * cdf + dmu
            ez2 = (mu_pre * mu_pre + var) * cdf + mu_pre * s * pdf
            var_post = fnp.maximum(ez2 - mu * mu, 1e-12) * (1.0 + EPS * fnp.diag(k4slice))
            corr = fnp.outer(h2, h1) * slice3
            cov = (fnp.outer(h1, h1) * cov_pre
                   + fnp.outer(h2, h2) * (cov_pre * cov_pre) * 0.5
                   + EPS * 0.5 * (corr + fnp.transpose(corr)))
            fnp.fill_diagonal(cov, var_post)
            cov = flops.as_symmetric(0.5 * (cov + fnp.transpose(cov)), symmetry=(0, 1))
            rows.append(mu)

            # guard: cheap invariants, Python branch, bail early on trip
            finite = bool(fnp.all(fnp.isfinite(mu))) and bool(fnp.all(fnp.isfinite(cov)))
            bounded = (float(fnp.max(fnp.abs(mu))) < ENVELOPE
                       and float(fnp.max(var_post)) < ENVELOPE)
            if not (finite and bounded):
                return None, True

        return fnp.stack(rows, axis=0), False

    # ------------------------------------------------------------------ #
    def _k2_all_layers(self, mlp: MLP) -> fnp.ndarray:
        """Plain Gaussian K=2 propagation; the free (<1% of B) fallback."""
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
            cov = flops.as_symmetric(0.5 * (cov + fnp.transpose(cov)), symmetry=(0, 1))
            rows.append(mu)
        return fnp.stack(rows, axis=0)
