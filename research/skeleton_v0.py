"""SK-1..SK-3: walking skeleton of the lazy-CP kappa3/kappa4 closure.

Runs the FULL op profile of the target architecture under flopscope with
crude/random coefficients, and reports a per-component FLOP ledger. This file
makes NO accuracy claims: its only claims are cost, op coverage, and fallback
behaviour (see SKELETON_PLAN.md gates SK-1..SK-3).

Op-profile components (all shapes final, all coefficients placeholder):
  A. (mu, Sigma) K=2 propagation        einsum symmetric idiom (proven)
  B. lazy-CP kappa3 carry               transport matvec-batch + births
  C. QR+SVD recompression               cost proxy; CP re-fit math comes later
  D. star slice (exact contraction)     matmul+reshape ONLY (einsum trap)
  E. S extra source families            placeholder n^3 contractions
  F. pointwise learned correction stub  (n^2,F)@(F,h) MLP + tanh, + fnp.interp
  G. kappa4 learned-local slice stub    second pointwise head
  H. per-layer guards + K2 bail-early   Python branching on computed values

Run (from whest-starterkit, which owns the venv):
  uv run python ../research/skeleton_v0.py
  uv run python ../research/skeleton_v0.py --force-trip 16   # SK-3
"""

from __future__ import annotations

import argparse
import time

import numpy as np  # OFFLINE ONLY: simulates shipped-npz tables (0-FLOP load)
import flopscope as flops
import flopscope.numpy as fnp

BUDGET = 2.72e11
FLOOR = 0.10 * BUDGET

# ---- architecture knobs (skeleton defaults sized to the plan's ledger) ----
R_MAX = 128          # carried CP directions cap
BIRTHS = 12          # new source directions born per layer
RECOMPRESS_EVERY = 4 # layers between QR+SVD recompressions
S_EXTRA = 8          # extra source-family contractions per layer (placeholders)
PW_HIDDEN = 8        # pointwise stub hidden width
EPS = 1e-3           # stub-correction magnitude (keeps state sane; physics later)
ENVELOPE = 1e3       # guard magnitude envelope (offline-fitted later)


def build_mlp_weights(n: int, depth: int, seed: int) -> list[np.ndarray]:
    """He-init weights, x @ W convention (matches starterkit)."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, n)).astype(np.float32) * np.sqrt(2.0 / n)
            for _ in range(depth)]


class Ledger:
    """Per-component FLOP accounting via BudgetContext deltas."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.rows: dict[str, float] = {}
        self._mark = ctx.flops_used

    def tick(self, name: str) -> None:
        now = self.ctx.flops_used
        self.rows[name] = self.rows.get(name, 0.0) + (now - self._mark)
        self._mark = now


def shipped_tables(n: int, depth: int, seed: int = 7):
    """Placeholder for the offline npz: random coefficient tables.
    fnp.asarray of host data models flopscope.Module.from_file (0 FLOPs)."""
    rng = np.random.default_rng(seed)
    t = {
        # pointwise stub heads: features F=6 -> hidden -> {k3 corr, k4 slice}
        "pw_w1": fnp.asarray(rng.standard_normal((6, PW_HIDDEN)).astype(np.float32) * 0.1),
        "pw_w2": fnp.asarray(rng.standard_normal((PW_HIDDEN, 2)).astype(np.float32) * 0.1),
        # 1-D gain table for the interp stub (identity-ish)
        "tab_x": fnp.asarray(np.linspace(-8, 8, 64).astype(np.float32)),
        "tab_y": fnp.asarray((np.linspace(-8, 8, 64) * 0.0 + 1.0).astype(np.float32)),
        # per-layer per-family coefficients for the S_EXTRA placeholder families
        "fam_c": fnp.asarray((rng.standard_normal((depth, S_EXTRA)) * 0.01).astype(np.float32)),
        # birth projector: which combinations of star columns seed new directions
        "birth_p": fnp.asarray(rng.standard_normal((n, BIRTHS)).astype(np.float32) / np.sqrt(n)),
    }
    return t


def k2_fallback(weights) -> fnp.ndarray:
    """Plain Gaussian (K=2) propagation, the bail-early target. ~2.2e9 FLOPs."""
    n = weights[0].shape[0]
    mu = fnp.zeros(n, dtype=fnp.float32)
    cov = fnp.eye(n, dtype=fnp.float32)
    for w_host in weights:
        w = fnp.asarray(w_host)
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
        w1 = cdf
        w2 = pdf / s
        cov = fnp.outer(w1, w1) * cov_pre + fnp.outer(w2, w2) * (cov_pre * cov_pre) * 0.5
        fnp.fill_diagonal(cov, var_post)
    return mu


def run_skeleton(weights, tables, ledger: Ledger, force_trip: int = -1):
    """One full closure pass. Returns (mu_final, tripped, layers_run)."""
    n = weights[0].shape[0]
    depth = len(weights)

    mu = fnp.zeros(n, dtype=fnp.float32)
    cov = fnp.eye(n, dtype=fnp.float32)
    V = fnp.zeros((n, BIRTHS), dtype=fnp.float32)   # lazy-CP carry directions
    c = fnp.zeros(BIRTHS, dtype=fnp.float32)        # their coefficients

    for layer, w_host in enumerate(weights):
        w = fnp.asarray(w_host)

        # ---- A. linear step + Gaussian ReLU gains -------------------------
        mu_pre = fnp.matmul(mu, w)
        cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)   # symmetric-tag idiom
        var = fnp.maximum(fnp.diag(cov_pre), 1e-10)
        s = fnp.sqrt(var)
        t = mu_pre / s
        pdf = flops.stats.norm.pdf(t)
        cdf = flops.stats.norm.cdf(t)
        h1 = cdf                       # E[ReLU'] gain
        h2 = pdf / s                   # 2nd Hermite-ish gain (shape-correct)
        ledger.tick("A_mu_sigma")

        # ---- F(i). table-interp stub on a per-neuron feature (op proof) ---
        gain_tweak = fnp.interp(t, tables["tab_x"], tables["tab_y"])
        h1 = h1 * gain_tweak
        ledger.tick("F_interp_stub")

        # ---- B. lazy-CP transport + births --------------------------------
        # transport: V <- W^T (h1 o V); one (n,n)@(n,R) matmul
        V = fnp.matmul(fnp.transpose(w), h1[:, None] * V)
        # births from star ingredient B = Sigma diag(h1) W (real cost, real shapes)
        Bm = fnp.matmul(cov_pre * h1[None, :], w)
        newV = fnp.matmul(Bm, tables["birth_p"])          # (n, BIRTHS)
        newc = fnp.full(BIRTHS, 1e-3, dtype=fnp.float32)
        V = fnp.concatenate([V, newV], axis=1)
        c = fnp.concatenate([c, newc], axis=0)
        ledger.tick("B_carry")

        # ---- C. recompression (cost proxy; symmetric-CP re-fit later) -----
        if V.shape[1] > R_MAX and layer % RECOMPRESS_EVERY == 0:
            Q, Rm = fnp.linalg.qr(V)                      # (n,R),(R,R)
            U, S, _ = fnp.linalg.svd(Rm * c[None, :])
            keep = min(R_MAX, U.shape[1])
            V = fnp.matmul(Q, U[:, :keep] * S[:keep][None, :])
            c = fnp.full(keep, 1.0, dtype=fnp.float32)
            ledger.tick("C_recompress")

        # ---- D. star slice, matmul+reshape only (NO einsum: symmetry trap)
        Cm = h2[:, None] * w * Bm
        Dm = h2[:, None] * (Bm * Bm)
        star = 2.0 * fnp.matmul(fnp.transpose(Cm), Bm) + fnp.matmul(fnp.transpose(Dm), w)
        ledger.tick("D_star")

        # ---- E. S extra source families (placeholder n^3 contractions) ----
        # CSE: the two Sigma*diag(h)*W bases are shared across all families
        # (first run measured 6.3% of B without sharing -- the plan's CSE rule,
        # demonstrated by the ledger). Each family is then ONE n^3 matmul.
        F1 = fnp.matmul(cov_pre * h1[None, :], w)
        F2 = fnp.matmul(cov_pre * h2[None, :], w)
        fam_acc = fnp.zeros((n, n), dtype=fnp.float32)
        for k in range(S_EXTRA):
            base = F2 if k % 2 else F1
            rhs = Bm if k % 3 == 0 else w
            ck = float(tables["fam_c"][layer, k])   # host scalar: avoids np auto-route
            fam_acc = fam_acc + ck * fnp.matmul(fnp.transpose(base), rhs)
        ledger.tick("E_families")

        # ---- slice extraction from the carry -------------------------------
        slice3 = fnp.matmul((V * V) * c[None, :], fnp.transpose(V)) + star + fam_acc
        ledger.tick("B_slice_extract")

        # ---- F(ii)+G. pointwise stub: k3 correction + k4 slice heads ------
        tp = fnp.broadcast_to(t[:, None], (n, n))
        tq = fnp.broadcast_to(t[None, :], (n, n))
        rho = cov_pre / fnp.outer(s, s)
        lf = fnp.full((n, n), (layer + 1.0) / depth, dtype=fnp.float32)
        feats = fnp.stack(
            [tp, tq, rho, slice3, fnp.asarray(star), lf], axis=-1
        ).reshape(n * n, 6)
        hid = fnp.tanh(fnp.matmul(feats, tables["pw_w1"]))
        heads = fnp.matmul(hid, tables["pw_w2"]).reshape(n, n, 2)
        slice3 = slice3 + EPS * heads[:, :, 0]
        k4slice = EPS * heads[:, :, 1]
        ledger.tick("FG_pointwise")

        # ---- apply cumulant corrections (shape-correct, magnitude EPS) ----
        dmu = EPS * fnp.sum(slice3 * rho, axis=1) / n
        mu = s * pdf + mu_pre * cdf + dmu
        ez2 = (mu_pre * mu_pre + var) * cdf + mu_pre * s * pdf
        var_post = fnp.maximum(ez2 - mu * mu, 1e-12) * (1.0 + EPS * fnp.diag(k4slice))
        corr = fnp.outer(h2, h1) * slice3
        cov = (fnp.outer(h1, h1) * cov_pre
               + fnp.outer(h2, h2) * (cov_pre * cov_pre) * 0.5
               + EPS * 0.5 * (corr + fnp.transpose(corr)))
        fnp.fill_diagonal(cov, var_post)
        # Explicit symmetrize + re-tag: keeps next layer's covariance einsum on
        # the symmetric-credit path (first run lost the tag every layer).
        # as_symmetric VALIDATES, so symmetrize first -- exact by commutativity.
        cov = flops.as_symmetric(0.5 * (cov + fnp.transpose(cov)), symmetry=(0, 1))
        ledger.tick("A_mu_sigma")

        # ---- H. guard (Python branch on computed values) -------------------
        if layer == force_trip:
            mu = mu * fnp.asarray(np.float32(np.nan))     # injected fault (SK-3)
        finite = bool(fnp.all(fnp.isfinite(mu))) and bool(fnp.all(fnp.isfinite(cov)))
        bounded = float(fnp.max(fnp.abs(mu))) < ENVELOPE and float(fnp.max(var_post)) < ENVELOPE
        ledger.tick("H_guards")
        if not (finite and bounded):
            return None, True, layer + 1

    return mu, False, depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-trip", type=int, default=-1,
                    help="inject NaN after this layer to exercise SK-3")
    args = ap.parse_args()

    weights = build_mlp_weights(args.width, args.depth, args.seed)

    # Symmetry-loss warnings were audited on the first run: cov is now re-tagged
    # explicitly (credit-bearing); the remaining losses (rho, feats, dmu, corr)
    # are consumed elementwise where no symmetric credit exists. Re-enable this
    # when porting new contractions -- it is the einsum-trap early-warning.
    flops.configure(symmetry_warnings=False)

    t0 = time.perf_counter()
    with flops.BudgetContext(flop_budget=int(BUDGET), quiet=True) as ctx:
        ledger = Ledger(ctx)
        tables = shipped_tables(args.width, args.depth)
        ledger.tick("_tables_load")
        mu, tripped, layers_run = run_skeleton(weights, tables, ledger,
                                               force_trip=args.force_trip)
        if tripped:
            mu = k2_fallback(weights)
            ledger.tick("H_fallback_k2")
        total = ctx.flops_used
    wall = time.perf_counter() - t0

    print(f"\n=== skeleton_v0 ledger (width={args.width} depth={args.depth} "
          f"R_max={R_MAX} births={BIRTHS} S_extra={S_EXTRA}) ===")
    for name in sorted(ledger.rows, key=ledger.rows.get, reverse=True):
        f = ledger.rows[name]
        print(f"  {name:<18} {f:>14,.0f}  ({100*f/BUDGET:5.2f}% of B)")
    print(f"  {'TOTAL':<18} {total:>14,.0f}  ({100*total/BUDGET:5.2f}% of B, "
          f"floor is 10.00%)")
    print(f"  wall {wall*1000:.0f} ms/MLP (local; grader multiple deferred)")
    print(f"  tripped={tripped} layers_run={layers_run}")

    ok_cost = total <= 0.075 * BUDGET if not tripped else total <= 0.10 * BUDGET
    ok_out = mu is not None and bool(fnp.all(fnp.isfinite(mu)))
    gate = "SK-3" if args.force_trip >= 0 else "SK-1/SK-2"
    print(f"\n  [{gate}] cost gate: {'PASS' if ok_cost else 'FAIL'}   "
          f"finite output: {'PASS' if ok_out else 'FAIL'}")


if __name__ == "__main__":
    main()
