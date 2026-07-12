"""Your estimator. Edit `predict()`. Run `python estimator.py` to iterate.

Stage 1 of the WhestBench ladder: just `flopscope` and the local engine. No CLI
knowledge required. Once `predict()` returns something interesting, climb to
Stage 2: `whest validate --estimator estimator.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from pathlib import Path

import flopscope
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

# The harness imports estimator.py by path and does NOT put its directory on
# sys.path, so a bare `import closure` can fail on the grader. Load it by path
# relative to this file, falling back to a plain import.
try:
    import closure
except ModuleNotFoundError:  # pragma: no cover - grader path
    _spec = importlib.util.spec_from_file_location(
        "closure", Path(__file__).resolve().parent / "closure.py")
    closure = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(closure)


def _closure_flops(n: int, depth: int, r: int, r4: int = 8) -> float:
    """Analytic FLOP estimate for the K=2+kappa3+kappa4 closure. Dominated by the
    O(n^3) covariance and star matmuls; kappa3's core adds O(n r^3) and kappa4's
    O(n r^4) -- which is why r4 is kept small."""
    per_layer = (6.0 * n ** 3 + 4.0 * n * n * r
                 + 2.0 * n * r ** 3 + 2.0 * r ** 4
                 + 2.0 * n * r4 ** 4 + 2.0 * r4 ** 5)
    return depth * per_layer

POINTS_FILE = "points.npz"
CLOSURE_FILE = "closure.npz"


class ClosureParams(flopscope.Module):
    """Calibrated coefficients for the analytic K=2 + kappa3 + kappa4 closure
    (produced by `research/calibrate_k3k4.py --export`). Offline => 0-FLOP load.

    theta : (depth, 13) per-layer coefficients
    meta  : [rank, w_fuse]
            w_fuse = weight on the closure when fusing with QMC at the final
            layer. 0 => pure QMC (exactly v4's behaviour); 1 => pure analytic.
            Set offline from held-out MLPs, so a weak closure cannot regress us.
    """

    def __init__(self) -> None:
        self.theta = fnp.zeros((1, 1))
        self.meta = fnp.zeros((2,))
        self.q0 = fnp.zeros((1, 1))     # subspace-iteration start block (shipped)


# Per-layer closure corrections (p, q, v, g1, g2, m) for calibrated K=2
# covariance propagation, learned offline end-to-end (fresh random MLP per
# gradient step; removes 66% of the Gaussian-closure error: held-out MSE
# 7.2e-5 -> 2.4e-5). Shape (32, 6); only valid for the phase-1 depth.
_THETA = [
    [+0.000473, +0.000000, -0.001964, -0.000713, +0.000232, +0.000473],
    [+0.000173, -0.000015, -0.001929, -0.000808, +0.000032, +0.000083],
    [+0.000671, -0.000124, -0.001951, -0.000798, +0.000408, +0.000374],
    [+0.000454, -0.000049, -0.001989, -0.000630, +0.000494, +0.000074],
    [+0.000357, -0.000083, -0.001994, -0.000467, +0.000050, +0.000061],
    [+0.000515, -0.000060, -0.002048, -0.000417, -0.000179, +0.000007],
    [+0.000320, -0.000105, -0.002091, -0.000188, +0.000046, -0.000191],
    [+0.000555, +0.000018, -0.002151, +0.000220, -0.000128, -0.000137],
    [-0.000026, -0.000063, -0.002136, +0.000118, -0.000419, -0.000293],
    [+0.000250, +0.000021, -0.002066, +0.000209, -0.000739, +0.000002],
    [+0.000111, -0.000031, -0.002112, +0.000313, -0.000530, -0.000281],
    [+0.000226, +0.000024, -0.002169, +0.000624, -0.000339, -0.000174],
    [+0.000062, +0.000093, -0.002119, +0.000732, -0.001215, -0.000224],
    [+0.000220, -0.000053, -0.002140, +0.000719, -0.000674, -0.000200],
    [+0.000239, +0.000104, -0.002135, +0.000886, -0.000800, -0.000238],
    [+0.000130, -0.000037, -0.002103, +0.000929, -0.001192, -0.000217],
    [+0.000143, +0.000038, -0.002097, +0.001043, -0.001277, -0.000181],
    [-0.000111, +0.000019, -0.002108, +0.001309, -0.001036, -0.000326],
    [-0.000030, +0.000046, -0.002103, +0.001365, -0.001177, -0.000403],
    [-0.000018, +0.000009, -0.002086, +0.001389, -0.001105, -0.000263],
    [-0.000102, +0.000112, -0.002128, +0.001597, -0.000819, -0.000424],
    [-0.000366, +0.000033, -0.001990, +0.001609, -0.001641, -0.000359],
    [-0.000491, +0.000040, -0.001985, +0.001811, -0.001264, -0.000456],
    [-0.000478, +0.000033, -0.001958, +0.001918, -0.000472, -0.000381],
    [-0.000084, +0.000026, -0.001977, +0.002018, -0.001692, -0.000431],
    [-0.000444, -0.000003, -0.001977, +0.002096, -0.001170, -0.000628],
    [-0.000250, -0.000007, -0.002022, +0.002193, -0.001108, -0.000508],
    [-0.000141, +0.000032, -0.002153, +0.002442, -0.001787, -0.000480],
    [-0.000305, -0.000041, -0.002235, +0.002522, -0.001446, -0.000553],
    [-0.000059, -0.000028, -0.002604, +0.002938, -0.000908, -0.000494],
    [-0.000347, -0.000145, -0.003678, +0.004106, -0.001495, -0.000448],
    [-0.003080, -0.001935, +0.000000, +0.000000, +0.000000, -0.001290],
]

# Global fusion weight on the calibrated-K2 estimate at the final layer,
# fit offline on 24 MLPs, validated on 24 held-out (errors independent:
# corr -0.04; fused val MSE 3.44e-6 vs 4.21e-6 QMC-only).
_W_K2 = 0.137


class Points(flopscope.Module):
    """Shipped QMC point set: scrambled Sobol -> Gaussian (inverse CDF),
    whitened to identity covariance, then projected to the sphere of radius
    E||x|| = sqrt(2)*Gamma(128.5)/Gamma(128). The network has no biases, so it
    is positively 1-homogeneous and the radial integral is exact — only
    directional discrepancy remains. Computed offline (off-budget); loading
    via ``from_file`` costs 0 FLOPs."""

    def __init__(self) -> None:
        self.x = fnp.zeros((1, 1))


class Estimator(BaseEstimator):
    """Quasi-Monte Carlo at the score-multiplier floor.

    The multiplier `max(0.1, C/B)` makes compute below 10% of budget free, and
    at depth 32 the final-layer activation variance is small (~0.012), so
    sampling at the floor beats every in-budget analytic method we measured
    (kprop K=2 ~8e-5 avg; factorized K=3 exceeds the budget at this depth;
    MC+K=2 fusion loses to pure MC even with oracle weights).

    Measured at 256x32 over 8 MLPs (6.3k points): iid MC 8.4e-6 final-layer
    MSE; shipped moment-matched Sobol points 2.7e-6 (~3.2x better). Antithetic
    pairing gave ~no gain (the output has almost no odd dependence on the
    input after 32 ReLU layers), so plain forward passes over the fixed point
    set are optimal here.
    """

    # Uses all 6400 shipped points (~10.06% grader utilization). The floor is
    # not a cliff: above it the multiplier is linear and trimmed-set MSE is
    # ~linear in 1/N, so the score surface is flat to first order — measured:
    # trimming to 9.95% traded +1.1% MSE for a 0.6% multiplier refund (net
    # +0.5% worse). Only the 100% budget line is a hard cliff.
    _BUDGET_FRACTION = 0.0991

    def __init__(self) -> None:
        self._setup_rng = None  # set from ctx.seed inside setup()
        self._points = None
        self._theta = None      # (depth, 13) calibrated K3/K4 closure coefficients
        self._rank = 0
        self._rank4 = 8
        self._q0 = None
        self._n_pts = 0         # QMC points to use alongside the closure
        self._w = None          # 3-way fusion weights [QMC, K2, K3K4], sum = 1

    def setup(self, ctx) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)
        if getattr(ctx, "submission_dir", None) is None:
            return
        # str, not Path: the grader's flopscope-client requires a string.
        d = Path(ctx.submission_dir)
        if (d / POINTS_FILE).exists():
            self._points = Points.from_file(str(d / POINTS_FILE)).x
        if (d / CLOSURE_FILE).exists():
            cp = ClosureParams.from_file(str(d / CLOSURE_FILE))
            meta = fnp.asarray(cp.meta)
            # [rank, n_pts, w_qmc, w_k2, w_k3k4]; written only when the offline
            # held-out fit says the 3-way beats v4 on SCORE (accuracy x compute).
            if meta.shape[0] >= 5:
                self._theta = cp.theta
                self._rank = int(float(meta[0]))
                self._n_pts = int(float(meta[1]))
                self._w = [float(meta[2]), float(meta[3]), float(meta[4])]
                self._rank4 = int(float(meta[5])) if meta.shape[0] >= 6 else 8
                self._q0 = cp.q0

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width, depth = mlp.width, mlp.depth
        pts = self._points
        have_pts = pts is not None and pts.shape[1] == width
        shape_ok = depth == len(_THETA) and have_pts

        # PURE-ANALYTIC mode: w_qmc == 0 => no sampling at all. The closure
        # already computes mu at every layer, so it fills all rows on its own.
        if (self._theta is not None and self._w is not None
                and abs(self._w[0]) < 1e-6 and depth == self._theta.shape[0]):
            out = closure.predict_all_layers(mlp, self._theta, self._rank,
                                             self._rank4, self._q0)
            if out is not None:
                chk = fnp.asarray(out, dtype=fnp.float64)
                if (bool(fnp.all(fnp.isfinite(chk)))
                        and float(fnp.max(fnp.abs(chk))) < 10.0):
                    return out
            # fall through to the sampling path if the closure misbehaved

        # Three-way fusion when calibrated params shipped: QMC + calibrated-K2 +
        # the K3/K4 closure. Weights and the QMC point count are fitted offline on
        # held-out MLPs against the SCORE (accuracy x compute), and the file is
        # only written when that beats v4 -- so this path cannot regress us.
        analytic = None
        if self._theta is not None and shape_ok:
            analytic = closure.predict_final_mean(mlp, self._theta, self._rank,
                                                  self._rank4, self._q0)
            if analytic is not None:
                chk = fnp.asarray(analytic, dtype=fnp.float64)
                if not bool(fnp.all(fnp.isfinite(chk))) or float(fnp.max(fnp.abs(chk))) > 10.0:
                    analytic = None            # never ship a diverged closure

        # Per-sample cost: matmul (2n^2) + relu (n) + mean accumulation (n) per layer.
        flops_per_sample = depth * (2 * width * width + 2 * width)
        if have_pts:
            n = max(min(int(self._BUDGET_FRACTION * budget / flops_per_sample),
                        pts.shape[0]), 1)
            if analytic is not None and self._n_pts > 0:
                n = min(n, self._n_pts)        # pay for the closure with QMC points
            x = pts[:n]
        else:
            # Fallback for shapes the shipped points don't cover (+16 FLOPs/elem RNG).
            rng = fnp.random.default_rng(mlp.seed)
            n = max(int(self._BUDGET_FRACTION * budget
                        / (flops_per_sample + 16 * width)), 1)
            x = rng.standard_normal((n, width)).astype(fnp.float32)

        rows = []
        for w in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            rows.append(fnp.mean(x, axis=0))

        if shape_ok:
            # Only run K2 if it actually carries weight. The K3/K4 closure
            # supersedes it (same covariance propagation, plus cumulants), and the
            # held-out fit sets w_k2 = 0. Computing it anyway would burn 2.2e9
            # FLOPs (0.8% of budget) on a value multiplied by zero -- enough to
            # push us from 9.6% back over the 10% multiplier floor.
            wq, wk2, wc = self._w if self._w is not None else (0.0, 0.0, 0.0)
            need_k2 = analytic is None or self._w is None or abs(wk2) > 1e-6
            k2, k2_ok = None, False
            if need_k2:
                k2 = self._calibrated_k2_final(mlp)
                k2_np = fnp.asarray(k2, dtype=fnp.float64)
                k2_ok = (bool(fnp.all(fnp.isfinite(k2_np)))
                         and float(fnp.max(fnp.abs(k2_np))) < 10.0)

            if analytic is not None and self._w is not None:
                if abs(wk2) > 1e-6 and k2_ok:
                    rows[-1] = wq * rows[-1] + wk2 * k2 + wc * analytic
                else:  # renormalise over the components actually used
                    tot = wq + wc
                    rows[-1] = (wq / tot) * rows[-1] + (wc / tot) * analytic
            elif k2_ok:
                # v4 fallback: QMC + calibrated-K2 at the offline-fitted weight.
                rows[-1] = (1.0 - _W_K2) * rows[-1] + _W_K2 * k2
        return fnp.stack(rows, axis=0)

    def _calibrated_k2_final(self, mlp: MLP) -> fnp.ndarray:
        """Covariance propagation (K=2, Wick series to 2nd order off-diagonal)
        with offline-learned per-layer closure corrections. ~2e9 FLOPs."""
        width = mlp.width
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = fnp.eye(width, dtype=fnp.float32)
        for layer, w in enumerate(mlp.weights):
            p, q, v, g1, g2, m = _THETA[layer]
            mu_pre = fnp.matmul(mu, w)
            # einsum keeps the symmetric tag (see examples/03) and its FLOP credit.
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)
            var = fnp.maximum(fnp.diag(cov_pre), 1e-10)
            s = fnp.sqrt(var)
            a = mu_pre / s
            pdf = flops.stats.norm.pdf(a)
            cdf = flops.stats.norm.cdf(a)
            mu = s * pdf * (1.0 + p) + mu_pre * cdf * (1.0 + q) + m * s
            ez2 = (mu_pre * mu_pre + var) * cdf + mu_pre * s * pdf
            var_post = fnp.maximum(ez2 - mu * mu, 1e-12) * (1.0 + v)
            w1 = cdf * (1.0 + g1)
            w2 = (pdf / s) * (1.0 + g2)
            cov = fnp.outer(w1, w1) * cov_pre + fnp.outer(w2, w2) * (cov_pre * cov_pre) * 0.5
            fnp.fill_diagonal(cov, var_post)
        return mu


def _load_baseline(name: str) -> type[BaseEstimator]:
    """Load the `Estimator` class from `examples/<name>.py` or `examples/0N_<name>.py`."""
    examples_dir = Path(__file__).resolve().parent / "examples"
    candidates = [examples_dir / f"{name}.py", *examples_dir.glob(f"??_{name}.py")]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.Estimator
    raise SystemExit(
        f"\n[whest-starterkit] Could not find baseline `{name}` in examples/.\n"
        f"Available: {sorted(p.name for p in examples_dir.glob('*.py'))}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterate on your estimator locally.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Compare your estimator against an example: 'random', 'mean_propagation', "
        "or 'covariance_propagation'.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)  # phase-1 competition shape (warmup was 8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from local_engine import build_mlp, compare_against_monte_carlo

    mlp = build_mlp(width=args.width, depth=args.depth, seed=args.seed)

    print("--- Your estimator ---")
    compare_against_monte_carlo(Estimator(), mlp)

    if args.baseline:
        baseline_cls = _load_baseline(args.baseline)
        print(f"\n--- Baseline: {args.baseline} ---")
        compare_against_monte_carlo(baseline_cls(), mlp)
