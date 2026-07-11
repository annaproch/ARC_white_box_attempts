"""Your estimator. Edit `predict()`. Run `python estimator.py` to iterate.

Stage 1 of the WhestBench ladder: just `flopscope` and the local engine. No CLI
knowledge required. Once `predict()` returns something interesting, climb to
Stage 2: `whest validate --estimator estimator.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator


class Estimator(BaseEstimator):
    """Antithetic Monte Carlo sized to the score-multiplier floor.

    The leaderboard multiplier `max(0.1, C/B)` makes all compute below 10% of
    the budget free, and at depth 32 the final-layer activation variance is
    small (~0.012), so plain sampling at the floor (~2-5e-6 final-layer MSE)
    beats every in-budget analytic method we measured (kprop K=2 ~4.7e-5;
    factorized K=3 exceeds the budget at this depth). Antithetic pairs (x, -x)
    remove the odd component of the integrand for ~free variance reduction.
    """

    # Aim below the 10% floor: margin for RNG FLOPs, reduction ops, and the
    # residual wall-time bucket (lambda * R_m) that also counts toward C_m.
    _BUDGET_FRACTION = 0.09

    def __init__(self) -> None:
        self._setup_rng = None  # set from ctx.seed inside setup()

    def setup(self, ctx) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        rng = fnp.random.default_rng(mlp.seed)
        width, depth = mlp.width, mlp.depth

        # Per-sample cost: matmul (2n^2) + relu (n) + mean accumulation (n)
        # per layer, plus ~16 FLOPs/element to draw the input.
        flops_per_sample = depth * (2 * width * width + 2 * width) + 16 * width
        n_pairs = max(int(self._BUDGET_FRACTION * budget / (2 * flops_per_sample)), 1)

        x = rng.standard_normal((n_pairs, width)).astype(fnp.float32)
        x = fnp.concatenate([x, -x], axis=0)  # antithetic pairs

        rows = []
        for w in mlp.weights:
            x = fnp.maximum(fnp.matmul(x, w), 0.0)
            rows.append(fnp.mean(x, axis=0))
        return fnp.stack(rows, axis=0)


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
