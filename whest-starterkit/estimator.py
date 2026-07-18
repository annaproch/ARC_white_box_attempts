"""v5 (WIP) — grader-proven kappa3 closure + the LEARNED ANCHOR, fused with QMC.

STATUS, HONESTLY: the anchor is real but NOT yet good enough to beat v4.
  measured (A1, 8 MLPs, 4 held out, R=32 core carry -- the shippable rank):
      kappa3 slice error   32.0%  ->  23.5%   with the anchor (+DAgger)
      end-to-end b^2       3.0e-5 ->  1.95e-5
      fusion bar to beat v4 (3.396e-7):  b^2 < 1.02e-5      <-- STILL 1.9x SHORT
  This file carries the numbers we ACTUALLY MEASURED. v4 is preserved at
  _archive/estimator_v4.py and remains the best submission.

WHAT IS PROVEN (STRATEGY.md 7):
  * The moment update reads ONLY the kappa3 slice (n^2) -> that slice IS the
    whole problem. Oracle slice => 3.8-4.5e-7 end-to-end (beats every leader).
  * The core (Tucker) carry is verified: 25.7% exact, 34% at R=32 (D1/D2 --
    three independent codepaths agree).
  * The anchor (per-layer poly map on [t_i, t_j, rho, carried_slice, star])
    beats an optimal per-layer gain => the residual E13 called "a shape error a
    scalar cannot touch" IS partially learnable.
  * kappa4 slices are learnable and worth ~16x in the moment update (G1c).

TO CROSS THE BAR the slice must reach ~12-15% (from 23.5%). Untried levers:
more training MLPs (A1 used FOUR of 64 available), richer features (kappa4
slice, h1/h2 gains, Sigma spectrum -- all already computed => free at inference).

All learned numbers live in v5.npz (0-FLOP load) => refit without code changes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import flopscope
import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator

# The grader imports estimator.py BY PATH and does not add its directory to
# sys.path, so closure.py must be loaded by path -- and never fatally.
closure = None
try:
    import closure  # noqa: F811
except Exception:  # pragma: no cover - grader path
    try:
        _p = Path(__file__).resolve().parent / "closure.py"
        if _p.exists():
            _spec = importlib.util.spec_from_file_location("closure", _p)
            closure = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(closure)
    except Exception:
        closure = None

V5_FILE = "v5.npz"
POINTS_FILE = "points.npz"
ENVELOPE = 10.0


class V5Params(flopscope.Module):
    """theta  : (depth, N_PARAMS)  closure calibration (zeros = uncalibrated)
    anchor : (depth, N_ANCHOR)  per-layer kappa3-slice correction  <- the A1 fit
    q0     : subspace-iteration start block
    meta   : [rank, rank4, n_qmc, w_qmc, anchor_on, k4_on]
    """

    def __init__(self) -> None:
        self.theta = fnp.zeros((1, 1))
        self.anchor = fnp.zeros((1, 1))
        self.q0 = fnp.zeros((1, 1))
        self.meta = fnp.zeros((6,))


class Points(flopscope.Module):
    def __init__(self) -> None:
        self.x = fnp.zeros((1, 1))


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._p = None
        self._points = None
        # config parsed ONCE in setup() -> no per-MLP scalar syncs in predict()
        self._rank = 0
        self._rank4 = 8
        self._n_qmc = 0
        self._w_qmc = 1.0            # no params => pure QMC
        self._no_k4 = True
        self._anchor = None

    def setup(self, ctx) -> None:
        if getattr(ctx, "submission_dir", None) is None:
            return
        d = Path(ctx.submission_dir)
        if (d / POINTS_FILE).exists():
            self._points = Points.from_file(str(d / POINTS_FILE)).x
        if (d / V5_FILE).exists():
            self._p = V5Params.from_file(str(d / V5_FILE))
            import hashlib, sys
            h = hashlib.md5((d / V5_FILE).read_bytes()).hexdigest()[:8]
            m = self._p.meta
            print(f"[v5] {V5_FILE} md5={h} rank={float(m[0]):.0f} "
                  f"rank4={float(m[1]):.0f} n_qmc={float(m[2]):.0f} "
                  f"w_qmc={float(m[3]):.2f} anchor={'ON' if float(m[4])>0.5 else 'OFF'} "
                  f"k4={'ON' if float(m[5])>0.5 else 'OFF'} "
                  f"theta_absmax={float(fnp.max(fnp.abs(self._p.theta))):.4f}",
                  file=sys.stderr, flush=True)
            # hoist every meta scalar sync here (once), out of predict()
            self._rank = int(float(m[0]))
            self._rank4 = int(float(m[1]))
            self._n_qmc = int(float(m[2]))
            self._w_qmc = float(m[3])
            self._no_k4 = float(m[5]) < 0.5
            self._anchor = self._p.anchor if float(m[4]) > 0.5 else None

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        n, depth = mlp.width, mlp.depth
        p = self._p
        rows = None

        if p is not None and closure is not None and p.theta.shape[0] == depth:
            out = closure.predict_all_layers(
                mlp, p.theta, self._rank, self._rank4,
                p.q0, no_k4=self._no_k4, anchor=self._anchor)
            if out is not None:
                # ONE guard on the whole stacked output (2 syncs, not per-layer)
                chk = fnp.asarray(out, dtype=fnp.float64)
                if (bool(fnp.all(fnp.isfinite(chk)))
                        and float(fnp.max(fnp.abs(chk))) < ENVELOPE):
                    if self._w_qmc <= 0.0:
                        return out                 # pure closure: no restack, no QMC
                    rows = [out[i] for i in range(depth)]

        if rows is None:                    # guard tripped / no params => QMC only
            rows = [fnp.zeros(n, dtype=fnp.float32) for _ in range(depth)]
            w_qmc = 1.0
        else:
            w_qmc = self._w_qmc

        # QMC on the remaining floor, fused at every layer. w_qmc is fitted
        # OFFLINE on held-out MLPs against SCORE, so a weak closure drives
        # w_qmc -> 1 and we degrade to pure QMC, never below it.
        pts = self._points
        if pts is not None and pts.shape[1] == n and w_qmc > 0.0:
            # cap: on a guard trip (w_qmc==1, closure unusable) FILL THE FLOOR --
            # the n_qmc knob only throttles fusion on healthy MLPs, it must never
            # starve the emergency fallback down to a single sample (n_qmc=0 -> k=1).
            cap = pts.shape[0] if w_qmc >= 1.0 else self._n_qmc
            per = depth * (2 * n * n + 2 * n)
            k = max(min(int(0.062 * budget / per), pts.shape[0], cap), 1)
            x = pts[:k]
            mc = []
            for w in mlp.weights:
                x = fnp.maximum(fnp.matmul(x, w), 0.0)
                mc.append(fnp.mean(x, axis=0))
            for i in range(depth):
                rows[i] = w_qmc * mc[i] + (1.0 - w_qmc) * rows[i]
        return fnp.stack(rows, axis=0)
