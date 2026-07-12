"""Price v5 with flopscope BEFORE any fitting. Placeholder constants; the
question is COST, not accuracy. Writes a provisional v5.npz so the estimator
runs its full op profile."""
from __future__ import annotations

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

import estimator_v5 as E

BUDGET = 2.72e11
n, depth = 256, 32
rng = np.random.default_rng(0)

# --- provisional constants (structure final, numbers to be refit offline) ---
p = E.V5Params()
p.anchor_w1 = fnp.asarray((rng.standard_normal((7, 8)) * 0.05).astype(np.float32))
p.anchor_w2 = fnp.asarray((rng.standard_normal((8, 1)) * 0.05).astype(np.float32))
p.k4_w1 = fnp.asarray((rng.standard_normal((7, 8)) * 0.05).astype(np.float32))
p.k4_w2 = fnp.asarray((rng.standard_normal((8, 1)) * 0.05).astype(np.float32))
p.birth_p = fnp.asarray((rng.standard_normal((n, E.BIRTHS)) / np.sqrt(n)).astype(np.float32))
p.gains = fnp.asarray(np.zeros((depth, 4), dtype=np.float32))
#            [rank, n_qmc, w_qmc, w_closure, anchor_on, k4_on]
p.meta = fnp.asarray(np.array([E.RANK, 4000, 0.5, 0.5, 1.0, 1.0], dtype=np.float32))
p.save("v5.npz")
print("wrote provisional v5.npz")


class _MLP:
    def __init__(self, w, d):
        self.width, self.depth = w, d
        g = np.random.default_rng(1)
        self.weights = [fnp.asarray((g.standard_normal((w, w)) * np.sqrt(2.0 / w))
                                    .astype(np.float32)) for _ in range(d)]
        self.seed = 1


class _Ctx:
    submission_dir = "."
    seed = 0


flops.configure(symmetry_warnings=False)
mlp = _MLP(n, depth)
est = E.Estimator()

with flops.BudgetContext(flop_budget=int(BUDGET), quiet=True) as ctx:
    est.setup(_Ctx())
    after_setup = ctx.flops_used
    out = est.predict(mlp, int(BUDGET))
    total = ctx.flops_used

print(f"\n=== v5 PRICED (flopscope, measured) ===")
print(f"  setup / npz load : {after_setup:>14,.0f}  ({100*after_setup/BUDGET:5.2f}% of B)")
print(f"  predict          : {total-after_setup:>14,.0f}  ({100*(total-after_setup)/BUDGET:5.2f}% of B)")
print(f"  TOTAL            : {total:>14,.0f}  ({100*total/BUDGET:5.2f}% of B)")
print(f"  output shape     : {tuple(out.shape)}  finite: {bool(fnp.all(fnp.isfinite(out)))}")
print(f"\n  floor = 10.00% of B (compute below it is FREE)")
print(f"  verdict: {'IN BUDGET (under the free floor)' if total <= 0.10*BUDGET else 'OVER THE FLOOR -- trim'}")
