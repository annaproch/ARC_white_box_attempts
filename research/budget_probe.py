"""BUDGET AUDIT — measure, never estimate.

IMPROVEMENTS.md records a 2.7x analytic FLOP miscount that lost a submission,
and our pure-analytic run was charged 8.1% effective vs 4.2% analytical (the
grader's residual wall-time markup). So before committing to an architecture,
count the complete-formula closure's FLOPs with flopscope itself.

Components priced separately, at the phase-1 shape (n=256, L=32, B=2.72e11):
  1. covariance propagation      W^T Sigma W      (symmetric einsum)
  2. the complete diagram series (10 families x p<=PMAX, n^2 elementwise)
  3. kappa3 propagation, symmetric CP rank R      (u -> W^T diag(h1) u)
  4. kappa3 slice extraction     k3_iij = sum_r lam_r u_ri^2 u_rj
  5. kappa4, crude, rank R4
Run:  uv run python research/budget_probe.py
"""
from __future__ import annotations

import sys

import flopscope
import flopscope.numpy as fnp

N, L, B = 256, 32, 2.72e11
PMAX = 6


def cost(label, fn, *a, **k):
    with flopscope.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        fn(*a, **k)
        f = ctx.flops_used
        r = ctx.residual_wall_time_s
    print(f"  {label:<44} {f:>12,.0f}  {f/B*100:>6.2f}%  "
          f"(wall {(ctx.wall_time_s or 0)*1000:6.0f} ms, residual {(r or 0)*1000:5.1f} ms)")
    return f


def cov_prop():
    cov = fnp.eye(N, dtype=fnp.float64)
    for _ in range(L):
        w = fnp.ones((N, N), dtype=fnp.float64) * 0.01
        cov = fnp.einsum("ij,ia,jb->ab", cov, w, w)


def diagram_series():
    """10 families x PMAX terms: outer(g_a, g_b) * K * S^p / p!  -- all n^2."""
    S = fnp.ones((N, N), dtype=fnp.float64) * 0.01
    K = fnp.ones((N, N), dtype=fnp.float64) * 0.01
    g = [fnp.ones(N, dtype=fnp.float64) for _ in range(16)]
    for _ in range(L):
        Sp = fnp.ones((N, N), dtype=fnp.float64)
        acc = fnp.zeros((N, N), dtype=fnp.float64)
        for p in range(PMAX + 1):
            for fam in range(10):
                acc = acc + fnp.outer(g[p], g[p + 1]) * K * Sp
            Sp = Sp * S


def k3_cp(R):
    """CP: pass-through u -> W^T (h1 * u), then slice k3_iij = (U^2 L) U^T."""
    U = fnp.ones((N, R), dtype=fnp.float64) * 0.01
    lam = fnp.ones(R, dtype=fnp.float64)
    for _ in range(L):
        w = fnp.ones((N, N), dtype=fnp.float64) * 0.01
        h1 = fnp.ones(N, dtype=fnp.float64)
        U = fnp.matmul(w.T, U * h1[:, None])          # n^2 R
        U2 = U * U                                     # n R
        _k3_iij = fnp.matmul(U2 * lam[None, :], U.T)   # n^2 R
        _k3_iii = fnp.sum(U2 * U * lam[None, :], axis=1)


def k4_cp(R4):
    """kappa4 only needs 10-20% accuracy -> small rank."""
    U = fnp.ones((N, R4), dtype=fnp.float64) * 0.01
    for _ in range(L):
        w = fnp.ones((N, N), dtype=fnp.float64) * 0.01
        h1 = fnp.ones(N, dtype=fnp.float64)
        U = fnp.matmul(w.T, U * h1[:, None])
        U2 = U * U
        _k4_iijj = fnp.matmul(U2, U2.T)
        _k4_iiij = fnp.matmul(U2 * U, U.T)
        _k4_iiii = fnp.sum(U2 * U2, axis=1)


def star():
    """Freshly generated kappa3 (star diagram): B = Sigma diag(h1) W, then
    two n^3 matmuls for the slice."""
    for _ in range(L):
        S = fnp.ones((N, N), dtype=fnp.float64) * 0.01
        w = fnp.ones((N, N), dtype=fnp.float64) * 0.01
        Bm = fnp.matmul(S, w)                 # 2n^3
        C = Bm * w
        _ = fnp.matmul(C.T, Bm)               # 2n^3
        _ = fnp.matmul((Bm * Bm).T, w)        # 2n^3


if __name__ == "__main__":
    print(f"BUDGET AUDIT | n={N} L={L} B={B:.2e} | floor = 10% = {0.1*B:.2e}\n")
    tot = 0
    tot += cost("1. covariance propagation W^T S W", cov_prop)
    tot += cost(f"2. complete diagram series (10 fam x p<={PMAX})", diagram_series)
    tot += cost("3. star (fresh kappa3 generation)", star)
    ks = {}
    for R in (16, 32, 64, 128, 256):
        ks[R] = cost(f"4. kappa3 CP propagation + slices, R={R:<3d}", k3_cp, R)
    k4c = cost("5. kappa4 CP, R4=16 (crude is fine)", k4_cp, 16)

    print(f"\n  fixed cost (1+2+3+5)                        {tot+k4c:>12,.0f}  "
          f"{(tot+k4c)/B*100:>6.2f}%")
    print("\n  TOTAL by kappa3 rank (the accuracy knob):")
    for R, f in ks.items():
        t = tot + k4c + f
        print(f"    R={R:<4d} {t:>13,.0f}  {t/B*100:>6.2f}% of budget"
              f"   {'OK (under 10% floor)' if t < 0.1*B else '** OVER FLOOR **'}")
    print("\n  NB: grader adds residual wall-time at 1e11 FLOP/s. Our pure-analytic")
    print("      run was charged 8.1% effective vs 4.2% analytical (~2x markup).")
