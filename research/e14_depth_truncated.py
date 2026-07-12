"""E14 — DEPTH-TRUNCATED SAMPLING (STRATEGY.md 4.12).

Propagate the distribution analytically to layer k, SAMPLE z^(k) from it, then
run only the remaining layers EXACTLY.

  * cost/sample = (32-k) * 2n^2   [ generation z = mu + L xi, then 31-k matmuls ]
    k=0 is exactly plain MC (z^(0) = W0^T x is EXACTLY Gaussian -> no bias)
  * layers k..31 have ZERO closure error -- the ReLUs there are exact
  * bias = the closure's error in (mu_z, Sigma_z) at layer k, PLUS the
    non-Gaussianity of z^(k). It has not compounded to layer 32 yet.

We sample z (the PRE-activation), not a: a = ReLU(.) is non-negative with an
atom at 0 and is badly non-Gaussian, while z is a sum of 256 terms and is close
to Gaussian (S1: RMS skew 0.43, marginally near-Gaussian).

Two bias sources are measured separately:
  TRUE   (mu_z*, Sigma_z*) from the corpus -> the pure Gaussianisation bias (the
         FLOOR of the method, independent of how good our closure is)
  CLOSURE(mu_hat, Sigma_hat) from the K=2 complete-formula closure -> realistic

MSE(k) = bias(k)^2 + Var/N(k),  N(k) = (floor budget - closure cost) / cost(k)
"""
from __future__ import annotations

import math
import statistics as st
import sys
import time

import torch

from harness import DT, load_corpus, mse, propagate

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256
B_FLOOR = 2.72e10            # the 10% multiplier floor
COST_SAMPLE = 2 * n * n      # per layer, per sample (matches the MC reference)
CLOSURE_32 = 2.2e9           # measured (budget_probe): cov prop + diagram series
KS = [0, 8, 16, 20, 24, 26, 28, 29, 30, 31]
NBIG = 400_000
CHUNK = 50_000


def sample_forward(mu_z, cov_z, Ws_rest, N, seed):
    """Sample z ~ N(mu_z, cov_z), ReLU, then run the remaining layers exactly.
    Returns the mean of the final activations and their per-neuron variance."""
    # Cholesky fails: the corpus covariance is a 2e6-sample estimate and the
    # closure's can have tiny negative eigenvalues. Use eigh with clamping --
    # a valid square root of the nearest PSD matrix.
    ev, evec = torch.linalg.eigh(cov_z)
    L = evec @ torch.diag(torch.sqrt(torch.clamp(ev, min=0.0)))
    Ld = L.to(DEV, F32)
    mud = mu_z.to(DEV, F32)
    Wd = [W.to(DEV, F32) for W in Ws_rest]
    g = torch.Generator(device="cpu").manual_seed(seed)
    s1 = torch.zeros(n, dtype=torch.float64)
    s2 = torch.zeros(n, dtype=torch.float64)
    done = 0
    while done < N:
        b = min(CHUNK, N - done)
        xi = torch.randn(b, n, generator=g, dtype=F32).to(DEV)
        A = torch.clamp(xi @ Ld.T + mud, min=0.0)        # ReLU(z^(k))
        for W in Wd:
            A = torch.clamp(A @ W, min=0.0)
        Ad = A.cpu().double()
        s1 += Ad.sum(0)
        s2 += (Ad * Ad).sum(0)
        done += b
    m = s1 / N
    v = s2 / N - m * m
    return m, v


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    recs = load_corpus(M)
    print(f"E14 | {M} MLPs | {NBIG:,} samples for the bias | device {DEV}\n",
          flush=True)

    rows = []
    for k in KS:
        t0 = time.time()
        b_true, b_clo, varr = [], [], []
        for m in range(M):
            rec = recs[m]
            Ws = [rec["W"][l] for l in range(32)]
            truth = rec["mu"][31]

            # --- TRUE state at layer k (corpus)
            mu_t, cov_t = rec["zmu"][k], rec["zcov"][k]
            # --- CLOSURE state at layer k (plain K=2, the one we already have)
            if k == 0:
                mu_c = torch.zeros(n, dtype=DT)
                cov_c = Ws[0].T @ Ws[0]
            else:
                mu_a, cov_a = torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT)
                mu_a, cov_a = propagate(mu_a, cov_a, Ws[:k])
                mu_c = mu_a @ Ws[k]
                cov_c = Ws[k].T @ cov_a @ Ws[k]

            rest = Ws[k + 1:]
            mt, vt = sample_forward(mu_t, cov_t, rest, NBIG, 1000 + m)
            mc, _ = sample_forward(mu_c, cov_c, rest, NBIG, 1000 + m)
            b_true.append(mse(mt, truth))
            b_clo.append(mse(mc, truth))
            varr.append(float(vt.mean()))

        # cost model
        cost = (32 - k) * COST_SAMPLE
        clo_cost = CLOSURE_32 * k / 32.0
        N = max((B_FLOOR - clo_cost) / cost, 1.0)
        bt, bc, V = st.mean(b_true), st.mean(b_clo), st.mean(varr)
        # NBIG's own MC noise (V/NBIG) is inside the measured bias -> subtract
        bt = max(bt - V / NBIG, 0.0)
        bc = max(bc - V / NBIG, 0.0)
        mse_t = bt + V / N
        mse_c = bc + V / N
        rows.append((k, N, bt, bc, V / N, mse_t, mse_c))
        print(f"  k={k:>2}  N={N:>8.0f}  bias(true)={bt:.2e}  "
              f"bias(closure)={bc:.2e}  var={V/N:.2e}  "
              f"MSE={mse_c:.2e}   [{time.time()-t0:.0f}s]", flush=True)

    print("\n=== DEPTH-TRUNCATED SAMPLING ===")
    print(f"{'k':>3} {'N':>9} {'bias^2(true)':>13} {'bias^2(clo)':>12} "
          f"{'Var/N':>10} {'MSE(true)':>11} {'MSE(closure)':>13}")
    for k, N, bt, bc, v, mt, mc in rows:
        print(f"{k:>3} {N:>9.0f} {bt:>13.2e} {bc:>12.2e} {v:>10.2e} "
              f"{mt:>11.2e} {mc:>13.2e}")
    best = min(rows, key=lambda r: r[6])
    print(f"\n  BEST (closure state): k={best[0]}  MSE={best[6]:.2e}  "
          f"-> score {best[6]*0.1:.2e}")
    print("  refs: v4 3.40e-7 | leaders 8.22e-8 / 9.16e-8 | MC@floor 6.5e-7")
    print("  (iid sampling here; QMC at layer k should cut Var by ~1.8x more)")


if __name__ == "__main__":
    main()
