"""E6 — HOW ACCURATE MUST THE CUMULANT SLICES BE?  (the deciding experiment)

The complete-formula closure reaches 5.59e-7 with ORACLE slices (E5). A real
estimator must produce the 5 slices itself. This measures the accuracy spec two
ways, and thereby picks the architecture:

(A) SAMPLING HYBRID -- estimate the slices from N real forward-propagated
    samples, propagate (mu,Sigma) analytically with the complete formula.
    Gives MSE(N) directly (no noise model). With the FLOP cost of N -> the SCORE.
(B) RELATIVE-ERROR SPEC -- corrupt the oracle slices with r% noise. Gives the
    spec a rank-R propagation / learned-table route must hit.

PERF: all MLPs are batched together (bmm) and every reduction runs on-device in
float32; only the (256,) / (256,256) accumulators cross to CPU float64. Two
passes (mean, then centred moments) keep float32 accumulation accurate to ~1e-4
relative, which is far below the noise levels under test.
"""
from __future__ import annotations

import math
import statistics as st
import sys
import time

import torch

from harness import DT, load_corpus, mse, relu_step_full

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_TRUTH = 2_000_000
CHUNK = 200_000
B_BUDGET = 2.72e11
NS = [1_000, 3_000, 10_000, 30_000, 100_000]
RS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
n = 256


def sample_slices(Wb, n_samp, seed, want_truth=False):
    """Batched over MLPs. Wb: (M,32,n,n) float32 on DEV.
    Returns cums[m][l] dict of float64 CPU slices, + true final mu per MLP."""
    M, L = Wb.shape[0], Wb.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)

    # ---- pass 1: mean of z at every layer
    s1 = torch.zeros(M, L, n, dtype=torch.float64)
    amu = torch.zeros(M, n, dtype=torch.float64)
    done = 0
    while done < n_samp:
        b = min(CHUNK, n_samp - done)
        X = torch.randn(b, n, generator=g, dtype=torch.float32)
        A = X.to(DEV).expand(M, b, n).contiguous()
        for l in range(L):
            Z = torch.bmm(A, Wb[:, l])
            s1[:, l] += Z.sum(1).cpu().double()
            A = torch.clamp(Z, min=0.0)
        amu += A.sum(1).cpu().double()
        done += b
    zmu = (s1 / n_samp)
    amu /= n_samp
    zmu_d = zmu.to(DEV, torch.float32)

    # ---- pass 2: centred moments (all on-device, float32; f64 accumulators)
    acc = {k: torch.zeros(M, L, *sh, dtype=torch.float64)
           for k, sh in (("m2", (n, n)), ("m3d", (n,)), ("m4d", (n,)),
                         ("m21", (n, n)), ("m22", (n, n)), ("m31", (n, n)))}
    g = torch.Generator(device="cpu").manual_seed(seed)
    done = 0
    while done < n_samp:
        b = min(CHUNK, n_samp - done)
        X = torch.randn(b, n, generator=g, dtype=torch.float32)
        A = X.to(DEV).expand(M, b, n).contiguous()
        for l in range(L):
            Z = torch.bmm(A, Wb[:, l])
            A = torch.clamp(Z, min=0.0)
            d = Z - zmu_d[:, l].unsqueeze(1)
            d2 = d * d
            d3 = d2 * d
            dT = d.transpose(1, 2)
            acc["m2"][:, l] += torch.bmm(dT, d).cpu().double()
            acc["m3d"][:, l] += d3.sum(1).cpu().double()
            acc["m4d"][:, l] += (d2 * d2).sum(1).cpu().double()
            acc["m21"][:, l] += torch.bmm(d2.transpose(1, 2), d).cpu().double()
            acc["m22"][:, l] += torch.bmm(d2.transpose(1, 2), d2).cpu().double()
            acc["m31"][:, l] += torch.bmm(d3.transpose(1, 2), d).cpu().double()
        done += b

    N = n_samp
    cums = []
    for m in range(M):
        per_layer = []
        for l in range(L):
            S = acc["m2"][m, l] / N
            v = torch.diagonal(S).clone()
            k4d = acc["m4d"][m, l] / N - 3 * v ** 2
            per_layer.append({
                "k3_iii": acc["m3d"][m, l] / N,
                "k3_iij": acc["m21"][m, l] / N,
                "k4_iiii": k4d,
                "k4_iijj": (acc["m22"][m, l] / N - torch.outer(v, v)
                            - 2 * S ** 2),
                "k4_iiij": acc["m31"][m, l] / N - 3 * v[:, None] * S,
            })
        cums.append(per_layer)
    return cums, (amu if want_truth else None)


def run_closure(W64, cums):
    mu, C = torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT)
    for l in range(32):
        mu, C = relu_step_full(mu, C, W64[l], cums[l])
    return mu


def mc_mean(Wb, N, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(N, n, generator=g, dtype=torch.float32)
    A = X.to(DEV).expand(Wb.shape[0], N, n).contiguous()
    for l in range(32):
        A = torch.clamp(torch.bmm(A, Wb[:, l]), min=0.0)
    return A.mean(1).cpu().double()


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    recs = load_corpus(n_mlps)
    W64 = [[r["W"][l] for l in range(32)] for r in recs]
    Wb = torch.stack([r["W"] for r in recs]).to(DEV, torch.float32)
    M = Wb.shape[0]

    t0 = time.time()
    oracle, truth = sample_slices(Wb, N_TRUTH, 7000, want_truth=True)
    print(f"  oracle slices ({N_TRUTH:,} samples x {M} MLPs): "
          f"{time.time()-t0:.0f}s", flush=True)

    resA, mcres = {}, {}
    for N in NS:
        cums, _ = sample_slices(Wb, N, 31_000)
        resA[N] = [mse(run_closure(W64[m], cums[m]), truth[m]) for m in range(M)]
        mcm = mc_mean(Wb, N, 31_000)
        mcres[N] = [mse(mcm[m], truth[m]) for m in range(M)]
        print(f"  N={N}: done ({time.time()-t0:.0f}s)", flush=True)

    resB = {}
    gen = torch.Generator().manual_seed(500)
    for r in RS:
        resB[r] = []
        for m in range(M):
            c = [{k: v * (1 + r * torch.randn(v.shape, generator=gen, dtype=DT))
                  for k, v in cl.items()} for cl in oracle[m]]
            resB[r].append(mse(run_closure(W64[m], c), truth[m]))

    # --- SPLIT SPEC: noise kappa3 and kappa4 SEPARATELY (where to spend effort?)
    K3 = ("k3_iii", "k3_iij")
    grid = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    split = {}
    for r3 in grid:
        for r4 in grid:
            vals = []
            for m in range(M):
                c = []
                for cl in oracle[m]:
                    d = {}
                    for k, v in cl.items():
                        r = r3 if k in K3 else r4
                        d[k] = v * (1 + r * torch.randn(v.shape, generator=gen,
                                                        dtype=DT)) if r else v
                    c.append(d)
                vals.append(mse(run_closure(W64[m], c), truth[m]))
            split[(r3, r4)] = st.mean(vals)
    # kappa4 dropped entirely (complete kappa3 formula only)
    zero4 = []
    for m in range(M):
        c = [{k: (v if k in K3 else torch.zeros_like(v)) for k, v in cl.items()}
             for cl in oracle[m]]
        zero4.append(mse(run_closure(W64[m], c), truth[m]))

    print("\n=== SPLIT SPEC: rows = kappa3 err, cols = kappa4 err (MSE) ===")
    print("  k3\\k4 " + " ".join(f"{c*100:>9.0f}%" for c in grid))
    for r3 in grid:
        print(f"{r3*100:>6.0f}% " + " ".join(f"{split[(r3,r4)]:>10.2e}"
                                             for r4 in grid))
    print(f"\nkappa4 DROPPED entirely (oracle k3, complete formula): "
          f"{st.mean(zero4):.2e}")
    print("targets: leaders 9.3e-7 | v4 3.4e-6 | MC@floor 6.5e-6")

    def sem(v):
        return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0

    print(f"\nE6 | {M} MLPs | complete-formula closure | {time.time()-t0:.0f}s")
    print("oracle ceiling 5.59e-7 | v4 3.4e-6 | leaders 9.3e-7 | MC@floor 6.5e-6\n")

    print("=== (A) SAMPLING HYBRID: slices from N samples ===")
    print(f"{'N':>8} {'closure MSE':>13} {'plain-MC':>11} {'FLOPs':>9} "
          f"{'%budg':>7} {'SCORE':>10}  vs v4(3.4e-7)")
    for N in NS:
        F = 4 * N * n * n * 32 + 2.5e9
        util = F / B_BUDGET
        s = st.mean(resA[N]) * max(0.1, util)
        print(f"{N:>8} {st.mean(resA[N]):>13.2e} {st.mean(mcres[N]):>11.2e} "
              f"{F:>9.1e} {util*100:>6.1f}% {s:>10.2e}  "
              f"{'WINS' if s < 3.4e-7 else ''}")

    print("\n=== (B) RELATIVE-ERROR SPEC (oracle slices + r% noise) ===")
    print(f"{'r':>6} {'MSE':>12} {'SEM':>10}   verdict")
    for r in RS:
        v = st.mean(resB[r])
        vd = ("BEATS LEADERS (<9.3e-7)" if v < 9.3e-7 else
              "beats v4 (<3.4e-6)" if v < 3.4e-6 else "")
        print(f"{r*100:>5.0f}% {v:>12.2e} {sem(resB[r]):>10.1e}   {vd}")


if __name__ == "__main__":
    main()
