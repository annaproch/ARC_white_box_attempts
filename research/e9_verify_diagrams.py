"""E9 — machine-verify the kappa3(ReLU(z)) diagram expansion, TENSOR-level.

Small n (so kappa3/kappa4 tensors are tiny), real 2-layer net so z is genuinely
non-Gaussian. Compute the TRUE kappa3(a) and the true cumulants of z by
sampling, then check the programmatic diagram sum (diagrams.py) reproduces it,
order by order.

This isolates the FORMULA completely: no contraction, no rank, no accumulation.
Whatever truncation this says we need is what the propagator must implement.
"""
from __future__ import annotations

import sys
import time

import torch

from diagrams import enumerate_diagrams, k3_activation
from harness import DT, hermite_gains

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def central_cumulants(Z, chunk=1_000_000, want_k4=False):
    """Sigma, kappa3, kappa4 (full tensors) of Z (N, n).

    The raw 4th-moment einsum is N*n^4 -- do it CHUNKED on the GPU in float32
    and accumulate the tiny (n^4) result in float64 on the CPU. Running it as a
    single float64 CPU einsum is ~4e11 ops and hangs."""
    N, n = Z.shape
    mu = Z.mean(0)
    S = torch.zeros(n, n, dtype=DT)
    M3 = torch.zeros(n, n, n, dtype=DT)
    M4 = torch.zeros(n, n, n, n, dtype=DT) if want_k4 else None
    for i in range(0, N, chunk):
        d = (Z[i:i + chunk] - mu).to(DEV, torch.float32)
        S += torch.einsum("np,nq->pq", d, d).cpu().double()
        M3 += torch.einsum("np,nq,nr->pqr", d, d, d).cpu().double()
        if want_k4:
            M4 += torch.einsum("np,nq,nr,ns->pqrs", d, d, d, d).cpu().double()
    S, M3 = S / N, M3 / N
    if not want_k4:
        return S, M3, None
    M4 = M4 / N
    K4 = (M4
          - torch.einsum("pq,rs->pqrs", S, S)
          - torch.einsum("pr,qs->pqrs", S, S)
          - torch.einsum("ps,qr->pqrs", S, S))
    return S, M3, K4


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    N = int(float(sys.argv[2])) if len(sys.argv) > 2 else 20_000_000
    torch.manual_seed(0)
    W0 = torch.randn(n, n, dtype=DT) * (2.0 / n) ** 0.5
    W1 = torch.randn(n, n, dtype=DT) * (2.0 / n) ** 0.5

    t0 = time.time()
    Zs, As = [], []
    done, CH = 0, 2_000_000
    g = torch.Generator().manual_seed(7)
    accZ = torch.zeros(0, n, dtype=DT)
    Zall = []
    Aall = []
    while done < N:
        b = min(CH, N - done)
        x = torch.randn(b, n, generator=g, dtype=torch.float32).to(DEV)
        a0 = torch.clamp(x @ W0.to(DEV, torch.float32), min=0)
        z1 = a0 @ W1.to(DEV, torch.float32)
        a1 = torch.clamp(z1, min=0)
        Zall.append(z1.cpu().to(DT))
        Aall.append(a1.cpu().to(DT))
        done += b
    Z = torch.cat(Zall)
    A = torch.cat(Aall)
    print(f"  sampled {N:,} x n={n}: {time.time()-t0:.0f}s", flush=True)

    S, K3z, K4z = central_cumulants(Z)
    _, K3a_true, _ = central_cumulants(A)

    zmu = Z.mean(0)
    zvar = torch.diagonal(S)
    g_h, *_ = hermite_gains(zmu, zvar, kmax=12)

    def rel(P):
        return ((P - K3a_true).pow(2).sum() / K3a_true.pow(2).sum()).sqrt().item()

    print(f"\nE9 | kappa3(ReLU(z)) tensor-level, n={n}, N={N:,}")
    print(f"{'truncation':>44} {'#diag':>7} {'rel err':>9}")
    for ms, m3, tag in (
            (2, 0, "GEN star only (2-edge)  [SHIPPED]"),
            (3, 0, "GEN to 3 sigma edges"),
            (4, 0, "GEN to 4 sigma edges"),
            (0, 1, "pass-through LEADING only"),
            (1, 1, "GEN + pass, <=1 sigma edge"),
            (2, 1, "GEN + pass, <=2 sigma edges"),
            (3, 1, "GEN + pass, <=3 sigma edges"),
            (4, 1, "GEN + pass, <=4 sigma edges"),
            (5, 1, "GEN + pass, <=5 sigma edges"),
    ):
        nd = len(enumerate_diagrams(ms, m3, 0))
        P = k3_activation(g_h, S, K3z, K4z, ms, m3, 0)
        print(f"{tag:>44} {nd:>7} {rel(P)*100:>8.2f}%")

    print("\nSPEC: kappa3 <=5% rel err (E6). n is swept because the diagram")
    print("series is a 1/n expansion -- it only converges at large n.")


if __name__ == "__main__":
    main()
