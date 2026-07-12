"""E9b — verify the kappa3(ReLU(z)) diagram expansion AT THE REAL SHAPE (n=256).

n=12/32 is the wrong regime: the diagram series is a 1/n expansion, so it barely
converges there (star-only 62.8% at n=32 vs 9.5% at n=256). The real question is
which truncation clears the <=5% spec at n=256 -- and that IS tractable:
kappa3 is 256^3 = 1.7e7 entries (134 MB), trivial in 128 GB.

GPU-efficient: the N*n^3 third-moment einsum is done as n matmuls of
(n x N)(N x n) on MPS in float32 (the naive einsum materialises an (N,n,n)
intermediate and OOMs). Diagram tensors are evaluated on-device.
"""
from __future__ import annotations

import sys
import time

import torch

from diagrams import enumerate_diagrams, eval_diagram
from harness import CORPUS, DT, hermite_gains

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256


def third_moment(d, chunk=64):
    """M3[p,q,r] = mean_n d_np d_nq d_nr, as n matmuls -- no (N,n,n) intermediate."""
    N = d.shape[0]
    M3 = torch.empty(n, n, n, device=d.device, dtype=F32)
    for p0 in range(0, n, chunk):
        for p in range(p0, min(p0 + chunk, n)):
            M3[p] = (d * d[:, p:p + 1]).T @ d
    return M3 / N


def main():
    N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1_000_000
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    lay = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    rec = torch.load(CORPUS / f"mlp_{200000+m}.pt", map_location="cpu",
                     weights_only=False)
    W = [rec["W"][l].to(DEV, F32) for l in range(32)]

    t0 = time.time()
    g = torch.Generator(device="cpu").manual_seed(3)
    A = torch.randn(N, n, generator=g, dtype=F32).to(DEV)
    for l in range(lay):
        A = torch.clamp(A @ W[l], min=0.0)
    Z = A @ W[lay]                      # z at layer `lay`
    Aout = torch.clamp(Z, min=0.0)      # a = ReLU(z)
    print(f"  forward {N:,} samples to layer {lay}: {time.time()-t0:.0f}s",
          flush=True)

    t0 = time.time()
    dz = Z - Z.mean(0)
    da = Aout - Aout.mean(0)
    S = ((dz.T @ dz) / N)
    K3z = third_moment(dz)
    K3a = third_moment(da)
    print(f"  cumulant tensors (256^3): {time.time()-t0:.0f}s", flush=True)

    zmu = Z.mean(0).cpu().to(DT)
    zvar = torch.diagonal(S).cpu().to(DT)
    gh, *_ = hermite_gains(zmu, zvar, kmax=14)
    gh = [x.to(DEV, F32) for x in gh]
    S64, K3z64 = S, K3z
    den = K3a.pow(2).sum()

    print(f"\nE9b | n=256, layer {lay}, N={N:,}  (kappa3 spec: <=5%)")
    print(f"{'truncation':>40} {'#diag':>7} {'rel err':>9}")
    for ms, m3, tag in (
            (2, 0, "GEN star only (2-edge) [SHIPPED]"),
            (3, 0, "GEN to 3 sigma edges"),
            (0, 1, "pass-through LEADING only"),
            (2, 0, "-"),
            (1, 1, "GEN + pass, <=1 sigma edge"),
            (2, 1, "GEN + pass, <=2 sigma edges"),
            (3, 1, "GEN + pass, <=3 sigma edges"),
            (4, 1, "GEN + pass, <=4 sigma edges"),
    ):
        if tag == "-":
            continue
        P = torch.zeros(n, n, n, device=DEV, dtype=F32)
        dgs = enumerate_diagrams(ms, m3, 0)
        for blocks, k, coef in dgs:
            P += eval_diagram(blocks, k, coef, gh, S64, K3z64, None)
        err = ((P - K3a).pow(2).sum() / den).sqrt().item()
        print(f"{tag:>40} {len(dgs):>7} {err*100:>8.2f}%", flush=True)
        del P


if __name__ == "__main__":
    main()
