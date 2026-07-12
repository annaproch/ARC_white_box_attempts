"""G1b — sampling-noise floor of the corpus zk3/zk4 slice targets.

STRATEGY §2.2 flagged this and it was never measured: the corpus targets are
themselves sample estimates (N = 2e6). Any learned-slice held-out error e is
inflated by the TEST-target noise nu:  e^2 ~ model^2 + nu^2  (training-target
noise largely averages out: each per-layer ridge sees n_train x 65k rows).

Method: same weights (corpus seed), two INDEPENDENT halves of 1e6 samples,
identical moment math to build_moment_corpus.py. Per layer, off-diagonal:

    r = ||A - B||_F / (0.5 ||A + B||_F)
    noise(1e6) = r / sqrt(2)          noise(2e6, the corpus) = r / 2

Interpretation feeds G1 (kappa4 learnability) AND retro-audits E12's kappa3
numbers (its 26.8-33.9% depth errors carry the same inflation).
"""
from __future__ import annotations

import math
import sys
import time

import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
LAY = [1, 3, 7, 15, 23, 31]


def build_ws(seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(N_W, N_W, generator=g) * math.sqrt(2.0 / N_W)).to(DEV)
            for _ in range(DEPTH)]


def z_moments(ws, n_samples, batch, seed):
    torch.manual_seed(seed)
    z_ = lambda: [torch.zeros(N_W, N_W, dtype=torch.float64) for _ in range(DEPTH)]
    Z1 = [torch.zeros(N_W, dtype=torch.float64) for _ in range(DEPTH)]
    Z2, Z21, Z22 = z_(), z_(), z_()
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        x = torch.randn(b, N_W, device=DEV)
        for li, w in enumerate(ws):
            z = x @ w
            z2 = z * z
            Z1[li] += z.sum(0).cpu().double()
            Z2[li] += (z.T @ z).cpu().double()
            Z21[li] += (z2.T @ z).cpu().double()
            Z22[li] += (z2.T @ z2).cpu().double()
            x = torch.relu(z)
        done += b
    return Z1, Z2, Z21, Z22


def central(S1, S2, S21, S22, n):
    """Identical to build_moment_corpus.central."""
    mu = S1 / n
    m2 = S2 / n
    m21 = S21 / n
    m22 = S22 / n
    cov = m2 - torch.outer(mu, mu)
    d = torch.diagonal(m2)
    k3 = (m21 - mu[None, :] * d[:, None] - 2 * mu[:, None] * m2
          + 2 * (mu[:, None] ** 2) * mu[None, :])
    e22 = (m22 - 2 * mu[None, :] * m21 - 2 * mu[:, None] * m21.T
           + (mu[None, :] ** 2) * d[:, None] + (mu[:, None] ** 2) * d[None, :]
           + 4 * torch.outer(mu, mu) * m2 - 3 * torch.outer(mu ** 2, mu ** 2))
    v = torch.diagonal(cov)
    k4 = e22 - torch.outer(v, v) - 2 * cov ** 2
    return mu, cov, k3, k4


def slices_for_half(ws, n, batch, seed):
    Z1, Z2, Z21, Z22 = z_moments(ws, n, batch, seed)
    out = []
    for l in range(DEPTH):
        mu, cov, k3, k4 = central(Z1[l], Z2[l], Z21[l], Z22[l], n)
        s = torch.sqrt(torch.clamp(torch.diagonal(cov), min=1e-30))
        y3 = k3 / (s[:, None] ** 2 * s[None, :])
        y4 = k4 / (s[:, None] ** 2 * s[None, :] ** 2)
        out.append((y3, y4))
    return out


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n_half = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    batch = 65_536
    off = ~torch.eye(N_W, dtype=torch.bool)
    print(f"G1b | {n_mlps} MLPs | two independent halves of {n_half:,} "
          f"(corpus = 2e6) | device {DEV}")

    n3 = {l: [] for l in range(DEPTH)}
    n4 = {l: [] for l in range(DEPTH)}
    for m in range(n_mlps):
        seed = 200_000 + m
        ws = build_ws(seed)
        t0 = time.time()
        A = slices_for_half(ws, n_half, batch, seed * 2 + 1)
        B = slices_for_half(ws, n_half, batch, seed * 2 + 2)
        for l in range(DEPTH):
            for (a, b), store in (((A[l][0], B[l][0]), n3), ((A[l][1], B[l][1]), n4)):
                r = ((a - b)[off].pow(2).sum().sqrt()
                     / (0.5 * (a + b))[off].pow(2).sum().sqrt()).item()
                store[l].append(r / 2.0)          # -> noise at the corpus N=2e6
        print(f"  mlp {seed}: {time.time() - t0:.0f}s", flush=True)

    import statistics as st
    print(f"\n=== relative sampling-noise floor of corpus targets (N=2e6) ===")
    print(f"{'':>14} " + " ".join(f"{'L' + str(l):>8}" for l in LAY)
          + f"   {'mean(32)':>9}")
    for name, store in (("zk3 (kappa3)", n3), ("zk4 (kappa4)", n4)):
        row = " ".join(f"{st.mean(store[l]) * 100:>7.1f}%" for l in LAY)
        mean_all = st.mean(st.mean(store[l]) for l in range(DEPTH)) \
            if False else st.mean([e for l in range(DEPTH) for e in store[l]])
        print(f"{name:>14} {row}   {mean_all * 100:>8.1f}%")

    print("\n  deflation: model ~= sqrt(e_heldout^2 - nu^2) for the G1/E12 tables")
    print("  (also: if nu_k4 >~ 20-30%, G1 CANNOT be decided on this corpus --")
    print("   rebuild targets with more offline samples first.)")


if __name__ == "__main__":
    main()
