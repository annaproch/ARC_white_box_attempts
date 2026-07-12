"""E5 — end-to-end oracle ceiling with the COMPLETE diagram formulas.

E2 measured the oracle kappa3+kappa4 ceiling at 1.26e-6 -- but with a formula
that kept only the p=0 term of every Sigma_ij series and omitted three whole
diagram families (kappa3_iii, kappa4_iiij, kappa4_iiii). E4 showed the complete
formula cuts the one-step covariance residual 2.6-12x.

This re-measures the end-to-end ceiling with the complete formula. It also needs
kappa4_iiij, which IS NOT IN THE CORPUS -- so the true slices are recomputed
from fresh samples here (also removes any dependence on the corpus).

Reports, on the same MLPs:
  plain K=2                                   (canonical baseline)
  oracle k3+k4, TRUNCATED formula             (= E2's 1.26e-6, reproduced)
  oracle k3+k4, COMPLETE formula              (the real ceiling)
"""
from __future__ import annotations

import math
import statistics as st
import sys

import torch

from harness import (CORPUS, DT, load_corpus, mse, propagate, relu_step,
                     relu_step_full)

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_SAMP = 2_000_000
CHUNK = 500_000


def true_slices(Wl, n_samp, seed):
    """Sampled TRUE cumulant slices of z^(l) for every layer, + true mu of a^(l).
    Streams in chunks; accumulates in float64."""
    n, L = 256, len(Wl)
    acc = [{k: torch.zeros(*sh, dtype=DT) for k, sh in
            (("s1", (n,)), ("m2", (n, n)), ("m3d", (n,)), ("m4d", (n,)),
             ("m21", (n, n)), ("m22", (n, n)), ("m31", (n, n)))}
           for _ in range(L)]
    amu = [torch.zeros(n, dtype=DT) for _ in range(L)]
    done = 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    while done < n_samp:
        b = min(CHUNK, n_samp - done)
        A = torch.randn(b, n, generator=g, dtype=torch.float32).to(DEV)
        for l in range(L):
            Z = (A @ Wl[l]).cpu().to(DT)
            A = torch.clamp(A @ Wl[l], min=0.0)
            amu[l] += A.cpu().to(DT).sum(0)
            a = acc[l]
            a["s1"] += Z.sum(0)
            a["m2"] += Z.T @ Z
            Z2 = Z * Z
            a["m3d"] += (Z2 * Z).sum(0)
            a["m4d"] += (Z2 * Z2).sum(0)
            a["m21"] += Z2.T @ Z
            a["m22"] += Z2.T @ Z2
            a["m31"] += (Z2 * Z).T @ Z
        done += b

    out = []
    for l in range(L):
        a = acc[l]
        N = n_samp
        mu = a["s1"] / N
        # central moments from raw ones
        M2 = a["m2"] / N - torch.outer(mu, mu)
        v = torch.diagonal(M2).clone()
        m3d = a["m3d"] / N - 3 * mu * (a["m2"].diagonal() / N) + 2 * mu ** 3
        m4d = (a["m4d"] / N - 4 * mu * (a["m3d"] / N)
               + 6 * mu ** 2 * (a["m2"].diagonal() / N) - 3 * mu ** 4)
        k4d = m4d - 3 * v ** 2
        # E[d_i^2 d_j] and E[d_i^3 d_j], E[d_i^2 d_j^2] via raw->central
        E21 = (a["m21"] / N - 2 * mu[:, None] * (a["m2"] / N)
               - (a["m2"].diagonal() / N)[:, None] * mu[None, :]
               + 2 * (mu ** 2)[:, None] * mu[None, :])
        E22 = (a["m22"] / N
               - 2 * mu[:, None] * (a["m21"].T / N).T * 0)  # placeholder, fixed below
        # cleaner: recompute E22, E31 with a second centred pass (cheap: reuse)
        out.append({"mu": mu, "cov": M2, "v": v, "k3_iij": E21,
                    "k3_iii": m3d, "k4_iiii": k4d,
                    "_raw": a, "N": N, "amu": amu[l] / N})
    return out


def centred_slices(Wl, n_samp, seed, mus):
    """Second pass: centred moments E[d_i^2 d_j^2] and E[d_i^3 d_j] directly."""
    n, L = 256, len(Wl)
    E22 = [torch.zeros(n, n, dtype=DT) for _ in range(L)]
    E31 = [torch.zeros(n, n, dtype=DT) for _ in range(L)]
    done = 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    while done < n_samp:
        b = min(CHUNK, n_samp - done)
        A = torch.randn(b, n, generator=g, dtype=torch.float32).to(DEV)
        for l in range(L):
            Z = (A @ Wl[l]).cpu().to(DT)
            A = torch.clamp(A @ Wl[l], min=0.0)
            d = Z - mus[l]["mu"]
            d2 = d * d
            E22[l] += d2.T @ d2
            E31[l] += (d2 * d).T @ d
        done += b
    return [e / n_samp for e in E22], [e / n_samp for e in E31]


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    recs = load_corpus(n_mlps)
    r_k2, r_trunc, r_full = [], [], []

    for m, rec in enumerate(recs):
        Wl64 = [rec["W"][l] for l in range(32)]
        Wl = [rec["W"][l].to(DEV, torch.float32) for l in range(32)]
        S = true_slices(Wl, N_SAMP, 7000 + m)
        E22, E31 = centred_slices(Wl, N_SAMP, 7000 + m, S)
        cums = []
        for l in range(32):
            v, C = S[l]["v"], S[l]["cov"]
            k4_iijj = E22[l] - torch.outer(v, v) - 2 * C ** 2
            k4_iiij = E31[l] - 3 * v[:, None] * C
            cums.append({"k3_iii": S[l]["k3_iii"], "k3_iij": S[l]["k3_iij"],
                         "k4_iiii": S[l]["k4_iiii"], "k4_iijj": k4_iijj,
                         "k4_iiij": k4_iiij})
        truth = S[31]["amu"]

        n = 256
        # 1) plain K=2
        mu, C = torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT)
        mu, C = propagate(mu, C, Wl64)
        r_k2.append(mse(mu, truth))
        # 2) oracle, TRUNCATED formula (reproduces E2)
        mu, C = torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT)
        for l in range(32):
            mu, C = relu_step(mu, C, Wl64[l], 8,
                              cums[l]["k3_iij"], cums[l]["k4_iijj"])
        r_trunc.append(mse(mu, truth))
        # 3) oracle, COMPLETE formula
        mu, C = torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT)
        for l in range(32):
            mu, C = relu_step_full(mu, C, Wl64[l], cums[l])
        r_full.append(mse(mu, truth))
        print(f"  MLP {m}: K2 {r_k2[-1]:.2e} | trunc {r_trunc[-1]:.2e} | "
              f"FULL {r_full[-1]:.2e}", flush=True)

    def sem(v):
        return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0

    print(f"\nE5 | {n_mlps} MLPs | {N_SAMP:,} samples | end-to-end final-layer MSE")
    for tag, v in (("plain K=2", r_k2),
                   ("oracle k3+k4, TRUNCATED formula (= E2)", r_trunc),
                   ("oracle k3+k4, COMPLETE formula", r_full)):
        print(f"  {tag:>40}: {st.mean(v):.3e} +- {sem(v):.1e}")
    print(f"\n  complete/truncated = {st.mean(r_full)/st.mean(r_trunc):.2f}x")


if __name__ == "__main__":
    main()
