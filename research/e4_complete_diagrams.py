"""E4 — the correction formulas are TRUNCATED. Complete them and price each term.

Re-derived from the Hermite diagram summation formula (2605.05179 Thm A.2).
For r=2 (covariance) a diagram partitions k1 legs on neuron i and k2 on j into
blocks; blocks of size <=2 must be MIXED (span both neurons), blocks of size >=3
need not be. Each cumulant block therefore generates a SERIES in Sigma_ij, not a
single term:

  Gaussian :  sum_p (1/p!) h[p](i) h[p](j) S^p                    p>=1
  k3_iij   :  1/2 sum_p (1/p!) h[p+2](i) h[p+1](j) k3_iij S^p     p>=0
  k3_iii   :  1/6 sum_p (1/p!) h[p+3](i) h[p](j)   k3_iii S^p     p>=1
  k4_iijj  :  1/4 sum_p (1/p!) h[p+2](i) h[p+2](j) k4_iijj S^p    p>=0
  k4_iiij  :  1/6 sum_p (1/p!) h[p+3](i) h[p+1](j) k4_iiij S^p    p>=0
  k4_iiii  : 1/24 sum_p (1/p!) h[p+4](i) h[p](j)   k4_iiii S^p    p>=1
(+ transposes i<->j)

The shipped code and harness.py keep ONLY the p=0 term of each, and omit the
k3_iii / k4_iiij / k4_iiii families entirely.

For r=1 (the mean) only blocks of size >=3 survive, giving Gram-Charlier:
  E[phi] = h0 + (1/6)h3 k3 + (1/24)h4 k4 + (1/120)h5 k5
           + (1/720) h6 (k6 + 10 k3^2) + ...
The k3^2 term ("power cumulant") is MISSING from our code.

Every cumulant slice is computed from the SAME samples, so this is a clean
apples-to-apples test of the FORMULA, with no corpus dependency.
Verdict rule: if a term's coefficient is right, adding it must REDUCE the
residual. A term that increases it is mis-derived.
"""
from __future__ import annotations

import statistics as st
import sys

import torch

from harness import CORPUS, DT, hermite_gains, ncdf, npdf

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_SAMP = 1_000_000
LAYERS = [3, 7, 15, 23, 30]
PMAX = 6            # truncate each Sigma_ij series at p <= PMAX
KMAX = 12           # need h up to p+4


def cumulant_slices(d):
    """Central-moment cumulant slices of the (already centred) sample matrix d.
    Returns the n-vectors / n x n matrices the diagram formula needs."""
    N = d.shape[0]
    v = (d ** 2).mean(0)                                  # Var
    S = (d.T @ d) / N                                     # Sigma_ij
    k3_iii = (d ** 3).mean(0)
    k4_iiii = (d ** 4).mean(0) - 3 * v ** 2
    d2 = d ** 2
    k3_iij = (d2.T @ d) / N                               # E[di^2 dj]  (= kappa3_iij)
    # kappa4_iijj = E[di^2 dj^2] - Var_i Var_j - 2 Sigma_ij^2
    k4_iijj = (d2.T @ d2) / N - torch.outer(v, v) - 2 * S ** 2
    # kappa4_iiij = E[di^3 dj] - 3 Var_i Sigma_ij
    k4_iiij = ((d ** 3).T @ d) / N - 3 * v[:, None] * S
    k5_i = (d ** 5).mean(0) - 10 * k3_iii * v             # kappa5_iiiii
    m6 = (d ** 6).mean(0)
    k6_i = (m6 - 15 * k4_iiii * v - 10 * k3_iii ** 2
            - 15 * v ** 3)                                # kappa6_iiiiii
    return v, S, k3_iii, k3_iij, k4_iiii, k4_iijj, k4_iiij, k5_i, k6_i


def series(g, S, ai, aj, coef, pmin, pmax, K):
    """coef * sum_{p=pmin}^{pmax} (1/p!) g[p+ai](i) g[p+aj](j) * K * S^p"""
    out = torch.zeros_like(S)
    Sp = torch.ones_like(S) if pmin == 0 else S.clone()
    fact = 1.0
    for p in range(pmin, pmax + 1):
        if p > 0:
            fact *= p
        term = torch.outer(g[p + ai], g[p + aj]) * K * (Sp / fact)
        out = out + term
        Sp = Sp * S
    return coef * out


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    res = {}
    rho_stat = {}

    for m in range(n_mlps):
        rec = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                         weights_only=False)
        Wl = [rec["W"][l].to(DEV, torch.float32) for l in range(32)]
        gen = torch.Generator(device="cpu").manual_seed(4000 + m)
        A = torch.randn(N_SAMP, 256, generator=gen, dtype=torch.float32).to(DEV)

        for l in range(31):
            Anext = torch.clamp(A @ Wl[l], min=0.0)
            if l + 1 not in LAYERS:
                A = Anext
                continue
            lay = l + 1
            Z = (Anext @ Wl[l + 1]).cpu().to(DT)
            a_out = torch.clamp(Z, min=0.0)
            true_mu = a_out.mean(0)
            ac = a_out - true_mu
            true_cov = (ac.T @ ac) / Z.shape[0]

            zmu = Z.mean(0)
            d = Z - zmu
            v, S, k3d, k3m, k4d, k4m, k4t, k5d, k6d = cumulant_slices(d)
            g, s, t, pdf, cdf = hermite_gains(zmu, v, kmax=KMAX)

            # off-diagonal correlation, to justify (or not) the p=0 truncation
            sd = torch.sqrt(v)
            R = (S / torch.outer(sd, sd)).abs()
            R.fill_diagonal_(0)
            rho_stat.setdefault(lay, []).append(
                (R.mean().item(), R.flatten().quantile(0.95).item(), R.max().item()))

            # ---------------- MEAN ----------------
            m_g = g[0]
            m_k3 = m_g + k3d / 6 * g[3]
            m_k34 = m_k3 + k4d / 24 * g[4]                     # <- what we ship
            m_pow = m_k34 + (10.0 / 720.0) * g[6] * k3d ** 2   # + power cumulant
            m_k56 = m_pow + k5d / 120 * g[5] + k6d / 720 * g[6]
            for tag, p in (("mean Gauss", m_g), ("mean +k3", m_k3),
                           ("mean +k3k4 [SHIPPED]", m_k34),
                           ("mean +k3^2 (power)", m_pow),
                           ("mean +k5+k6", m_k56)):
                res.setdefault((tag, lay), []).append(
                    ((p - true_mu) ** 2).mean().item())

            # ---------------- COVARIANCE ----------------
            def finish(C):
                C = 0.5 * (C + C.T)
                C = C.clone()
                C.fill_diagonal_(0)
                T = true_cov.clone()
                T.fill_diagonal_(0)
                return ((C - T) ** 2).mean().item()      # off-diagonal only

            # Gaussian Wick series: sum_{k>=1} (1/k!) h[k](i) h[k](j) S^k
            # (offsets are 0,0 -- g[p] with S^p, NOT g[p+1])
            C_g = series(g, S, 0, 0, 1.0, 1, PMAX, torch.ones_like(S))
            # p=0 only  (= what harness.py / the shipped closure implement)
            C_ship = (C_g
                      + 0.5 * (torch.outer(g[2], g[1]) * k3m
                               + torch.outer(g[1], g[2]) * k3m.T)
                      + 0.25 * torch.outer(g[2], g[2]) * k4m)
            # full k3_iij series (all p)
            C_a = C_g + series(g, S, 2, 1, 0.5, 0, PMAX, k3m) \
                      + series(g, S, 1, 2, 0.5, 0, PMAX, k3m.T)
            # + k3_iii family
            k3i = k3d[:, None].expand_as(S)
            C_b = C_a + series(g, S, 3, 0, 1 / 6, 1, PMAX, k3i) \
                      + series(g, S, 0, 3, 1 / 6, 1, PMAX, k3i.T)
            # + full k4_iijj series
            C_c = C_b + series(g, S, 2, 2, 0.25, 0, PMAX, k4m)
            # + k4_iiij family  (NOT IN THE CORPUS AT ALL)
            C_d = C_c + series(g, S, 3, 1, 1 / 6, 0, PMAX, k4t) \
                      + series(g, S, 1, 3, 1 / 6, 0, PMAX, k4t.T)
            # + k4_iiii family
            k4i = k4d[:, None].expand_as(S)
            C_e = C_d + series(g, S, 4, 0, 1 / 24, 1, PMAX, k4i) \
                      + series(g, S, 0, 4, 1 / 24, 1, PMAX, k4i.T)

            for tag, C in (("cov Gauss", C_g), ("cov p=0 only [SHIPPED]", C_ship),
                           ("cov +k3_iij all p", C_a), ("cov +k3_iii", C_b),
                           ("cov +k4_iijj all p", C_c), ("cov +k4_iiij", C_d),
                           ("cov +k4_iiii", C_e)):
                res.setdefault((tag, lay), []).append(finish(C))
            A = Anext
        print(f"  MLP {m} done", flush=True)

    print(f"\nE4 | {n_mlps} MLPs | {N_SAMP:,} samples | pmax={PMAX}")
    print("\noff-diagonal |rho| of z  (justifies or kills the p=0 truncation)")
    print(f"{'layer':>6} {'mean':>8} {'p95':>8} {'max':>8}")
    for lay in LAYERS:
        a = rho_stat[lay]
        print(f"{lay:>6} {st.mean(x[0] for x in a):>8.3f} "
              f"{st.mean(x[1] for x in a):>8.3f} {st.mean(x[2] for x in a):>8.3f}")

    for grp in ("mean", "cov"):
        print(f"\n=== {grp.upper()}: one-step MSE vs sampled truth ===")
        tags = [t for (t, _) in res if t.startswith(grp)]
        tags = sorted(set(tags), key=lambda x: list(dict.fromkeys(
            t for (t, _) in res)).index(x))
        print(f"{'variant':>24} " + " ".join(f"{l:>10}" for l in LAYERS))
        for t in tags:
            row = f"{t:>24} "
            for lay in LAYERS:
                row += f"{st.mean(res[(t, lay)]):>10.2e} "
            print(row)


if __name__ == "__main__":
    main()
