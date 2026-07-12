"""G1 test: is the Hermite/Wick truncation our 10× gap?

Runs K=2 covariance propagation at the challenge shape with an increasing
number of Wick terms in the off-diagonal ReLU covariance update:

    Cov(ReLU(Zi), ReLU(Zj)) = sum_{k>=1} h_k(i) h_k(j) C_ij^k / k!

Shipped estimator uses 2 terms; torch experiments used 4. If deep-net
pairwise correlations are large, the series converges slowly and truncation
error compounds over 32 layers.

Also reports the distribution of |rho| (pre-activation correlations) per
layer, which is the direct diagnostic: if |rho| is small the series is fine
and the gap is elsewhere (=> go to G3, rank-truncated K=3).

Run from the kprop repo so torch resolves:
    cd ../mlp_cumulant_propagation && uv run python ../research/wick_terms_test.py
"""

import math

import numpy as np
import torch

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
GT_N, BATCH = 2_000_000, 131_072
SEEDS = 6
TERM_COUNTS = [2, 4, 8, 16, 32, 64]
SQ2PI = math.sqrt(2 * math.pi)


def build_ws(seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(N_W, N_W, generator=g) * math.sqrt(2.0 / N_W)).to(DEV)
            for _ in range(DEPTH)]


def gt_mean(ws, n):
    s = torch.zeros(N_W, dtype=torch.float64)
    d = 0
    while d < n:
        b = min(BATCH, n - d)
        x = torch.randn(b, N_W, device=DEV)
        for w in ws:
            x = torch.relu(x @ w)
        s += x.sum(0).cpu().double()
        d += b
    return s / n


def he_coefs(mu, s, var, a, pdf, cdf, k_max):
    """Wick coefficients h_k = E[d^k ReLU(Z)] for Z ~ N(mu, var), k = 1..k_max.

    k=1: Phi(a)
    k=2: phi(a)/s
    k>=3: (-1)^(k-2) s^(-(k-1)) He_{k-2}(a) phi(a)      [probabilists' Hermite]
    """
    coefs = [cdf, pdf / s]
    if k_max >= 3:
        # He_0 = 1, He_1 = a, He_{m+1} = a*He_m - m*He_{m-1}
        he_prev, he_cur = torch.ones_like(a), a
        for k in range(3, k_max + 1):
            m = k - 3  # He index for this term is k-2; start at He_1 for k=3
            he = he_cur if k == 3 else he_next
            coefs.append(((-1) ** (k - 2)) * s ** (-(k - 1)) * he * pdf)
            he_next = a * he - (m + 1) * (he_prev if k == 3 else he_prev)
            he_prev, he_cur = he, he_next
    return coefs[:k_max]


def k2_propagate(ws, n_terms, collect_rho=False):
    mu = torch.zeros(N_W, dtype=torch.float64, device=DEV)
    cov = torch.eye(N_W, dtype=torch.float64, device=DEV)
    rho_stats = []
    for W in ws:
        Wd = W.double()
        mu_pre = mu @ Wd
        cov_pre = Wd.T @ (cov @ Wd)
        var = torch.clamp(torch.diagonal(cov_pre), min=1e-12)
        s = var.sqrt()
        a = mu_pre / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        if collect_rho:
            rho = cov_pre / torch.outer(s, s)
            off = rho[~torch.eye(N_W, dtype=bool, device=DEV)].abs()
            rho_stats.append((float(off.mean()), float(off.quantile(0.95)), float(off.max())))
        mu_post = s * pdf + mu_pre * cdf
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu_post**2, min=0.0)

        coefs = he_coefs(mu_pre, s, var, a, pdf, cdf, n_terms)
        cov_post = torch.zeros_like(cov_pre)
        Ck = cov_pre.clone()
        for k in range(1, n_terms + 1):
            hk = coefs[k - 1]
            cov_post += torch.outer(hk, hk) * Ck / math.factorial(k)
            if k < n_terms:
                Ck = Ck * cov_pre
        cov_post = cov_post - torch.diag(torch.diagonal(cov_post)) + torch.diag(var_post)
        mu, cov = mu_post, cov_post
    return mu.cpu(), rho_stats


print(f"{'seed':>4s} " + " ".join(f"{'k=' + str(t):>10s}" for t in TERM_COUNTS))
acc = dict.fromkeys(TERM_COUNTS, 0.0)
for seed in range(SEEDS):
    ws = build_ws(60_000 + seed)
    gt = gt_mean(ws, GT_N)
    row = []
    for t in TERM_COUNTS:
        est, rho = k2_propagate(ws, t, collect_rho=(seed == 0 and t == TERM_COUNTS[0]))
        mse = float(((est - gt) ** 2).mean())
        acc[t] += mse / SEEDS
        row.append(mse)
        if rho:
            print("  |rho| off-diagonal by layer (mean / p95 / max):")
            for li in (0, 1, 3, 7, 15, 31):
                m, p95, mx = rho[li]
                print(f"    L{li:2d}: {m:.3f} / {p95:.3f} / {mx:.3f}")
            print()
    print(f"{seed:4d} " + " ".join(f"{v:10.3e}" for v in row))
print(f"{'MEAN':>4s} " + " ".join(f"{acc[t]:10.3e}" for t in TERM_COUNTS))
print("\nIf MSE keeps dropping with more terms => G1 (truncation) was the gap.")
print("If it plateaus by k=4-8 => truncation is fine; go to G3 (rank-r K=3).")
