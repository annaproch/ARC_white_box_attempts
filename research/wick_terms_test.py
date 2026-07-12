"""G1 test: is the Hermite/Wick truncation our 10x gap?

K=2 covariance propagation at the challenge shape with an increasing number
of Wick terms in the off-diagonal ReLU covariance update:

    Cov(ReLU(Zi), ReLU(Zj)) = sum_{k>=1} h_k(i) h_k(j) C_ij^k / k!

Stable dimensionless form used here (avoids sigma^-(k-1) overflow):
    g_k(i) := h_k(i) * sigma_i^(k-1) = (-1)^(k-2) He_{k-2}(alpha_i) phi(alpha_i)
              [g_1 = Phi(alpha)]
    Cov_post = outer(sigma, sigma) * sum_k rho^k outer(g_k, g_k) / k!
so convergence is governed purely by the correlation rho.

Shipped estimator uses 2 terms; torch experiments used 4. Also prints the
off-diagonal |rho| distribution per layer — the direct diagnostic.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/wick_terms_test.py
"""

import math

import torch

torch.set_grad_enabled(False)
SAMPLE_DEV = "mps" if torch.backends.mps.is_available() else "cpu"  # GT sampling (float32)
PROP_DEV = "cpu"  # propagation in float64
N_W, DEPTH = 256, 32
GT_N, BATCH = 2_000_000, 131_072
SEEDS = 6
TERM_COUNTS = [2, 4, 8, 16, 32, 64]
SQ2PI = math.sqrt(2 * math.pi)


def build_ws(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N_W, N_W, generator=g, dtype=torch.float64) * math.sqrt(2.0 / N_W)
            for _ in range(DEPTH)]


def gt_mean(ws, n):
    wd = [w.float().to(SAMPLE_DEV) for w in ws]
    s = torch.zeros(N_W, dtype=torch.float64)
    d = 0
    while d < n:
        b = min(BATCH, n - d)
        x = torch.randn(b, N_W, device=SAMPLE_DEV)
        for w in wd:
            x = torch.relu(x @ w)
        s += x.sum(0).cpu().double()
        d += b
    return s / n


def wick_g(a, pdf, cdf, k_max):
    """g_k for k = 1..k_max (dimensionless Wick coefficients)."""
    g = [cdf]                       # k=1: Phi(alpha)
    if k_max >= 2:
        g.append(pdf)               # k=2: phi(alpha)   [He_0 = 1]
    he_prev, he_cur = torch.ones_like(a), a  # He_0, He_1
    for k in range(3, k_max + 1):
        m = k - 2                   # need He_{k-2}; at k=3 that is He_1 = he_cur
        g.append(((-1) ** m) * he_cur * pdf)
        he_prev, he_cur = he_cur, a * he_cur - m * he_prev  # He_{m+1}
    return g[:k_max]


def k2_propagate(ws, n_terms, collect_rho=False):
    mu = torch.zeros(N_W, dtype=torch.float64, device=PROP_DEV)
    cov = torch.eye(N_W, dtype=torch.float64, device=PROP_DEV)
    rho_stats = []
    eye = torch.eye(N_W, dtype=torch.bool, device=PROP_DEV)
    for W in ws:
        mu_pre = mu @ W
        cov_pre = W.T @ (cov @ W)
        var = torch.clamp(torch.diagonal(cov_pre), min=1e-12)
        s = var.sqrt()
        a = mu_pre / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        rho = cov_pre / torch.outer(s, s)
        rho = torch.clamp(rho, -1.0, 1.0)
        if collect_rho:
            off = rho[~eye].abs()
            rho_stats.append((float(off.mean()), float(off.quantile(0.95)), float(off.max())))

        mu_post = s * pdf + mu_pre * cdf
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu_post**2, min=0.0)

        # Fold sqrt(k!) into each coefficient: term_k = rho^k * outer(g_k, g_k)/k!
        # = rho^k * outer(g_k/sqrt(k!), g_k/sqrt(k!)). Keeps He_k growth in check
        # and avoids bignum factorials.
        g = wick_g(a, pdf, cdf, n_terms)
        acc = torch.zeros_like(cov_pre)
        rho_k = rho.clone()
        sf = 1.0
        for k in range(1, n_terms + 1):
            sf *= math.sqrt(k)  # sf = sqrt(k!)
            gh = g[k - 1] / sf
            acc += torch.outer(gh, gh) * rho_k
            if k < n_terms:
                rho_k = rho_k * rho
        cov_post = torch.outer(s, s) * acc
        cov_post = cov_post - torch.diag(torch.diagonal(cov_post)) + torch.diag(var_post)
        mu, cov = mu_post, cov_post
    return mu, rho_stats


ws0 = build_ws(60_000)
_, rho = k2_propagate(ws0, 4, collect_rho=True)
print("off-diagonal |rho| by layer (mean / p95 / max):")
for li in (0, 1, 3, 7, 15, 23, 31):
    m, p95, mx = rho[li]
    print(f"  L{li:2d}: {m:.3f} / {p95:.3f} / {mx:.3f}")

print(f"\n{'seed':>4s} " + " ".join(f"{'k=' + str(t):>10s}" for t in TERM_COUNTS))
acc = dict.fromkeys(TERM_COUNTS, 0.0)
for seed in range(SEEDS):
    ws = build_ws(60_000 + seed)
    gt = gt_mean(ws, GT_N)
    row = []
    for t in TERM_COUNTS:
        est, _ = k2_propagate(ws, t)
        mse = float(((est - gt) ** 2).mean())
        acc[t] += mse / SEEDS
        row.append(mse)
    print(f"{seed:4d} " + " ".join(f"{v:10.3e}" for v in row))
print(f"{'MEAN':>4s} " + " ".join(f"{acc[t]:10.3e}" for t in TERM_COUNTS))
print("\nMSE keeps dropping with more terms => G1 (truncation) was the gap.")
print("Plateau by k=4-8            => truncation fine; go to G3 (rank-r K=3).")
