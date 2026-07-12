"""v4.1 candidate: per-MLP adaptive fusion weight.

w_m = vhat_m / (vhat_m + b2_global), where vhat_m is the QMC variance of the
mean estimated from the 6400 samples themselves (free in-budget) and
b2_global is the calibrated-K2 bias^2 fit offline. Compared against the
global w=0.137 of v4, on held-out MLPs; reports mean and worst-MLP MSE.
"""

import math

import numpy as np
import torch

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
GT_N, BATCH = 2_000_000, 131_072
M_TRAIN, M_VAL = 24, 32
SCRATCH = "/private/tmp/claude-501/-Users-lisek-Projects-ARC-white-box/fffd79fe-57cb-423a-9b9c-e89e35bd048e/scratchpad"

theta = torch.tensor(np.load(f"{SCRATCH}/theta_k2_v2.npy"), device=DEV)
P = torch.tensor(np.load(f"{SCRATCH}/points_v3.npy"), device=DEV)
W_GLOBAL = 0.137


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


def k2_cal(ws):
    mu = torch.zeros(N_W, device=DEV)
    cov = torch.eye(N_W, device=DEV)
    for l, W in enumerate(ws):
        p, q, v, g1, g2, m = 0.05 * torch.tanh(theta[l])
        mu_pre = mu @ W
        cov_pre = W.T @ (cov @ W)
        var = torch.clamp(torch.diagonal(cov_pre), min=1e-10)
        s = var.sqrt()
        a = mu_pre / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        mu_post = s * pdf * (1 + p) + mu_pre * cdf * (1 + q) + m * s
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu_post**2, min=1e-12) * (1 + v)
        w1 = cdf * (1 + g1)
        w2 = (pdf / s) * (1 + g2)
        cov_post = torch.outer(w1, w1) * cov_pre + torch.outer(w2, w2) * cov_pre**2 / 2
        cov_post = cov_post - torch.diag(torch.diagonal(cov_post)) + torch.diag(var_post)
        mu, cov = mu_post, cov_post
    return mu.cpu().double()


def qmc_with_var(ws):
    x = P
    for w in ws:
        x = torch.relu(x @ w)
    m = x.mean(0).cpu().double()
    vhat = float(x.var(dim=0).mean()) / P.shape[0]  # avg per-neuron variance of the mean
    return m, vhat


def collect(seeds):
    rows = []
    for sd in seeds:
        ws = build_ws(sd)
        gt = gt_mean(ws, GT_N)
        mq, vhat = qmc_with_var(ws)
        rows.append((mq - gt, k2_cal(ws) - gt, vhat))
    return rows


tr = collect(range(700_000, 700_000 + M_TRAIN))
va = collect(range(820_000, 820_000 + M_VAL))

b2 = float(np.mean([float((ek**2).mean()) for _, ek, _ in tr]))
print(f"b2_global (train): {b2:.3e}")

res = {"qmc_only": [], "global_w": [], "adaptive_w": []}
for eq, ek, vhat in va:
    res["qmc_only"].append(float((eq**2).mean()))
    f = (1 - W_GLOBAL) * eq + W_GLOBAL * ek
    res["global_w"].append(float((f**2).mean()))
    wm = vhat / (vhat + b2)
    f = (1 - wm) * eq + wm * ek
    res["adaptive_w"].append(float((f**2).mean()))

print(f"\nval ({M_VAL} MLPs):      mean        worst       p75")
for k, v in res.items():
    v = np.array(v)
    print(f"  {k:12s} {v.mean():.3e}  {v.max():.3e}  {np.percentile(v,75):.3e}")
ws_used = [vhat / (vhat + b2) for _, _, vhat in va]
print(f"\nadaptive w range: {min(ws_used):.3f} .. {max(ws_used):.3f} (global {W_GLOBAL})")
