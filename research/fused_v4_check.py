"""Honest end-to-end check of v4 candidate: calibrated-K2 + QMC fusion.

Budget accounting variants:
  A) 6400 QMC points + K2 (total ~11.2% of B, soft multiplier)
  B) 5510 QMC points + K2 (total ~10.06% like v3)
  C) v3 baseline: 6400 points, no K2
Global fusion weight w fit on train MLPs, evaluated on held-out MLPs.
Also reports the error correlation (independence check).
"""

import math

import numpy as np
import torch

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
GT_N, BATCH = 2_000_000, 131_072
M_TRAIN, M_VAL = 24, 24
SCRATCH = "/private/tmp/claude-501/-Users-lisek-Projects-ARC-white-box/fffd79fe-57cb-423a-9b9c-e89e35bd048e/scratchpad"
B = 2.72e11
FPS = DEPTH * (2 * N_W**2 + 2 * N_W)
K2_FLOPS = 7 * N_W**3 * DEPTH  # paper Table 1 leading term

theta = torch.tensor(np.load(f"{SCRATCH}/theta_k2_v2.npy"), device=DEV)
P = torch.tensor(np.load(f"{SCRATCH}/points_v3.npy"), device=DEV)


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


def qmc(ws, n_pts):
    x = P[:n_pts]
    for w in ws:
        x = torch.relu(x @ w)
    return x.mean(0).cpu().double()


def collect(seeds):
    rows = []
    for sd in seeds:
        ws = build_ws(sd)
        gt = gt_mean(ws, GT_N)
        rows.append((qmc(ws, 6400) - gt, qmc(ws, 5510) - gt, k2_cal(ws) - gt))
    return [torch.stack(c) for c in zip(*rows)]


tr = collect(range(700_000, 700_000 + M_TRAIN))
va = collect(range(800_000, 800_000 + M_VAL))

for name, (eq_full, eq_cut, ek) in [("train", tr), ("val", va)]:
    v_full = float((eq_full**2).mean())
    v_cut = float((eq_cut**2).mean())
    b2 = float((ek**2).mean())
    rho = float((eq_cut * ek).mean() / math.sqrt(v_cut * b2))
    print(f"{name}: qmc6400 {v_full:.3e}  qmc5510 {v_cut:.3e}  k2cal {b2:.3e}  corr {rho:+.3f}")

# fit global w on train (variant A and B), evaluate on val
for label, idx_q, n_pts, util in [("A: 6400+K2", 0, 6400, (6400 * FPS + K2_FLOPS) / B + 0.0016),
                                  ("B: 5510+K2", 1, 5510, (5510 * FPS + K2_FLOPS) / B + 0.0016)]:
    eq_tr, ek_tr = tr[idx_q], tr[2]
    d = ek_tr - eq_tr
    w = float(-(eq_tr * d).sum() / (d * d).sum())  # min ||eq + w(ek-eq)||^2
    eq_va, ek_va = va[idx_q], va[2]
    fused = (1 - w) * eq_va + w * ek_va
    mse = float((fused**2).mean())
    mult = max(0.1, util)
    print(f"{label}: w_k2={w:.3f}  val fused MSE {mse:.3e}  util {util:.4f}  "
          f"score {mse*mult:.3e}")

v3_mse = float((va[0] ** 2).mean())
print(f"C: v3 (6400, no K2): val MSE {v3_mse:.3e}  score {v3_mse*0.1006:.3e}")
