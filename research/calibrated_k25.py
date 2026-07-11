"""End-to-end calibrated K=2: learned per-layer closure corrections (192
universal scalars), trained by autograd through the covariance propagation
on fresh random MLPs each step (unbiased loss, no overfitting channel).
"""

import math
import time

import torch

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
GT_N, BATCH, M_VAL = 2_000_000, 131_072, 24
STEPS, X_BATCH, MLPS_STEP = 2500, 8192, 4


def build_ws(seed, dev=DEV):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(N_W, N_W, generator=g) * math.sqrt(2.0 / N_W)).to(dev)
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


def k2_forward(ws, theta):
    """K=2.5: calibrated covariance propagation + exact diagonal-3rd-cumulant
    tracking with Edgeworth correction terms, all scales learned (theta=0
    reproduces plain K=2)."""
    mu = torch.zeros(N_W, device=DEV)
    cov = torch.eye(N_W, device=DEV)
    k3 = torch.zeros(N_W, device=DEV)
    for l, W in enumerate(ws):
        t = 0.05 * torch.tanh(theta[l])
        p, q, v, g1, g2, m = t[0], t[1], t[2], t[3], t[4], t[5]
        pa, qa, g1a, g2a = t[6], t[7], t[8], t[9]
        e_mu, e_var = 20 * t[10], 20 * t[11]  # Edgeworth-term scales (O(1) range)
        mu_pre = mu @ W
        cov_pre = W.T @ (cov @ W)
        k3_pre = k3 @ (W**3)
        var = torch.clamp(torch.diagonal(cov_pre), min=1e-10)
        s = var.sqrt()
        a = mu_pre / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        apdf = a * pdf
        # Edgeworth mean/second-moment corrections from the tracked kappa3
        d_mu = -k3_pre * apdf / (6 * var)
        d_ez2 = k3_pre * pdf / (3 * s)
        mu_post = (s * pdf * (1 + p + pa * apdf) + mu_pre * cdf * (1 + q + qa * apdf)
                   + m * s + e_mu * d_mu)
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf + e_var * d_ez2
        var_post = torch.clamp(ez2 - mu_post**2, min=1e-12) * (1 + v)
        # post-ReLU diagonal 3rd cumulant: Gaussian-injected part (exact) +
        # linear-gain pass-through of the incoming kappa3
        ez3 = (mu_pre**3 + 3 * mu_pre * var) * cdf + (mu_pre**2 + 2 * var) * s * pdf
        k3 = (ez3 - 3 * mu_post * ez2 + 2 * mu_post**3) + (cdf**3) * k3_pre
        w1 = cdf * (1 + g1 + g1a * apdf)
        w2 = (pdf / s) * (1 + g2 + g2a * apdf)
        cov_post = torch.outer(w1, w1) * cov_pre + torch.outer(w2, w2) * cov_pre**2 / 2
        cov_post = cov_post - torch.diag(torch.diagonal(cov_post)) + torch.diag(var_post)
        mu, cov = mu_post, cov_post
    return mu


# validation (fixed MLPs + ground truth)
val = []
t0 = time.time()
for m in range(M_VAL):
    ws = build_ws(80_000 + m)
    val.append((ws, gt_mean(ws, GT_N)))
print(f"val ready ({time.time()-t0:.0f}s)")


def val_mse(theta):
    tot = 0.0
    for ws, gt in val:
        est = k2_forward(ws, theta).cpu().double()
        tot += float(((est - gt) ** 2).mean()) / M_VAL
    return tot


theta0 = torch.zeros(DEPTH, 12, device=DEV)
print(f"plain K=2 val MSE: {val_mse(theta0):.3e}")

torch.set_grad_enabled(True)
theta = torch.zeros(DEPTH, 12, device=DEV, requires_grad=True)
opt = torch.optim.Adam([theta], lr=2e-4)
best = (val_mse(theta0), theta0.clone())
t0 = time.time()
for step in range(1, STEPS + 1):
    opt.zero_grad()
    loss = torch.zeros((), device=DEV)
    for k in range(MLPS_STEP):
        ws = build_ws(2_000_000 + step * MLPS_STEP + k)
        est = k2_forward(ws, theta)
        with torch.no_grad():
            x = torch.randn(X_BATCH, N_W, device=DEV)
            for w in ws:
                x = torch.relu(x @ w)
            m_x = x.mean(0)
        loss = loss + ((est * est).sum() - 2 * (est * m_x).sum()) / MLPS_STEP
    loss.backward()
    torch.nn.utils.clip_grad_norm_([theta], 1.0)
    opt.step()
    if step % 150 == 0:
        with torch.no_grad():
            v = val_mse(theta.detach())
        if v < best[0]:
            best = (v, theta.detach().clone())
        print(f"step {step:5d}  val {v:.3e}  best {best[0]:.3e}  ({time.time()-t0:.0f}s)")

b2 = best[0]
v_q = 4.08e-06
fused = b2 * v_q / (b2 + v_q)
print(f"\nend-to-end calibrated K=2: {b2:.3e}  (plain: {val_mse(theta0):.3e})")
print(f"fused with floor-QMC -> MSE ~{fused:.3e}, score ~{fused*0.1:.3e} (v3: 3.99e-07)")
import numpy as np
np.save("/private/tmp/claude-501/-Users-lisek-Projects-ARC-white-box/fffd79fe-57cb-423a-9b9c-e89e35bd048e/scratchpad/theta_k25.npy",
        best[1].cpu().numpy())
