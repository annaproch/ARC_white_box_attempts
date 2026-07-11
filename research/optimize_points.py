"""Optimize the quadrature point set by SGD on the true MLP ensemble.

Loss per step (fresh random MLP W, fresh Gaussian batch X — no reuse, no
overfitting):  ||m_P||^2 - 2<m_P, m_X>  which is an unbiased estimator of
E_W ||m_P - E_x act||^2 + const.  Rows are re-projected to the exact-radius
sphere after each step (radial integration is analytic by 1-homogeneity).
"""

import math
import time

import numpy as np
import torch

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH, N_PTS = 256, 32, 6272
SCRATCH = "/private/tmp/claude-501/-Users-lisek-Projects-ARC-white-box/fffd79fe-57cb-423a-9b9c-e89e35bd048e/scratchpad"
E_R = math.sqrt(2.0) * math.exp(math.lgamma((N_W + 1) / 2) - math.lgamma(N_W / 2))

STEPS = 3000
MLPS_PER_STEP = 4
X_BATCH = 8192
EVAL_EVERY = 250
VAL_MLPS, GT_VAL, BATCH = 24, 2_000_000, 131_072


def build_ws(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [(torch.randn(N_W, N_W, generator=g) * math.sqrt(2.0 / N_W)).to(DEV)
            for _ in range(DEPTH)]


def fwd(ws, x):
    for w in ws:
        x = torch.relu(x @ w)  # weights stored (in,out)-equivalent: symmetric law
    return x


def gt_mean(ws, n):
    s = torch.zeros(N_W, dtype=torch.float64)
    d = 0
    while d < n:
        b = min(BATCH, n - d)
        s += fwd(ws, torch.randn(b, N_W, device=DEV)).sum(0).cpu().double()
        d += b
    return s / n


# validation set (fixed, with ground truth)
val = []
t0 = time.time()
for m in range(VAL_MLPS):
    ws = build_ws(990_000 + m)
    val.append((ws, gt_mean(ws, GT_VAL)))
print(f"val GT ready ({time.time()-t0:.0f}s)")


def val_mse(P):
    tot = 0.0
    for ws, gt in val:
        m = fwd(ws, P).mean(0).cpu().double()
        tot += float(((m - gt) ** 2).mean()) / len(val)
    return tot


P0 = torch.tensor(np.load(f"{SCRATCH}/points_sph.npy"), device=DEV)
print(f"init val MSE: {val_mse(P0):.3e}")

torch.set_grad_enabled(True)
P = P0.clone().requires_grad_()
opt = torch.optim.Adam([P], lr=2e-3)
best = (val_mse(P0), P0.clone())
t0 = time.time()
for step in range(1, STEPS + 1):
    opt.zero_grad()
    loss = torch.zeros((), device=DEV)
    for k in range(MLPS_PER_STEP):
        ws = build_ws(1_000_000 + step * MLPS_PER_STEP + k)
        m_P = fwd(ws, P).mean(0)
        with torch.no_grad():
            m_X = fwd(ws, torch.randn(X_BATCH, N_W, device=DEV)).mean(0)
        loss = loss + ((m_P * m_P).sum() - 2 * (m_P * m_X).sum()) / MLPS_PER_STEP
    loss.backward()
    opt.step()
    with torch.no_grad():
        P.data = P.data / P.data.norm(dim=1, keepdim=True) * E_R
    if step % EVAL_EVERY == 0:
        with torch.no_grad():
            v = val_mse(P.data)
        if v < best[0]:
            best = (v, P.data.clone())
        print(f"step {step:5d}  val {v:.3e}  best {best[0]:.3e}  ({time.time()-t0:.0f}s)")

np.save(f"{SCRATCH}/points_opt.npy", best[1].cpu().numpy())
print(f"final best val MSE {best[0]:.3e} (init {val_mse(P0):.3e}); saved points_opt.npy")
