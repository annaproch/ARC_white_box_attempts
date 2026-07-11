"""Measure QMC convergence exponent beta (MSE ~ N^-beta) at 256x32.

Decides the operating point: score(N) = MSE(N) * max(0.1, N/N_full).
If beta > 1, full budget beats the 10% floor.
N_full = 2.72e11 / 4.21e6 ~ 64.6k samples; floor ~ 6.46k.
"""

import math

import torch

from mlp_kprop.mlp import MLP

torch.set_grad_enabled(False)

N_W, DEPTH = 256, 32
GT_SAMPLES = 8_000_000
BATCH = 131_072
NS = [1614, 3229, 6459, 12919, 25839, 51678, 64598]  # 2.5%..80%..100% of budget
device = "mps" if torch.backends.mps.is_available() else "cpu"


def make_points(n_pts, seed):
    sob = torch.quasirandom.SobolEngine(N_W, scramble=True, seed=seed)
    u = sob.draw(n_pts, dtype=torch.float64).clamp(1e-9, 1 - 1e-9)
    x = math.sqrt(2.0) * torch.erfinv(2 * u - 1)
    x = x - x.mean(dim=0)
    L = torch.linalg.cholesky((x.T @ x) / n_pts)
    return torch.linalg.solve_triangular(L, x.T, upper=False).T.float()


POINTS = {n: make_points(n, 12345) for n in NS}
IID = {n: torch.randn(n, N_W, generator=torch.Generator().manual_seed(777)) for n in [6459, 64598]}

acc_q = dict.fromkeys(NS, 0.0)
acc_i = dict.fromkeys(IID, 0.0)
SEEDS = 8
for seed in range(SEEDS):
    torch.manual_seed(seed)
    mlp = MLP(hidden_dim=N_W, num_layers=DEPTH + 1, nonlin="relu",
              init_kind="manual", w_var=2.0)
    for Wm in mlp.Ws:
        Wm.weight.copy_(torch.randn(N_W, N_W) * math.sqrt(2.0 / N_W))
        if Wm.bias is not None:
            Wm.bias.zero_()
    Ws = [mlp.Ws[l].weight.to(device) for l in range(DEPTH)]

    sums = torch.zeros(N_W, dtype=torch.float64)
    done = 0
    while done < GT_SAMPLES:
        b = min(BATCH, GT_SAMPLES - done)
        x = torch.randn(b, N_W, device=device)
        for l in range(DEPTH):
            x = torch.relu(x @ Ws[l].T)
        sums += x.sum(dim=0).cpu().double()
        done += b
    gt = sums / GT_SAMPLES

    def fwd_mean(x0):
        x = x0.to(device)
        for l in range(DEPTH):
            x = torch.relu(x @ Ws[l].T)
        return x.mean(dim=0).cpu().double()

    for n, p in POINTS.items():
        acc_q[n] += float(((fwd_mean(p) - gt) ** 2).mean()) / SEEDS
    for n, p in IID.items():
        acc_i[n] += float(((fwd_mean(p) - gt) ** 2).mean()) / SEEDS

N_FULL = 64598
print(f"{'N':>7s} {'util':>6s} {'qmc_mse':>10s} {'score':>10s}   (score = mse * max(0.1, N/N_full))")
for n in NS:
    mult = max(0.1, n / N_FULL)
    print(f"{n:7d} {n/N_FULL:6.1%} {acc_q[n]:10.3e} {acc_q[n]*mult:10.3e}")
for n, v in acc_i.items():
    mult = max(0.1, n / N_FULL)
    print(f"iid {n:6d} {n/N_FULL:5.1%} {v:10.3e} {v*mult:10.3e}")

import numpy as np
lg = np.log(np.array(NS, dtype=float))
lm = np.log(np.array([acc_q[n] for n in NS]))
beta = -np.polyfit(lg, lm, 1)[0]
print(f"\nfitted beta (QMC): {beta:.3f}  (beta>1 => run above the floor)")
