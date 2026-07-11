"""Hybrid estimator test at challenge shape (256 x 32):
kprop K=2 (analytic) + Monte Carlo filling the free floor budget (2.72e10 FLOPs),
combined per-neuron by inverse-MSE weighting. 3 seeds.
"""

import math
import time

import torch

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop

torch.set_grad_enabled(False)

N_WIDTH, DEPTH = 256, 32
GT_SAMPLES = 8_000_000
BATCH = 131_072
FLOOR = 2.72e10  # 10% of budget: compute below this is free
FLOPS_PER_SAMPLE = DEPTH * (2 * N_WIDTH**2 + N_WIDTH)  # matmul + relu per layer
KPROP2_FLOPS = 2.3e9
RNG_MARGIN = 0.80  # leave headroom for RNG draws, residual wall-time, etc.
N_MC = int(RNG_MARGIN * (FLOOR - KPROP2_FLOPS) / FLOPS_PER_SAMPLE)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"MC samples within floor: {N_MC}  (floor {FLOOR:.2e}, {FLOPS_PER_SAMPLE:.2e} FLOPs/sample)")
print(f"{'seed':>4s} {'K2_mse':>10s} {'MC_mse':>10s} {'hyb_scalar':>10s} {'hyb_oracle':>10s}")

for seed in range(3):
    torch.manual_seed(seed)
    mlp = MLP(hidden_dim=N_WIDTH, num_layers=DEPTH + 1, nonlin="relu",
              init_kind="manual", w_var=2.0)
    for Wm in mlp.Ws:
        Wm.weight.copy_(torch.randn(N_WIDTH, N_WIDTH) * math.sqrt(2.0 / N_WIDTH))
        if Wm.bias is not None:
            Wm.bias.zero_()
    Ws = [mlp.Ws[l].weight.to(device) for l in range(DEPTH)]

    # ground truth (high-N MC)
    sums = torch.zeros(N_WIDTH, dtype=torch.float64)
    done = 0
    while done < GT_SAMPLES:
        b = min(BATCH, GT_SAMPLES - done)
        x = torch.randn(b, N_WIDTH, device=device)
        for l in range(DEPTH):
            x = torch.relu(x @ Ws[l].T)
        sums += x.sum(dim=0).cpu().double()
        done += b
    gt_final = sums / GT_SAMPLES

    # kprop K=2
    K_in = {1: torch.zeros(N_WIDTH), 2: torch.eye(N_WIDTH)}
    K_by_layer = mlp_kprop(mlp, K_in, k_max=2, output_all=True, output_d_max=1,
                           up_to_layer=f"act{DEPTH-1}")
    k2_final = K_by_layer[f"act{DEPTH-1}"][1].to_tensor().double()

    # budget-floor MC estimate (antithetic pairs) + per-neuron variance estimate
    x = torch.randn(N_MC // 2, N_WIDTH, device=device)
    x = torch.cat([x, -x], dim=0)
    for l in range(DEPTH):
        x = torch.relu(x @ Ws[l].T)
    mc_final = x.mean(dim=0).cpu().double()
    mc_var = x.var(dim=0).cpu().double() / x.shape[0]  # variance of the MC mean

    k2_mse = float(((k2_final - gt_final) ** 2).mean())
    mc_mse = float(((mc_final - gt_final) ** 2).mean())

    # scalar-weight hybrid: b2 = global K2 bias^2 (would come from public split offline)
    b2 = k2_mse
    w_mc = b2 / (b2 + mc_var)  # per-neuron MC weight, scalar b2
    hyb = w_mc * mc_final + (1 - w_mc) * k2_final
    hyb_mse = float(((hyb - gt_final) ** 2).mean())

    # oracle per-neuron weights (upper bound on what better weighting could buy)
    b2_or = (k2_final - gt_final) ** 2
    w_or = b2_or / (b2_or + mc_var)
    hyb_or = w_or * mc_final + (1 - w_or) * k2_final
    hyb_or_mse = float(((hyb_or - gt_final) ** 2).mean())

    print(f"{seed:4d} {k2_mse:10.3e} {mc_mse:10.3e} {hyb_mse:10.3e} {hyb_or_mse:10.3e}")
