"""Does cumulant propagation (kprop) beat the K=2 covariance baseline at the
challenge's phase-1 shape (width 256, 32 ReLU layers, He init, no bias)?

Compares kprop k_max=1/2/3(factorized) against Monte Carlo ground truth and
reports final-layer MSE, all-layers MSE, FLOPs (symmetric-adjusted and raw),
and wall time. Challenge budget: B=2.72e11, score floor at 0.1*B=2.72e10.
"""

import math
import time

import torch

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.flop_utils import NamedFlopCounter

torch.set_grad_enabled(False)
torch.manual_seed(0)

N_WIDTH, DEPTH = 256, 32  # 32 ReLU layers (challenge phase-1 shape)
MC_SAMPLES = 8_000_000
BATCH = 131_072
BUDGET = 2.72e11

# num_layers = DEPTH+1 linear layers => ReLU after the first DEPTH of them;
# we stop at act{DEPTH-1}, so the last linear layer is never used.
mlp = MLP(hidden_dim=N_WIDTH, num_layers=DEPTH + 1, nonlin="relu",
          init_kind="manual", w_var=2.0)
for Wm in mlp.Ws:
    Wm.weight.copy_(torch.randn(N_WIDTH, N_WIDTH) * math.sqrt(2.0 / N_WIDTH))
    if Wm.bias is not None:
        Wm.bias.zero_()

# ---------- Monte Carlo ground truth ----------
device = "mps" if torch.backends.mps.is_available() else "cpu"
Ws = [mlp.Ws[l].weight.to(device) for l in range(DEPTH)]
sums = torch.zeros(DEPTH, N_WIDTH, dtype=torch.float64)
sqsums = torch.zeros(DEPTH, N_WIDTH, dtype=torch.float64)
t0 = time.time()
done = 0
while done < MC_SAMPLES:
    b = min(BATCH, MC_SAMPLES - done)
    x = torch.randn(b, N_WIDTH, device=device)
    for l in range(DEPTH):
        x = torch.relu(x @ Ws[l].T)
        sums[l] += x.sum(dim=0).cpu().double()
        sqsums[l] += (x * x).sum(dim=0).cpu().double()
    done += b
mc_mean = sums / MC_SAMPLES
mc_var = sqsums / MC_SAMPLES - mc_mean**2
noise_floor_final = float(mc_var[-1].mean()) / MC_SAMPLES
noise_floor_all = float(mc_var.mean()) / MC_SAMPLES
print(f"MC ground truth: N={MC_SAMPLES:.1e} on {device}, {time.time()-t0:.1f}s")
print(f"  MC noise-floor MSE  final layer: {noise_floor_final:.3e}   all layers: {noise_floor_all:.3e}")
print(f"  final-layer mean magnitude: mean|mu|={float(mc_mean[-1].abs().mean()):.4f}, "
      f"mean var={float(mc_var[-1].mean()):.4f}")

# ---------- kprop variants ----------
K_in = {1: torch.zeros(N_WIDTH), 2: torch.eye(N_WIDTH)}
variants = [
    ("k_max=1 (mean prop class)", dict(k_max=1, factor=False)),
    ("k_max=2 (cov prop class)", dict(k_max=2, factor=False)),
    ("k_max=3 factorized", dict(k_max=3, factor=True)),
    ("k_max=3 factorized augmented", dict(k_max=3, factor=True, kind=Kind.AUGMENT)),
]

print(f"\n{'variant':32s} {'final_mse':>11s} {'all_mse':>11s} {'flops_adj':>10s} "
      f"{'flops_raw':>10s} {'%budget':>8s} {'time':>7s}")
for name, kwargs in variants:
    t0 = time.time()
    try:
        with NamedFlopCounter() as counter:
            K_by_layer = mlp_kprop(
                mlp, {k: v.clone() for k, v in K_in.items()},
                output_all=True, output_d_max=1,
                up_to_layer=f"act{DEPTH-1}", **kwargs,
            )
        flops_adj, flops_raw = counter.total(), counter.raw_total()
    except Exception as e:  # noqa: BLE001
        print(f"{name:32s} FAILED: {type(e).__name__}: {e}")
        continue
    dt = time.time() - t0
    means = torch.stack(
        [K_by_layer[f"act{l}"][1].to_tensor().to(torch.float64) for l in range(DEPTH)]
    )
    err = means - mc_mean
    final_mse = float((err[-1] ** 2).mean())
    all_mse = float((err**2).mean())
    print(f"{name:32s} {final_mse:11.3e} {all_mse:11.3e} {flops_adj:10.2e} "
          f"{flops_raw:10.2e} {100*flops_raw/BUDGET:7.2f}% {dt:6.1f}s")

print(f"\n(challenge budget B={BUDGET:.2e}; score-multiplier floor at {0.1*BUDGET:.2e} FLOPs)")
print("Reference (starterkit, public mini split): mean_prop final_mse ~9.5e-4, "
      "cov_prop(gain approx) ~8.4e-5")
