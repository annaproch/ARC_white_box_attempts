"""kprop K=2 kind=AUGMENT at 256x32 - the one affordable analytic variant not yet tested."""

import math

import torch

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from mlp_kprop.flop_utils import NamedFlopCounter

torch.set_grad_enabled(False)
N_W, DEPTH = 256, 32
GT_SAMPLES = 4_000_000
BATCH = 131_072
device = "mps" if torch.backends.mps.is_available() else "cpu"

for seed in range(3):
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

    K_in = {1: torch.zeros(N_W), 2: torch.eye(N_W)}
    with NamedFlopCounter() as counter:
        out = mlp_kprop(mlp, K_in, k_max=2, kind=Kind.AUGMENT, output_all=True,
                        output_d_max=1, up_to_layer=f"act{DEPTH-1}")
    est = out[f"act{DEPTH-1}"][1].to_tensor().double()
    print(f"seed {seed}: K2-augment final mse={float(((est-gt)**2).mean()):.3e} "
          f"flops={counter.raw_total():.2e}")
