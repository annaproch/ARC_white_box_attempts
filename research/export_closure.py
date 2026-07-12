"""Turn a calibration checkpoint into the shipped closure.npz.

Also FITS the fusion weight w on held-out MLPs, so a weak closure gets a small
weight and cannot regress the submission:

    w* = argmin_w  E|| (1-w)*qmc + w*closure - truth ||^2
       = <e_q, e_q - e_c> / ||e_q - e_c||^2      (e = errors)
    clipped to [0, 1]; if the closure alone is clearly better than the fused
    estimate, w is set to 1 (pure analytic).

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/export_closure.py
"""

import math
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KIT = ROOT / "whest-starterkit"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32

import sys
sys.path.insert(0, str(ROOT / "research"))
import calibrate_k3k4 as K   # import is side-effect free (training lives in main())

ck = torch.load(DATA / "calib_k3k4.pt", map_location=DEV)
theta = ck["best_theta"].to(DEV)
rank = int(ck.get("rank", 16))
K.set_rank(rank)               # the closure must run at the rank it was trained at
print(f"checkpoint: step {ck['step']}, best val {ck['best_val']:.3e}, rank {rank}")

# ---- held-out MLPs (disjoint from training and from the val set used to pick theta)
seeds = list(range(950_000, 950_000 + 16))
Ws = K.make_ws(seeds)
gt = K.mc_final(Ws, 2_000_000)
pts = torch.tensor(np.load(DATA / "points_v3.npy"), device=DEV)

est_c = K.closure(Ws, theta)
x = pts.unsqueeze(0).expand(len(seeds), -1, -1).contiguous()
for li in range(DEPTH):
    x = torch.relu(torch.bmm(x, Ws[:, li]))
est_q = x.mean(1)

e_c, e_q = est_c - gt, est_q - gt
mse_c = float((e_c ** 2).mean())
mse_q = float((e_q ** 2).mean())
d = e_q - e_c
w = float((e_q * d).sum() / (d * d).sum())
w = min(max(w, 0.0), 1.0)
fused = (1 - w) * est_q + w * est_c
mse_f = float(((fused - gt) ** 2).mean())
if mse_c <= mse_f:
    w, mse_f = 1.0, mse_c          # closure alone wins -> go pure analytic

print(f"held-out ({len(seeds)} MLPs): QMC {mse_q:.3e} | closure {mse_c:.3e} | "
      f"fused {mse_f:.3e}  -> w_fuse = {w:.3f}")
print(f"v4 raw MSE 3.12e-6  =>  {'BEATS v4' if mse_f < 3.12e-6 else 'does NOT beat v4 yet'}")

import flopscope
import flopscope.numpy as fnp


class ClosureParams(flopscope.Module):
    def __init__(self):
        self.theta = fnp.zeros((1, 1))
        self.meta = fnp.zeros((2,))


cp = ClosureParams()
cp.theta = fnp.array(theta.cpu().numpy().astype(np.float32))
cp.meta = fnp.array(np.array([float(rank), float(w)], dtype=np.float32))
out = KIT / "closure.npz"
cp.save(str(out))
print(f"\nwrote {out}  (theta {tuple(cp.theta.shape)}, rank {rank}, w_fuse {w:.3f})")
print("next:  uv run whest run --estimator . --split mini --runner subprocess")
