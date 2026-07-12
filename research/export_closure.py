"""Turn a calibration checkpoint into the shipped closure.npz.

Fits the fusion weight w on held-out MLPs, so a weak closure gets a small weight
and cannot regress the submission:

    w* = argmin_w  E|| (1-w) qmc + w closure - truth ||^2
    clipped to [0, 1]; if the closure alone beats the fused estimate, w = 1.

Compares like-for-like on the SAME held-out MLPs:
    QMC alone  |  QMC + v4's calibrated-K2 (what is on the leaderboard now)
               |  QMC + the new K=3/K=4 closure
The new closure only ships if it beats the v4 fusion. Otherwise we keep v4.

Writes whest-starterkit/closure.npz directly with numpy (flopscope.Module's
format is just np.savez of the public attributes plus a b'{}' __meta__ marker),
so this runs in the kprop venv.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/export_closure.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KIT = ROOT / "whest-starterkit"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)

sys.path.insert(0, str(ROOT / "research"))
import calibrate_k3k4 as K   # import is side-effect free (training lives in main())

ck = torch.load(DATA / "calib_k3k4.pt", map_location=DEV)
theta = ck["best_theta"].to(DEV)
rank = int(ck.get("rank", 16))
K.set_rank(rank)
print(f"checkpoint: step {ck['step']}, best val {ck['best_val']:.3e}, rank {rank}")

# ---- held-out MLPs (disjoint from training and from the val set that picked theta)
seeds = list(range(950_000, 950_000 + 16))
Ws = K.make_ws(seeds)
gt = K.mc_final(Ws, 2_000_000)
pts = torch.tensor(np.load(DATA / "points_v3.npy"), device=DEV)


def qmc_final(Ws):
    x = pts.unsqueeze(0).expand(Ws.shape[0], -1, -1).contiguous()
    for li in range(DEPTH):
        x = torch.relu(torch.bmm(x, Ws[:, li]))
    return x.mean(1)


def k2_v4(Ws):
    """v4's shipped calibrated-K=2 closure (6 params/layer), for a like-for-like baseline."""
    th = torch.tensor(np.load(ROOT / "research" / "theta_k2_v2.npy"), device=DEV)
    B = Ws.shape[0]
    mu = torch.zeros(B, N_W, device=DEV)
    cov = torch.eye(N_W, device=DEV).expand(B, N_W, N_W).contiguous()
    eye = torch.eye(N_W, device=DEV)
    for li in range(DEPTH):
        W = Ws[:, li]
        p, q, v, g1, g2, m = 0.05 * torch.tanh(th[li])
        mu_pre = torch.bmm(mu.unsqueeze(1), W).squeeze(1)
        cov_pre = torch.bmm(W.transpose(1, 2), torch.bmm(cov, W))
        var = torch.clamp(torch.diagonal(cov_pre, dim1=1, dim2=2), min=1e-10)
        s = var.sqrt()
        a = mu_pre / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        mu = s * pdf * (1 + p) + mu_pre * cdf * (1 + q) + m * s
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu * mu, min=1e-12) * (1 + v)
        w1 = cdf * (1 + g1)
        w2 = (pdf / s) * (1 + g2)
        cov = (w1.unsqueeze(2) * w1.unsqueeze(1) * cov_pre
               + w2.unsqueeze(2) * w2.unsqueeze(1) * cov_pre**2 / 2)
        cov = cov * (1 - eye) + torch.diag_embed(var_post)
    return mu


def fit_w(e_q, e_o):
    d = e_q - e_o
    w = float((e_q * d).sum() / (d * d).sum())
    return min(max(w, 0.0), 1.0)


est_q = qmc_final(Ws)
est_c = K.closure(Ws, theta)
est_k2 = k2_v4(Ws)
e_q, e_c, e_k2 = est_q - gt, est_c - gt, est_k2 - gt

mse = lambda e: float((e ** 2).mean())
w_new = fit_w(e_q, e_c)
w_v4 = fit_w(e_q, e_k2)
mse_new = mse((1 - w_new) * e_q + w_new * e_c)
mse_v4 = mse((1 - w_v4) * e_q + w_v4 * e_k2)
if mse(e_c) <= mse_new:
    w_new, mse_new = 1.0, mse(e_c)      # closure alone wins -> pure analytic

print(f"\nheld-out ({len(seeds)} MLPs), same MLPs for every row:")
print(f"  QMC alone                    {mse(e_q):.3e}")
print(f"  QMC + v4 calibrated-K2       {mse_v4:.3e}   (w={w_v4:.3f})  <- on the leaderboard")
print(f"  QMC + new K3/K4 closure      {mse_new:.3e}   (w={w_new:.3f})")
print(f"  closure alone                {mse(e_c):.3e}")

if mse_new < mse_v4:
    gain = mse_v4 / mse_new
    print(f"\n=> NEW CLOSURE WINS by {gain:.2f}x — shipping it.")
    np.savez(KIT / "closure.npz",
             theta=theta.cpu().numpy().astype(np.float32),
             meta=np.array([float(rank), float(w_new)], dtype=np.float32),
             **{"__meta__": np.frombuffer(b"{}", dtype=np.uint8)})
    print(f"wrote {KIT / 'closure.npz'}  (rank {rank}, w_fuse {w_new:.3f})")
    print("next:  cd ../whest-starterkit && uv run whest run --estimator . "
          "--split mini --runner subprocess")
else:
    short = mse_new / mse_v4
    print(f"\n=> not yet: the new closure's fusion is {short:.2f}x WORSE than v4's.")
    print(f"   Keep training (it resumes automatically); re-run this to re-check.")
    print(f"   closure needs val ~<{mse(e_k2):.1e} to overtake the K=2 fusion "
          f"(currently {mse(e_c):.1e}).")
    print("   closure.npz NOT written — the submitted estimator stays as v4.")
