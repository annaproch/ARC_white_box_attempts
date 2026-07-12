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


# ---- FLOP model (validated: predicts v4's grader utilisation to ~1%) --------
B_FLOP = 2.72e11
PT_FLOP = DEPTH * (2 * N_W * N_W + 2 * N_W)          # per QMC point
K2_FLOP = 2.2e9                                      # v4's calibrated-K2 closure
CL_FLOP = DEPTH * (6.0 * N_W ** 3 + 4.0 * N_W ** 2 * rank
                   + 2.0 * N_W * rank ** 4 + 2.0 * rank ** 5)
V4_PTS = 6400
V4_C = V4_PTS * PT_FLOP + K2_FLOP                    # ~2.9e10 = 10.9% (matches grader)


def score(mse_val, C):
    return mse_val * max(0.1, C / B_FLOP)


def min_var_weights(E):
    """Affine weights (sum=1) minimising the variance of sum_i w_i e_i.
    Preserves calibration; cannot do worse than the best single component."""
    C = (E @ E.T) / E.shape[1]
    C = C + 1e-12 * np.eye(len(C)) * np.trace(C)
    inv1 = np.linalg.solve(C, np.ones(len(C)))
    return inv1 / inv1.sum()


est_q_full = qmc_final(Ws)
est_c = K.closure(Ws, theta)
est_k2 = k2_v4(Ws)
mse = lambda e: float((e ** 2).mean())

print(f"\nFLOP model: QMC pt {PT_FLOP:.2e} | K2 {K2_FLOP:.1e} | "
      f"closure(r={rank}) {CL_FLOP:.2e} ({CL_FLOP/B_FLOP:.1%} of budget)")

# v4 baseline on THESE MLPs
e_q, e_k2, e_c = est_q_full - gt, est_k2 - gt, est_c - gt
w2 = min_var_weights(np.stack([e_q.cpu().numpy().ravel(), e_k2.cpu().numpy().ravel()]))
mse_v4 = mse(w2[0] * e_q + w2[1] * e_k2)
s_v4 = score(mse_v4, V4_C)
print(f"\nv4 (QMC{V4_PTS} + K2): mse {mse_v4:.3e}  C {V4_C:.2e} "
      f"({V4_C/B_FLOP:.1%})  score {s_v4:.3e}   <- to beat")

# 3-way, holding TOTAL compute at v4's level: pay for the closure with QMC points
best = None
for n_pts in (6400, 5800, 5200, 4600, 4000):
    C = n_pts * PT_FLOP + K2_FLOP + CL_FLOP
    x = pts[:n_pts].unsqueeze(0).expand(len(seeds), -1, -1).contiguous()
    for li in range(DEPTH):
        x = torch.relu(torch.bmm(x, Ws[:, li]))
    e_qn = x.mean(1) - gt
    E = np.stack([e_qn.cpu().numpy().ravel(), e_k2.cpu().numpy().ravel(),
                  e_c.cpu().numpy().ravel()])
    w = min_var_weights(E)
    m = mse(w[0] * e_qn + w[1] * e_k2 + w[2] * e_c)
    s = score(m, C)
    print(f"  QMC{n_pts:5d}+K2+closure: mse {m:.3e}  C {C:.2e} ({C/B_FLOP:5.1%})  "
          f"score {s:.3e}   w = [{w[0]:+.2f} {w[1]:+.2f} {w[2]:+.2f}]")
    if best is None or s < best[0]:
        best = (s, n_pts, w, m, C)

s_best, n_best, w_best, m_best, C_best = best
print(f"\nbest 3-way: QMC{n_best} + K2 + closure -> score {s_best:.3e} "
      f"(v4 {s_v4:.3e})")

if s_best < s_v4 * 0.995:
    print(f"=> WINS by {s_v4/s_best:.3f}x — shipping.")
    np.savez(KIT / "closure.npz",
             theta=theta.cpu().numpy().astype(np.float32),
             meta=np.array([float(rank), float(n_best),
                            float(w_best[0]), float(w_best[1]), float(w_best[2])],
                           dtype=np.float32),
             **{"__meta__": np.frombuffer(b"{}", dtype=np.uint8)})
    print(f"wrote {KIT/'closure.npz'} (rank {rank}, pts {n_best}, "
          f"w {w_best.round(3).tolist()})")
    print("next:  cd ../whest-starterkit && uv run whest run --estimator . "
          "--split mini --runner subprocess")
else:
    print("=> no gain over v4. closure.npz NOT written; submission stays v4.")
    print(f"   closure alone is {mse(e_c):.2e}; it needs to get closer to "
          f"QMC's {mse(e_q):.2e} to earn its FLOPs.")
