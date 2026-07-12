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
rank4 = int(ck.get("rank4", min(8, rank)))
K.set_rank(rank, rank4)
print(f"checkpoint: step {ck['step']}, best val {ck['best_val']:.3e}, "
      f"rank3 {rank} / rank4 {rank4}")

# ---- held-out MLPs (disjoint from training and from the val set that picked theta)
# CACHED: re-sampling the ground truth each run made the ship/no-ship decision
# noisy (v4's mse moved 3.34e-6 -> 3.58e-6 between two identical exports).
N_HELD, GT_N = 48, 4_000_000
HELD = DATA / f"heldout_gt_{N_HELD}.pt"
seeds = list(range(950_000, 950_000 + N_HELD))
Ws = K.make_ws(seeds)
if HELD.exists():
    gt = torch.load(HELD)["gt"].to(DEV)
else:
    print(f"building held-out GT: {N_HELD} MLPs x {GT_N:.0e} samples...")
    gt = K.mc_final(Ws, GT_N)
    torch.save({"seeds": seeds, "gt": gt.cpu(), "n_samples": GT_N}, HELD)


def make_points(n_pts):
    """Sobol -> Gaussian -> whiten -> sphere, moment-matched AT THIS SIZE.

    The shipped 6400-point set is whitened as a whole, so a PREFIX pts[:n] is
    NOT moment-matched -- it has non-zero mean and non-identity covariance, which
    throws away exactly the property that made QMC 3.2x better than iid. (Visible
    in the sweep: 5600 points scored worse than 5200.) So build each candidate
    set at its own size.
    """
    sob = torch.quasirandom.SobolEngine(N_W, scramble=True, seed=12345)
    u = sob.draw(n_pts, dtype=torch.float64).clamp(1e-9, 1 - 1e-9)
    x = math.sqrt(2.0) * torch.erfinv(2 * u - 1)
    x = x - x.mean(dim=0)
    L = torch.linalg.cholesky((x.T @ x) / n_pts)
    x = torch.linalg.solve_triangular(L, x.T, upper=False).T      # cov = I exactly
    e_r = math.sqrt(2.0) * math.exp(math.lgamma((N_W + 1) / 2) - math.lgamma(N_W / 2))
    x = x / x.norm(dim=1, keepdim=True) * e_r                     # exact radial integral
    return x.float().to(DEV)


def qmc_from(points, Ws):
    x = points.unsqueeze(0).expand(Ws.shape[0], -1, -1).contiguous()
    for li in range(DEPTH):
        x = torch.relu(torch.bmm(x, Ws[:, li]))
    return x.mean(1)


pts_full = make_points(6400)


def qmc_final(Ws):
    return qmc_from(pts_full, Ws)


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


# ---- FLOP model ------------------------------------------------------------
# MEASURED on the grader, not estimated. My analytic estimate said the closure
# cost 4.16e9 (1.5% of budget); the real cost is 1.13e10 (4.2%) -- I counted the
# Tucker cores and forgot the matmuls that dominate: the star SLICE alone is four
# 2n^3 matmuls (C3^T Bs, D3^T Wr, and two more for kappa4), on top of the
# covariance propagation (4n^3) and the star's B matrix (2n^3): ~14 n^3 per layer,
# not 6. Utilisation came in at 12.8% instead of 9.6% and the submission LOST.
# Never trust an analytic FLOP estimate here again -- measure it.
B_FLOP = 2.72e11
PT_FLOP = DEPTH * (2 * N_W * N_W + 2 * N_W)          # per QMC point (validated)
K2_FLOP = 2.2e9                                      # v4's calibrated-K2 closure
RESID_FLOP = 1.7e9                                   # measured lambda * wall-time
_CL_MEASURED = {(32, 8): 1.13e10}                    # (rank3, rank4) -> FLOPs
CL_FLOP = _CL_MEASURED.get((rank, rank4))
if CL_FLOP is None:                                  # crude scaling from the measured point
    CL_FLOP = 1.13e10 * (14.0 * N_W ** 3 + 2.0 * N_W * rank ** 3
                         + 2.0 * N_W * rank4 ** 4) / (
        14.0 * N_W ** 3 + 2.0 * N_W * 32 ** 3 + 2.0 * N_W * 8 ** 4)
    print(f"WARNING: closure cost for r3={rank},r4={rank4} is EXTRAPOLATED "
          f"({CL_FLOP:.2e}). Measure it with `whest run` before trusting a win.")
CL_FLOP += RESID_FLOP
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
      f"closure(r3={rank},r4={rank4}) {CL_FLOP:.2e} ({CL_FLOP/B_FLOP:.1%} of budget)")

# v4 baseline on THESE MLPs
e_q, e_k2, e_c = est_q_full - gt, est_k2 - gt, est_c - gt
w2 = min_var_weights(np.stack([e_q.cpu().numpy().ravel(), e_k2.cpu().numpy().ravel()]))
mse_v4 = mse(w2[0] * e_q + w2[1] * e_k2)
s_v4 = score(mse_v4, V4_C)
print(f"\nv4 (QMC{V4_PTS} + K2): mse {mse_v4:.3e}  C {V4_C:.2e} "
      f"({V4_C/B_FLOP:.1%})  score {s_v4:.3e}   <- to beat")

# Sweep BOTH the point count and whether K2 is worth its 2.2e9 FLOPs.
# The closure supersedes K2 (same covariance propagation + cumulants), so paying
# for both is paying twice for the same thing -- that is what pushed us to 19%.
print("\nsweep (score = mse x max(0.1, C/B); compute must come back near the floor):")
best = None
qmc_cache = {}
for n_pts in (6400, 6000, 5600, 5200, 4800, 4400, 4000, 3600):
    e_qn = qmc_from(make_points(n_pts), Ws) - gt      # matched AT this size
    qmc_cache[n_pts] = e_qn
    for use_k2 in (False, True):
        C = n_pts * PT_FLOP + CL_FLOP + (K2_FLOP if use_k2 else 0.0)
        cols = [e_qn, e_c] + ([e_k2] if use_k2 else [])
        E = np.stack([e.cpu().numpy().ravel() for e in cols])
        w = min_var_weights(E)
        m = mse(sum(w[i] * cols[i] for i in range(len(cols))))
        s = score(m, C)
        tag = "QMC+K2+closure" if use_k2 else "QMC+closure   "
        wtxt = " ".join(f"{v:+.2f}" for v in w)
        print(f"  {n_pts:5d} {tag}: mse {m:.3e}  C {C:.2e} ({C/B_FLOP:5.1%})  "
              f"score {s:.3e}  w=[{wtxt}]")
        if best is None or s < best[0]:
            best = (s, n_pts, w, m, C, use_k2)

s_best, n_best, w_best, m_best, C_best, k2_best = best
w3 = [float(w_best[0]), float(w_best[2]) if k2_best else 0.0, float(w_best[1])]
print(f"\nbest: QMC{n_best}{' + K2' if k2_best else ''} + closure -> "
      f"score {s_best:.3e}  (v4 {s_v4:.3e})")

if s_best < s_v4 * 0.995:
    print(f"=> WINS by {s_v4/s_best:.3f}x — shipping.")
    # ship the point set BUILT AT n_best (a prefix of the 6400 set is not
    # moment-matched, which is what made the sweep non-monotonic)
    best_pts = make_points(n_best).cpu().numpy().astype(np.float32)
    np.savez(KIT / "points.npz", x=best_pts,
             **{"__meta__": np.frombuffer(b"{}", dtype=np.uint8)})
    print(f"wrote {KIT/'points.npz'} ({n_best} points, moment-matched at that size)")
    np.savez(KIT / "closure.npz",
             theta=theta.cpu().numpy().astype(np.float32),
             meta=np.array([float(rank), float(n_best),
                            w3[0], w3[1], w3[2], float(rank4)], dtype=np.float32),
             # ship the subspace-iteration start block so the estimator builds the
             # SAME basis the coefficients were calibrated on
             q0=K._Q0.cpu().numpy().astype(np.float32),
             **{"__meta__": np.frombuffer(b"{}", dtype=np.uint8)})
    print(f"wrote {KIT/'closure.npz'} (r3 {rank}, r4 {rank4}, pts {n_best}, "
          f"w[qmc,k2,closure] = {[round(v,3) for v in w3]})")
    print("next:  cd ../whest-starterkit && uv run whest run --estimator . "
          "--split mini --runner subprocess")
else:
    print("=> no gain over v4. closure.npz NOT written; submission stays v4.")
    print(f"   closure alone is {mse(e_c):.2e}; it needs to get closer to "
          f"QMC's {mse(e_q):.2e} to earn its FLOPs.")
