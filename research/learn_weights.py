"""Learn an offline quadrature rule for depth-32 He-ReLU MLPs.

Points: Sobol->Gauss, whitened, projected to the sphere of radius E||x||
        (exact radial integration via positive 1-homogeneity).
Weights: ridge-regularized empirical Bayesian quadrature
        (G + lam I) w = b with G, b accumulated over offline MLPs,
        optionally with a mean-propagation feature column (global coeff).
Validation: fresh MLPs; compares uniform/spherical/learned variants.
"""

import math
import time

import numpy as np
import torch

from mlp_kprop.mlp import MLP

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH, N_PTS = 256, 32, 6272
M_TRAIN, M_VAL = 128, 32
GT_TRAIN, GT_VAL, BATCH = 1_500_000, 4_000_000, 131_072
SCRATCH = "/private/tmp/claude-501/-Users-lisek-Projects-ARC-white-box/fffd79fe-57cb-423a-9b9c-e89e35bd048e/scratchpad"

# ---------- point sets ----------
sob = torch.quasirandom.SobolEngine(N_W, scramble=True, seed=12345)
u = sob.draw(N_PTS, dtype=torch.float64).clamp(1e-9, 1 - 1e-9)
g = math.sqrt(2.0) * torch.erfinv(2 * u - 1)
g = g - g.mean(dim=0)
L = torch.linalg.cholesky((g.T @ g) / N_PTS)
g = torch.linalg.solve_triangular(L, g.T, upper=False).T  # cov = I exactly
E_R = math.sqrt(2.0) * math.exp(math.lgamma((N_W + 1) / 2) - math.lgamma(N_W / 2))
sph = (g / g.norm(dim=1, keepdim=True)) * E_R  # spherical projection
P_MM = g.float()
P_SPH = sph.float()


def build_mlp(seed):
    torch.manual_seed(seed)
    mlp = MLP(hidden_dim=N_W, num_layers=DEPTH + 1, nonlin="relu",
              init_kind="manual", w_var=2.0)
    for Wm in mlp.Ws:
        Wm.weight.copy_(torch.randn(N_W, N_W) * math.sqrt(2.0 / N_W))
        if Wm.bias is not None:
            Wm.bias.zero_()
    return [mlp.Ws[l].weight.to(DEV) for l in range(DEPTH)]


def fwd_acts(Ws, x0):
    x = x0.to(DEV)
    for l in range(DEPTH):
        x = torch.relu(x @ Ws[l].T)
    return x  # (N_pts, n) final activations


def gt_mean(Ws, n_samples):
    sums = torch.zeros(N_W, dtype=torch.float64)
    done = 0
    while done < n_samples:
        b = min(BATCH, n_samples - done)
        x = torch.randn(b, N_W, device=DEV)
        for l in range(DEPTH):
            x = torch.relu(x @ Ws[l].T)
        sums += x.sum(dim=0).cpu().double()
        done += b
    return sums / n_samples


def mean_prop(Ws):
    mu = torch.zeros(N_W, dtype=torch.float64)
    var = torch.ones(N_W, dtype=torch.float64)
    for l in range(DEPTH):
        W = Ws[l].cpu().double()
        mu_p = W @ mu
        var_p = torch.clamp((W * W) @ var, min=1e-12)
        s = var_p.sqrt()
        a = mu_p / s
        pdf = torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        mu = s * pdf + mu_p * cdf
        ez2 = (mu_p**2 + var_p) * cdf + mu_p * s * pdf
        var = torch.clamp(ez2 - mu * mu, min=0.0)
    return mu  # final-layer mean-prop estimate


# ---------- accumulate normal equations on training MLPs ----------
D = N_PTS + 1  # +1 mean-prop feature column
G = torch.zeros(D, D, dtype=torch.float64)
b = torch.zeros(D, dtype=torch.float64)
t0 = time.time()
for m in range(M_TRAIN):
    Ws = build_mlp(20_000 + m)
    A = fwd_acts(Ws, P_SPH).T  # (n, N_pts) float32 on DEV
    mp = mean_prop(Ws).float().to(DEV)
    Af = torch.cat([A, mp[:, None]], dim=1)  # (n, D)
    y = gt_mean(Ws, GT_TRAIN)
    G += (Af.T @ Af).cpu().double()
    b += (Af.T @ y.float().to(DEV)).cpu().double()
    if m % 16 == 15:
        print(f"  train MLP {m+1}/{M_TRAIN}  ({time.time()-t0:.0f}s)")

# ---------- solve ridge sweeps via eigendecomposition ----------
evals, Q = torch.linalg.eigh(G)
w0 = torch.zeros(D, dtype=torch.float64)
w0[:N_PTS] = 1.0 / N_PTS  # uniform-mean prior
r = b - G @ w0
Qr = Q.T @ r
lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
cands = {}
for lam in lams:
    scale = lam * float(evals.max())
    delta = Q @ (Qr / (evals + scale))
    cands[f"learned lam={lam:g}"] = w0 + delta

# ---------- validation ----------
uniform = torch.zeros(D, dtype=torch.float64)
uniform[:N_PTS] = 1.0 / N_PTS
variants = {"uniform+mm(v2)": ("mm", uniform), "uniform+sph": ("sph", uniform)}
variants.update({k: ("sph", w) for k, w in cands.items()})

mses = dict.fromkeys(variants, 0.0)
for m in range(M_VAL):
    Ws = build_mlp(90_000 + m)
    gt = gt_mean(Ws, GT_VAL)
    mp = mean_prop(Ws)
    A_mm = fwd_acts(Ws, P_MM).T.cpu().double()
    A_sph = fwd_acts(Ws, P_SPH).T.cpu().double()
    feats = {"mm": torch.cat([A_mm, mp[:, None]], dim=1),
             "sph": torch.cat([A_sph, mp[:, None]], dim=1)}
    for k, (pset, w) in variants.items():
        pred = feats[pset] @ w
        mses[k] += float(((pred - gt) ** 2).mean()) / M_VAL

print(f"\nvalidation over {M_VAL} fresh MLPs (final-layer MSE):")
for k, v in mses.items():
    extra = ""
    if k.startswith("learned"):
        w = variants[k][1]
        extra = f"  (mp coeff {float(w[-1]):+.3f}, |w-u|/|u| {float((w[:N_PTS]-1/N_PTS).norm()/ (1/math.sqrt(N_PTS))):.2f})"
    print(f"  {k:22s} {v:.3e}{extra}")

best = min((k for k in cands), key=lambda k: mses[k])
print(f"\nbest: {best}  -> saving weights + spherical points")
np.save(f"{SCRATCH}/points_sph.npy", P_SPH.numpy())
np.save(f"{SCRATCH}/quad_w.npy", variants[best][1].numpy())
