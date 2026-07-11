"""How much of covariance-propagation (K=2) error is ensemble-systematic
(→ removable by offline-learned calibration of the Gaussian closure) vs
per-MLP idiosyncratic (→ irreducible given the K=2 state)?

Levels: raw K=2 -> minus global mean error -> ridge regression of the error
on the K=2 internal state (est, mu_pre, sigma_pre, alpha, crude diag-3rd-
cumulant proxy), trained on 64 MLPs, evaluated on 24 held-out MLPs.
Then: projected score of fusing calibrated-K2 with floor QMC (global weight).
"""

import math

import torch

from mlp_kprop.mlp import MLP

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
M_TRAIN, M_VAL = 64, 24
GT_N, BATCH = 1_500_000, 131_072
SQ2PI = math.sqrt(2 * math.pi)


def build_ws(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N_W, N_W, generator=g) * math.sqrt(2.0 / N_W)
            for _ in range(DEPTH)]


def gt_mean(ws, n):
    wd = [w.to(DEV) for w in ws]
    s = torch.zeros(N_W, dtype=torch.float64)
    d = 0
    while d < n:
        b = min(BATCH, n - d)
        x = torch.randn(b, N_W, device=DEV)
        for w in wd:
            x = torch.relu(x @ w)
        s += x.sum(0).cpu().double()
        d += b
    return s / n


def k2_with_state(ws, n_terms=4):
    """K=2 covariance propagation; returns final estimate + internal state."""
    mu = torch.zeros(N_W, dtype=torch.float64)
    cov = torch.eye(N_W, dtype=torch.float64)
    k3 = torch.zeros(N_W, dtype=torch.float64)  # crude diagonal 3rd-cumulant proxy
    for W in ws:
        Wd = W.double()  # forward is x @ W: columns are output neurons
        mu_pre = Wd.T @ mu
        cov_pre = Wd.T @ cov @ Wd
        k3_pre = (Wd**3).T @ k3
        var = torch.clamp(torch.diag(cov_pre), min=1e-12)
        s = var.sqrt()
        a = mu_pre / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        mu_post = s * pdf + mu_pre * cdf
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu_post**2, min=0.0)
        # 3rd central moment of ReLU(N(mu,s^2)) (Gaussian-injected skew), plus
        # linear-gain pass-through of incoming k3 (crude closure).
        ez3 = (mu_pre**3 + 3 * mu_pre * var) * cdf + (mu_pre**2 + 2 * var) * s * pdf
        skew_inj = ez3 - 3 * mu_post * ez2 + 2 * mu_post**3
        k3 = skew_inj + (cdf**3) * k3_pre
        wick = [cdf, pdf / s, -a * pdf / var, (a * a - 1) * pdf / (var * s)]
        cov_post = torch.zeros_like(cov_pre)
        Ck = cov_pre.clone()
        for k in range(1, n_terms + 1):
            cov_post += torch.outer(wick[k - 1], wick[k - 1]) * Ck / math.factorial(k)
            if k < n_terms:
                Ck = Ck * cov_pre
        cov_post.fill_diagonal_(0.0)
        cov_post += torch.diag(var_post)
        mu, cov = mu_post, cov_post
        state = (mu_pre, s, a)
    mu_pre, s, a = state
    feats = torch.stack([
        mu, mu_pre, s, a, k3,
        mu * mu, s * s, a * a, mu * s, torch.clamp(a, min=0),
        k3 / torch.clamp(s**3, min=1e-9), torch.ones_like(mu),
    ], dim=1)
    return mu, feats


def collect(seeds):
    E, F = [], []
    for sd in seeds:
        ws = build_ws(sd)
        est, feats = k2_with_state(ws)
        gt = gt_mean(ws, GT_N)
        E.append(est - gt)
        F.append(feats)
    return torch.cat(E), torch.cat(F, dim=0)


err_tr, X_tr = collect(range(30_000, 30_000 + M_TRAIN))
err_va, X_va = collect(range(80_000, 80_000 + M_VAL))

raw = float((err_va**2).mean())
c0 = err_tr.mean()
lvl1 = float(((err_va - c0) ** 2).mean())

# standardize features, ridge sweep
mu_f, sd_f = X_tr.mean(0), X_tr.std(0).clamp(min=1e-12)
Z_tr, Z_va = (X_tr - mu_f) / sd_f, (X_va - mu_f) / sd_f
G = Z_tr.T @ Z_tr
best = None
for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    th = torch.linalg.solve(G + lam * float(G.diagonal().max()) * torch.eye(G.shape[0], dtype=torch.float64),
                            Z_tr.T @ err_tr)
    v = float(((err_va - Z_va @ th) ** 2).mean())
    if best is None or v < best[0]:
        best = (v, lam)
lvl2, lam_best = best

print(f"K=2 error decomposition (val, {M_VAL} MLPs):")
print(f"  raw MSE                    {raw:.3e}")
print(f"  - global mean offset       {lvl1:.3e}   (systematic scalar: {1-lvl1/raw:.1%})")
print(f"  - state-feature ridge      {lvl2:.3e}   (systematic total:  {1-lvl2/raw:.1%}, lam={lam_best:g})")

v_qmc = 4.0e-06  # floor-QMC MSE (mini-split scale)
v_qmc_red = v_qmc * 1.02  # QMC keeps ~98% of samples after paying for K2
b2 = lvl2
fused = (b2 * v_qmc_red) / (b2 + v_qmc_red)
print(f"\nfusion projection (independent errors, global weights):")
print(f"  calibrated-K2 bias^2 {b2:.3e}  QMC var {v_qmc_red:.3e}")
print(f"  fused MSE ~ {fused:.3e}  -> score ~ {fused*0.1:.3e}  (current v3 ~ 3.99e-07)")
