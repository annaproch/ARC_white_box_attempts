"""LAYERWISE FITTING against the corpus — replaces end-to-end SGD.

Why: every configuration of the end-to-end trainer plateaus at ~1.05e-5, because
each layer's coefficients get their gradient from ONE distant loss (256 numbers,
backpropagated through 32 layers). The corpus has the TRUE mu, Sigma, kappa3,
kappa4 at every layer, so each piece can be supervised against its own target
(32 x 65k targets instead of 256).

Two decoupled problems, in the right order:

  (A) MOMENT CORRECTIONS -- closed form.
      Feed each layer the TRUE pre-activation state (m, S, k3, k4) from the corpus
      and fit, by LINEAR LEAST SQUARES, the coefficients of the correction terms
      against the TRUE post-activation mu / Sigma:
          mu_hat  = gauss + c3m*A3 + c4m*A4
          Sig_hat = wick*(1+s_off) + c3c*C3 + c4c*C4
      These are all multipliers on additive terms => no SGD needed at all.
      The RESIDUAL of each fit says exactly what is still missing, per layer.

  (B) CUMULANT PROPAGATION -- the actual bottleneck.
      Teacher-force the moments (inject true m, S each layer) so cumulant error is
      not confounded with moment drift, then train the generation/pass-through
      coefficients so OUR propagated kappa3/kappa4 slices match the corpus's
      SAMPLED slices at every layer. Also fits the kappa4 CHAIN diagram, which I
      never derived -- kappa4's star and chain have the same number of Sigma edges
      (3), so star-only was computing about HALF of kappa4's leading generation.
      That is why kappa4 has been contributing nothing.

Outputs a theta compatible with calibrate_k3k4.py / closure.py, plus per-layer
residuals so we can see what the closure still cannot express.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/fit_layerwise.py
"""

import math
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
N_TERMS = 4
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "moment_corpus"
DATA = ROOT / "data"


def wick_cov(a, pdf, cdf, s, rho):
    g = [cdf, pdf]
    he_prev, he_cur = torch.ones_like(a), a
    for k in range(3, N_TERMS + 1):
        m = k - 2
        g.append(((-1) ** m) * he_cur * pdf)
        he_prev, he_cur = he_cur, a * he_cur - m * he_prev
    acc = torch.zeros_like(rho)
    rho_k, sf = rho.clone(), 1.0
    for k in range(1, N_TERMS + 1):
        sf *= math.sqrt(k)
        gh = g[k - 1] / sf
        acc += torch.outer(gh, gh) * rho_k
        if k < N_TERMS:
            rho_k = rho_k * rho
    return torch.outer(s, s) * acc


def basics(m, S):
    var = torch.clamp(torch.diagonal(S), min=1e-8)
    s = var.sqrt()
    a = torch.clamp(m / s, -12.0, 12.0)
    pdf = torch.exp(-0.5 * a * a) / SQ2PI
    cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
    rho = torch.clamp(S / torch.outer(s, s), -1.0, 1.0)
    return var, s, a, pdf, cdf, rho


files = sorted(CORPUS.glob("mlp_*.pt"))
print(f"corpus: {len(files)} MLPs")
if len(files) < 16:
    print("WARNING: small corpus; rebuild with --n-mlps 64 for a trustworthy fit")

# =============================================================================
# (A) MOMENT CORRECTIONS — closed-form least squares, per layer
# =============================================================================
print("\n(A) moment corrections: least squares with TRUE cumulants injected")
print(f"{'L':>3s} {'mean: gauss':>12s} {'fitted':>10s} {'gain':>7s} "
      f"{'c3m':>6s} {'c4m':>6s} | {'cov: gauss':>11s} {'fitted':>10s} {'gain':>7s} "
      f"{'c3c':>6s} {'c4c':>6s}")

fit_mean, fit_cov = {}, {}
for li in range(DEPTH):
    Am, bm, Ac, bc = [], [], [], []
    g_mean_err = g_cov_err = 0.0
    for f in files:
        r = torch.load(f, map_location="cpu")
        m, S = r["zmu"][li].double(), r["zcov"][li].double()
        K3, K4 = r["zk3"][li].double(), r["zk4"][li].double()
        mu_t, cov_t = r["mu"][li].double(), r["cov"][li].double()
        var, s, a, pdf, cdf, rho = basics(m, S)

        # mean: gauss + c3m * A3 + c4m * A4   (A3, A4 are the Edgeworth terms)
        g0 = s * pdf + m * cdf
        A3 = -(1.0 / 6.0) * a * pdf * (torch.diagonal(K3) / var)
        A4 = (1.0 / 24.0) * (a * a - 1) * pdf * (torch.diagonal(K4) / (var * s))
        Am.append(torch.stack([A3, A4], 1))
        bm.append(mu_t - g0)
        g_mean_err += float(((g0 - mu_t) ** 2).mean()) / len(files)

        # covariance (off-diagonal): wick*(1+s_off) + c3c*C3 + c4c*C4
        eye = torch.eye(N_W, dtype=torch.bool)
        W0 = wick_cov(a, pdf, cdf, s, rho)
        q3 = K3 / s[:, None]
        c3t = pdf[:, None] * cdf[None, :] * q3
        C3 = 0.5 * (c3t + c3t.T)
        C4 = 0.25 * (pdf[:, None] * pdf[None, :] * (K4 / torch.outer(s, s)))
        Ac.append(torch.stack([W0[~eye], C3[~eye], C4[~eye]], 1))
        bc.append(cov_t[~eye])
        g_cov_err += float(((W0[~eye] - cov_t[~eye]) ** 2).mean()) / len(files)

    Am, bm = torch.cat(Am), torch.cat(bm)
    x_m = torch.linalg.lstsq(Am, bm.unsqueeze(1)).solution.squeeze(1)
    res_m = float(((Am @ x_m - bm) ** 2).mean())

    Ac, bc = torch.cat(Ac), torch.cat(bc)
    x_c = torch.linalg.lstsq(Ac, bc.unsqueeze(1)).solution.squeeze(1)
    res_c = float(((Ac @ x_c - bc) ** 2).mean())

    fit_mean[li] = (float(x_m[0]), float(x_m[1]))
    fit_cov[li] = (float(x_c[0]), float(x_c[1]), float(x_c[2]))
    if li in (0, 1, 3, 7, 15, 23, 31):
        print(f"{li:3d} {g_mean_err:12.3e} {res_m:10.3e} "
              f"{g_mean_err/max(res_m,1e-30):6.1f}x {x_m[0]:6.2f} {x_m[1]:6.2f} | "
              f"{g_cov_err:11.3e} {res_c:10.3e} {g_cov_err/max(res_c,1e-30):6.1f}x "
              f"{x_c[1]:6.2f} {x_c[2]:6.2f}")

np.savez(DATA / "fit_moments.npz",
         mean=np.array([fit_mean[l] for l in range(DEPTH)], dtype=np.float32),
         cov=np.array([fit_cov[l] for l in range(DEPTH)], dtype=np.float32))
print(f"\nsaved {DATA/'fit_moments.npz'}")
print("The fitted coefficients are the BEST POSSIBLE moment corrections given")
print("perfect cumulants. Residual = what the correction FORM cannot express.")
print("\nNext: (B) supervise the cumulant propagation itself (fit_cumulants.py),")
print("including the kappa4 CHAIN diagram -- star and chain both have 3 Sigma")
print("edges, so star-only computes about half of kappa4's leading generation.")
