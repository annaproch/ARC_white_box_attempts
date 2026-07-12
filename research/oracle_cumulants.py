"""ORACLE: how much do 3rd/4th cumulants actually buy? (prices the whole build)

Scalar calibration has plateaued at ~2.4-2.6e-5 under both +-5% and +-60%
bounds => the correction the closure needs is STRUCTURED, not a uniform scale.
The structured form (first order in the cumulants) is

    Cov(ReLU_i, ReLU_j) = Wick_Gaussian(alpha_i, alpha_j, rho_ij)
        + 1/2 [ g2(i) g1(j) kappa3(i,i,j) + g1(i) g2(j) kappa3(i,j,j) ]
        + 1/4  g2(i) g2(j) kappa4(i,i,j,j)
        + (higher order)

where g_k are the ReLU Wick coefficients of the pre-activation.

We cannot propagate kappa3/kappa4 yet -- but the corpus HAS them (sampled).
So we can run the closure with the TRUE cumulant slices injected at every
layer and measure the final-layer MSE. That is the ceiling of the whole
third/fourth-order program, obtained before building any propagator:

  variants:
    K2            : Gaussian closure (baseline, ~6e-5)
    K2 + kappa3   : + true third-cumulant correction to the covariance
    K2 + k3 + k4  : + true fourth-cumulant correction
    oracle-Sigma  : replace the propagated Sigma with the TRUE sampled Sigma at
                    every layer (the absolute ceiling of any covariance fix)

Interpretation:
  * "+kappa3" ~ 1e-6  -> kappa3 IS the answer; build/predict it (n^2 slices).
  * needs k4 too       -> plan a 4th-order term as well.
  * even oracle-Sigma  -> if that is still >> 1e-6, the covariance is NOT the
                          whole story and the mean path matters too.

Requires: research/build_moment_corpus.py to have been run.
Run:  cd ../mlp_cumulant_propagation && uv run python ../research/oracle_cumulants.py
"""

import math
from pathlib import Path

import torch

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
N_TERMS = 4
CORPUS = Path(__file__).resolve().parent.parent / "data" / "moment_corpus"


def wick_g(a, pdf, cdf, k_max):
    g = [cdf, pdf]
    he_prev, he_cur = torch.ones_like(a), a
    for k in range(3, k_max + 1):
        m = k - 2
        g.append(((-1) ** m) * he_cur * pdf)
        he_prev, he_cur = he_cur, a * he_cur - m * he_prev
    return g[:k_max]


def propagate(Ws, mode, k3_true=None, k4_true=None, cov_true=None):
    """mode: 'k2' | 'k3' | 'k3k4' | 'oracle_cov'.

    The cumulant slices are indexed [layer][i, j] = kappa(a_i, a_i, a_j) etc.,
    i.e. cumulants of the POST-activation of that layer -- which is exactly the
    input distribution of the next linear layer. We inject the correction into
    the covariance of the pre-activation of the NEXT layer via the standard
    first-order diagram terms.
    """
    mu = torch.zeros(N_W, dtype=torch.float64)
    cov = torch.eye(N_W, dtype=torch.float64)
    eye = torch.eye(N_W, dtype=torch.bool)
    for li in range(DEPTH):
        W = Ws[li]
        mu_pre = mu @ W
        cov_pre = W.T @ (cov @ W)
        var = torch.clamp(torch.diagonal(cov_pre), min=1e-12)
        s = var.sqrt()
        a = mu_pre / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        rho = torch.clamp(cov_pre / torch.outer(s, s), -1.0, 1.0)

        mu_new = s * pdf + mu_pre * cdf
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu_new**2, min=0.0)

        g = wick_g(a, pdf, cdf, N_TERMS)
        acc = torch.zeros_like(cov_pre)
        rho_k, sf = rho.clone(), 1.0
        for k in range(1, N_TERMS + 1):
            sf *= math.sqrt(k)
            gh = g[k - 1] / sf
            acc += torch.outer(gh, gh) * rho_k
            if k < N_TERMS:
                rho_k = rho_k * rho
        cov_new = torch.outer(s, s) * acc

        # --- inject the true cumulant corrections (oracle) ---
        # g1 = Phi(alpha) (dimensionless), g2 = phi(alpha)/sigma has units 1/sigma.
        # In the stable form used above g[0]=Phi, g[1]=phi; the physical Wick
        # coefficients are h1 = Phi, h2 = phi/sigma.
        h1 = g[0]
        h2 = g[1] / s
        if mode in ("k3", "k3k4") and k3_true is not None:
            K3 = k3_true[li].double()                       # kappa3(a_i, a_i, a_j)
            corr3 = 0.5 * (torch.outer(h2, h1) * K3 + torch.outer(h1, h2) * K3.T)
            cov_new = cov_new + corr3
        if mode == "k3k4" and k4_true is not None:
            K4 = k4_true[li].double()                       # kappa4(a_i,a_i,a_j,a_j)
            cov_new = cov_new + 0.25 * torch.outer(h2, h2) * K4

        cov_new = cov_new.clone()
        cov_new[eye] = var_post
        mu = mu_new
        cov = cov_true[li].double() if mode == "oracle_cov" else cov_new
    return mu


files = sorted(CORPUS.glob("mlp_*.pt"))
if not files:
    raise SystemExit(f"no corpus in {CORPUS} — run build_moment_corpus.py first")
print(f"{len(files)} MLPs in corpus\n")
print(f"{'seed':>7s} {'K2':>10s} {'K2+k3':>10s} {'K2+k3+k4':>10s} {'oracle-cov':>11s}")
tot = {"k2": 0.0, "k3": 0.0, "k3k4": 0.0, "oracle_cov": 0.0}
for f in files:
    r = torch.load(f)
    Ws = r["W"].double()
    gt = r["mu"][-1].double()                              # true final-layer mean
    row = {}
    for mode in ("k2", "k3", "k3k4", "oracle_cov"):
        est = propagate(Ws, mode, r["k3"], r["k4"], r["cov"])
        row[mode] = float(((est - gt) ** 2).mean())
        tot[mode] += row[mode] / len(files)
    print(f"{r['seed']:7d} {row['k2']:10.3e} {row['k3']:10.3e} {row['k3k4']:10.3e} "
          f"{row['oracle_cov']:11.3e}")
print(f"{'MEAN':>7s} {tot['k2']:10.3e} {tot['k3']:10.3e} {tot['k3k4']:10.3e} "
      f"{tot['oracle_cov']:11.3e}")
print(f"\ngains: k3 {tot['k2']/max(tot['k3'],1e-30):.1f}x   "
      f"k3+k4 {tot['k2']/max(tot['k3k4'],1e-30):.1f}x   "
      f"oracle-cov {tot['k2']/max(tot['oracle_cov'],1e-30):.1f}x")
print("\nReference: QMC at floor ~4.1e-6; competitor ~3.2e-7 (pure analytic, floor).")
print("If k3 lands near 1e-6: build the rank-r kappa3 propagator (or LEARN the")
print("kappa3 slices from (alpha_i, alpha_j, rho_ij) -- n^2 cost, no tensors).")
