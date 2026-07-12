"""ORACLE: how much do 3rd/4th cumulants actually buy? (prices the whole build)

v2. The first version was buggy in two ways:
  * it injected kappa3 of the POST-activation a, but the diagram formula needs
    cumulants of the PRE-activation z. (And kappa3(z) = W^T-transform of
    kappa3(a) needs the FULL n^3 tensor: the n^2 slices of a cannot produce the
    slices of z. Slices suffice to MEASURE/CORRECT, not to PROPAGATE -- which is
    exactly why ARC's implementation carries the full factored tensor. Here we
    sample kappa3(z) directly from the corpus.)
  * its 'oracle-cov' mode came out WORSE than plain K=2, which is impossible for
    correct code -- so we now include a sanity mode that must reproduce a value
    we have already measured independently.

Modes:
  k2          : Gaussian closure (baseline)
  k3          : + true kappa3(z) corrections   [mean and covariance]
  k3k4        : + true kappa4(z) corrections
  oracle_cov  : replace the propagated Sigma with the TRUE Sigma each layer
  oracle_all  : replace BOTH mu and Sigma with the truth each layer
                --> the answer is then just the Gaussian ReLU formula applied to
                    the true final moments, which error_attribution.py already
                    measured as ~1.0e-6.  *** SANITY CHECK ***
                    If oracle_all != ~1e-6, the corpus is broken; fix that first.
                    If oracle_all ~ 1e-6 but oracle_cov is bad, then K=2 is a
                    self-consistent fixed point (covariance underestimate partly
                    cancels the accumulated Gaussian-shape error in the mean) and
                    mean+covariance must be corrected TOGETHER.

First-order diagram terms used (h_k = E[d^k ReLU(Z)]: h1=Phi(a), h2=phi(a)/s,
h3=-a*phi(a)/s^2, h4=(a^2-1)*phi(a)/s^3):
  mean:  E[ReLU(z_i)] += (1/6) h3_i kappa3_iii + (1/24) h4_i kappa4_iiii
  cov :  Cov_ij       += (1/2)[h2_i h1_j k3(z_i,z_i,z_j) + h1_i h2_j k3(z_i,z_j,z_j)]
                       + (1/4) h2_i h2_j kappa4(z_i,z_i,z_j,z_j)

Requires: build_moment_corpus.py (v2, with zk3/zk4) to have been run.
Run:  cd ../mlp_cumulant_propagation && uv run python ../research/oracle_cumulants.py
"""

import math
from pathlib import Path

import torch

torch.set_grad_enabled(False)
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
N_TERMS = 4
CORPUS = Path(__file__).resolve().parent.parent / "data" / "moment_corpus"


def propagate(r, mode):
    Ws = r["W"].double()
    zk3, zk4 = r["zk3"].double(), r["zk4"].double()
    cov_t, mu_t = r["cov"].double(), r["mu"].double()
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

        # physical Wick coefficients
        h1 = cdf
        h2 = pdf / s
        h3 = -a * pdf / var
        h4 = (a * a - 1) * pdf / (var * s)

        mu_new = s * pdf + mu_pre * cdf
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu_new**2, min=0.0)

        # Gaussian (converged k<=4 Wick series), stable dimensionless form
        g = [cdf, pdf]
        he_prev, he_cur = torch.ones_like(a), a
        for k in range(3, N_TERMS + 1):
            m = k - 2
            g.append(((-1) ** m) * he_cur * pdf)
            he_prev, he_cur = he_cur, a * he_cur - m * he_prev
        acc = torch.zeros_like(cov_pre)
        rho_k, sf = rho.clone(), 1.0
        for k in range(1, N_TERMS + 1):
            sf *= math.sqrt(k)
            gh = g[k - 1] / sf
            acc += torch.outer(gh, gh) * rho_k
            if k < N_TERMS:
                rho_k = rho_k * rho
        cov_new = torch.outer(s, s) * acc

        if mode in ("k3", "k3k4"):
            K3 = zk3[li]                                  # kappa3(z_i, z_i, z_j)
            mu_new = mu_new + (1.0 / 6.0) * h3 * torch.diagonal(K3)
            cov_new = cov_new + 0.5 * (torch.outer(h2, h1) * K3
                                       + torch.outer(h1, h2) * K3.T)
        if mode == "k3k4":
            K4 = zk4[li]                                  # kappa4(z_i,z_i,z_j,z_j)
            mu_new = mu_new + (1.0 / 24.0) * h4 * torch.diagonal(K4)
            cov_new = cov_new + 0.25 * torch.outer(h2, h2) * K4

        cov_new = cov_new.clone()
        cov_new[eye] = var_post

        if mode == "oracle_all":
            mu, cov = mu_t[li], cov_t[li]
        elif mode == "oracle_cov":
            mu, cov = mu_new, cov_t[li]
        else:
            mu, cov = mu_new, cov_new

    # for oracle_all the reported answer must be the ANALYTIC last-layer output,
    # not the injected truth -- recompute it from the true penultimate moments.
    if mode == "oracle_all":
        W = Ws[DEPTH - 1]
        m_prev = mu_t[DEPTH - 2]
        c_prev = cov_t[DEPTH - 2]
        mu_pre = m_prev @ W
        var = torch.clamp(torch.diagonal(W.T @ (c_prev @ W)), min=1e-12)
        s = var.sqrt()
        a = mu_pre / s
        return s * torch.exp(-0.5 * a * a) / SQ2PI + mu_pre * 0.5 * (
            1 + torch.erf(a / math.sqrt(2)))
    return mu


files = sorted(CORPUS.glob("mlp_*.pt"))
if not files:
    raise SystemExit(f"no corpus in {CORPUS} — run build_moment_corpus.py first")
probe = torch.load(files[0])
if "zk3" not in probe:
    raise SystemExit("corpus predates the pre-activation cumulants — delete "
                     f"{CORPUS} and re-run build_moment_corpus.py")

print(f"{len(files)} MLPs in corpus\n")
print(f"{'seed':>7s} {'K2':>10s} {'K2+k3':>10s} {'K2+k3+k4':>10s} "
      f"{'oracle-cov':>11s} {'oracle-ALL':>11s}")
modes = ("k2", "k3", "k3k4", "oracle_cov", "oracle_all")
tot = dict.fromkeys(modes, 0.0)
for f in files:
    r = torch.load(f)
    gt = r["mu"][-1].double()
    row = {}
    for mode in modes:
        est = propagate(r, mode)
        row[mode] = float(((est - gt) ** 2).mean())
        tot[mode] += row[mode] / len(files)
    print(f"{r['seed']:7d} {row['k2']:10.3e} {row['k3']:10.3e} {row['k3k4']:10.3e} "
          f"{row['oracle_cov']:11.3e} {row['oracle_all']:11.3e}")
print(f"{'MEAN':>7s} {tot['k2']:10.3e} {tot['k3']:10.3e} {tot['k3k4']:10.3e} "
      f"{tot['oracle_cov']:11.3e} {tot['oracle_all']:11.3e}")

print("\n*** SANITY: oracle-ALL must be ~1e-6 (independently measured shape error).")
if tot["oracle_all"] > 1e-5:
    print("    It is NOT -> the corpus (or the weight/convention handling) is broken.")
else:
    print("    OK -> corpus is sound; the other columns are trustworthy.")
print(f"\ngains vs K2: k3 {tot['k2']/max(tot['k3'],1e-30):.1f}x   "
      f"k3+k4 {tot['k2']/max(tot['k3k4'],1e-30):.1f}x   "
      f"oracle-cov {tot['k2']/max(tot['oracle_cov'],1e-30):.1f}x")
print("Reference: QMC at floor ~4.1e-6; competitor ~3.2e-7 (pure analytic, floor).")
