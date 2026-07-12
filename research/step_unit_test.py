"""UNIT-TEST ONE CLOSURE STEP against truth (no propagation, no compounding).

The oracle showed:
    perfect mu + perfect Sigma  -> 1e-6   (the closure formula is fine)
    perfect Sigma, propagated mu -> 1.9e-4 (WORSE than K=2's 1.3e-4)
=> K=2 is a self-consistent fixed point: the mean update's per-layer shape bias
   compounds over depth, and the covariance underestimate partially cancels it.
   Fixing either alone makes the answer worse. The mean update is the real
   target, and mean+covariance must be corrected together.

So: stop guessing from end-to-end runs. Feed the TRUE pre-activation moments of
layer l (mu_z, Sigma_z, kappa3(z), kappa4(z), all sampled) into a single ReLU
step and compare against the TRUE post-activation mu/Sigma of that same layer.
This isolates the step from the dynamics and validates the correction formulas
(coefficients AND signs) one at a time.

MEAN step:      E[ReLU(z_i)]
  gauss     : s*phi(a) + m*Phi(a)
  +edge3    : + (1/6)  h3_i kappa3_iii          h3 = -a phi(a)/s^2
  +edge4    : + (1/24) h4_i kappa4_iiii         h4 = (a^2-1) phi(a)/s^3

COV step:       Cov(ReLU(z_i), ReLU(z_j))
  gauss     : sum_k rho^k g_k(i) g_k(j) s_i s_j / k!         (converged, k<=4)
  +k3       : + 1/2 [h2_i h1_j k3(z_i,z_i,z_j) + h1_i h2_j k3(z_i,z_j,z_j)]
  +k4       : + 1/4  h2_i h2_j kappa4(z_i,z_i,z_j,z_j)

If a correction REDUCES the single-step error -> the formula is right and the
term matters. If it INCREASES it -> the coefficient/sign is wrong, and every
end-to-end result built on it was meaningless.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/step_unit_test.py
"""

import math
from pathlib import Path

import torch

torch.set_grad_enabled(False)
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
N_TERMS = 4
CORPUS = Path(__file__).resolve().parent.parent / "data" / "moment_corpus"

files = sorted(CORPUS.glob("mlp_*.pt"))
if not files:
    raise SystemExit("run build_moment_corpus.py first")
probe = torch.load(files[0], map_location="cpu")
if "zk3" not in probe:
    raise SystemExit("stale corpus (no zk3) — re-run build_moment_corpus.py")

eye = torch.eye(N_W, dtype=torch.bool)
LAYERS = (0, 1, 3, 7, 15, 23, 31)

mean_err = {k: dict.fromkeys(LAYERS, 0.0) for k in ("gauss", "edge3", "edge34")}
cov_err = {k: dict.fromkeys(LAYERS, 0.0) for k in ("gauss", "k3", "k3k4")}

for f in files:
    r = torch.load(f, map_location="cpu")
    zmu, zcov = r["zmu"].double(), r["zcov"].double()
    zk3, zk4 = r["zk3"].double(), r["zk4"].double()
    mu_t, cov_t = r["mu"].double(), r["cov"].double()

    for li in LAYERS:
        m = zmu[li]
        var = torch.clamp(torch.diagonal(zcov[li]), min=1e-12)
        s = var.sqrt()
        a = m / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        h1 = cdf
        h2 = pdf / s
        h3 = -a * pdf / var
        h4 = (a * a - 1) * pdf / (var * s)

        # ---- MEAN step ----
        g0 = s * pdf + m * cdf
        g3 = g0 + (1.0 / 6.0) * h3 * torch.diagonal(zk3[li])
        g4 = g3 + (1.0 / 24.0) * h4 * torch.diagonal(zk4[li])
        tgt = mu_t[li]
        for key, est in (("gauss", g0), ("edge3", g3), ("edge34", g4)):
            mean_err[key][li] += float(((est - tgt) ** 2).mean()) / len(files)

        # ---- COVARIANCE step (off-diagonal only; the diagonal is exact 1-D) ----
        rho = torch.clamp(zcov[li] / torch.outer(s, s), -1.0, 1.0)
        gk = [cdf, pdf]
        he_prev, he_cur = torch.ones_like(a), a
        for k in range(3, N_TERMS + 1):
            mm = k - 2
            gk.append(((-1) ** mm) * he_cur * pdf)
            he_prev, he_cur = he_cur, a * he_cur - mm * he_prev
        acc = torch.zeros_like(rho)
        rho_k, sf = rho.clone(), 1.0
        for k in range(1, N_TERMS + 1):
            sf *= math.sqrt(k)
            gh = gk[k - 1] / sf
            acc += torch.outer(gh, gh) * rho_k
            if k < N_TERMS:
                rho_k = rho_k * rho
        c0 = torch.outer(s, s) * acc
        K3, K4 = zk3[li], zk4[li]
        c3 = c0 + 0.5 * (torch.outer(h2, h1) * K3 + torch.outer(h1, h2) * K3.T)
        c4 = c3 + 0.25 * torch.outer(h2, h2) * K4
        tgt_c = cov_t[li]
        for key, est in (("gauss", c0), ("k3", c3), ("k3k4", c4)):
            cov_err[key][li] += float(((est[~eye] - tgt_c[~eye]) ** 2).mean()) / len(files)

print(f"single-step MEAN error (MSE vs true post-activation mean), {len(files)} MLPs")
print(f"{'L':>3s} {'gauss':>11s} {'+edge3':>11s} {'+edge3+4':>11s} {'gain':>7s}")
for li in LAYERS:
    g, e3, e4 = mean_err["gauss"][li], mean_err["edge3"][li], mean_err["edge34"][li]
    print(f"{li:3d} {g:11.3e} {e3:11.3e} {e4:11.3e} {g/max(e4,1e-30):6.1f}x")

print(f"\nsingle-step COVARIANCE error (off-diagonal MSE vs true post-act. cov)")
print(f"{'L':>3s} {'gauss':>11s} {'+k3':>11s} {'+k3+k4':>11s} {'gain':>7s}")
for li in LAYERS:
    g, c3, c4 = cov_err["gauss"][li], cov_err["k3"][li], cov_err["k3k4"][li]
    print(f"{li:3d} {g:11.3e} {c3:11.3e} {c4:11.3e} {g/max(c4,1e-30):6.1f}x")

print("\nBoth corrections reduce their step error -> formulas are right; the whole")
print("problem is then obtaining kappa3(z)/kappa4(z) in-budget (propagate or learn).")
print("A correction that INCREASES the step error has a wrong sign/coefficient.")
