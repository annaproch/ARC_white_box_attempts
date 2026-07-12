"""Rank-1 non-Gaussian core closure ("K=2 + skewed common mode").

Diagnosis (layerwise_cov_drift.py): the marginals of the pre-activations are
Gaussian (skew ~0.02), but the DOMINANT EIGENMODE of the activation covariance
is strongly non-Gaussian (|skew| ~0.3-0.5, kurt ~0.1-0.3) and carries 50-83%
of the variance. The Gaussian closure therefore mis-propagates that one mode
and underestimates lambda1 by 27-43%, which biases sigma_z by ~30% and causes
98% of our final-layer error.

Closure:
    a = mu + sqrt(lambda) * u * c + eps,     eps ~ N(0, R),  R = Sigma - lambda*u u^T
with c a SCALAR non-Gaussian variable (mean 0, var 1) tracked by its skew and
kurtosis and represented by a Q-node quadrature (Gauss-Hermite nodes warped by
a cubic Cornish-Fisher-style map).

Per layer:
    z          = W^T a                      -> m(c) = mu_z + sqrt(lambda) p c,  p = W^T u
    Sigma_eps  = W^T R W                    -> conditional covariance of z given c
    a'_i       = ReLU(z_i)
    mu'        = sum_q w_q M(c_q)                             [M = exact 1-D ReLU mean]
    Sigma'     = sum_q w_q [C(c_q) + M M^T] - mu' mu'^T       [law of total covariance]
where C(c_q) is the conditional ReLU covariance from the (converged, k<=4)
Wick series around the conditional Gaussian.

Then re-extract the core (u', lambda' = top eigenpair of Sigma'), and push the
c-distribution forward:  c' | c ~ N(t(c), v(c)) with
    t(c) = u'^T (M(c) - mu'),   v(c) = u'^T C(c) u',
giving the new (skew, kurt) in closed form from the mixture moments. The
non-Gaussianity is generated automatically by the nonlinearity of t(.), so we
can start from a Gaussian c at layer 0 (measured true skew there: 0.02).

Cost at n=256, L=32, Q=16, K=4: ~2n^3 + Q*K*n^2 per layer ~= 1.5e9 FLOPs total
(~0.6% of budget) -- affordable pure-analytic at the 10% floor.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/core_closure.py
"""

import math

import numpy as np
import torch

torch.set_grad_enabled(False)
SDEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
GT_N, BATCH = 4_000_000, 131_072
SEEDS = 6
Q_NODES = 16
N_TERMS = 4          # Wick series (converged; see wick_terms_test.py)
SQ2PI = math.sqrt(2 * math.pi)
DT = torch.float64


def build_ws(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N_W, N_W, generator=g, dtype=DT) * math.sqrt(2.0 / N_W)
            for _ in range(DEPTH)]


def gt_mean(ws, n):
    wd = [w.float().to(SDEV) for w in ws]
    s = torch.zeros(N_W, dtype=DT)
    d = 0
    while d < n:
        b = min(BATCH, n - d)
        x = torch.randn(b, N_W, device=SDEV)
        for w in wd:
            x = torch.relu(x @ w)
        s += x.sum(0).cpu().double()
        d += b
    return s / n


# ---- quadrature over the non-Gaussian core -------------------------------
_xi, _wt = np.polynomial.hermite_e.hermegauss(Q_NODES)   # probabilists' GH
XI = torch.tensor(_xi, dtype=DT)
WT = torch.tensor(_wt, dtype=DT)
WT = WT / WT.sum()


def core_nodes(skew, kurt):
    """Nodes/weights for a standardized c with the given skew and excess kurtosis.

    c = h(xi), xi ~ N(0,1),  h(xi) = xi + a2 (xi^2 - 1) + a3 (xi^3 - 3 xi)
    (Gram-Charlier/Cornish-Fisher form: to leading order skew = 6*a2, kurt = 24*a3).
    Then standardize numerically on the nodes so mean=0, var=1 exactly.
    """
    a2 = skew / 6.0
    a3 = kurt / 24.0
    a2 = float(np.clip(a2, -0.25, 0.25))
    a3 = float(np.clip(a3, -0.10, 0.10))
    c = XI + a2 * (XI**2 - 1) + a3 * (XI**3 - 3 * XI)
    m = (WT * c).sum()
    v = (WT * (c - m) ** 2).sum()
    return (c - m) / v.sqrt(), WT


def wick_g(a, pdf, cdf, k_max):
    g = [cdf, pdf]
    he_prev, he_cur = torch.ones_like(a), a
    for k in range(3, k_max + 1):
        m = k - 2
        g.append(((-1) ** m) * he_cur * pdf)
        he_prev, he_cur = he_cur, a * he_cur - m * he_prev
    return g[:k_max]


def relu_cond(m, s, rho):
    """Conditional ReLU mean, variance and off-diagonal covariance given a
    Gaussian z ~ N(m, diag(s^2)) with correlation matrix rho."""
    a = m / s
    pdf = torch.exp(-0.5 * a * a) / SQ2PI
    cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
    M = s * pdf + m * cdf
    ez2 = (m * m + s * s) * cdf + m * s * pdf
    V = torch.clamp(ez2 - M * M, min=0.0)
    g = wick_g(a, pdf, cdf, N_TERMS)
    acc = torch.zeros_like(rho)
    rho_k, sf = rho.clone(), 1.0
    for k in range(1, N_TERMS + 1):
        sf *= math.sqrt(k)
        gh = g[k - 1] / sf
        acc += torch.outer(gh, gh) * rho_k
        if k < N_TERMS:
            rho_k = rho_k * rho
    C = torch.outer(s, s) * acc
    C = C - torch.diag(torch.diagonal(C)) + torch.diag(V)
    return M, C


def propagate(ws, use_core=True, track=None):
    """Rank-1 non-Gaussian core closure. use_core=False -> plain K=2."""
    mu = torch.zeros(N_W, dtype=DT)
    cov = torch.eye(N_W, dtype=DT)
    lam, u = 0.0, torch.zeros(N_W, dtype=DT)
    skew, kurt = 0.0, 0.0

    for li, W in enumerate(ws):
        if use_core and lam > 0:
            cq, wq = core_nodes(skew, kurt)
        else:
            cq = torch.zeros(1, dtype=DT)
            wq = torch.ones(1, dtype=DT)

        R = cov - lam * torch.outer(u, u) if (use_core and lam > 0) else cov
        mu_z = mu @ W
        p = (u @ W) * math.sqrt(lam) if (use_core and lam > 0) else torch.zeros(N_W, dtype=DT)
        Se = W.T @ (R @ W)
        s = torch.clamp(torch.diagonal(Se), min=1e-12).sqrt()
        rho = torch.clamp(Se / torch.outer(s, s), -1.0, 1.0)

        Ms, Cs = [], []
        for q in range(len(cq)):
            m_q = mu_z + p * cq[q]
            M, C = relu_cond(m_q, s, rho)
            Ms.append(M)
            Cs.append(C)

        mu_new = sum(float(wq[q]) * Ms[q] for q in range(len(cq)))
        cov_new = torch.zeros_like(cov)
        for q in range(len(cq)):
            cov_new += float(wq[q]) * (Cs[q] + torch.outer(Ms[q], Ms[q]))
        cov_new -= torch.outer(mu_new, mu_new)
        cov_new = 0.5 * (cov_new + cov_new.T)

        # re-extract the core and push the c-distribution forward
        ev, U = torch.linalg.eigh(cov_new)
        u_new = U[:, -1]
        lam_new = float(torch.clamp(ev[-1], min=0.0))
        if use_core and lam_new > 1e-12:
            t = torch.tensor([float(u_new @ (Ms[q] - mu_new)) for q in range(len(cq))], dtype=DT)
            v = torch.tensor([float(u_new @ (Cs[q] @ u_new)) for q in range(len(cq))], dtype=DT)
            m1 = float((wq * t).sum())
            m2 = float((wq * (t**2 + v)).sum())
            m3 = float((wq * (t**3 + 3 * t * v)).sum())
            m4 = float((wq * (t**4 + 6 * t**2 * v + 3 * v**2)).sum())
            var = max(m2 - m1**2, 1e-16)
            skew = (m3 - 3 * m1 * m2 + 2 * m1**3) / var**1.5
            kurt = (m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4) / var**2 - 3.0
            skew = float(np.clip(skew, -1.5, 1.5))
            kurt = float(np.clip(kurt, -1.0, 3.0))
        else:
            skew, kurt = 0.0, 0.0

        mu, cov, u, lam = mu_new, cov_new, u_new, lam_new
        if track is not None and li in track:
            track[li] = (lam, skew, kurt)

    return mu


print(f"{'seed':>4s} {'K=2':>11s} {'core(rank1)':>12s} {'gain':>7s}")
acc2 = acc_c = 0.0
for seed in range(SEEDS):
    ws = build_ws(60_000 + seed)
    gt = gt_mean(ws, GT_N)
    e2 = float(((propagate(ws, use_core=False) - gt) ** 2).mean())
    tr = {31: None}
    ec = float(((propagate(ws, use_core=True, track=tr) - gt) ** 2).mean())
    acc2 += e2 / SEEDS
    acc_c += ec / SEEDS
    lam, sk, ku = tr[31]
    print(f"{seed:4d} {e2:11.3e} {ec:12.3e} {e2/ec:6.2f}x   "
          f"(L31 core: lam={lam:.4f} skew={sk:+.3f} kurt={ku:+.3f})")
print(f"{'MEAN':>4s} {acc2:11.3e} {acc_c:12.3e} {acc2/acc_c:6.2f}x")
print("\nReference: QMC at the floor ~4.1e-6; competitor ~3.2e-7 (pure analytic, floor).")
print("Next if this works: end-to-end calibration on top, then port to flopscope.")
