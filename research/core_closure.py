"""Rank-1 non-Gaussian core closure ("K=2 + skewed common mode"), v2.

v1 used a cubic Cornish-Fisher warp of Gauss-Hermite nodes to represent the
core's distribution; at Q=16 the outer GH nodes sit at |xi| ~ 6.6, the cubic
warp sent them to |c| ~ 12, the conditional means exploded, and the skew/kurt
recursion ran away into the clamps (observed: skew=+1.5, kurt=+3.0, lambda1
13.6 vs truth ~2). The closure was fed a garbage core, hence no gain.

v2 represents the core's distribution NONPARAMETRICALLY as Q equally-weighted
quantile particles, pushed forward exactly:

    a       = mu + sqrt(lam) * u * c + eps,   eps ~ N(0, R),  R = Sigma - lam u u^T
    z | c   ~ N(mu_z + p c, Sigma_eps),       p = sqrt(lam) W^T u,  Sigma_eps = W^T R W
    mu'     = sum_q w_q M(c_q)                             [exact 1-D ReLU mean]
    Sigma'  = sum_q w_q [C(c_q) + M M^T] - mu' mu'^T       [law of total covariance]
    u',lam' = top eigenpair of Sigma'
    c' | c  ~ N( t(c), v(c) ),  t(c)=u'^T(M(c)-mu'),  v(c)=u'^T C(c) u'

The pushforward is expanded with a small Gauss-Hermite sub-grid and re-gridded
back to Q quantiles, so the particle cloud is always a valid distribution
(bounded by real conditional means + finite noise). Var(c')=1 holds by
construction since lam' = Var(t) + E[v].

NOTE: if c were Gaussian, a would be jointly Gaussian and this closure reduces
exactly to K=2 -- all the gain comes from c's non-Gaussianity (measured true
skew ~0.4, kurt ~0.25 at depth). The per-layer printout checks exactly that.

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
Q = 24          # core particles (equal weights = quantiles)
G = 5           # Gauss-Hermite sub-nodes for the pushforward noise
N_TERMS = 4     # Wick series (converged; see wick_terms_test.py)
SQ2PI = math.sqrt(2 * math.pi)
DT = torch.float64

_g_xi, _g_w = np.polynomial.hermite_e.hermegauss(G)
GX = torch.tensor(_g_xi, dtype=DT)
GW = torch.tensor(_g_w / _g_w.sum(), dtype=DT)


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


def wick_g(a, pdf, cdf, k_max):
    g = [cdf, pdf]
    he_prev, he_cur = torch.ones_like(a), a
    for k in range(3, k_max + 1):
        m = k - 2
        g.append(((-1) ** m) * he_cur * pdf)
        he_prev, he_cur = he_cur, a * he_cur - m * he_prev
    return g[:k_max]


def relu_cond(m, s, rho, ss):
    """Conditional ReLU mean M and covariance C for z ~ N(m, diag(s^2)) with
    correlation rho (ss = outer(s, s), precomputed)."""
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
    C = ss * acc
    C = C - torch.diag(torch.diagonal(C)) + torch.diag(V)
    return M, C


def regrid(y, w, q):
    """Re-compress a weighted cloud to q equally-weighted quantile particles."""
    idx = torch.argsort(y)
    ys, wsx = y[idx], w[idx]
    cdf = torch.cumsum(wsx, 0) - 0.5 * wsx
    targets = (torch.arange(q, dtype=DT) + 0.5) / q
    out = torch.from_numpy(
        np.interp(targets.numpy(), cdf.numpy(), ys.numpy())
    ).to(DT)
    out = out - out.mean()
    sd = out.std(unbiased=False)
    return out / torch.clamp(sd, min=1e-12)


def propagate(ws, use_core=True, verbose=False):
    mu = torch.zeros(N_W, dtype=DT)
    cov = torch.eye(N_W, dtype=DT)
    lam, u = 0.0, torch.zeros(N_W, dtype=DT)
    c = torch.zeros(1, dtype=DT)           # layer 0: input is exactly Gaussian
    w = torch.ones(1, dtype=DT)            # -> single node, no core

    if verbose:
        print(f"  {'L':>3s} {'lam1':>8s} {'skew(c)':>8s} {'kurt(c)':>8s}")

    for li, W in enumerate(ws):
        core_on = use_core and lam > 1e-12
        R = cov - lam * torch.outer(u, u) if core_on else cov
        mu_z = mu @ W
        p = (u @ W) * math.sqrt(lam) if core_on else None
        Se = W.T @ (R @ W)
        s = torch.clamp(torch.diagonal(Se), min=1e-12).sqrt()
        ss = torch.outer(s, s)
        rho = torch.clamp(Se / ss, -1.0, 1.0)

        nodes = c if core_on else torch.zeros(1, dtype=DT)
        wts = w if core_on else torch.ones(1, dtype=DT)
        Ms, Cs = [], []
        for q in range(len(nodes)):
            m_q = mu_z + p * nodes[q] if core_on else mu_z
            M, C = relu_cond(m_q, s, rho, ss)
            Ms.append(M)
            Cs.append(C)

        mu_new = torch.zeros(N_W, dtype=DT)
        cov_new = torch.zeros(N_W, N_W, dtype=DT)
        for q in range(len(nodes)):
            mu_new += float(wts[q]) * Ms[q]
        for q in range(len(nodes)):
            cov_new += float(wts[q]) * (Cs[q] + torch.outer(Ms[q], Ms[q]))
        cov_new -= torch.outer(mu_new, mu_new)
        cov_new = 0.5 * (cov_new + cov_new.T)

        ev, U = torch.linalg.eigh(cov_new)
        u_new = U[:, -1]
        lam_new = float(torch.clamp(ev[-1], min=0.0))

        if use_core and lam_new > 1e-12:
            t = torch.stack([u_new @ (Ms[q] - mu_new) for q in range(len(nodes))])
            v = torch.stack([u_new @ (Cs[q] @ u_new) for q in range(len(nodes))])
            v = torch.clamp(v, min=0.0)
            # pushforward: c' | c ~ N(t, v); expand with GH sub-nodes, then regrid
            y = (t[:, None] + v.sqrt()[:, None] * GX[None, :]).reshape(-1)
            wy = (wts[:, None] * GW[None, :]).reshape(-1)
            c = regrid(y, wy, Q)
            w = torch.full((Q,), 1.0 / Q, dtype=DT)
        else:
            c = torch.zeros(1, dtype=DT)
            w = torch.ones(1, dtype=DT)

        mu, cov, u, lam = mu_new, cov_new, u_new, lam_new
        if verbose and li in (0, 1, 2, 3, 7, 15, 23, 31):
            sk = float((w * c**3).sum()) if len(c) > 1 else 0.0
            ku = float((w * c**4).sum()) - 3.0 if len(c) > 1 else 0.0
            print(f"  {li:3d} {lam:8.4f} {sk:8.3f} {ku:8.3f}")

    return mu


print("core diagnostics, seed 0 (compare: true skew(c) ~ +-0.4, kurt ~ 0.25,")
print("true lambda1 at L31 ~ 1.5-2.5; K=2 underestimates lambda1 by 27-43%)")
propagate(build_ws(60_000), use_core=True, verbose=True)

print(f"\n{'seed':>4s} {'K=2':>11s} {'core(rank1)':>12s} {'gain':>7s}")
acc2 = acc_c = 0.0
for seed in range(SEEDS):
    ws = build_ws(60_000 + seed)
    gt = gt_mean(ws, GT_N)
    e2 = float(((propagate(ws, use_core=False) - gt) ** 2).mean())
    ec = float(((propagate(ws, use_core=True) - gt) ** 2).mean())
    acc2 += e2 / SEEDS
    acc_c += ec / SEEDS
    print(f"{seed:4d} {e2:11.3e} {ec:12.3e} {e2/ec:6.2f}x")
print(f"{'MEAN':>4s} {acc2:11.3e} {acc_c:12.3e} {acc2/acc_c:6.2f}x")
print("\nReference: QMC at the floor ~4.1e-6; competitor ~3.2e-7 (pure analytic, floor).")
