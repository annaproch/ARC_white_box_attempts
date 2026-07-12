"""Where and how does the K=2 covariance go wrong, layer by layer?

The final-layer error is 98% propagated-moment error, dominated by a ~30%
error in the pre-activation variance. sigma_z^2 = w^T Sigma_a w is driven by
Sigma's top eigenvalue (lambda1 ~ n * mean-off-diagonal), so a small
systematic bias in the off-diagonal update is amplified by n and compounds
over 32 layers.

This script compares K=2 propagation against sampled truth AT EVERY LAYER:
  mean            : rms relative error of mu
  var (diag)      : mean relative error of the diagonal
  lambda1         : relative error of Sigma's top eigenvalue  <-- the amplifier
  offdiag         : relative error of the mean off-diagonal covariance
  skew(c), kurt(c): non-Gaussianity of the dominant-mode coefficient
                    c = u1^T (a - mu) / sqrt(lambda1)

Read-out:
  * lambda1 error growing ~geometrically with depth => multiplicative per-layer
    bias => a rank-1 / low-rank kappa3 correction (or a learned per-layer
    covariance-scale correction) is the right fix, and rank can be small.
  * skew(c) large => the dominant mode is genuinely non-Gaussian => track its
    distribution (quadrature closure) rather than just its variance.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/layerwise_cov_drift.py
"""

import math

import torch

torch.set_grad_enabled(False)
SDEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
GT_N, BATCH = 2_000_000, 131_072
SEEDS = 3
SQ2PI = math.sqrt(2 * math.pi)
N_TERMS = 4  # converged (see wick_terms_test.py)


def build_ws(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N_W, N_W, generator=g, dtype=torch.float64) * math.sqrt(2.0 / N_W)
            for _ in range(DEPTH)]


def k2_layers(ws):
    """K=2 propagation; yields (mu, cov) of the post-ReLU activations per layer."""
    mu = torch.zeros(N_W, dtype=torch.float64)
    cov = torch.eye(N_W, dtype=torch.float64)
    out = []
    for W in ws:
        mu_pre = mu @ W
        cov_pre = W.T @ (cov @ W)
        var = torch.clamp(torch.diagonal(cov_pre), min=1e-12)
        s = var.sqrt()
        a = mu_pre / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        rho = torch.clamp(cov_pre / torch.outer(s, s), -1.0, 1.0)
        mu = s * pdf + mu_pre * cdf
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu * mu, min=0.0)
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
        cov = torch.outer(s, s) * acc
        cov = cov - torch.diag(torch.diagonal(cov)) + torch.diag(var_post)
        out.append((mu.clone(), cov.clone()))
    return out


def true_layers(ws):
    """Sampled (mu, cov) per layer, plus skew/kurtosis of the dominant mode."""
    wd = [w.float().to(SDEV) for w in ws]
    s1 = [torch.zeros(N_W, dtype=torch.float64) for _ in range(DEPTH)]
    s2 = [torch.zeros(N_W, N_W, dtype=torch.float64) for _ in range(DEPTH)]
    chunks = []
    done = 0
    while done < GT_N:
        b = min(BATCH, GT_N - done)
        x = torch.randn(b, N_W, device=SDEV)
        acts = []
        for w in wd:
            x = torch.relu(x @ w)
            acts.append(x)
        for li, a in enumerate(acts):
            s1[li] += a.sum(0).cpu().double()
            s2[li] += (a.T @ a).cpu().double()
        chunks.append([a[:8192].cpu() for a in acts])  # keep a slice for higher moments
        done += b
    out = []
    for li in range(DEPTH):
        mu = s1[li] / GT_N
        cov = s2[li] / GT_N - torch.outer(mu, mu)
        out.append((mu, cov))
    return out, chunks


def mode_moments(cov, mu, chunks, li):
    """skew/kurtosis of c = u1^T (a - mu)/sqrt(lambda1) from the retained slices."""
    ev, U = torch.linalg.eigh(cov)
    u1 = U[:, -1].float()
    lam1 = float(ev[-1])
    cs = []
    for ch in chunks:
        a = ch[li]
        cs.append(((a - mu.float()) @ u1) / math.sqrt(max(lam1, 1e-12)))
    c = torch.cat(cs).double()
    c = c - c.mean()
    sd = c.std()
    skew = float((c**3).mean() / sd**3)
    kurt = float((c**4).mean() / sd**4 - 3.0)
    return lam1, skew, kurt


for seed in range(SEEDS):
    ws = build_ws(70_000 + seed)
    est = k2_layers(ws)
    tru, chunks = true_layers(ws)
    print(f"\n=== seed {seed} ===")
    print(f"{'L':>3s} {'mu err':>8s} {'var err':>8s} {'lam1 err':>9s} {'offdiag err':>11s} "
          f"{'skew(c)':>8s} {'kurt(c)':>8s} {'eig1frac':>8s}")
    for li in (0, 1, 2, 3, 5, 7, 11, 15, 19, 23, 27, 31):
        mu_e, cov_e = est[li]
        mu_t, cov_t = tru[li]
        eye = torch.eye(N_W, dtype=torch.bool)
        mu_err = float(((mu_e - mu_t) ** 2).mean().sqrt() / (mu_t**2).mean().sqrt())
        d_e, d_t = torch.diagonal(cov_e), torch.diagonal(cov_t)
        var_err = float(((d_e - d_t) / d_t.clamp(min=1e-12)).mean())
        lam_t, skew, kurt = mode_moments(cov_t, mu_t, chunks, li)
        lam_e = float(torch.linalg.eigvalsh(cov_e)[-1])
        lam_err = (lam_e - lam_t) / max(lam_t, 1e-12)
        off_e = float(cov_e[~eye].mean())
        off_t = float(cov_t[~eye].mean())
        off_err = (off_e - off_t) / abs(off_t) if abs(off_t) > 1e-12 else float("nan")
        eig1frac = lam_t / float(torch.diagonal(cov_t).sum())
        print(f"{li:3d} {mu_err:8.4f} {var_err:8.4f} {lam_err:9.4f} {off_err:11.4f} "
              f"{skew:8.3f} {kurt:8.3f} {eig1frac:8.3f}")

print("\nlam1 err growing with depth => multiplicative bias in the off-diagonal update")
print("   => low-rank kappa3 correction (rank set by eig1frac) is the fix.")
print("skew(c)/kurt(c) large => dominant mode genuinely non-Gaussian => quadrature closure.")
