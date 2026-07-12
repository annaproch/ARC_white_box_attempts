"""Where does our K=2 final-layer error actually come from?

The final-layer estimate is  E[ReLU(z)] ~= sigma*phi(alpha) + mu*Phi(alpha),
with (mu, sigma) the propagated moments of the final PRE-activation z = z_31.
Three error sources:

  (A) upstream moment error : our propagated (mu_hat, sigma_hat) != true
  (B) closure/shape error   : z is not Gaussian, so the formula is wrong even
                              when fed the TRUE (mu, sigma)
  (C) their interaction

Sampling gives the true mu_z, sigma_z, skew/kurtosis of z, and the true
E[ReLU(z)]. So:
   err_total   = g(mu_hat, sig_hat) - truth
   err_shape   = g(mu_true, sig_true) - truth        <- (B) alone
   err_moments = g(mu_hat, sig_hat) - g(mu_true, sig_true)   <- (A) alone

If (B) dominates -> fix the 1-D marginal model of z (Edgeworth/quadrature on
the true skew) -- cheap, no higher-order tensor propagation needed.
If (A) dominates -> the moments themselves are wrong -> we need third-order
information upstream (rank-r K=3), and we should also check WHICH moment
(mean vs variance) is off.

Also prints the covariance spectrum of the last hidden layer and the
non-Gaussianity along its top mode -- the input to the "non-Gaussian core +
Gaussian remainder" closure idea.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/error_attribution.py
"""

import math

import torch

torch.set_grad_enabled(False)
SDEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
GT_N, BATCH = 4_000_000, 131_072
SEEDS = 6
SQ2PI = math.sqrt(2 * math.pi)


def build_ws(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(N_W, N_W, generator=g, dtype=torch.float64) * math.sqrt(2.0 / N_W)
            for _ in range(DEPTH)]


def relu_mean(mu, var):
    s = torch.clamp(var, min=1e-12).sqrt()
    a = mu / s
    pdf = torch.exp(-0.5 * a * a) / SQ2PI
    cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
    return s * pdf + mu * cdf


def k2_final_preact(ws, n_terms=4):
    """K=2 covariance propagation; returns propagated (mu, var) of the FINAL
    pre-activation z_31, plus the last hidden layer's (mu, cov)."""
    mu = torch.zeros(N_W, dtype=torch.float64)
    cov = torch.eye(N_W, dtype=torch.float64)
    for li, W in enumerate(ws):
        mu_pre = mu @ W
        cov_pre = W.T @ (cov @ W)
        var = torch.clamp(torch.diagonal(cov_pre), min=1e-12)
        if li == DEPTH - 1:
            return mu_pre, var, mu, cov
        s = var.sqrt()
        a = mu_pre / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        rho = torch.clamp(cov_pre / torch.outer(s, s), -1.0, 1.0)
        mu_post = s * pdf + mu_pre * cdf
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu_post**2, min=0.0)
        g = [cdf, pdf]
        he_prev, he_cur = torch.ones_like(a), a
        for k in range(3, n_terms + 1):
            m = k - 2
            g.append(((-1) ** m) * he_cur * pdf)
            he_prev, he_cur = he_cur, a * he_cur - m * he_prev
        acc = torch.zeros_like(cov_pre)
        rho_k, sf = rho.clone(), 1.0
        for k in range(1, n_terms + 1):
            sf *= math.sqrt(k)
            gh = g[k - 1] / sf
            acc += torch.outer(gh, gh) * rho_k
            if k < n_terms:
                rho_k = rho_k * rho
        cov = torch.outer(s, s) * acc
        cov = cov - torch.diag(torch.diagonal(cov)) + torch.diag(var_post)
        mu = mu_post
    raise RuntimeError


def true_final_stats(ws):
    """Sampled moments of the final pre-activation z_31 and the true E[ReLU(z)],
    plus the last hidden layer's covariance spectrum and top-mode skew."""
    wd = [w.float().to(SDEV) for w in ws]
    s1 = torch.zeros(N_W, dtype=torch.float64)
    s2 = torch.zeros(N_W, dtype=torch.float64)
    s3 = torch.zeros(N_W, dtype=torch.float64)
    sr = torch.zeros(N_W, dtype=torch.float64)          # E[ReLU(z)]
    a_sum = torch.zeros(N_W, dtype=torch.float64)       # last hidden mean
    a_cov = torch.zeros(N_W, N_W, dtype=torch.float64)  # last hidden 2nd moment
    done = 0
    while done < GT_N:
        b = min(BATCH, GT_N - done)
        x = torch.randn(b, N_W, device=SDEV)
        for w in wd[:-1]:
            x = torch.relu(x @ w)
        z = x @ wd[-1]                                   # final pre-activation
        # reduce on-device (float32), move only the small results to CPU float64
        s1 += z.sum(0).cpu().double()
        s2 += (z * z).sum(0).cpu().double()
        s3 += (z * z * z).sum(0).cpu().double()
        sr += torch.relu(z).sum(0).cpu().double()
        a_sum += x.sum(0).cpu().double()
        a_cov += (x.T @ x).cpu().double()
        done += b
    m1 = s1 / GT_N
    var = s2 / GT_N - m1**2
    m3 = s3 / GT_N - 3 * m1 * (s2 / GT_N) + 2 * m1**3   # 3rd central moment
    skew = m3 / var.clamp(min=1e-12) ** 1.5
    truth = sr / GT_N
    a_mean = a_sum / GT_N
    a_cov = a_cov / GT_N - torch.outer(a_mean, a_mean)
    return m1, var, skew, truth, a_cov


print(f"{'seed':>4s} {'total':>10s} {'shape(B)':>10s} {'moments(A)':>11s} "
      f"{'|dmu|rms':>9s} {'|dsig|rms':>9s} {'skew(z)':>8s} {'eig1frac':>8s}")
tot = {"total": 0.0, "shape": 0.0, "moments": 0.0}
for seed in range(SEEDS):
    ws = build_ws(60_000 + seed)
    mu_hat, var_hat, _, _ = k2_final_preact(ws)
    mu_t, var_t, skew_t, truth, a_cov = true_final_stats(ws)

    g_hat = relu_mean(mu_hat, var_hat)
    g_true = relu_mean(mu_t, var_t)
    e_total = float(((g_hat - truth) ** 2).mean())
    e_shape = float(((g_true - truth) ** 2).mean())
    e_mom = float(((g_hat - g_true) ** 2).mean())
    tot["total"] += e_total / SEEDS
    tot["shape"] += e_shape / SEEDS
    tot["moments"] += e_mom / SEEDS

    dmu = float(((mu_hat - mu_t) ** 2).mean() ** 0.5)
    dsig = float(((var_hat.sqrt() - var_t.sqrt()) ** 2).mean() ** 0.5)
    ev = torch.linalg.eigvalsh(a_cov).flip(0).clamp(min=0)
    eig1 = float(ev[0] / ev.sum())
    print(f"{seed:4d} {e_total:10.3e} {e_shape:10.3e} {e_mom:11.3e} "
          f"{dmu:9.2e} {dsig:9.2e} {float(skew_t.mean()):8.3f} {eig1:8.3f}")

print(f"{'MEAN':>4s} {tot['total']:10.3e} {tot['shape']:10.3e} {tot['moments']:11.3e}")
print("\nshape(B) >> moments(A) -> fix the 1-D marginal of z (Edgeworth/quadrature): cheap.")
print("moments(A) >> shape(B) -> upstream moments are wrong -> need rank-r K=3.")
print("eig1frac = share of last-hidden-layer variance in the top mode (common mode).")
