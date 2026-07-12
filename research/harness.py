"""E0 — the canonical harness. ONE implementation of each closure, ONE benchmark.

Everything downstream cites this file. Rules (STRATEGY.md §0):
  * float64 on CPU (cov propagation over 32 layers; MPS has no float64)
  * paired stats over a FIXED corpus, mean +- SEM of the paired difference
  * self-tests for conventions run before any science

Conventions (match the challenge + build_moment_corpus.py):
  W[l] : (in, out).  z^(l) = W[l]^T a^(l-1)   (i.e. a @ W)
  a^(l) = ReLU(z^(l)),  l = 0..L-1,  a^(-1) = x ~ N(0, I)
Corpus per MLP: mu/cov/k3/k4[l] = moments of a^(l);  zmu/zcov/zk3/zk4[l] of z^(l).
  k3[i,j] = kappa3(v_i, v_i, v_j)      (n,n slice)
  k4[i,j] = kappa4(v_i, v_i, v_j, v_j) (n,n slice)
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "moment_corpus"
DT = torch.float64

SQRT2PI = math.sqrt(2.0 * math.pi)


def npdf(t):
    return torch.exp(-0.5 * t * t) / SQRT2PI


def ncdf(t):
    return 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))


# ---------------------------------------------------------------- corpus ----
def load_corpus(n_mlps: int = 24, seed0: int = 200_000):
    """Fixed MLP set. Never regenerate; add MLPs only by extending the range."""
    recs = []
    for m in range(n_mlps):
        p = CORPUS / f"mlp_{seed0 + m}.pt"
        if not p.exists():
            break
        d = torch.load(p, map_location="cpu", weights_only=False)
        recs.append({k: (v.to(DT) if torch.is_tensor(v) else v) for k, v in d.items()})
    if not recs:
        raise SystemExit(f"no corpus in {CORPUS}")
    return recs


# ------------------------------------------------------- the closure step ----
def hermite_gains(zmu, zvar, kmax: int = 8):
    """ReLU Hermite coefficients w.r.t. N(zmu, zvar), hhat_k = E[ReLU^(k)(Y)].

        hhat_0 = s*pdf + mu*cdf      (= E[ReLU])
        hhat_1 = cdf
        hhat_k = (-1)^k He_{k-2}(t) * pdf / s^(k-1)      for k >= 2

    Derivation: ReLU^(k) = delta^(k-2) for k>=2, and E[delta^(m)(Y)] =
    (-1)^m p^(m)(0) = (-1)^m He_m(t) pdf / s^(m+1).  Verified in self_test().
    """
    s = torch.sqrt(torch.clamp(zvar, min=1e-30))
    t = zmu / s
    pdf, cdf = npdf(t), ncdf(t)
    g = [s * pdf + zmu * cdf, cdf]
    # probabilists' Hermite He_m(t), m = 0..kmax-2
    he = [torch.ones_like(t), t]
    for m in range(1, kmax):
        he.append(t * he[m] - m * he[m - 1])
    for k in range(2, kmax + 1):
        g.append(((-1.0) ** k) * he[k - 2] * pdf / s ** (k - 1))
    return g, s, t, pdf, cdf


def relu_step(mu_a, cov_a, W, kmax: int = 8, zk3=None, zk4=None):
    """One layer: linear (EXACT) then ReLU (the closure).

    zk3/zk4 = (n,n) TRUE cumulant slices of z at this layer. If given, adds the
    third/fourth-order corrections; if None, this is the pure Gaussian (K=2)
    closure. Returns (mu', cov') of a' = ReLU(z).
    """
    zmu = mu_a @ W                       # W is (in,out): z = W^T a
    zcov = W.T @ cov_a @ W
    zvar = torch.clamp(torch.diagonal(zcov), min=1e-30)

    g, s, t, pdf, cdf = hermite_gains(zmu, zvar, kmax)

    # --- mean.  E[phi(z)] = E_G[phi] + k3/3! * E_G[phi'''] + k4/4! * E_G[phi'''']
    #     and E_G[phi^(k)] = hhat_k, so the corrections are just g[3], g[4].
    mu = g[0].clone()
    # --- E[a^2] for the exact diagonal.  psi = ReLU^2: psi'''=2*delta,
    #     psi''''=2*delta'  =>  E_G[psi''']=2*pdf/s, E_G[psi'''']=-2*t*pdf/s^2
    ez2 = (zmu * zmu + zvar) * cdf + zmu * s * pdf
    if zk3 is not None:
        k3d = torch.diagonal(zk3)                       # kappa3(z_i,z_i,z_i)
        mu = mu + (k3d / 6.0) * g[3]
        ez2 = ez2 + (k3d / 6.0) * (2.0 * pdf / s)
    if zk4 is not None:
        k4d = torch.diagonal(zk4)                       # kappa4(z_i,...,z_i)
        mu = mu + (k4d / 24.0) * g[4]
        ez2 = ez2 + (k4d / 24.0) * (-2.0 * t * pdf / s ** 2)
    var = torch.clamp(ez2 - mu * mu, min=1e-30)

    # --- off-diagonal covariance: Wick/Hermite series (Cor. A.3 of 2605.05179)
    #     Cov[phi(z_i),phi(z_j)] = sum_{k>=1} hhat_k(i) hhat_k(j) C_ij^k / k!
    C = zcov
    cov = torch.zeros_like(C)
    Ck = C.clone()
    fact = 1.0
    for k in range(1, kmax + 1):
        fact *= k
        cov = cov + torch.outer(g[k], g[k]) * Ck / fact
        Ck = Ck * C
    if zk3 is not None:
        # dCov_ij = 1/2 [ g2(i) g1(j) k3(i,i,j) + g1(i) g2(j) k3(i,j,j) ]
        cov = cov + 0.5 * (torch.outer(g[2], g[1]) * zk3
                           + torch.outer(g[1], g[2]) * zk3.T)
    if zk4 is not None:
        # leading kappa4 term: 1/4 g2(i) g2(j) k4(i,i,j,j).  (Terms needing
        # k4_iiij are NOT in the corpus -- this correction is incomplete.)
        cov = cov + 0.25 * torch.outer(g[2], g[2]) * zk4
    cov = 0.5 * (cov + cov.T)
    idx = torch.arange(cov.shape[0])
    cov[idx, idx] = var
    return mu, cov


def _series(g, S, ai, aj, coef, pmin, pmax, K):
    """coef * sum_{p=pmin}^{pmax} (1/p!) g[p+ai](i) g[p+aj](j) * K * S^p"""
    out = torch.zeros_like(S)
    Sp = torch.ones_like(S) if pmin == 0 else S.clone()
    fact = 1.0
    for p in range(pmin, pmax + 1):
        if p > 0:
            fact *= p
        out = out + torch.outer(g[p + ai], g[p + aj]) * K * (Sp / fact)
        Sp = Sp * S
    return coef * out


def relu_step_full(mu_a, cov_a, W, cum, pmax: int = 6, kmax: int = 12):
    """COMPLETE Hermite diagram expansion (E4). `cum` holds the TRUE cumulant
    slices of z at this layer: k3_iii, k3_iij, k4_iiii, k4_iijj, k4_iiij.

    Every cumulant block generates a SERIES in Sigma_ij (the shipped code kept
    only p=0), and three kappa4 families exist (iijj, iiij, iiii) -- they are the
    same order and only work TOGETHER. See STRATEGY.md 4.8.
    """
    zmu = mu_a @ W
    zcov = W.T @ cov_a @ W
    zvar = torch.clamp(torch.diagonal(zcov), min=1e-30)
    g, s, t, pdf, cdf = hermite_gains(zmu, zvar, kmax)
    k3d, k3m, k4d, k4m, k4t = (cum["k3_iii"], cum["k3_iij"],
                               cum["k4_iiii"], cum["k4_iijj"], cum["k4_iiij"])

    # ---- mean: Gram-Charlier incl. the kappa3^2 POWER-CUMULANT term
    mu = (g[0] + k3d / 6.0 * g[3] + k4d / 24.0 * g[4]
          + (10.0 / 720.0) * g[6] * k3d ** 2)
    ez2 = ((zmu * zmu + zvar) * cdf + zmu * s * pdf
           + (k3d / 6.0) * (2.0 * pdf / s)
           + (k4d / 24.0) * (-2.0 * t * pdf / s ** 2))
    var = torch.clamp(ez2 - mu * mu, min=1e-30)

    # ---- covariance: full series, all diagram families
    S = zcov
    ones = torch.ones_like(S)
    k3i = k3d[:, None].expand_as(S)
    k4i = k4d[:, None].expand_as(S)
    cov = (_series(g, S, 0, 0, 1.0, 1, pmax, ones)
           + _series(g, S, 2, 1, 0.5, 0, pmax, k3m)
           + _series(g, S, 1, 2, 0.5, 0, pmax, k3m.T)
           + _series(g, S, 3, 0, 1 / 6, 1, pmax, k3i)
           + _series(g, S, 0, 3, 1 / 6, 1, pmax, k3i.T)
           + _series(g, S, 2, 2, 0.25, 0, pmax, k4m)
           + _series(g, S, 3, 1, 1 / 6, 0, pmax, k4t)
           + _series(g, S, 1, 3, 1 / 6, 0, pmax, k4t.T)
           + _series(g, S, 4, 0, 1 / 24, 1, pmax, k4i)
           + _series(g, S, 0, 4, 1 / 24, 1, pmax, k4i.T))
    cov = 0.5 * (cov + cov.T)
    idx = torch.arange(cov.shape[0])
    cov[idx, idx] = var
    return mu, cov


def propagate(mu, cov, Ws, kmax: int = 8):
    """Run the pure Gaussian closure over the given weight list."""
    for W in Ws:
        mu, cov = relu_step(mu, cov, W, kmax)
    return mu, cov


# ------------------------------------------------------------ self-tests ----
def self_test(rec, atol_lin=2e-3):
    """Catch convention/indexing bugs BEFORE any science. Fails loudly."""
    W, mu, cov, zmu, zcov = rec["W"], rec["mu"], rec["cov"], rec["zmu"], rec["zcov"]
    n, L = mu.shape[1], mu.shape[0]
    out = {}

    # (1) linear step is EXACT: z^(l) = W[l]^T a^(l-1).
    #     For l>=1 the corpus mu[l-1] and zmu[l] are empirical moments of the
    #     SAME samples, and z = W^T a holds per-sample, so the relation must
    #     hold to FLOAT PRECISION (not sampling noise). Any drift = convention bug.
    #     l=0 is excluded: the true E[z^0]=0 and Cov=W^T W are known exactly,
    #     while the corpus stores noisy sample estimates of them -- there the
    #     analytic value is the correct one (verified in (1b)).
    e_mu, e_cov = [], []
    for l in range(1, L):
        zm = mu[l - 1] @ W[l]
        zc = W[l].T @ cov[l - 1] @ W[l]
        e_mu.append((zm - zmu[l]).pow(2).mean().sqrt().item()
                    / zmu[l].pow(2).mean().sqrt().item())
        e_cov.append((zc - zcov[l]).pow(2).mean().sqrt().item()
                     / zcov[l].pow(2).mean().sqrt().item())
    out["linear_mu_relerr"] = max(e_mu)
    out["linear_cov_relerr"] = max(e_cov)
    assert out["linear_mu_relerr"] < 1e-5, (
        f"LINEAR STEP MISMATCH (rel {out['linear_mu_relerr']:.2e}) -> convention bug")
    assert out["linear_cov_relerr"] < 1e-4, (
        f"LINEAR COV MISMATCH (rel {out['linear_cov_relerr']:.2e}) -> convention bug")

    # (1b) layer 0: corpus deviation from the exact analytic value must be
    #      consistent with MC noise sqrt(Var/N), confirming the corpus (not us)
    #      is the noisy one.
    dz0 = (torch.zeros(n, dtype=DT) @ W[0] - zmu[0]).pow(2).mean().sqrt().item()
    pred = math.sqrt(torch.diagonal(zcov[0]).mean().item() / rec["n_samples"])
    out["layer0_noise_ratio"] = dz0 / pred
    assert 0.5 < out["layer0_noise_ratio"] < 2.0, (
        f"layer-0 deviation {dz0:.2e} not explained by MC noise {pred:.2e}")

    # (2) Hermite gains. hhat_k = E[ReLU^(k)(Y)] = d^k/dmu^k E[ReLU(Y)], so
    #     finite-difference the closed form E[ReLU] = s*pdf(t)+mu*cdf(t) in mu.
    #     (Do NOT quadrature E[ReLU'] directly: the integrand is discontinuous,
    #     so a grid gives O(dz) error and falsely fails.)
    #     First confirm the closed form itself against smooth quadrature.
    zm = zmu[L - 1][:8].clone()
    zv = torch.diagonal(zcov[L - 1])[:8].clone()
    s0 = torch.sqrt(zv)
    q = torch.linspace(-14, 14, 200_001, dtype=DT)
    wq = npdf(q) * (q[1] - q[0])
    quad0 = torch.stack([(torch.clamp(zm[i] + s0[i] * q, min=0) * wq).sum()
                         for i in range(8)])
    g0 = s0 * npdf(zm / s0) + zm * ncdf(zm / s0)
    err0 = (quad0 - g0).abs().max().item()
    assert err0 < 1e-6, f"E[ReLU] closed form wrong: {err0:.2e}"

    #     Then check hhat_k against its DEFINITION (paper eq. 1):
    #         hhat_k = (1/s^k) E[ ReLU(Y) * He_k((Y-mu)/s) ]
    #     The integrand is smooth (ReLU x polynomial x Gaussian), so quadrature
    #     is exact here -- unlike E[ReLU^(k)], which is a distribution.
    KM = 6
    g, *_ = hermite_gains(zm, zv, kmax=KM)
    he = [torch.ones_like(q), q]                  # He_m(q), q = standardised
    for m in range(1, KM + 1):
        he.append(q * he[m] - m * he[m - 1])
    errs = []
    for k in range(1, KM + 1):
        ref = torch.stack([
            (torch.clamp(zm[i] + s0[i] * q, min=0) * he[k] * wq).sum() / s0[i] ** k
            for i in range(8)])
        errs.append(((ref - g[k]).abs() / (g[k].abs() + 1e-12)).max().item())
    out["hermite_max_relerr"] = max(errs)
    assert out["hermite_max_relerr"] < 1e-6, (
        f"HERMITE GAIN FORMULA WRONG: rel err per k = "
        f"{[f'{e:.1e}' for e in errs]}")

    # (3) GT sampling noise floor on the final-layer mean (for interpreting MSE).
    var_final = torch.diagonal(cov[L - 1]).mean().item()
    out["gt_mse_noise_floor"] = var_final / rec["n_samples"]
    return out


def mse(pred, truth):
    return ((pred - truth) ** 2).mean().item()


if __name__ == "__main__":
    recs = load_corpus(4)
    print(f"corpus: {len(recs)} MLPs, n_samples={recs[0]['n_samples']:,}")
    for r in recs[:2]:
        o = self_test(r)
        print(f"  seed {r['seed']}: linear rel-err mu {o['linear_mu_relerr']:.1e} "
              f"cov {o['linear_cov_relerr']:.1e} | layer0 dev/noise "
              f"{o['layer0_noise_ratio']:.2f} | "
              f"hermite rel-err {o['hermite_max_relerr']:.1e} | "
              f"GT MSE noise floor {o['gt_mse_noise_floor']:.1e}")
    print("SELF-TESTS PASSED")
