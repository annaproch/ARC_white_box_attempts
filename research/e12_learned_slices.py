"""E12 (P1) — can the kappa3 slice be READ OFF the local state?

The claim (mean-field universality): the scale-normalised slice

    y_ij = kappa3(z)_iij / (sigma_i^2 sigma_j)

is a UNIVERSAL function of (t_i, t_j, rho_ij, layer), where t = zmu/sigma and
rho = Sigma_ij/(sigma_i sigma_j). If so we never propagate a tensor at all:
fit the function OFFLINE, ship it as a table (0-FLOP load), evaluate it at n^2
per layer (~1% of budget), and leave ~9% of the floor for QMC.

Evidence for the claim: S1 measured corr(gamma_i, t_i) = 0.92 for the diagonal.

Everything needed is ALREADY in the corpus (zmu, zcov, zk3 per layer) -- no
sampling. Fit per-layer ridge on a polynomial basis in (t_i, t_j, rho), train on
one set of MLPs and TEST ON HELD-OUT ones: if it does not transfer, it is not a
universal function, it is overfitting.

GPU: the design matrix is built and the normal equations accumulated on MPS in
float32; only the tiny (P x P) Gram matrices cross to CPU float64 to solve.
"""
from __future__ import annotations

import itertools
import statistics as st
import sys
import time

import torch

from harness import CORPUS, DT, hermite_gains

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256


def basis_exponents(deg):
    return [(a, b, c) for a in range(deg + 1) for b in range(deg + 1)
            for c in range(deg + 1) if a + b + c <= deg]


def features(zmu, zcov, target=None):
    """Return (X, y) for one layer, flattened over the n^2 (i,j) pairs.
    All on DEV in float32."""
    s = torch.sqrt(torch.clamp(torch.diagonal(zcov), min=1e-30))
    t = zmu / s
    rho = zcov / torch.outer(s, s)
    ti = t[:, None].expand(n, n)
    tj = t[None, :].expand(n, n)
    y = None
    if target is not None:
        y = target / (s[:, None] ** 2 * s[None, :])
    return ti, tj, rho, y


def design(ti, tj, rho, exps, star=None):
    cols = []
    for a, b, c in exps:
        cols.append((ti ** a) * (tj ** b) * (rho ** c))
    X = torch.stack([c.reshape(-1) for c in cols], 1)         # (n^2, P)
    if star is not None:
        # the STAR is the non-local piece the local features cannot see:
        # k3_iij = 2(C^T B)_ij + (D^T W)_ij,  B = Sigma diag(h1) W  -- a
        # contraction over p. Give it to the regression, plus its interactions
        # with the low-order local basis.
        sv = star.reshape(-1, 1)
        inter = [sv]
        for a, b, c in exps:
            if a + b + c <= 2:
                inter.append(sv * ((ti ** a) * (tj ** b) * (rho ** c)).reshape(-1, 1))
        X = torch.cat([X] + inter, 1)
    return X


def star_slice(zcov_prev, W, zmu_prev):
    """Freshly-generated kappa3 of z at THIS layer, from the previous layer's
    Gaussian state: B = Sigma diag(h1) W ; 2(C^T B) + (D^T W)."""
    s = torch.sqrt(torch.clamp(torch.diagonal(zcov_prev), min=1e-30))
    # hermite_gains needs float64; MPS has none -> compute on CPU, move back
    g, *_ = hermite_gains(zmu_prev.cpu().double(), (s ** 2).cpu().double(), kmax=4)
    h1 = g[1].to(W.device, W.dtype)
    h2 = g[2].to(W.device, W.dtype)
    B = zcov_prev @ (h1[:, None] * W)
    C = (h2[:, None] * W) * B
    D = h2[:, None] * B * B
    return 2.0 * (C.T @ B) + (D.T @ W)


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_train = n_mlps // 2
    global USE_STAR
    USE_STAR = (len(sys.argv) > 3 and sys.argv[3] == "star")
    exps = basis_exponents(deg)
    P = len(exps)
    Pfull = P + (1 + sum(1 for e in exps if sum(e) <= 2)) if USE_STAR else P
    print(f"E12 | {n_mlps} MLPs ({n_train} train / {n_mlps-n_train} HELD OUT) | "
          f"poly deg {deg} -> {P} coeffs | star={USE_STAR} | device {DEV}")

    recs = []
    for m in range(n_mlps):
        d = torch.load(CORPUS / f"mlp_{200000+m}.pt", map_location="cpu",
                       weights_only=False)
        recs.append({k: d[k] for k in ("zmu", "zcov", "zk3", "W", "mu", "cov")})

    t0 = time.time()
    lam = 1e-6
    coefs = []
    for l in range(32):
        G = torch.zeros(Pfull, Pfull, dtype=torch.float64)
        b = torch.zeros(Pfull, dtype=torch.float64)
        for m in range(n_train):
            zmu = recs[m]["zmu"][l].to(DEV, F32)
            zcov = recs[m]["zcov"][l].to(DEV, F32)
            k3 = recs[m]["zk3"][l].to(DEV, F32)
            ti, tj, rho, y = features(zmu, zcov, k3)
            st_ = None
            if l > 0:
                st_ = star_slice(recs[m]["zcov"][l-1].to(DEV, F32),
                                 recs[m]["W"][l].to(DEV, F32),
                                 recs[m]["zmu"][l-1].to(DEV, F32))
                sn = torch.sqrt(torch.clamp(torch.diagonal(zcov), min=1e-30))
                st_ = st_ / (sn[:, None] ** 2 * sn[None, :])
            else:
                st_ = torch.zeros(n, n, device=DEV, dtype=F32)
            X = design(ti, tj, rho, exps, st_ if USE_STAR else None)
            yv = y.reshape(-1)
            G += (X.T @ X).cpu().double()
            b += (X.T @ yv).cpu().double()
        G += lam * torch.eye(Pfull, dtype=torch.float64) * torch.diagonal(G).mean()
        coefs.append(torch.linalg.solve(G, b))
    print(f"  fit: {time.time()-t0:.0f}s", flush=True)

    # ---- evaluate on HELD-OUT MLPs
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)
    LAY = [1, 3, 7, 15, 23, 31]
    per_layer = {l: [] for l in LAY}
    allerr = []
    for m in range(n_train, n_mlps):
        for l in range(32):
            zmu = recs[m]["zmu"][l].to(DEV, F32)
            zcov = recs[m]["zcov"][l].to(DEV, F32)
            k3 = recs[m]["zk3"][l].to(DEV, F32)
            ti, tj, rho, y = features(zmu, zcov, k3)
            if l > 0:
                st_ = star_slice(recs[m]["zcov"][l-1].to(DEV, F32),
                                 recs[m]["W"][l].to(DEV, F32),
                                 recs[m]["zmu"][l-1].to(DEV, F32))
                sn = torch.sqrt(torch.clamp(torch.diagonal(zcov), min=1e-30))
                st_ = st_ / (sn[:, None] ** 2 * sn[None, :])
            else:
                st_ = torch.zeros(n, n, device=DEV, dtype=F32)
            X = design(ti, tj, rho, exps, st_ if USE_STAR else None)
            s = torch.sqrt(torch.clamp(torch.diagonal(zcov), min=1e-30))
            pred_y = (X @ coefs[l].to(DEV, F32)).reshape(n, n)
            pred = pred_y * (s[:, None] ** 2 * s[None, :])     # un-normalise
            e = ((pred - k3)[off].pow(2).sum()
                 / k3[off].pow(2).sum()).sqrt().item()
            allerr.append(e)
            if l in LAY:
                per_layer[l].append(e)

    print("\n=== HELD-OUT relative error of the learned kappa3_iij slice ===")
    print(f"{'layer':>6} " + " ".join(f"{'L'+str(l):>8}" for l in LAY)
          + f"   {'mean(all 32)':>13}")
    print(f"{'err':>6} "
          + " ".join(f"{st.mean(per_layer[l])*100:>7.1f}%" for l in LAY)
          + f"   {st.mean(allerr)*100:>12.1f}%")
    print("\n  SPEC (E6): kappa3 <=5% -> b^2 8.8e-7 | 10% -> 2.07e-6 | 20% -> 7e-6")
    print("  FUSION (4.9.3): b^2 < 1.02e-5 beats v4; b^2 ~3e-6 -> 1.8x v4")
    print("  Propagated-tensor closure for comparison: 11% (L1) -> 47% (L31)")


if __name__ == "__main__":
    main()
