"""G1 — can the kappa4 slice be LEARNED from local state + kappa3 features?

The kappa3 version of this (E12/P1) FAILED at 27-84% because kappa3 depends on
accumulated history that local pairwise state cannot see. G1 asks the same
question for kappa4_iijj with two differences that both work in our favour:

  1. The spec is 4-10x looser: E6 measured kappa4 tolerance 10-20% (vs 2-5%
     for kappa3). At kappa3=2%: kappa4 10% -> MSE 6.6e-7, 20% -> 1.04e-6.
  2. The regression gets the kappa3 slices AS FEATURES -- they carry exactly
     the accumulated history that killed P1. Physically motivated: the
     kappa4-generation diagrams at leading order are built from {Sigma,
     kappa3} blocks (power cumulants ~ kappa3^2 are the same order).

Cells (all held-out on unseen MLPs, per-layer ridge as in E12):
  A  local only            t_i, t_j, rho poly basis          (P1 analogue)
  B  + star                + the kappa3 star slice           (P1's best)
  C  + TRUE kappa3         + zk3_ij, zk3_ji, products        (upper bound)
  D  + NOISY kappa3 (25%)  simulates propagated-quality kappa3 features

PASS: held-out relative error <= 20% (mean over layers) in cell C, and D not
catastrophically worse. FAIL: kappa4 must be propagated (factored, complete
families) -- the expensive route.

Target normalisation: y_ij = kappa4(z)_iijj / (sigma_i^2 sigma_j^2).
Metric: off-diagonal relative Frobenius error, per layer + mean, HELD OUT.

Mac/MPS notes: everything n^2-sized stays on MPS in float32; the per-(m,l)
shared work (star slice incl. the CPU hermite_gains hop, transfers, local
poly basis) is computed ONCE and reused by all four cells; only the tiny
(P x P) Grams cross to CPU float64.
"""
from __future__ import annotations

import statistics as st
import sys
import time

import torch

from harness import CORPUS, hermite_gains

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256
LAY = [1, 3, 7, 15, 23, 31]
CELLS = [("A", "local only", 0.0),
         ("B", "+ star", 0.0),
         ("C", "+ TRUE kappa3 feats", 0.0),
         ("D", "+ NOISY kappa3 (25%)", 0.25)]


def basis_exponents(deg):
    return [(a, b, c) for a in range(deg + 1) for b in range(deg + 1)
            for c in range(deg + 1) if a + b + c <= deg]


def star_slice(zcov_prev, W, zmu_prev):
    """kappa3 star of z at this layer from the previous Gaussian state (E12)."""
    s = torch.sqrt(torch.clamp(torch.diagonal(zcov_prev), min=1e-30))
    g, *_ = hermite_gains(zmu_prev.cpu().double(), (s ** 2).cpu().double(), kmax=4)
    h1 = g[1].to(W.device, W.dtype)
    h2 = g[2].to(W.device, W.dtype)
    B = zcov_prev @ (h1[:, None] * W)
    C = (h2[:, None] * W) * B
    D = h2[:, None] * B * B
    return 2.0 * (C.T @ B) + (D.T @ W)


def shared_arrays(rec, l):
    """Everything the cells share for one (mlp, layer): computed ONCE."""
    zmu = rec["zmu"][l].to(DEV, F32)
    zcov = rec["zcov"][l].to(DEV, F32)
    zk3 = rec["zk3"][l].to(DEV, F32)
    zk4 = rec["zk4"][l].to(DEV, F32)
    s = torch.sqrt(torch.clamp(torch.diagonal(zcov), min=1e-30))
    t = zmu / s
    rho = zcov / torch.outer(s, s)
    ti = t[:, None].expand(n, n)
    tj = t[None, :].expand(n, n)
    y3 = zk3 / (s[:, None] ** 2 * s[None, :])
    y4 = zk4 / (s[:, None] ** 2 * s[None, :] ** 2)
    if l > 0:
        stf = star_slice(rec["zcov"][l - 1].to(DEV, F32),
                         rec["W"][l].to(DEV, F32),
                         rec["zmu"][l - 1].to(DEV, F32))
        stf = stf / (s[:, None] ** 2 * s[None, :])
    else:
        stf = torch.zeros(n, n, device=DEV, dtype=F32)
    return ti, tj, rho, y3, stf, y4


def local_design(ti, tj, rho, exps):
    """(n^2, P) local polynomial basis, with cached power tables."""
    dmax = max(max(e) for e in exps)
    pi = [torch.ones_like(ti)]
    pj = [torch.ones_like(tj)]
    pr = [torch.ones_like(rho)]
    for _ in range(dmax):
        pi.append(pi[-1] * ti)
        pj.append(pj[-1] * tj)
        pr.append(pr[-1] * rho)
    cols = [(pi[a] * pj[b] * pr[c]).reshape(-1) for a, b, c in exps]
    return torch.stack(cols, 1)


def cell_extras(cell, y3, stf, noise_rng):
    if cell == "A":
        return []
    if cell == "B":
        return [stf]
    if cell == "D" and noise_rng is not None:
        y3 = y3 * (1.0 + 0.25 * torch.randn(y3.shape, generator=noise_rng,
                                            device="cpu").to(y3.device))
    y3t = y3.T
    # kappa3-bearing features: both orientations, product & squares (power
    # cumulants ~ kappa3^2 are exactly kappa4's diagram order), plus the star.
    return [y3, y3t, y3 * y3t, y3 ** 2, y3t ** 2, stf]


def full_design(Xloc, ti, tj, rho, exps_low_cols, extras):
    """Local basis + extras + extras x low-order-local interactions."""
    if not extras:
        return Xloc
    blocks = [Xloc]
    for ex in extras:
        ev = ex.reshape(-1, 1)
        blocks.append(ev)
        blocks.append(ev * exps_low_cols)
    return torch.cat(blocks, 1)


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n_train = n_mlps // 2
    exps = basis_exponents(deg)
    low = [(a, b, c) for a, b, c in exps if a + b + c <= 2]
    low_idx = [exps.index(e) for e in low]
    P = len(exps)
    sizes = {c: P + len(cell_extras(c, torch.zeros(1, 1), torch.zeros(1, 1), None))
                * (1 + len(low)) for c, _, _ in CELLS}
    print(f"G1 | {n_mlps} MLPs ({n_train} train / {n_mlps - n_train} HELD OUT) | "
          f"poly deg {deg} ({P} local coeffs) | device {DEV}")

    recs = []
    for m in range(n_mlps):
        d = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                       weights_only=False)
        recs.append({k: d[k] for k in ("zmu", "zcov", "zk3", "zk4", "W")})
    print(f"  corpus n_samples per MLP: {d['n_samples']:,}")

    lam = 1e-6
    rng = torch.Generator().manual_seed(0)
    t0 = time.time()

    # ---- fit: one pass over (l, m); all cells share the per-(m,l) arrays ----
    G = {c: [torch.zeros(sizes[c], sizes[c], dtype=torch.float64) for _ in range(32)]
         for c, _, _ in CELLS}
    bvec = {c: [torch.zeros(sizes[c], dtype=torch.float64) for _ in range(32)]
            for c, _, _ in CELLS}
    for l in range(32):
        for m in range(n_train):
            ti, tj, rho, y3, stf, y4 = shared_arrays(recs[m], l)
            Xloc = local_design(ti, tj, rho, exps)
            low_cols = Xloc[:, low_idx]          # low terms are a subset of exps
            yv = y4.reshape(-1)
            for cell, _, _ in CELLS:
                X = full_design(Xloc, ti, tj, rho, low_cols,
                                cell_extras(cell, y3, stf, rng))
                G[cell][l] += (X.T @ X).cpu().double()
                bvec[cell][l] += (X.T @ yv).cpu().double()
    coefs = {}
    for cell, _, _ in CELLS:
        coefs[cell] = []
        for l in range(32):
            Gl = G[cell][l] + lam * torch.eye(sizes[cell], dtype=torch.float64) \
                 * torch.diagonal(G[cell][l]).mean()
            coefs[cell].append(torch.linalg.solve(Gl, bvec[cell][l]))
    print(f"  fit: {time.time() - t0:.0f}s", flush=True)

    # ---- eval on held-out, same shared-arrays pass ----
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)
    per_layer = {c: {l: [] for l in LAY} for c, _, _ in CELLS}
    allerr = {c: [] for c, _, _ in CELLS}
    for m in range(n_train, n_mlps):
        for l in range(32):
            ti, tj, rho, y3, stf, y4 = shared_arrays(recs[m], l)
            Xloc = local_design(ti, tj, rho, exps)
            low_cols = Xloc[:, low_idx]
            for cell, _, _ in CELLS:
                X = full_design(Xloc, ti, tj, rho, low_cols,
                                cell_extras(cell, y3, stf, rng))
                pred = (X @ coefs[cell][l].to(DEV, F32)).reshape(n, n)
                e = ((pred - y4)[off].pow(2).sum()
                     / y4[off].pow(2).sum()).sqrt().item()
                allerr[cell].append(e)
                if l in LAY:
                    per_layer[cell][l].append(e)

    print(f"\n{'cell':<26} " + " ".join(f"{'L' + str(l):>8}" for l in LAY)
          + f"   {'mean(32)':>9}")
    for cell, label, _ in CELLS:
        row = " ".join(f"{st.mean(per_layer[cell][l]) * 100:>7.1f}%" for l in LAY)
        print(f"{cell + ': ' + label:<26} {row}   {st.mean(allerr[cell]) * 100:>8.1f}%")

    print(f"\n  total: {time.time() - t0:.0f}s")
    print("  SPEC (E6): kappa4 <=10% costs ~1.1x | 20% ~1.5x | 50% ~4x | dropped ~10x")
    print("  PASS: cell C mean <=20% (and D not catastrophic) -> kappa4 needs")
    print("  NO propagation: n^2 learned map at inference; the kappa4(z)-block")
    print("  wall in kappa3 generation falls. FAIL -> factored kappa4 transport.")
    print("  (kappa3 reference, E12: local 39.8%, +star 35.2% -- FAILED its 2-5% spec)")


if __name__ == "__main__":
    main()
