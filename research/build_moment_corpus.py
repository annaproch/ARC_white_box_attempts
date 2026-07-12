"""Build a cached corpus of sampled per-layer moments — including the K=3/K=4
slices the closure actually needs.

KEY POINT: the cumulant corrections to the covariance update only ever use
CONTRACTED cumulants, not the full tensors. To first order,

    dCov(ReLU_i, ReLU_j) ~ 1/2 [ g2(i) g1(j) kappa3(i,i,j)
                               + g1(i) g2(j) kappa3(i,j,j) ] + (kappa4 terms)

so we need kappa3_iij and kappa4_iijj: n x n objects, sampleable with one extra
n x n matmul each per batch. No n^3 / n^4 tensor ever has to exist.

What this enables:
  1. Measure the exact third/fourth-order correction needed at every layer.
  2. Test whether those slices are PREDICTABLE from the K=2 state
     (alpha_i, alpha_j, rho_ij, layer). If yes -> we get the K=3/K=4 effect at
     n^2 cost with a learned map ("absorb"), instead of n^3*rank propagation
     ("compute"). If no -> we must build the real rank-r kappa3 propagator.
  3. Supervised LAYERWISE training of the closure (match true Sigma at every
     layer) instead of only end-to-end final-layer MSE: 32 x 65k targets per
     MLP instead of 256 numbers.
  4. All future closure experiments stop re-sampling ground truth.

Stored per MLP (float32): weights, and per layer mu (n), Sigma (n,n),
K3 = kappa3_iij (n,n), K4 = kappa4_iijj (n,n).
Size: ~25 MB per MLP -> 32 MLPs ~ 800 MB in data/ (gitignored).

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/build_moment_corpus.py
      (add --n-mlps 8 for a quick smoke test)
"""

import argparse
import math
import time
from pathlib import Path

import torch

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
OUT = Path(__file__).resolve().parent.parent / "data" / "moment_corpus"

ap = argparse.ArgumentParser()
ap.add_argument("--n-mlps", type=int, default=32)
ap.add_argument("--samples", type=int, default=2_000_000)
ap.add_argument("--batch", type=int, default=65_536)
ap.add_argument("--seed0", type=int, default=200_000)
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)


def build_ws(seed):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(N_W, N_W, generator=g) * math.sqrt(2.0 / N_W)).to(DEV)
            for _ in range(DEPTH)]


def moments(ws, n_samples, batch):
    """Accumulate raw moments per layer for BOTH the post-activation a = ReLU(z)
    and the PRE-activation z = a_prev @ W.

    The pre-activation cumulants are the ones the diagram formula actually needs:
    cumulants of ReLU(z) are expressed in terms of cumulants of z. (And note
    kappa3(z) = W^T-transform of kappa3(a), which requires the FULL n^3 tensor --
    the n^2 slices of a cannot produce the slices of z. So slices suffice to
    measure/correct, but not to propagate. Here we sample kappa3(z) directly.)
    """
    z_ = lambda: [torch.zeros(N_W, N_W, dtype=torch.float64) for _ in range(DEPTH)]
    v_ = lambda: [torch.zeros(N_W, dtype=torch.float64) for _ in range(DEPTH)]
    S1, S2, S21, S22 = v_(), z_(), z_(), z_()              # post-activation a
    Z1, Z2, Z21, Z22 = v_(), z_(), z_(), z_()              # pre-activation z
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        x = torch.randn(b, N_W, device=DEV)
        for li, w in enumerate(ws):
            z = x @ w
            z2 = z * z
            Z1[li] += z.sum(0).cpu().double()
            Z2[li] += (z.T @ z).cpu().double()
            Z21[li] += (z2.T @ z).cpu().double()     # E[z_i^2 z_j]
            Z22[li] += (z2.T @ z2).cpu().double()    # E[z_i^2 z_j^2]
            x = torch.relu(z)
            x2 = x * x
            S1[li] += x.sum(0).cpu().double()
            S2[li] += (x.T @ x).cpu().double()
            S21[li] += (x2.T @ x).cpu().double()     # E[a_i^2 a_j] (rows i, cols j)
            S22[li] += (x2.T @ x2).cpu().double()    # E[a_i^2 a_j^2]
        done += b
    return (S1, S2, S21, S22), (Z1, Z2, Z21, Z22)


def central(S1, S2, S21, S22, n):
    """Central moments -> Sigma, kappa3_iij, kappa4_iijj."""
    mu = S1 / n
    m2 = S2 / n
    m21 = S21 / n
    m22 = S22 / n
    cov = m2 - torch.outer(mu, mu)
    d = torch.diagonal(m2)                       # E[a_i^2]
    # E[(a_i-mu_i)^2 (a_j-mu_j)] = m21_ij - mu_j*E[a_i^2] - 2 mu_i m2_ij + 2 mu_i^2 mu_j
    k3 = (m21
          - mu[None, :] * d[:, None]
          - 2 * mu[:, None] * m2
          + 2 * (mu[:, None] ** 2) * mu[None, :])
    # E[(a_i-mu_i)^2 (a_j-mu_j)^2]
    e22 = (m22
           - 2 * mu[None, :] * m21
           - 2 * mu[:, None] * m21.T
           + (mu[None, :] ** 2) * d[:, None]
           + (mu[:, None] ** 2) * d[None, :]
           + 4 * torch.outer(mu, mu) * m2
           - 3 * torch.outer(mu**2, mu**2))
    v = torch.diagonal(cov)
    k4 = e22 - torch.outer(v, v) - 2 * cov**2    # fourth cumulant kappa4_iijj
    return mu, cov, k3, k4


t_all = time.time()
for m in range(args.n_mlps):
    seed = args.seed0 + m
    path = OUT / f"mlp_{seed}.pt"
    if path.exists():
        # schema check, not just existence: older records lack the pre-activation
        # cumulants (zk3/zk4), which are the ones the diagram formula needs.
        try:
            cached_ok = "zk3" in torch.load(path, map_location="cpu")
        except Exception:
            cached_ok = False
        if cached_ok:
            print(f"[{m+1}/{args.n_mlps}] seed {seed}: cached, skipping")
            continue
        print(f"[{m+1}/{args.n_mlps}] seed {seed}: stale schema, rebuilding")
    t0 = time.time()
    ws = build_ws(seed)
    (S1, S2, S21, S22), (Z1, Z2, Z21, Z22) = moments(ws, args.samples, args.batch)
    rec = {"seed": seed, "n_samples": args.samples,
           "W": torch.stack([w.cpu() for w in ws]).to(torch.float32)}
    mus, covs, k3s, k4s = [], [], [], []
    zmus, zcovs, zk3s, zk4s = [], [], [], []
    for li in range(DEPTH):
        mu, cov, k3, k4 = central(S1[li], S2[li], S21[li], S22[li], args.samples)
        mus.append(mu.float()); covs.append(cov.float())
        k3s.append(k3.float()); k4s.append(k4.float())
        zmu, zcov, zk3, zk4 = central(Z1[li], Z2[li], Z21[li], Z22[li], args.samples)
        zmus.append(zmu.float()); zcovs.append(zcov.float())
        zk3s.append(zk3.float()); zk4s.append(zk4.float())
    rec["mu"] = torch.stack(mus)        # (L, n)    post-activation mean
    rec["cov"] = torch.stack(covs)      # (L, n, n) post-activation covariance
    rec["k3"] = torch.stack(k3s)        # (L, n, n) kappa3(a_i, a_i, a_j)
    rec["k4"] = torch.stack(k4s)        # (L, n, n) kappa4(a_i, a_i, a_j, a_j)
    rec["zmu"] = torch.stack(zmus)      # (L, n)    PRE-activation mean
    rec["zcov"] = torch.stack(zcovs)    # (L, n, n) PRE-activation covariance
    rec["zk3"] = torch.stack(zk3s)      # (L, n, n) kappa3(z_i, z_i, z_j)  <- what the
    rec["zk4"] = torch.stack(zk4s)      # (L, n, n) kappa4(z_i,z_i,z_j,z_j)   formula needs
    torch.save(rec, path)
    mb = path.stat().st_size / 1e6
    print(f"[{m+1}/{args.n_mlps}] seed {seed}: {time.time()-t0:.0f}s, {mb:.0f} MB")

print(f"\ncorpus in {OUT}  ({time.time()-t_all:.0f}s total)")

# quick read-out on the first record: how big are the cumulant corrections?
first = sorted(OUT.glob("mlp_*.pt"))[0]
r = torch.load(first)
print(f"\nscale check ({first.name}):")
print(f"{'L':>3s} {'rms Sigma':>10s} {'rms k3_iij':>11s} {'rms k4_iijj':>12s} "
      f"{'k3/Sigma':>9s} {'k4/Sigma^2':>11s}")
for li in (0, 1, 3, 7, 15, 23, 31):
    cov, k3, k4 = r["cov"][li], r["k3"][li], r["k4"][li]
    off = ~torch.eye(N_W, dtype=torch.bool)
    rc = float((cov[off] ** 2).mean().sqrt())
    r3 = float((k3[off] ** 2).mean().sqrt())
    r4 = float((k4[off] ** 2).mean().sqrt())
    print(f"{li:3d} {rc:10.3e} {r3:11.3e} {r4:12.3e} {r3/rc:9.3f} {r4/rc**2:11.3f}")
print("\nk3/Sigma large => third cumulants materially bias the covariance update.")
