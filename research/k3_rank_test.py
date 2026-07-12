"""Is the accumulated kappa3(z) LOW-RANK? (decides whether K=3 is affordable)

analytic_k3_test.py showed: the freshly-generated (star) part of kappa3 is exact
at layer 1 (corr 0.996) but decays 26x by L31 while the true kappa3 stays
constant (rms ~3e-2) -- so at depth the ACCUMULATED (passed-through) kappa3
dominates. Carrying it exactly is what costs ARC L^2 (the CP rank grows every
layer).

Hope: by depth the covariance is dominated by ONE common mode (eig1frac 0.5-0.8)
and that mode is skewed (|skew| ~ 0.4). If the accumulated kappa3 has simply
concentrated onto that mode, then

    kappa3 ~ kappa * u (x) u (x) u        (symmetric rank-1)
    => slice  kappa3_iij ~ kappa * u_i^2 * u_j     <- a RANK-1 MATRIX
                                                      (left factor u^2, right u)

and propagation is trivial: factors scale by h1 elementwise and hit W^T --
O(n^2 r) per layer. Then K=3 costs about what K=2 costs.

Tests, on the sampled kappa3(z) slice matrix M_ij = kappa3(z_i, z_i, z_j):
  1. SVD energy: what fraction of M's Frobenius energy is in the top 1/2/4/8
     singular values? (Low-rank matrix => low-rank tensor structure.)
  2. Alignment: is M's top RIGHT singular vector aligned with the top eigenvector
     u of Sigma_z, and is its top LEFT singular vector aligned with u^2?
     (That is the signature of kappa * u (x) u (x) u.)
  3. Rank-r reconstruction error using u_m from Sigma_z's top eigenvectors and
     least-squares coefficients -- the practical estimator we would build.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/k3_rank_test.py
"""

import math
from pathlib import Path

import torch

torch.set_grad_enabled(False)
N_W, DEPTH = 256, 32
CORPUS = Path(__file__).resolve().parent.parent / "data" / "moment_corpus"
LAYERS = (1, 3, 7, 15, 23, 31)
RANKS = (1, 2, 4, 8, 16)

files = sorted(CORPUS.glob("mlp_*.pt"))
if not files:
    raise SystemExit("run build_moment_corpus.py first")

print("SVD energy of the true kappa3(z) slice matrix (fraction in top-k)")
print(f"{'L':>3s} {'top1':>7s} {'top2':>7s} {'top4':>7s} {'top8':>7s} "
      f"{'align(right,u)':>15s} {'align(left,u^2)':>16s}")
for li in LAYERS:
    e1 = e2 = e4 = e8 = ar = al = 0.0
    for f in files:
        r = torch.load(f, map_location="cpu")
        M = r["zk3"][li].double()
        S = r["zcov"][li].double()
        sv = torch.linalg.svdvals(M)
        tot = float((sv**2).sum())
        e1 += float((sv[:1] ** 2).sum()) / tot / len(files)
        e2 += float((sv[:2] ** 2).sum()) / tot / len(files)
        e4 += float((sv[:4] ** 2).sum()) / tot / len(files)
        e8 += float((sv[:8] ** 2).sum()) / tot / len(files)
        U, _, Vh = torch.linalg.svd(M)
        ev, EV = torch.linalg.eigh(S)
        u = EV[:, -1]
        u2 = u * u
        u2 = u2 / u2.norm()
        ar += abs(float(Vh[0] @ u)) / len(files)
        al += abs(float(U[:, 0] @ u2)) / len(files)
    print(f"{li:3d} {e1:7.3f} {e2:7.3f} {e4:7.3f} {e8:7.3f} {ar:15.3f} {al:16.3f}")

print("\nrank-r reconstruction of kappa3 slice from Sigma's top eigenvectors")
print("(the estimator we would actually build: kappa3 ~ sum_m c_m u_m^2 (x) u_m)")
print(f"{'L':>3s} " + " ".join(f"{'r=' + str(r):>9s}" for r in RANKS) + f" {'rel err r=8':>11s}")
for li in LAYERS:
    acc = dict.fromkeys(RANKS, 0.0)
    rel8 = 0.0
    for f in files:
        r = torch.load(f, map_location="cpu")
        M = r["zk3"][li].double()
        S = r["zcov"][li].double()
        ev, EV = torch.linalg.eigh(S)
        for rk in RANKS:
            U = EV[:, -rk:]                       # (n, rk) top eigenvectors
            # basis of candidate slices: (u_m^2) outer (u_m); least-squares in c
            basis = torch.stack([torch.outer(U[:, m] ** 2, U[:, m]).flatten()
                                 for m in range(rk)], dim=1)   # (n^2, rk)
            c, *_ = torch.linalg.lstsq(basis, M.flatten().unsqueeze(1))
            pred = (basis @ c).reshape(N_W, N_W)
            corr = float(torch.corrcoef(torch.stack([pred.flatten(),
                                                     M.flatten()]))[0, 1])
            acc[rk] += corr / len(files)
            if rk == 8:
                rel8 += float(((pred - M) ** 2).mean().sqrt()
                              / (M**2).mean().sqrt()) / len(files)
    print(f"{li:3d} " + " ".join(f"{acc[rk]:9.3f}" for rk in RANKS) + f" {rel8:11.3f}")

print("\ncorr >~0.9 at small r => kappa3 is low-rank in Sigma's eigenbasis:")
print("  -> propagate a rank-r kappa3 (factors: h1-scale + W^T), O(n^2 r)/layer,")
print("     K=3 accuracy at ~K=2 cost. Otherwise we must carry a growing CP rank.")
