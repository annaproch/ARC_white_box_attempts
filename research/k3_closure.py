"""THE BUILD: K=2 closure + kappa3 propagated as [exact star] + [rank-r Tucker core].

Forced by measurements, not taste:
  * corrections validated (step_unit_test): mean up to 1300x, cov up to 193x
  * they must be applied to mean AND covariance TOGETHER (oracle: partial fixes
    break K=2's error cancellation and make things worse)
  * kappa3's freshly generated part is EXACT from the star diagrams, O(n^3)
    (analytic_k3_test: corr 0.996 at L1) but decays 26x by L31
  * the accumulated part dominates at depth and is nearly RANK-1 there
    (k3_rank_test: top-1 SVD energy 0.98, alignment with Sigma's top eigvec 0.998)

Representation:  kappa3(z) = Tucker core T (r,r,r) in V = top-r eigenbasis of S.
Per layer:
  slices     k3_iij, k3_iii  from (T, V)
  moments    mu_a  = gauss + (1/6) h3 * k3_iii
             Sig_a = wick  + 1/2 [h2_i h1_j k3_iij + h1_i h2_j k3_ijj]  (off-diag)
  propagate  z' = W'^T ReLU(z):
      B    = S diag(h1) W'                      (n,n)   -- star factors
      A    = W' V'  ,  Bv = B V'                (n,r)
      T_star[mkl] = sum_i h2_i [ A_im Bv_ik Bv_il + Bv_im A_ik Bv_il
                                 + Bv_im Bv_ik A_il ]           (n r^3)
      M    = V'^T W'^T diag(h1) V               (r,r)   -- pass-through
      T'   = T x M x M x M + T_star                      (r^4)

Cost/layer: 2n^3 (Sigma) + O(n^2 r + n r^3 + r^4) => ~0.5% of budget overall.

Validation: per-layer correlation of the PROPAGATED kappa3 against the corpus's
SAMPLED kappa3(z), then the final-layer MSE against ground truth.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/k3_closure.py
      flags: --rank 16 | --rank 32 | --no-star (ablation)
"""

import argparse
import math
from pathlib import Path

import torch

torch.set_grad_enabled(False)
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
N_TERMS = 4
CORPUS = Path(__file__).resolve().parent.parent / "data" / "moment_corpus"

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--no-star", action="store_true")
ap.add_argument("--mlps", type=int, default=8)
args = ap.parse_args()
R = args.rank


def wick_series(a, pdf, cdf, s, rho):
    g = [cdf, pdf]
    he_prev, he_cur = torch.ones_like(a), a
    for k in range(3, N_TERMS + 1):
        m = k - 2
        g.append(((-1) ** m) * he_cur * pdf)
        he_prev, he_cur = he_cur, a * he_cur - m * he_prev
    acc = torch.zeros_like(rho)
    rho_k, sf = rho.clone(), 1.0
    for k in range(1, N_TERMS + 1):
        sf *= math.sqrt(k)
        gh = g[k - 1] / sf
        acc += torch.outer(gh, gh) * rho_k
        if k < N_TERMS:
            rho_k = rho_k * rho
    return torch.outer(s, s) * acc


def core_slices(T, V):
    """kappa3_iij (n,n) and kappa3_iii (n,) from Tucker core T (r,r,r), basis V (n,r)."""
    G = (V.unsqueeze(2) * V.unsqueeze(1)).reshape(N_W, -1)     # (n, r*r): V_im V_ik
    P = G @ T.reshape(-1, T.shape[-1])                         # (n, r)
    slice_ij = P @ V.T                                         # (n, n)
    diag = ((G @ T.reshape(-1, T.shape[-1])) * V).sum(1)       # sum_{mkl} T V V V
    return slice_ij, diag


def propagate(Ws, use_k3=True, use_star=True, truth=None):
    """Ws: (L, n, n) float64. Returns final-layer mean and a per-layer kappa3 log."""
    # layer 0 pre-activation: x ~ N(0, I) exactly => m = 0, S = W0^T W0, kappa3 = 0
    m = torch.zeros(N_W, dtype=torch.float64)
    S = Ws[0].T @ Ws[0]
    T = torch.zeros(R, R, R, dtype=torch.float64)
    V = torch.linalg.eigh(S)[1][:, -R:]
    log = []

    for li in range(DEPTH):
        var = torch.clamp(torch.diagonal(S), min=1e-12)
        s = var.sqrt()
        a = m / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        h1, h2 = cdf, pdf / s
        h3 = -a * pdf / var
        rho = torch.clamp(S / torch.outer(s, s), -1.0, 1.0)

        if use_k3:
            k3_slice, k3_diag = core_slices(T, V)
        else:
            k3_slice = torch.zeros_like(S)
            k3_diag = torch.zeros(N_W, dtype=torch.float64)

        if truth is not None:
            tk = truth[li].double()
            c = (float(torch.corrcoef(torch.stack([k3_slice.flatten(), tk.flatten()]))[0, 1])
                 if use_k3 and float(k3_slice.abs().max()) > 0 else 0.0)
            log.append((li, c, float((k3_slice**2).mean().sqrt()),
                        float((tk**2).mean().sqrt())))

        # ---- corrected post-activation moments (mean AND covariance together)
        mu_a = s * pdf + m * cdf + (1.0 / 6.0) * h3 * k3_diag
        ez2 = (m * m + var) * cdf + m * s * pdf + (1.0 / 3.0) * h2 * k3_diag
        var_a = torch.clamp(ez2 - mu_a * mu_a, min=1e-14)
        Sig_a = wick_series(a, pdf, cdf, s, rho)
        Sig_a = Sig_a + 0.5 * (torch.outer(h2, h1) * k3_slice
                               + torch.outer(h1, h2) * k3_slice.T)
        Sig_a = Sig_a - torch.diag(torch.diagonal(Sig_a)) + torch.diag(var_a)

        if li == DEPTH - 1:
            return mu_a, log

        # ---- push through the next linear layer
        W = Ws[li + 1]
        m_next = mu_a @ W
        S_next = W.T @ (Sig_a @ W)
        V_next = torch.linalg.eigh(S_next)[1][:, -R:]

        if use_k3:
            hW = h1[:, None] * W                       # diag(h1) W
            # pass-through: T x M x M x M
            M = V_next.T @ (W.T @ (h1[:, None] * V))   # (r, r)
            T_next = torch.einsum("mkl,am,bk,cl->abc", T, M, M, M)
            if use_star:
                B = S @ hW                             # (n, n): B_iq
                A = W @ V_next                         # (n, r): (V'^T w_i)_m
                Bv = B @ V_next                        # (n, r): (V'^T B_i)_m
                hA = h2[:, None] * A
                t1 = torch.einsum("im,ik,il->mkl", hA, Bv, Bv)
                t2 = torch.einsum("im,ik,il->mkl", h2[:, None] * Bv, A, Bv)
                t3 = torch.einsum("im,ik,il->mkl", h2[:, None] * Bv, Bv, A)
                T_next = T_next + t1 + t2 + t3
            T = T_next

        m, S, V = m_next, S_next, V_next

    raise RuntimeError


files = sorted(CORPUS.glob("mlp_*.pt"))[: args.mlps]
if not files:
    raise SystemExit("run build_moment_corpus.py first")

print(f"rank r={R}, star={'off' if args.no_star else 'on'}, {len(files)} MLPs\n")

# per-layer kappa3 fidelity on the first MLP
r0 = torch.load(files[0], map_location="cpu")
_, log = propagate(r0["W"].double(), use_k3=True, use_star=not args.no_star,
                   truth=r0["zk3"])
print("propagated kappa3 vs sampled truth (first MLP):")
print(f"{'L':>3s} {'corr':>7s} {'rms ours':>10s} {'rms true':>10s}")
for li, c, ro, rt in log:
    if li in (0, 1, 3, 7, 15, 23, 31):
        print(f"{li:3d} {c:7.3f} {ro:10.3e} {rt:10.3e}")

print(f"\n{'seed':>7s} {'K2':>11s} {'K2+kappa3':>11s} {'gain':>7s}")
a2 = a3 = 0.0
for f in files:
    r = torch.load(f, map_location="cpu")
    Ws = r["W"].double()
    gt = r["mu"][-1].double()
    e2 = float(((propagate(Ws, use_k3=False)[0] - gt) ** 2).mean())
    e3 = float(((propagate(Ws, use_k3=True, use_star=not args.no_star)[0] - gt) ** 2).mean())
    a2 += e2 / len(files)
    a3 += e3 / len(files)
    print(f"{r['seed']:7d} {e2:11.3e} {e3:11.3e} {e2/max(e3,1e-30):6.2f}x")
print(f"{'MEAN':>7s} {a2:11.3e} {a3:11.3e} {a2/max(a3,1e-30):6.2f}x")
print("\nReference: QMC floor ~4.1e-6 | v4 (submitted) 3.4e-7 score | competitor 3.2e-7 MSE")
print("Next if kappa3 lands: add kappa4 (same Tucker machinery), then calibrate,")
print("then port to flopscope and go pure-analytic at the floor.")
