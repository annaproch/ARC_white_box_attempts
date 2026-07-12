"""Can we COMPUTE kappa3(z) analytically at O(n^3), instead of propagating tensors?

step_unit_test.py proved the corrections are right: with true kappa3(z)/kappa4(z)
the single-step errors collapse (mean 1300x, covariance 193x) and the projected
final-layer error is ~5e-8 -- better than the competitor. So everything now
hinges on OBTAINING kappa3(z) in-budget.

Key structural fact: kappa3 of the ACTIVATIONS is not an arbitrary tensor. At
leading order (z Gaussian) the diagram formula gives the "star" terms

    kappa3(a)_ijk = h2_i h1_j h1_k S_ij S_ik + h1_i h2_j h1_k S_ij S_jk
                                              + h1_i h1_j h2_k S_ik S_jk
    (S = Sigma_z, h1 = Phi(alpha), h2 = phi(alpha)/sigma)

plus a "triangle" term  h2_i h2_j h2_k S_ij S_jk S_ki,
plus a "pass-through" of the incoming kappa3(z)  (h1_i h1_j h1_k kappa3(z)_ijk).

After the linear map z' = W^T a, the SLICE we need collapses to matmuls:

    B = S diag(h1) W
    C_ip = h2_i W_ip B_ip ,  D_ip = h2_i B_ip^2
    kappa3(z')_ppq = 2 (C^T B)_pq + (D^T W)_pq          <-- O(n^3), 3 matmuls

This script checks that formula against the corpus's SAMPLED kappa3(z), and
prices what the star terms alone capture:

  * star-only accurate  -> we have an O(n^3) analytic kappa3 -> build the closure.
  * poor                -> the triangle and/or pass-through matter; the
                           pass-through is the term that forces a factored
                           tensor (it is why ARC's implementation is expensive).

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/analytic_k3_test.py
"""

import math
from pathlib import Path

import torch

torch.set_grad_enabled(False)
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
CORPUS = Path(__file__).resolve().parent.parent / "data" / "moment_corpus"
LAYERS = (0, 1, 3, 7, 15, 23, 30)   # predict zk3 of layer l+1 from layer l


def star_k3_slice(m, S, W):
    """kappa3(z')_ppq from the star diagrams. m, S = pre-activation mean/cov of
    layer l; W = weights of layer l+1. Returns (n, n): rows p, cols q."""
    var = torch.clamp(torch.diagonal(S), min=1e-12)
    s = var.sqrt()
    a = m / s
    pdf = torch.exp(-0.5 * a * a) / SQ2PI
    cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
    h1, h2 = cdf, pdf / s
    B = S @ (h1[:, None] * W)                 # B_iq = sum_j S_ij h1_j W_jq
    C = h2[:, None] * W * B                   # C_ip = h2_i W_ip B_ip
    D = h2[:, None] * B * B                   # D_ip = h2_i B_ip^2
    return 2.0 * (C.T @ B) + (D.T @ W)


def triangle_k3_slice(m, S, W, rank=None):
    """kappa3(z')_ppq from the triangle diagram.
    Exact form is O(n^4); with rank truncation of S it is O(n^2 r^2).
    Here: exact via einsum (validation only)."""
    var = torch.clamp(torch.diagonal(S), min=1e-12)
    s = var.sqrt()
    a = m / s
    pdf = torch.exp(-0.5 * a * a) / SQ2PI
    P = (pdf / s)[:, None] * W                # P_ip = h2_i W_ip
    # sum_ijk P_ip P_jp P_kq S_ij S_jk S_ki
    return torch.einsum("ip,jp,kq,ij,jk,ki->pq", P, P, P, S, S, S)


files = sorted(CORPUS.glob("mlp_*.pt"))
if not files:
    raise SystemExit("run build_moment_corpus.py first")

print(f"predicting kappa3(z) at layer l+1 from (mu_z, Sigma_z) at layer l, "
      f"{len(files)} MLPs\n")
print(f"{'L+1':>4s} {'rms true':>10s} {'rms star':>10s} {'err star':>9s} "
      f"{'corr star':>10s} {'err +tri':>9s} {'corr +tri':>10s}")

for li in LAYERS:
    rms_t = rms_s = 0.0
    err_s = corr_s = err_t = corr_t = 0.0
    for f in files:
        r = torch.load(f, map_location="cpu")
        m = r["zmu"][li].double()
        S = r["zcov"][li].double()
        W = r["W"][li + 1].double()
        true = r["zk3"][li + 1].double()

        star = star_k3_slice(m, S, W)
        tri = triangle_k3_slice(m, S, W)
        both = star + tri

        def stats(est):
            e = float(((est - true) ** 2).mean().sqrt())
            c = float(torch.corrcoef(torch.stack([est.flatten(), true.flatten()]))[0, 1])
            return e, c

        es, cs = stats(star)
        et, ct = stats(both)
        rms_t += float((true**2).mean().sqrt()) / len(files)
        rms_s += float((star**2).mean().sqrt()) / len(files)
        err_s += es / len(files)
        corr_s += cs / len(files)
        err_t += et / len(files)
        corr_t += ct / len(files)
    print(f"{li+1:4d} {rms_t:10.3e} {rms_s:10.3e} {err_s:9.2e} {corr_s:10.3f} "
          f"{err_t:9.2e} {corr_t:10.3f}")

print("\ncorr ~ 1 and err << rms(true) => the analytic kappa3 is good enough:")
print("  -> build the closure with computed kappa3 (O(n^3)/layer, no tensors).")
print("low corr => the pass-through term (needs the full/factored kappa3 tensor)")
print("  dominates -- that is precisely what makes ARC's implementation expensive.")
