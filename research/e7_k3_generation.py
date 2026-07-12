"""E7 — how accurate is the kappa3 GENERATION formula?  (the build gate)

E6 spec: kappa3 slices need <=2-5% relative error (5% -> 8.8e-7, 10% -> 2.0e-6).
The logged figure for our propagated kappa3 is 6.7% -- right on the cliff. Is
that the GENERATION formula or the rank-r carry?

Layer 1 isolates GENERATION exactly: z^(0) = W0^T x is EXACTLY Gaussian, so
kappa3(z^(0)) = 0 and there is NO pass-through. Any error here is the formula.

DERIVATION (Thm A.2, r=3, Gaussian z). Connected diagrams are indexed by edge
multiplicities (m_pq, m_pr, m_qr), >=2 of them nonzero:

  k3(a)_pqr = sum  [S_pq^a/a!][S_pr^b/b!][S_qr^c/c!]
                   h[a+b](p) h[a+c](q) h[b+c](r)

Take the c=0 (no q-r edge) family -- it contracts to pure matmuls. With
  M_a[p,i] = (S^a / a!) @ (diag(h_a) W)
and contracting k3(z')_iij = sum_pqr k3(a)_pqr W_pi W_qi W_rj:

  hub=p (q->i, r->j):  sum_p h[a+b](p) W_pi M_a[p,i] M_b[p,j]
  hub=q (p->i, r->j):  identical after contraction (S symmetric)  => factor 2
  hub=r (p->i, q->i):  sum_p h[a+b](p) M_a[p,i] M_b[p,i] W_pj

  k3(z')_iij = sum_{a,b>=1} { 2 sum_p h[a+b](p) W_pi M_a[p,i] M_b[p,j]
                              +  sum_p h[a+b](p) M_a[p,i] M_b[p,i] W_pj }

a=b=1 collapses to 2(C^T B) + (D^T W) with B = S diag(h1) W  -- EXACTLY the
shipped star. Everything with a+b > 2 is what we throw away.
(The triangle family, c>=1, needs O(n^4) to contract and is NOT included here;
if the a,b series alone closes the gap, we never need it.)
"""
from __future__ import annotations

import math
import statistics as st
import sys
import time

import torch

from harness import CORPUS, DT, hermite_gains

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_SAMP = 4_000_000
CHUNK = 250_000
n = 256


def sampled_k3_z1(Wb, n_samp, seed):
    """True kappa3_iij and kappa3_iii of z^(1), by sampling. Batched over MLPs."""
    M = Wb.shape[0]

    def fwd(X):
        A = torch.clamp(torch.bmm(X, Wb[:, 0]), min=0.0)
        return torch.bmm(A, Wb[:, 1])

    s1, done = torch.zeros(M, n, dtype=torch.float64), 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    while done < n_samp:
        b = min(CHUNK, n_samp - done)
        X = torch.randn(b, n, generator=g, dtype=torch.float32).to(DEV)
        s1 += fwd(X.expand(M, b, n).contiguous()).sum(1).cpu().double()
        done += b
    mu = (s1 / n_samp).to(DEV, torch.float32)

    m21 = torch.zeros(M, n, n, dtype=torch.float64)
    m3d = torch.zeros(M, n, dtype=torch.float64)
    g = torch.Generator(device="cpu").manual_seed(seed)
    done = 0
    while done < n_samp:
        b = min(CHUNK, n_samp - done)
        X = torch.randn(b, n, generator=g, dtype=torch.float32).to(DEV)
        d = fwd(X.expand(M, b, n).contiguous()) - mu.unsqueeze(1)
        d2 = d * d
        m21 += torch.bmm(d2.transpose(1, 2), d).cpu().double()
        m3d += (d2 * d).sum(1).cpu().double()
        done += b
    return m21 / n_samp, m3d / n_samp


def gen_k3_iij(S, W, amax, kmax=14):
    """kappa3(z')_iij from the Gaussian state of z. amax=1 -> star only."""
    zvar = torch.diagonal(S)
    g, *_ = hermite_gains(torch.zeros(n, dtype=DT), zvar, kmax=kmax)
    M = [None] + [((S ** a) / math.factorial(a)) @ (g[a][:, None] * W)
                  for a in range(1, amax + 1)]
    out = torch.zeros(n, n, dtype=DT)
    for a in range(1, amax + 1):
        for b in range(1, amax + 1):
            h = g[a + b]
            C = h[:, None] * W * M[a]                 # C[p,i]
            D = h[:, None] * M[a] * M[b]              # D[p,i]
            out += 2.0 * (C.T @ M[b]) + (D.T @ W)
    return out


def triangle_k3_iij(S, W, a, b, c, kmax=14):
    """Triangle family (all three Sigma edges present), multiplicities (a,b,c):

        sum_pqr (S^a)_pq (S^b)_pr (S^c)_qr h[a+b](p)W_pi h[a+c](q)W_qi
                h[b+c](r)W_rj  / (a! b! c!)

    Contract as: G_i[p,q] = (S^a)_pq * x_i[p] * y_i[q]; then
        inner[i,r] = ((S^b)^T G_i (S^c))_rr,  and  T = inner @ diag(h[b+c]) W.
    O(n^4) -- fine offline; the point here is to test the FORMULA, not its cost.
    """
    zvar = torch.diagonal(S)
    g, *_ = hermite_gains(torch.zeros(n, dtype=DT), zvar, kmax=kmax)
    Sa, Sb, Sc = S ** a, S ** b, S ** c
    X = (g[a + b][:, None] * W)          # x_i[p] = h[a+b](p) W_pi  -> X[p,i]
    Y = (g[a + c][:, None] * W)          # Y[q,i]
    dev = torch.device("mps") if DEV.type == "mps" else DEV
    Sa32, Sb32, Sc32 = (t.to(dev, torch.float32) for t in (Sa, Sb, Sc))
    X32, Y32 = X.to(dev, torch.float32), Y.to(dev, torch.float32)
    inner = torch.empty(n, n, device=dev, dtype=torch.float32)
    CH = 32
    for i0 in range(0, n, CH):
        xi = X32[:, i0:i0 + CH].T                       # (ch, p)
        yi = Y32[:, i0:i0 + CH].T                       # (ch, q)
        G = Sa32.unsqueeze(0) * xi.unsqueeze(2) * yi.unsqueeze(1)   # (ch,p,q)
        # inner[i,r] = sum_pq G[i,p,q] Sb[p,r] Sc[q,r]
        T1 = torch.einsum("ipq,pr->iqr", G, Sb32)
        inner[i0:i0 + CH] = torch.einsum("iqr,qr->ir", T1, Sc32)
    inner = inner.cpu().to(DT)
    Z = g[b + c][:, None] * W                                        # Z[r,j]
    return (inner @ Z) / (math.factorial(a) * math.factorial(b)
                          * math.factorial(c))


def rel(pred, truth, mask):
    d = (pred - truth)[mask]
    return (d.pow(2).sum() / truth[mask].pow(2).sum()).sqrt().item()


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    recs = [torch.load(CORPUS / f"mlp_{200000+m}.pt", map_location="cpu",
                       weights_only=False) for m in range(n_mlps)]
    Wb = torch.stack([r["W"] for r in recs]).to(DEV, torch.float32)

    t0 = time.time()
    k3_iij_t, k3_iii_t = sampled_k3_z1(Wb, N_SAMP, 12345)
    print(f"  sampled truth ({N_SAMP:,} x {n_mlps} MLPs): {time.time()-t0:.0f}s",
          flush=True)

    W0 = torch.stack([r["W"][0] for r in recs]).to(DT)
    W1 = torch.stack([r["W"][1] for r in recs]).to(DT)
    S0 = torch.bmm(W0.transpose(1, 2), W0)         # Cov(z^0), EXACT

    off = ~torch.eye(n, dtype=torch.bool)
    dia = torch.eye(n, dtype=torch.bool)
    print("\nE7 | kappa3 GENERATION at layer 1 "
          "(z^0 exactly Gaussian => no pass-through)")
    print(f"{'formula':>26} {'rel err k3_iij':>16} {'rel err k3_iii':>16}")
    all1 = torch.ones(n, dtype=torch.bool)

    def report(tag, preds):
        e1 = st.mean(rel(preds[m], k3_iij_t[m], off) for m in range(n_mlps))
        e2 = st.mean(rel(torch.diagonal(preds[m]), k3_iii_t[m], all1)
                     for m in range(n_mlps))
        print(f"{tag:>34} {e1*100:>15.2f}% {e2*100:>15.2f}%")

    # order-2 only: the star (a=b=1). This is what ships.
    star = [gen_k3_iij(S0[m], W1[m], 1) for m in range(n_mlps)]
    report("2-edge: STAR only [SHIPPED]", star)

    # unbalanced: (a,b) series without the triangle -- expect WORSE
    ab2 = [gen_k3_iij(S0[m], W1[m], 2) for m in range(n_mlps)]
    report("3-edge, (a,b) only (UNBALANCED)", ab2)

    # order-consistent 3-edge: star + (2,1) + (1,2) + triangle(1,1,1)
    t0 = time.time()
    tri = [triangle_k3_iij(S0[m], W1[m], 1, 1, 1) for m in range(n_mlps)]
    print(f"  [triangle O(n^4): {time.time()-t0:.0f}s]", flush=True)
    report("3-edge COMPLETE (+ triangle)", [ab2[m] + tri[m] for m in range(n_mlps)])
    report("  (star + triangle, no (a,b))", [star[m] + tri[m] for m in range(n_mlps)])

    print("\nSPEC (E6): kappa3 needs <=2-5% rel err.  5% -> 8.8e-7, 10% -> 2.0e-6")


if __name__ == "__main__":
    main()
