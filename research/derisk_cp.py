"""DERISK THE CP CARRY — verify the algebra, THEN measure compression.

Two bugs were found in estimator_v5's carry by doing this derivation properly:
  BUG 1  the ansatz was symmetric-CP (u(x)u(x)u) -- but the star is
         sum_i h2(i) sym(a_i (x) b_i (x) b_i) with TWO distinct vectors, so the
         ansatz could not represent it AT ANY RANK. It carried noise.
  BUG 2  the star was attached to the WRONG LAYER. The star built from
         Sigma(z^l), h1/h2(z^l) and W[l+1] belongs to kappa3(z^{l+1}) -- it goes
         into the carry for the NEXT layer, not into kappa3(z^l).

Correct recursion (derived, and verified below against a dense tensor):

    kappa3(z^l) = sum_t c_t sym(a_t (x) b_t (x) b_t)          [carried CP terms]

    slice:      kappa3_ppq = 2 (c.A.B)^T B + (c.B.B)^T A      [2 matmuls]
    pass-thru:  a_t <- W^T (h1 . a_t),  b_t <- W^T (h1 . b_t) [2 matmuls, EXACT]
    births:     a_i = W[l+1][i,:],  b_i = B[i,:],  c_i = h2(i)
                with B = Sigma(z^l) diag(h1) W[l+1]           [n new terms/layer]

MODE 1 (--verify, n=32): uncompressed CP  ==  dense n^3 tensor propagation,
        same diagrams. Must agree to float precision. This is the check that was
        skipped and that would have caught both bugs in five minutes.
MODE 2 (--sweep, n=256): with the algebra verified, measure the CARRIED SLICE
        error vs corpus truth per layer for R in {32,64,128,256,all}. This is
        the number the whole anchor story rests on (G5 assumed ~25-36%).

GPU: factors are (R,n) -- tiny. Transport/slice are 4 matmuls/layer on MPS
float32. Seconds per MLP.
"""
from __future__ import annotations

import argparse
import statistics as st

import torch

from harness import CORPUS, DT, hermite_gains, relu_step

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
LAY = [1, 3, 7, 15, 23, 31]


# ----------------------------------------------------------------- CP core ---
def cp_slice(A, B, c):
    """kappa3_ppq from CP terms. A,B: (T,n); c: (T,).  2 matmuls."""
    cab = (c[:, None] * A * B)
    cbb = (c[:, None] * B * B)
    return 2.0 * (cab.T @ B) + (cbb.T @ A)


def cp_transport(A, B, c, h1, W):
    """kappa3(a) -> kappa3(z'): a <- W^T(h1.a), b <- W^T(h1.b). EXACT."""
    return (A * h1[None, :]) @ W, (B * h1[None, :]) @ W, c


def cp_births(zcov, h1, h2, Wnext):
    """The star born at this layer, as CP terms in z^{l+1} space."""
    Bm = (zcov * h1[None, :]) @ Wnext          # (n, n) : B[i,:] rows
    return Wnext.clone(), Bm, h2.clone()       # a_i = Wnext[i,:], b_i = Bm[i,:]


def cp_compress(A, B, c, R):
    """Magnitude pruning: keep the R terms with the largest |c| ||a|| ||b||^2.
    UNPROVEN (ignores cancellation) -- which is precisely what --sweep measures."""
    if R <= 0 or A.shape[0] <= R:
        return A, B, c
    w = c.abs() * A.norm(dim=1) * B.norm(dim=1) ** 2
    idx = torch.topk(w, R).indices
    return A[idx], B[idx], c[idx]


# ------------------------------------------------------- MODE 1: verify -----
def dense_star_tensor(zcov, h1, h2, Wn):
    """kappa3(z') star as a DENSE tensor, built independently of the CP path."""
    Bm = (zcov * h1[None, :]) @ Wn
    T = (torch.einsum("i,im,ik,il->mkl", h2, Wn, Bm, Bm)
         + torch.einsum("i,im,ik,il->mkl", h2, Bm, Wn, Bm)
         + torch.einsum("i,im,ik,il->mkl", h2, Bm, Bm, Wn))
    return T


def verify(n=32, depth=6, seed=0):
    torch.manual_seed(seed)
    Ws = [(torch.randn(n, n) * (2.0 / n) ** 0.5).to(DEV, F32) for _ in range(depth)]
    mu = torch.zeros(n, dtype=DT)
    C = torch.eye(n, dtype=DT)

    K3 = torch.zeros(n, n, n, device=DEV, dtype=F32)        # dense truth-path
    A = torch.zeros(0, n, device=DEV, dtype=F32)            # CP path
    B = torch.zeros(0, n, device=DEV, dtype=F32)
    c = torch.zeros(0, device=DEV, dtype=F32)

    print(f"verify | n={n} depth={depth} | CP (uncompressed) vs DENSE tensor")
    print(f"{'layer':>6} {'slice rel diff':>16} {'tensor rel diff':>18}")
    worst = 0.0
    for l in range(depth):
        W64 = Ws[l].cpu().to(DT)
        zmu64 = mu @ W64
        zcov64 = W64.T @ C @ W64
        zcov = zcov64.to(DEV, F32)
        var = torch.clamp(torch.diagonal(zcov), min=1e-30)
        g, _, _, pdf, cdf = hermite_gains(zmu64, torch.diagonal(zcov64), kmax=6)
        h1 = cdf.to(DEV, F32)
        h2 = (pdf / torch.sqrt(var.cpu().to(DT)).to(DEV, F32).cpu()).to(DEV, F32) \
            if False else (pdf.to(DEV, F32) / torch.sqrt(var))

        # --- compare the two representations of kappa3(z^l)
        idx = torch.arange(n, device=DEV)
        sl_dense = K3[idx, idx, :]
        sl_cp = cp_slice(A, B, c) if A.shape[0] else torch.zeros(n, n, device=DEV,
                                                                 dtype=F32)
        d_sl = ((sl_cp - sl_dense).norm() / (sl_dense.norm() + 1e-30)).item()
        # rebuild the CP tensor densely (only affordable at small n) and compare
        if A.shape[0]:
            T_cp = (torch.einsum("t,tm,tk,tl->mkl", c, A, B, B)
                    + torch.einsum("t,tm,tk,tl->mkl", c, B, A, B)
                    + torch.einsum("t,tm,tk,tl->mkl", c, B, B, A))
            d_t = ((T_cp - K3).norm() / (K3.norm() + 1e-30)).item()
        else:
            d_t = 0.0
        worst = max(worst, d_sl, d_t)
        print(f"{l:>6} {d_sl:>16.2e} {d_t:>18.2e}")

        mu, C = relu_step(mu, C, W64, 8, sl_cp.cpu().to(DT), None)

        if l + 1 < depth:
            Wn = Ws[l + 1]
            # dense: pass-through (h1^3) + star
            K3p = torch.einsum("ijk,ia,jb,kc->abc",
                               K3 * h1[:, None, None] * h1[None, :, None]
                               * h1[None, None, :], Wn, Wn, Wn) if A.shape[0] \
                else torch.zeros(n, n, n, device=DEV, dtype=F32)
            K3 = K3p + dense_star_tensor(zcov, h1, h2, Wn)
            # CP: transport carried terms, then append the star's own terms
            if A.shape[0]:
                A, B, c = cp_transport(A, B, c, h1, Wn)
            aN, bN, cN = cp_births(zcov, h1, h2, Wn)
            A = torch.cat([A, aN], 0)
            B = torch.cat([B, bN], 0)
            c = torch.cat([c, cN], 0)

    ok = worst < 1e-4
    print(f"\n  {'PASS' if ok else 'FAIL'}: worst relative deviation {worst:.1e}")
    print("  (float32 at n=32; anything <1e-4 is numerics, not algebra)")
    return ok


# -------------------------------------------------------- MODE 2: sweep -----
def sweep(n_mlps=8, ranks=(32, 64, 128, 256, 0)):
    n = 256
    res = {R: {l: [] for l in range(32)} for R in ranks}
    nterms = {R: [] for R in ranks}
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)

    for m in range(n_mlps):
        rec = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                         weights_only=False)
        W = [rec["W"][l].to(DEV, F32) for l in range(32)]
        W64 = [rec["W"][l].to(DT) for l in range(32)]
        for R in ranks:
            mu = torch.zeros(n, dtype=DT)
            C = torch.eye(n, dtype=DT)
            A = torch.zeros(0, n, device=DEV, dtype=F32)
            B = torch.zeros(0, n, device=DEV, dtype=F32)
            c = torch.zeros(0, device=DEV, dtype=F32)
            for l in range(32):
                zmu64 = mu @ W64[l]
                zcov64 = W64[l].T @ C @ W64[l]
                zcov = zcov64.to(DEV, F32)
                var = torch.clamp(torch.diagonal(zcov), min=1e-30)
                _, _, _, pdf, cdf = hermite_gains(zmu64, torch.diagonal(zcov64),
                                                  kmax=6)
                h1 = cdf.to(DEV, F32)
                h2 = pdf.to(DEV, F32) / torch.sqrt(var)

                sl = cp_slice(A, B, c) if A.shape[0] else torch.zeros(
                    n, n, device=DEV, dtype=F32)
                truth = rec["zk3"][l].to(DEV, F32)
                e = ((sl - truth)[off].norm() / truth[off].norm()).item()
                res[R][l].append(e)

                mu, C = relu_step(mu, C, W64[l], 8, sl.cpu().to(DT),
                                  rec["zk4"][l].to(DT))
                if l + 1 < 32:
                    Wn = W[l + 1]
                    if A.shape[0]:
                        A, B, c = cp_transport(A, B, c, h1, Wn)
                    aN, bN, cN = cp_births(zcov, h1, h2, Wn)
                    A = torch.cat([A, aN], 0)
                    B = torch.cat([B, bN], 0)
                    c = torch.cat([c, cN], 0)
                    A, B, c = cp_compress(A, B, c, R)
            nterms[R].append(A.shape[0])
        print(f"  mlp {m} done", flush=True)

    print(f"\nsweep | {n_mlps} MLPs | CARRIED kappa3 slice error vs corpus truth")
    print(f"{'R':>8} {'#terms':>7} " + " ".join(f"{'L' + str(l):>7}" for l in LAY)
          + f" {'mean(32)':>9}")
    for R in ranks:
        row = " ".join(f"{st.mean(res[R][l]) * 100:>6.1f}%" for l in LAY)
        allm = st.mean([st.mean(res[R][l]) for l in range(32)])
        tag = "ALL" if R == 0 else str(R)
        print(f"{tag:>8} {int(st.mean(nterms[R])):>7} {row} {allm * 100:>8.1f}%")
    print("\n  R=ALL is the CEILING of the star+pass-through diagram set itself.")
    print("  G5 assumed a ~25-36% propagated slice (it used a 3-sigma TENSOR carry).")
    print("  If R=64 is close to R=ALL, magnitude pruning is fine and v5 ships.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--mlps", type=int, default=8)
    a = ap.parse_args()
    if a.verify:
        ok = verify()
        if not ok:
            raise SystemExit("CP algebra FAILED -- do not proceed")
    if a.sweep:
        sweep(a.mlps)
