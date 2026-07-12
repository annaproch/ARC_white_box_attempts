"""(B) Supervise the CUMULANT PROPAGATION directly — the only thing left.

fit_layerwise.py proved the correction form is exact: with TRUE cumulants the
one-step error drops to ~2e-9 (mean) and ~6e-8 (covariance), and the fitted
coefficients match my theoretical ones (c3m ~ 1.01, c3c ~ 0.98). So 100% of the
closure's remaining error (1.05e-5) is in how accurately we PROPAGATE kappa3 and
kappa4.

Ours today: kappa3 corr ~0.95 but only ~65% of the true magnitude; kappa4 is
missing its CHAIN diagram entirely.

Why the chain matters: count Sigma-edges (that is the order in the small
parameter).
    kappa3: star = 2 edges, triangle = 3   -> star dominates (measured: corr 0.996 at L1)
    kappa4: star = 3 edges, CHAIN  = 3     -> SAME ORDER. star-only computes half.

Chain, derived:  kappa4_chain(a)_ijkl = h1_i h2_j h2_k h1_l S_ij S_jk S_kl
After the linear map, with P = diag(h1) W, R = S P, and projecting onto basis V:
    RV = R V,  WV = W V,  Y_j(m,n) = h2_j RV_jm WV_jn
    core_chain[(m,n),(o,p)] = (Y^T S Y)[(m,n),(o,p)]     -> two matmuls
then symmetrised over the 4 slots.

This script teacher-forces the MOMENTS (injects the corpus's true m, S at every
layer, so cumulant error is not confounded with moment drift) and fits the
generation / pass-through coefficients -- including the chain's -- so that OUR
propagated kappa3/kappa4 slices match the SAMPLED ones at every layer.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/fit_cumulants.py
"""

import math
import time
from pathlib import Path

import numpy as np
import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "moment_corpus"
DATA = ROOT / "data"
R, R4 = 32, 8
STEPS, LR = 120, 5e-2
N_MLPS = 8      # the loss loops over MLPs sequentially; 32 made a step ~30s

_Q0 = torch.randn(N_W, R, generator=torch.Generator().manual_seed(7)).to(DEV)


def top_basis(S):
    """Same procedure as the trainer/estimator. MPS has no QR/eigh kernel, so do
    those two small ops on the CPU (bases are computed once, under no_grad)."""
    Q = _Q0
    for _ in range(2):
        Y = (S @ Q).cpu()
        Q = torch.linalg.qr(Y).Q.to(S.device)
    Sr = (Q.T @ S @ Q).cpu()
    Sr = 0.5 * (Sr + Sr.T)
    _, U = torch.linalg.eigh(Sr)
    return (Q.cpu() @ torch.flip(U, dims=[1])).to(S.device)


def sym4(T):
    """Symmetrise a (r,r,r,r) tensor over all 24 index permutations."""
    import itertools
    out = torch.zeros_like(T)
    for p in itertools.permutations(range(4)):
        out = out + T.permute(*p)
    return out / 24.0


def load(n_mlps=32):
    recs = []
    for f in sorted(CORPUS.glob("mlp_*.pt"))[:n_mlps]:
        r = torch.load(f, map_location="cpu")
        recs.append({k: r[k].to(DEV) for k in ("W", "zmu", "zcov", "zk3", "zk4")})
    return recs


@torch.no_grad()
def bases(rec):
    return [top_basis(rec["zcov"][li]) for li in range(DEPTH)]


def propagate(rec, Vs, coef, use_chain=True):
    """Teacher-forced moments; our cores. Returns per-layer (k3_slice, k4_slice)."""
    T3 = torch.zeros(R, R, R, device=DEV)
    T4 = torch.zeros(R4, R4, R4, R4, device=DEV)
    star = None
    out = []
    for li in range(DEPTH):
        S, m, V = rec["zcov"][li], rec["zmu"][li], Vs[li]
        var = torch.clamp(torch.diagonal(S), min=1e-8)
        s = var.sqrt()
        a = torch.clamp(m / s, -12.0, 12.0)
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        h1, h2, h3 = cdf, pdf / s, -a * pdf / var

        G2 = (V.unsqueeze(2) * V.unsqueeze(1)).reshape(N_W, R * R)
        P3 = G2 @ T3.reshape(R * R, R)
        k3 = P3 @ V.T
        V4 = V[:, :R4]
        G2b = (V4.unsqueeze(2) * V4.unsqueeze(1)).reshape(N_W, R4 * R4)
        k4 = (G2b @ T4.reshape(R4 * R4, R4 * R4)) @ G2b.T
        if star is not None:
            h2s, h3s, Wr, Bs = star
            C3 = h2s[:, None] * Wr * Bs
            D3 = h2s[:, None] * Bs * Bs
            k3 = k3 + 2.0 * (C3.T @ Bs) + (D3.T @ Wr)
            C4 = h3s[:, None] * Wr * Bs
            D4 = h3s[:, None] * Bs * Bs
            k4 = k4 + 2.0 * (C4.T @ (Bs * Bs)) + 2.0 * (D4.T @ (Wr * Bs))
        out.append((k3, k4))

        if li == DEPTH - 1:
            break
        W = rec["W"][li + 1]
        Vn = Vs[li + 1]
        Z = h1[:, None] * (W @ Vn)
        M = Z.T @ V
        M4 = M[:R4, :R4]
        T3 = coef["c3p"][li] * torch.einsum("mkl,am,nk,ol->ano", T3, M, M, M)
        T4 = coef["c4p"][li] * torch.einsum("mklp,am,nk,ol,qp->anoq", T4, M4, M4, M4, M4)

        if star is not None:
            h2s, h3s, Wr, Bs = star
            A, Bv = Wr @ Z, Bs @ Z
            T3 = T3 + coef["c3g"][li] * (
                torch.einsum("i,im,ik,il->mkl", h2s, A, Bv, Bv)
                + torch.einsum("i,im,ik,il->mkl", h2s, Bv, A, Bv)
                + torch.einsum("i,im,ik,il->mkl", h2s, Bv, Bv, A))
            A4, B4 = A[:, :R4], Bv[:, :R4]
            T4 = T4 + coef["c4g"][li] * sym4(
                torch.einsum("i,im,ik,il,ip->mklp", h3s, A4, B4, B4, B4) * 4.0)

        if use_chain:                       # kappa4 CHAIN: the missing half
            P = h1[:, None] * W
            Rm = S @ P
            RV = (Rm @ Vn)[:, :R4]
            WV = (W @ Vn)[:, :R4]
            Y = (h2[:, None, None] * RV[:, :, None] * WV[:, None, :]).reshape(N_W, R4 * R4)
            chain = (Y.T @ (S @ Y)).reshape(R4, R4, R4, R4)
            T4 = T4 + coef["c4chain"][li] * sym4(chain)

        star = (h2, h3, W, S @ (h1[:, None] * W))
    return out


recs = load(N_MLPS)
print(f"{len(recs)} MLPs; precomputing bases...", flush=True)
t0 = time.time()
VS = [bases(r) for r in recs]
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

raw = torch.zeros(DEPTH, 5, device=DEV, requires_grad=True)   # c3g,c3p,c4g,c4p,c4chain
opt = torch.optim.Adam([raw], lr=LR)


def coefs(x):
    e = torch.exp(1.5 * torch.tanh(x))
    return {"c3g": e[:, 0], "c3p": e[:, 1], "c4g": e[:, 2],
            "c4p": e[:, 3], "c4chain": e[:, 4]}


def losses(coef, use_chain, grad=True):
    l3 = l4 = 0.0
    for rec, Vs in zip(recs, VS):
        outs = propagate(rec, Vs, coef, use_chain)
        for li in range(1, DEPTH):
            t3, t4 = rec["zk3"][li], rec["zk4"][li]
            k3, k4 = outs[li]
            l3 = l3 + ((k3 - t3) ** 2).mean() / (t3**2).mean()
            l4 = l4 + ((k4 - t4) ** 2).mean() / (t4**2).mean()
    n = len(recs) * (DEPTH - 1)
    return l3 / n, l4 / n


with torch.no_grad():
    b3, b4 = losses(coefs(torch.zeros(DEPTH, 5, device=DEV)), use_chain=False)
    c3, c4 = losses(coefs(torch.zeros(DEPTH, 5, device=DEV)), use_chain=True)
print(f"\nrelative error of the PROPAGATED cumulants (1.0 = as bad as predicting 0):", flush=True)
print(f"  star only     : kappa3 {float(b3):.3f}   kappa4 {float(b4):.3f}", flush=True)
print(f"  star + chain  : kappa3 {float(c3):.3f}   kappa4 {float(c4):.3f}  <- chain effect", flush=True)

print("\nfitting generation / pass-through / chain coefficients...")
t0 = time.time()
for step in range(1, STEPS + 1):
    coef = coefs(raw)
    l3, l4 = losses(coef, use_chain=True)
    loss = l3 + l4
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([raw], 1.0)
    opt.step()
    if step % 10 == 0:
        print(f"  step {step:4d}  kappa3 {float(l3):.4f}  kappa4 {float(l4):.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

torch.save({"raw": raw.detach().cpu(), "rank": R, "rank4": R4}, DATA / "fit_cumulants.pt")
print(f"\nsaved {DATA/'fit_cumulants.pt'}")
print("Next: fold these into theta (with the fitted moment coefficients from")
print("fit_layerwise.py) and evaluate the closure end-to-end.")
