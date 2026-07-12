"""(B) Supervise the CUMULANT PROPAGATION directly — batched on the GPU.

fit_layerwise.py proved the correction form is exact: with TRUE cumulants the
one-step error drops to ~2e-9 (mean) / ~6e-8 (covariance), and the fitted
coefficients match my theoretical ones (c3m ~ 1.01, c3c ~ 0.98). So 100% of the
closure's remaining error (1.05e-5) is in how accurately we PROPAGATE kappa3/4.

Ours today: kappa3 corr ~0.95 but only ~65% of true magnitude; kappa4 missing its
CHAIN diagram entirely.

Why the chain matters -- count Sigma-edges (the order in the small parameter):
    kappa3: star = 2 edges, triangle = 3  -> star dominates (corr 0.996 at L1) ✓
    kappa4: star = 3 edges, CHAIN  = 3    -> SAME ORDER; star-only computes HALF.

Chain, derived:  kappa4_chain(a)_ijkl = h1_i h2_j h2_k h1_l S_ij S_jk S_kl
After the linear map, with P = diag(h1)W, Rm = S P, projected onto basis V:
    RV = Rm V,  WV = W V,  Y_j(m,n) = h2_j RV_jm WV_jn
    core_chain[(m,n),(o,p)] = (Y^T S Y)   -> two matmuls, then symmetrised.

Teacher-forces the MOMENTS (injects the corpus's true m, S each layer, so cumulant
error is not confounded with moment drift) and fits generation / pass-through /
chain coefficients so OUR propagated kappa3/kappa4 slices match the SAMPLED ones
at every layer: 32 layers x 65k entries x B MLPs of supervision, versus the 256
numbers the end-to-end loss provides.

ALL MLPs are propagated in one batch on the GPU (128 GB unified memory: 64 MLPs of
corpus state is ~2 GB).

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/fit_cumulants.py
"""

import itertools
import math
import time
from pathlib import Path

import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "moment_corpus"
DATA = ROOT / "data"
R, R4 = 32, 8
N_MLPS, STEPS, LR = 48, 250, 5e-2
PERMS4 = list(itertools.permutations(range(4)))

_Q0 = torch.randn(N_W, R, generator=torch.Generator().manual_seed(7)).to(DEV)


@torch.no_grad()
def top_basis_batched(S):
    """(B,n,n) -> (B,n,R), columns ordered by eigenvalue. MPS has no QR/eigh, so
    those two small ops run on the CPU (batched, once)."""
    B = S.shape[0]
    Q = _Q0.expand(B, N_W, R).contiguous()
    for _ in range(2):
        Q = torch.linalg.qr(torch.bmm(S, Q).cpu()).Q.to(DEV)
    Sr = torch.bmm(Q.transpose(1, 2), torch.bmm(S, Q)).cpu()
    Sr = 0.5 * (Sr + Sr.transpose(1, 2))
    _, U = torch.linalg.eigh(Sr)
    return torch.bmm(Q, torch.flip(U, dims=[2]).to(DEV))


def sym4(T):
    out = torch.zeros_like(T)
    for p in PERMS4:
        out = out + T.permute(0, *[i + 1 for i in p])
    return out / 24.0


print(f"loading {N_MLPS} MLPs...", flush=True)
t0 = time.time()
files = sorted(CORPUS.glob("mlp_*.pt"))[:N_MLPS]
W = torch.stack([torch.load(f, map_location="cpu")["W"] for f in files]).to(DEV)
ZMU = torch.stack([torch.load(f, map_location="cpu")["zmu"] for f in files]).to(DEV)
ZCOV = torch.stack([torch.load(f, map_location="cpu")["zcov"] for f in files]).to(DEV)
ZK3 = torch.stack([torch.load(f, map_location="cpu")["zk3"] for f in files]).to(DEV)
ZK4 = torch.stack([torch.load(f, map_location="cpu")["zk4"] for f in files]).to(DEV)
B = W.shape[0]
print(f"  {B} MLPs on {DEV} ({time.time()-t0:.0f}s); precomputing bases...", flush=True)
VS = [top_basis_batched(ZCOV[:, li]) for li in range(DEPTH)]
print(f"  bases done ({time.time()-t0:.0f}s)", flush=True)

# normalisers so kappa3 and kappa4 losses are comparable across layers
N3 = torch.stack([(ZK3[:, li] ** 2).mean() for li in range(DEPTH)])
N4 = torch.stack([(ZK4[:, li] ** 2).mean() for li in range(DEPTH)])


def propagate(coef, use_chain=True):
    T3 = torch.zeros(B, R, R, R, device=DEV)
    T4 = torch.zeros(B, R4, R4, R4, R4, device=DEV)
    star = None
    l3 = torch.zeros((), device=DEV)
    l4 = torch.zeros((), device=DEV)
    for li in range(DEPTH):
        S, m, V = ZCOV[:, li], ZMU[:, li], VS[li]
        var = torch.clamp(torch.diagonal(S, dim1=1, dim2=2), min=1e-8)
        s = var.sqrt()
        a = torch.clamp(m / s, -12.0, 12.0)
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        h1, h2, h3 = cdf, pdf / s, -a * pdf / var

        G2 = (V.unsqueeze(3) * V.unsqueeze(2)).reshape(B, N_W, R * R)
        k3 = torch.bmm(torch.bmm(G2, T3.reshape(B, R * R, R)), V.transpose(1, 2))
        V4 = V[:, :, :R4]
        G2b = (V4.unsqueeze(3) * V4.unsqueeze(2)).reshape(B, N_W, R4 * R4)
        k4 = torch.bmm(torch.bmm(G2b, T4.reshape(B, R4 * R4, R4 * R4)),
                       G2b.transpose(1, 2))
        if star is not None:
            h2s, h3s, Wr, Bs = star
            C3 = h2s.unsqueeze(2) * Wr * Bs
            D3 = h2s.unsqueeze(2) * Bs * Bs
            k3 = k3 + 2.0 * torch.bmm(C3.transpose(1, 2), Bs) \
                    + torch.bmm(D3.transpose(1, 2), Wr)
            C4 = h3s.unsqueeze(2) * Wr * Bs
            D4 = h3s.unsqueeze(2) * Bs * Bs
            k4 = k4 + 2.0 * torch.bmm(C4.transpose(1, 2), Bs * Bs) \
                    + 2.0 * torch.bmm(D4.transpose(1, 2), Wr * Bs)

        if li >= 1:
            l3 = l3 + ((k3 - ZK3[:, li]) ** 2).mean() / N3[li]
            l4 = l4 + ((k4 - ZK4[:, li]) ** 2).mean() / N4[li]

        if li == DEPTH - 1:
            break
        Wn, Vn = W[:, li + 1], VS[li + 1]
        Z = h1.unsqueeze(2) * torch.bmm(Wn, Vn)
        M = torch.bmm(Z.transpose(1, 2), V)
        M4 = M[:, :R4, :R4]
        T3 = coef["c3p"][li] * torch.einsum("bmkl,bam,bnk,bol->bano", T3, M, M, M)
        T4 = coef["c4p"][li] * torch.einsum("bmklp,bam,bnk,bol,bqp->banoq",
                                            T4, M4, M4, M4, M4)
        if star is not None:
            h2s, h3s, Wr, Bs = star
            A, Bv = torch.bmm(Wr, Z), torch.bmm(Bs, Z)
            T3 = T3 + coef["c3g"][li] * (
                torch.einsum("bi,bim,bik,bil->bmkl", h2s, A, Bv, Bv)
                + torch.einsum("bi,bim,bik,bil->bmkl", h2s, Bv, A, Bv)
                + torch.einsum("bi,bim,bik,bil->bmkl", h2s, Bv, Bv, A))
            A4, B4 = A[:, :, :R4], Bv[:, :, :R4]
            T4 = T4 + coef["c4g"][li] * sym4(
                4.0 * torch.einsum("bi,bim,bik,bil,bip->bmklp", h3s, A4, B4, B4, B4))
        if use_chain:                      # kappa4 CHAIN -- the missing half
            Rm = torch.bmm(S, h1.unsqueeze(2) * Wn)
            RV = torch.bmm(Rm, Vn)[:, :, :R4]
            WV = torch.bmm(Wn, Vn)[:, :, :R4]
            Y = (h2[:, :, None, None] * RV[:, :, :, None]
                 * WV[:, :, None, :]).reshape(B, N_W, R4 * R4)
            chain = torch.bmm(Y.transpose(1, 2), torch.bmm(S, Y)).reshape(
                B, R4, R4, R4, R4)
            T4 = T4 + coef["c4chain"][li] * sym4(chain)

        star = (h2, h3, Wn, torch.bmm(S, h1.unsqueeze(2) * Wn))
    n = DEPTH - 1
    return l3 / n, l4 / n


def coefs(x):
    e = torch.exp(1.5 * torch.tanh(x))
    return {"c3g": e[:, 0], "c3p": e[:, 1], "c4g": e[:, 2],
            "c4p": e[:, 3], "c4chain": e[:, 4]}


zero = torch.zeros(DEPTH, 5, device=DEV)
with torch.no_grad():
    b3, b4 = propagate(coefs(zero), use_chain=False)
    c3, c4 = propagate(coefs(zero), use_chain=True)
print("\nrelative error of the PROPAGATED cumulants "
      "(1.0 = as bad as predicting zero):", flush=True)
print(f"  star only    : kappa3 {float(b3):.3f}   kappa4 {float(b4):.3f}", flush=True)
print(f"  star + chain : kappa3 {float(c3):.3f}   kappa4 {float(c4):.3f}"
      f"   <- chain effect", flush=True)

raw = torch.zeros(DEPTH, 5, device=DEV, requires_grad=True)
opt = torch.optim.Adam([raw], lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.05)
print("\nfitting generation / pass-through / chain coefficients...", flush=True)
t0 = time.time()
best = (float("inf"), raw.detach().clone())
for step in range(1, STEPS + 1):
    l3, l4 = propagate(coefs(raw), use_chain=True)
    loss = l3 + l4
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([raw], 1.0)
    opt.step()
    sched.step()
    if float(loss) < best[0]:
        best = (float(loss), raw.detach().clone())
    if step % 10 == 0:
        print(f"  step {step:4d}  kappa3 {float(l3):.4f}  kappa4 {float(l4):.4f}  "
              f"({step/(time.time()-t0):.1f} steps/s)", flush=True)

torch.save({"raw": best[1].cpu(), "rank": R, "rank4": R4, "loss": best[0]},
           DATA / "fit_cumulants.pt")
print(f"\nbest total {best[0]:.4f} -> saved {DATA/'fit_cumulants.pt'}")
print("Next: fold these into theta with the fitted moment coefficients and")
print("evaluate the closure end-to-end.")
