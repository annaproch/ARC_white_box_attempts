"""G2 — do the kappa4(z) BLOCKS break the 10.6% one-step kappa3 floor?

E9b measured kappa3(ReLU(z)) reconstruction at n=256 and found a 10.6% floor
that is NOT sigma-edge-limited (converged by 4 edges), NOT rank-limited
(R=128 = exact), NOT calibration-limited. STRATEGY 4.14 attributes it to the
missing kappa4(z) BLOCKS -- but that was never tested, because
`diagrams.eval_diagram` feeds kappa4 to einsum as a FULL n^4 tensor: 17 GB at
n=256, and the n^4 moment accumulation is ~3e16 FLOPs. Untestable as written.

THE KEY STRUCTURAL FACT (STRATEGY 4.14): a kappa4 block in a kappa3(a) diagram
puts 4 legs on at most 3 VERTICES. So every kappa4 operand that can ever appear
is a SLICE with <=3 distinct indices:

    (2,1,1) -> k4_ppqr   n^3   67 MB   <- the largest object we ever need
    (2,2,0) -> k4_ppqq   n^2
    (3,1,0) -> k4_pppq   n^2
    (4,0,0) -> k4_pppp   n

Same for kappa3 blocks: (1,1,1) -> k3_pqr (n^3), (2,1,0) -> k3_ppq, (3,0,0).

This module provides:
  * `k_slices`      sampled Sigma, kappa3 and kappa4 slices at the real shape
                    (n matmuls of (n,N)(N,n), no N*n^3 intermediate)
  * `eval_slice`    a SLICE-AWARE diagram evaluator (drop-in for eval_diagram)
  * `--verify`      machine check: slice evaluator == full-tensor evaluator,
                    exactly, on random symmetric cumulants at small n. This is
                    the mandated verification (hand tensor algebra is BANNED);
                    it tests the SLICE BOOKKEEPING, not the physics.
  * `--measure`     the G2 number at n=256: kappa3(ReLU(z)) relative error with
                    kappa4 blocks OFF vs ON.

PASS: adding kappa4 blocks takes the one-step kappa3 error from ~10.6% to <=5%
(the E6 spec). FAIL: the diagram route cannot reach the spec per-step, and
error-anchoring (G3) becomes the primary program rather than a garnish.
"""
from __future__ import annotations

import argparse
import time

import torch

from diagrams import enumerate_diagrams, eval_diagram
from harness import CORPUS, DT, hermite_gains

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
LETTERS = "pqr"


# --------------------------------------------------------------- slices ----
# Triple products as ONE matmul via a Khatri-Rao (column-wise Kronecker) lift:
#
#     M3[p,q,r] = sum_n d_np d_nq d_nr  =  (KR^T @ d).reshape(n,n,n),
#     KR[:, p*n+q] = d_np * d_nq                                   (B, n^2)
#
# The naive form (`for p: (d * d[:,p]).T @ d`) materialises an (N,n) temporary
# 256 times per tensor -- ~2 TB of memory traffic at N=4e6, i.e. bandwidth-bound
# for a job that should be compute-bound. The lift turns it into one
# (n^2 x B)(B x n) matmul per chunk: ~70 FLOPs/byte, GPU-saturating.
# B=4096 keeps KR at ~1 GB, so several fit comfortably in 128 GB.
KR_CHUNK = 4096


def _kr(a, b):
    """Khatri-Rao: (B,n),(B,n) -> (B, n*n) with column p*n+q = a_p * b_q."""
    B, n = a.shape
    return (a[:, :, None] * b[:, None, :]).reshape(B, n * n)


def stream_moments(W, lay, N, fwd_chunk=131_072, seed=3):
    """Two-pass streaming: pass 1 = means (forward only), pass 2 = the centred
    n^3 moments of z and a. Nothing but the (n^3) accumulators is retained, so N
    is limited by time, not memory."""
    n = 256
    dev = W[0].device

    def forward(g, b):
        A = torch.randn(b, n, generator=g, dtype=F32).to(dev)
        for l in range(lay):
            A = torch.clamp(A @ W[l], min=0.0)
        Z = A @ W[lay]
        return Z, torch.clamp(Z, min=0.0)

    # ---- pass 1: means
    # MPS has no float64: accumulate per-chunk sums on-device in float32, add
    # them into a CPU float64 running total (one small transfer per chunk).
    g = torch.Generator(device="cpu").manual_seed(seed)
    sz = torch.zeros(n, dtype=DT)
    sa = torch.zeros(n, dtype=DT)
    done = 0
    while done < N:
        b = min(fwd_chunk, N - done)
        Z, A = forward(g, b)
        sz += Z.sum(0).cpu().to(DT)
        sa += A.sum(0).cpu().to(DT)
        done += b
    zmu = (sz / N).to(dev, F32)
    amu = (sa / N).to(dev, F32)

    # ---- pass 2: centred moments (C, M3z, M4z_ppqr, M3a)
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = torch.zeros(n, n, device=dev, dtype=F32)
    M3z = torch.zeros(n * n, n, device=dev, dtype=F32)
    M4z = torch.zeros(n * n, n, device=dev, dtype=F32)
    M3a = torch.zeros(n * n, n, device=dev, dtype=F32)
    done = 0
    while done < N:
        b = min(fwd_chunk, N - done)
        Z, A = forward(g, b)
        dz_full = Z - zmu
        da_full = A - amu
        C += dz_full.T @ dz_full
        for s in range(0, b, KR_CHUNK):            # KR lift in ~1 GB slabs
            dz = dz_full[s:s + KR_CHUNK]
            da = da_full[s:s + KR_CHUNK]
            M3z += _kr(dz, dz).T @ dz
            M4z += _kr(dz * dz, dz).T @ dz
            M3a += _kr(da, da).T @ da
        done += b
    inv = 1.0 / N
    C *= inv
    M3z = (M3z * inv).reshape(n, n, n)
    M4z = (M4z * inv).reshape(n, n, n)
    M3a = (M3a * inv).reshape(n, n, n)

    v = torch.diagonal(C)
    # kappa4_ppqr = M4_ppqr - [C_pp C_qr + 2 C_pq C_pr]   (3 pairings of ppqr)
    K4 = (M4z - v[:, None, None] * C[None, :, :]
          - 2.0 * C[:, :, None] * C[:, None, :])
    return {"C": C, "K3": M3z, "K4": K4, "zmu": zmu, "K3a": M3a}


def _k4_view(K4, pattern):
    """kappa4 operand for a size-4 block with leg pattern `pattern` over the 3
    vertices, expressed via the stored K4[p,q,r] = kappa4(p,p,q,r).

    kappa4 is FULLY SYMMETRIC, so a pattern is characterised by its multiset of
    leg counts; we return (tensor, subscript over the vertices it touches)."""
    counts = [(v, c) for v, c in enumerate(pattern) if c > 0]
    counts.sort(key=lambda x: -x[1])
    shape = tuple(c for _, c in counts)
    if shape == (2, 1, 1):                       # k4(p,p,q,r) -> stored directly
        (vp, _), (vq, _), (vr, _) = counts
        return K4, [vp, vq, vr]
    if shape == (2, 2):                          # k4(p,p,q,q) = diag over (q,r)
        (vp, _), (vq, _) = counts
        return torch.diagonal(K4, dim1=1, dim2=2), [vp, vq]
    if shape == (3, 1):                          # k4(p,p,p,q) = K4[p,p,q]
        (vp, _), (vq, _) = counts
        return torch.diagonal(K4, dim1=0, dim2=1).T, [vp, vq]
    if shape == (4,):                            # k4(p,p,p,p)
        (vp, _) = counts[0]
        return torch.diagonal(torch.diagonal(K4, dim1=0, dim2=1), dim1=0, dim2=1), [vp]
    raise ValueError(f"unreachable kappa4 pattern {pattern}")


def _k3_view(K3, pattern):
    """kappa3 operand for a size-3 block, from the full K3[p,q,r]."""
    counts = [(v, c) for v, c in enumerate(pattern) if c > 0]
    counts.sort(key=lambda x: -x[1])
    shape = tuple(c for _, c in counts)
    if shape == (1, 1, 1):
        return K3, [counts[0][0], counts[1][0], counts[2][0]]
    if shape == (2, 1):                          # k3(p,p,q)
        (vp, _), (vq, _) = counts
        return torch.diagonal(K3, dim1=0, dim2=1).T, [vp, vq]
    if shape == (3,):                            # k3(p,p,p)
        d1 = torch.diagonal(K3, dim1=0, dim2=1)
        return torch.diagonal(d1, dim1=0, dim2=1), [counts[0][0]]
    raise ValueError(f"unreachable kappa3 pattern {pattern}")


def eval_slice(blocks, k, coef, g, S, K3, K4):
    """SLICE-AWARE diagram evaluator. Same contract as diagrams.eval_diagram,
    but kappa3/kappa4 blocks are contracted from <=3-index slices, so no n^4
    tensor is ever built. Verified equal to eval_diagram in --verify."""
    operands, subs = [], []
    for size, b in blocks:
        if size == 2:                             # a Sigma edge: 2 legs, 2 vertices
            vs = [v for v in range(3) for _ in range(b[v])]
            operands.append(S)
            subs.append("".join(LETTERS[v] for v in vs))
        elif size == 3:
            t, vs = _k3_view(K3, b)
            operands.append(t)
            subs.append("".join(LETTERS[v] for v in vs))
        else:
            t, vs = _k4_view(K4, b)
            operands.append(t)
            subs.append("".join(LETTERS[v] for v in vs))
    expr = ",".join(subs) + ",p,q,r->pqr"
    operands += [g[k[0]], g[k[1]], g[k[2]]]
    return coef * torch.einsum(expr, *operands)


def k3_of_relu(g, S, K3, K4, max_sigma, max_k3, max_k4):
    n = S.shape[0]
    out = torch.zeros(n, n, n, device=S.device, dtype=S.dtype)
    for blocks, k, coef in enumerate_diagrams(max_sigma, max_k3, max_k4):
        out += eval_slice(blocks, k, coef, g, S, K3, K4)
    return out


# ------------------------------------------------------------- COMPILED ----
# EVERY diagram output is a pure ELEMENTWISE product: all operand indices are
# subsets of {p,q,r} and the output is pqr, so there are NO summed indices.
# Hence the whole sum collapses to, per n^3-operand group,
#
#     out = einsum('dpq,dpr,dqr->pqr', M1, M2, M3) * (K3_pqr and/or K4 factor)
#
# with every Sigma power, k3_ppq / k4_ppqq / k4_pppq matrix and gain vector
# folded into the three (D,n,n) batched matrices. That is ~8 kernels instead of
# 10,000 (the per-diagram einsum loop was launch-bound: ~10 min/layer, and G3
# would call it ~160k times). Verified against eval_slice in --verify-compiled.
def compile_diagrams(max_sigma, max_k3, max_k4):
    """Group diagrams by their n^3-operand signature. Returns a plan: a list of
    (n3_key, [(coef, blocks, k), ...]) where n3_key says which n^3 tensors
    multiply that group."""
    from collections import defaultdict
    groups = defaultdict(list)
    for blocks, k, coef in enumerate_diagrams(max_sigma, max_k3, max_k4):
        n3 = []                                   # which n^3 operands appear
        for size, b in blocks:
            counts = sorted([(v, c) for v, c in enumerate(b) if c > 0],
                            key=lambda x: -x[1])
            shape = tuple(c for _, c in counts)
            if size == 3 and shape == (1, 1, 1):
                n3.append(("K3", tuple(v for v, _ in counts)))
            elif size == 4 and shape == (2, 1, 1):
                n3.append(("K4", tuple(v for v, _ in counts)))
        groups[tuple(sorted(n3))].append((coef, blocks, k))
    return list(groups.items())


def _n3_tensor(key, K3, K4):
    """Materialise the (permuted) n^3 factor for a group key."""
    n = K3.shape[0] if K3 is not None else K4.shape[0]
    out = None
    for kind, verts in key:
        T = K3 if kind == "K3" else K4          # K4[a,b,c] = kappa4(a,a,b,c)
        # verts lists the vertices in the tensor's own index order; permute to pqr
        perm = [0, 0, 0]
        for axis, v in enumerate(verts):
            perm[v] = axis
        Tp = T.permute(*perm)
        out = Tp if out is None else out * Tp
    return out


def eval_compiled(plan, g, S, K3, K4):
    """Evaluate a compiled plan. One batched einsum per n^3 group."""
    n = S.shape[0]
    dev, dt = S.device, S.dtype
    total = torch.zeros(n, n, n, device=dev, dtype=dt)
    Spow = [torch.ones(n, n, device=dev, dtype=dt)]
    for _ in range(6):
        Spow.append(Spow[-1] * S)

    for key, diags in plan:
        D = len(diags)
        M = [torch.ones(D, n, n, device=dev, dtype=dt) for _ in range(3)]  # pq,pr,qr
        PAIR = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
        for d, (coef, blocks, k) in enumerate(diags):
            mats = [torch.ones(n, n, device=dev, dtype=dt) for _ in range(3)]
            vecs = [g[k[0]].clone(), g[k[1]].clone(), g[k[2]].clone()]
            for size, b in blocks:
                counts = sorted([(v, c) for v, c in enumerate(b) if c > 0],
                                key=lambda x: -x[1])
                shape = tuple(c for _, c in counts)
                vs = [v for v, _ in counts]
                if size == 2:                       # Sigma edge (mixed, 2 vertices)
                    i, j = sorted(vs)
                    mats[PAIR[(i, j)]] = mats[PAIR[(i, j)]] * (
                        S if i == vs[0] or True else S)
                elif size == 3 and shape == (2, 1):
                    t, _ = _k3_view(K3, b)          # k3_ppq matrix on (vs[0], vs[1])
                    i, j = vs[0], vs[1]
                    mats[PAIR[tuple(sorted((i, j)))]] = mats[
                        PAIR[tuple(sorted((i, j)))]] * (t if i < j else t.T)
                elif size == 3 and shape == (3,):
                    t, _ = _k3_view(K3, b)
                    vecs[vs[0]] = vecs[vs[0]] * t
                elif size == 4 and shape in ((2, 2), (3, 1)):
                    t, _ = _k4_view(K4, b)
                    i, j = vs[0], vs[1]
                    mats[PAIR[tuple(sorted((i, j)))]] = mats[
                        PAIR[tuple(sorted((i, j)))]] * (t if i < j else t.T)
                elif size == 4 and shape == (4,):
                    t, _ = _k4_view(K4, b)
                    vecs[vs[0]] = vecs[vs[0]] * t
                # (1,1,1)/(2,1,1) n^3 blocks are handled by the group factor
            mats[0] = mats[0] * vecs[0][:, None] * vecs[1][None, :]   # p, q
            mats[1] = mats[1] * vecs[2][None, :]                      # r
            mats[0] = mats[0] * coef
            for a in range(3):
                M[a][d] = mats[a]
        # MPS caps tensor dims at INT_MAX and this einsum materialises a
        # (D,n,n,n) intermediate, so chunk the diagram axis to keep it < ~2e8
        # elements. Still ~D/8 kernels instead of D.
        dchunk = max(1, int(2e8 // (n ** 3)))
        part = torch.zeros(n, n, n, device=dev, dtype=dt)
        for c in range(0, D, dchunk):
            part += torch.einsum("dpq,dpr,dqr->pqr",
                                 M[0][c:c + dchunk], M[1][c:c + dchunk],
                                 M[2][c:c + dchunk])
        if key:
            part = part * _n3_tensor(key, K3, K4)
        total += part
        del M
    return total


# --------------------------------------------------------------- verify ----
def verify(n=6, seed=0):
    """Machine check of the SLICE BOOKKEEPING: eval_slice must equal
    eval_diagram (full-tensor einsum) exactly, on random FULLY SYMMETRIC
    cumulants where the n^4 tensor is affordable. Physics-free by design."""
    torch.manual_seed(seed)
    A = torch.randn(n, n, dtype=torch.float64)
    S = A @ A.T / n                                      # SPD, symmetric
    R3 = torch.randn(n, n, n, dtype=torch.float64)
    K3 = sum(R3.permute(p) for p in
             [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]) / 6
    R4 = torch.randn(n, n, n, n, dtype=torch.float64)
    K4full = torch.zeros_like(R4)
    import itertools
    for p in itertools.permutations(range(4)):
        K4full += R4.permute(p)
    K4full /= 24
    K4slice = K4full[torch.arange(n)[:, None, None], torch.arange(n)[:, None, None],
                     torch.arange(n)[None, :, None], torch.arange(n)[None, None, :]]
    g = [torch.randn(n, dtype=torch.float64) for _ in range(32)]

    print(f"verify | n={n} | slice evaluator vs full-tensor evaluator")
    print(f"{'truncation':>34} {'#diag':>6} {'max abs diff':>14} {'rel':>10}")
    worst = 0.0
    for ms, m3, m4 in ((2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 1, 1),
                       (3, 1, 1), (4, 1, 1), (2, 0, 2), (3, 2, 2)):
        dgs = enumerate_diagrams(ms, m3, m4)
        Pf = torch.zeros(n, n, n, dtype=torch.float64)
        Ps = torch.zeros(n, n, n, dtype=torch.float64)
        for blocks, k, coef in dgs:
            Pf += eval_diagram(blocks, k, coef, g, S, K3, K4full)
            Ps += eval_slice(blocks, k, coef, g, S, K3, K4slice)
        dif = (Pf - Ps).abs().max().item()
        rel = dif / Pf.abs().max().item()
        worst = max(worst, rel)
        tag = f"sigma<={ms}, k3<={m3}, k4<={m4}"
        print(f"{tag:>34} {len(dgs):>6} {dif:>14.2e} {rel:>9.1e}")
    ok = worst < 1e-10
    print(f"\n  {'PASS' if ok else 'FAIL'}: worst relative deviation {worst:.1e} "
          f"({'float noise' if ok else 'REAL DISAGREEMENT'})")
    return ok


# -------------------------------------------------------------- measure ----
def measure(N, m, lay, noise_check=True):
    """The G2 number at the real shape: kappa3(ReLU(z)) relative error with the
    kappa4(z) blocks OFF vs ON. Reproduces E9b's rows, then extends them.

    Also reports the TRUTH tensor's own sampling-noise floor (two independent
    half-N estimates). Per the G1b lesson: a 5% spec cannot be measured against
    a truth that is itself 5% noise."""
    n = 256
    rec = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                     weights_only=False)
    W = [rec["W"][l].to(DEV, F32) for l in range(32)]

    t0 = time.time()
    sl = stream_moments(W, lay, N, seed=3)
    K3a = sl["K3a"]
    print(f"  streamed {N:,} samples -> Sigma, k3(z), k4(z)_ppqr, k3(a): "
          f"{time.time() - t0:.0f}s", flush=True)

    nf = None
    if noise_check:
        t1 = time.time()
        h1 = stream_moments(W, lay, N // 2, seed=11)["K3a"]
        h2 = stream_moments(W, lay, N // 2, seed=22)["K3a"]
        # halves differ by sqrt(2)x the noise of a half => /2 gives the full-N floor
        nf = ((h1 - h2).pow(2).sum() / (0.5 * (h1 + h2)).pow(2).sum()).sqrt().item() / 2
        del h1, h2
        print(f"  truth-tensor noise floor at N={N:,}: {nf * 100:.2f}%  "
              f"({time.time() - t1:.0f}s)", flush=True)

    zmu = sl["zmu"].cpu().to(DT)
    zvar = torch.diagonal(sl["C"]).cpu().to(DT)
    gh, *_ = hermite_gains(zmu, zvar, kmax=16)
    gh = [x.to(DEV, F32) for x in gh]
    den = K3a.pow(2).sum()

    print(f"\nG2 | n=256, MLP {m}, layer {lay}, N={N:,}  (E6 spec: kappa3 <=5%)")
    if nf:
        print(f"     truth noise floor {nf * 100:.2f}% -- errors below ~{nf * 200:.1f}% "
              f"are not resolvable here")
    print(f"{'truncation':>44} {'#diag':>6} {'rel err':>9}")
    rows = [(2, 0, 0, "star only (2-edge) [SHIPPED]"),
            (2, 1, 0, "GEN + pass-through  [E9b best ~10.6%]"),
            (3, 1, 0, "  + 3 sigma edges"),
            (4, 1, 0, "  + 4 sigma edges"),
            (2, 1, 1, "GEN + pass + KAPPA4 BLOCKS  <- G2"),
            (3, 1, 1, "  + 3 sigma edges"),
            (4, 1, 1, "  + 4 sigma edges")]
    for ms, m3, m4, tag in rows:
        t1 = time.time()
        dgs = enumerate_diagrams(ms, m3, m4)
        P = torch.zeros(n, n, n, device=DEV, dtype=F32)
        for blocks, k, coef in dgs:
            P += eval_slice(blocks, k, coef, gh, sl["C"], sl["K3"], sl["K4"])
        err = ((P - K3a).pow(2).sum() / den).sqrt().item()
        print(f"{tag:>44} {len(dgs):>6} {err * 100:>8.2f}%   ({time.time() - t1:.0f}s)",
              flush=True)
        del P


def noise_test(N, m, lay):
    """SHIPPABILITY #2: does the kappa4-block win SURVIVE realistic kappa4 error?

    G2 fed the blocks with SAMPLED (oracle) kappa4. The shipped closure gets them
    from G1's LEARNED slices: ppqq 28.5%, pppq 50.2%, pppp 31.0% held-out -- and
    ppqr (n^3) is not learned at all. If the win evaporates under that much error,
    G2 is an oracle-only result and does not ship.

    Corrupts each kappa4 family with iid relative noise at its G1 level and
    re-measures the 3-sigma + kappa4-block reconstruction."""
    n = 256
    rec = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                     weights_only=False)
    W = [rec["W"][l].to(DEV, F32) for l in range(32)]
    sl = stream_moments(W, lay, N, seed=3)
    K3a, C, K3, K4 = sl["K3a"], sl["C"], sl["K3"], sl["K4"]
    zmu = sl["zmu"].cpu().to(DT)
    gh, *_ = hermite_gains(zmu, torch.diagonal(C).cpu().to(DT), kmax=16)
    gh = [x.to(DEV, F32) for x in gh]
    den = K3a.pow(2).sum()
    dgs = enumerate_diagrams(3, 1, 1)
    g = torch.Generator(device="cpu").manual_seed(5)

    def run(K4x):
        P = torch.zeros(n, n, n, device=DEV, dtype=F32)
        for blocks, k, coef in dgs:
            P += eval_slice(blocks, k, coef, gh, C, K3, K4x)
        return ((P - K3a).pow(2).sum() / den).sqrt().item()

    print(f"\nG2-NOISE | layer {lay}, MLP {m}, N={N:,}, 3 sigma + kappa4 blocks")
    print(f"{'kappa4 quality':>34} {'rel err':>9}")
    print(f"{'ORACLE kappa4 (the G2 number)':>34} {run(K4) * 100:>8.2f}%")
    for lvl, tag in ((0.10, "10% noise"), (0.30, "30% noise (~G1 level)"),
                     (0.50, "50% noise"), (1.00, "100% noise")):
        K4n = K4 * (1.0 + lvl * torch.randn(K4.shape, generator=g).to(DEV))
        print(f"{tag:>34} {run(K4n) * 100:>8.2f}%", flush=True)
    Z = torch.zeros_like(K4)
    print(f"{'kappa4 blocks DROPPED':>34} {run(Z) * 100:>8.2f}%")
    print("  (no-kappa4 reference at this layer: L3 10.29 / L7 13.14 / L15 12.62%)")


def ablate(N, m, lay):
    """WHICH kappa4 blocks earn the 2.7x? THE SHIPPABILITY QUESTION.

    G2 (layer 3, N=4e6): 3 sigma edges + kappa4 blocks took the per-step kappa3
    error 10.29% -> 3.79%. But the kappa4 blocks include the (2,1,1) leg pattern,
    which needs **kappa4_ppqr -- an n^3 object**. G1 only learned n^2 SLICES
    (k4_iijj at 28.5%). So:

      A  no kappa4 blocks                      (baseline)
      B  kappa4 blocks with n^2/n^1 legs ONLY  -> (2,2,0),(3,1,0),(4,0,0)
                                                  needs k4_ppqq, k4_pppq, k4_pppp
                                                  == exactly what G1 can LEARN
      C  kappa4 blocks with (2,1,1) ONLY       -> needs k4_ppqr  (n^3)
      D  all kappa4 blocks                     (= the 3.79% cell)

    If B ~ D: SHIPPABLE with G1's learned n^2 slices, today.
    If C carries it: we need a cheap route to the n^3 kappa4_ppqr.
    """
    n = 256
    rec = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                     weights_only=False)
    W = [rec["W"][l].to(DEV, F32) for l in range(32)]
    t0 = time.time()
    sl = stream_moments(W, lay, N, seed=3)
    K3a = sl["K3a"]
    print(f"  streamed {N:,}: {time.time() - t0:.0f}s", flush=True)

    zmu = sl["zmu"].cpu().to(DT)
    gh, *_ = hermite_gains(zmu, torch.diagonal(sl["C"]).cpu().to(DT), kmax=16)
    gh = [x.to(DEV, F32) for x in gh]
    den = K3a.pow(2).sum()

    def k4_pattern(b):
        return tuple(sorted((c for c in b if c > 0), reverse=True))

    def keep(blocks, mode):
        pats = [k4_pattern(b) for size, b in blocks if size == 4]
        if mode == "A":
            return not pats
        if mode == "B":
            return all(p != (2, 1, 1) for p in pats)
        if mode == "C":
            return all(p == (2, 1, 1) for p in pats)
        return True

    print(f"\nG2-ABLATE | n=256, MLP {m}, layer {lay}, N={N:,}, 3 sigma edges")
    print(f"{'kappa4 blocks kept':>42} {'#diag':>6} {'rel err':>9}  {'ships?':>8}")
    rows = [("A", "NONE (baseline)", "n/a"),
            ("B", "n^2 legs only: (2,2,0),(3,1,0),(4,0,0)", "YES - G1"),
            ("C", "(2,1,1) only  -> needs n^3 k4_ppqr", "n^3 !"),
            ("D", "ALL", "mixed")]
    # TWO metrics, and the distinction is the whole ballgame:
    #   TENSOR err -- what TRANSPORT (carrying kappa3 history) consumes
    #   SLICE  err -- what the MOMENT UPDATE consumes (kappa3(a)_ppq only).
    # On the (p,p,q) plane a (2,1,1) kappa4 operand collapses to k4_pppq/k4_ppqq
    # -- n^2 objects G1 ALREADY LEARNS. So the slice may ship even if the tensor
    # needs the n^3 k4_ppqr.
    idx = torch.arange(n, device=DEV)
    truth_sl = K3a[idx, idx, :]
    den_sl = truth_sl[~torch.eye(n, dtype=torch.bool, device=DEV)].pow(2).sum().sqrt()
    offm = ~torch.eye(n, dtype=torch.bool, device=DEV)
    print(f"{'':>42} {'':>6} {'TENSOR':>8} {'SLICE':>8}")
    for mode, tag, ship in rows:
        dgs = [d for d in enumerate_diagrams(3, 1, 1) if keep(d[0], mode)]
        P = torch.zeros(n, n, n, device=DEV, dtype=F32)
        for blocks, k, coef in dgs:
            P += eval_slice(blocks, k, coef, gh, sl["C"], sl["K3"], sl["K4"])
        err = ((P - K3a).pow(2).sum() / den).sqrt().item()
        esl = ((P[idx, idx, :] - truth_sl)[offm].pow(2).sum().sqrt() / den_sl).item()
        print(f"{tag:>42} {len(dgs):>6} {err * 100:>7.2f}% {esl * 100:>7.2f}%  {ship:>8}",
              flush=True)
        del P


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--samples", type=float, default=1e6)
    ap.add_argument("--mlp", type=int, default=0)
    ap.add_argument("--mlps", type=int, default=1, help="repeat over this many MLPs")
    ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()
    if a.verify:
        verify(a.n)
    if a.measure:
        for mm in range(a.mlp, a.mlp + a.mlps):
            measure(int(a.samples), mm, a.layer, noise_check=(mm == a.mlp))
    if a.ablate:
        for mm in range(a.mlp, a.mlp + a.mlps):
            ablate(int(a.samples), mm, a.layer)
    if a.noise:
        noise_test(int(a.samples), a.mlp, a.layer)
