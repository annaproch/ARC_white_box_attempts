"""GPU-resident sampled cumulant slices — the shared engine for the gates.

Replaces `e5_endtoend.true_slices` + `centred_slices`, which were the compute
wall (142 s/MLP): they moved every (chunk x 256) activation batch to CPU and
contracted it in FLOAT64 -- 4 N*n^2 matmuls per layer per chunk, x32 layers,
x2 full network passes. ~1.7e13 CPU float64 FLOPs per MLP, with the GPU idle.

Three changes:

1. **Contract on MPS in float32; accumulate (n,n) results in CPU float64.**
   The per-chunk transfer drops from N*n to n*n and the heavy work hits the
   GPU. Sums of <=250k float32 terms carry ~1e-4 relative error -- three
   orders below the 2-6% sampling-noise floor of the slices themselves
   (measured in G1b).

2. **Two passes -> one cheap + one heavy.** Pass 1 (forward only, no n^2
   contractions) gets mu; pass 2 accumulates CENTRED moments d = z - mu
   directly. This is both faster AND better conditioned than the old
   raw-moment route (no k4 = m4 - 3m2^2 cancellation at large mu).

3. **Batch M MLPs through bmm.** 128 GB unified memory: at M=4, chunk=200k
   the live tensors are ~5 GB. Independent input streams per MLP.

Public API:
    sampled_cums(W32_list, n_samp, seed) -> (cums_per_mlp, amu_per_mlp)
        cums[m][l] = dict(mu, cov, k3_iii, k3_iij, k4_iiii, k4_iijj, k4_iiij)
        amu[m]     = E[ReLU(z^(31))]  (the truth vector the gates score against)
    Cached to data/slice_cache/ keyed by (mlp seed, n_samp, sample seed).

Validate against the old slow path (same sample stream):
    python fastmoments.py --validate
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from harness import DT

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
N = 256
CACHE = Path(__file__).resolve().parent.parent / "data" / "slice_cache"

# 128 GB unified memory. Live tensors per chunk are 5 x (M*chunk*n) float32:
# at M=6, chunk=500k that is ~15 GB, well inside the machine, and it keeps the
# bmms large enough to saturate the GPU.
DEFAULT_CHUNK = 500_000
DEFAULT_MBATCH = 6
# Accumulate on-device in float32 and flush to CPU float64 every FLUSH chunks.
# The first version transferred 4 (M,n,n) blocks to CPU per LAYER per CHUNK --
# ~5k pipeline stalls per batch, which is what made it sync-bound (69 s/MLP).
# Flushing every 2M samples bounds float32 accumulator growth; the validate
# harness re-checks that this introduces no bias.
FLUSH_CHUNKS = 4


def _stack_weights(W32_list):
    """[(32 x (n,n) on DEV)] per MLP -> per-layer (M, n, n)."""
    M = len(W32_list)
    return [torch.stack([W32_list[m][l] for m in range(M)]) for l in range(32)], M


def _pass_mu(Wb, M, n_samp, seed, chunk):
    """Pass 1: per-layer E[z] (and E[ReLU(z)]) only -- forward matmuls, no n^2
    contractions. Cheap, and it lets pass 2 centre exactly."""
    s_z = [torch.zeros(M, N, dtype=F32, device=DEV) for _ in range(32)]
    s_a = [torch.zeros(M, N, dtype=F32, device=DEV) for _ in range(32)]
    acc_z = [torch.zeros(M, N, dtype=DT) for _ in range(32)]
    acc_a = [torch.zeros(M, N, dtype=DT) for _ in range(32)]
    done = 0
    ci = 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    while done < n_samp:
        b = min(chunk, n_samp - done)
        A = torch.randn(M, b, N, generator=g, dtype=F32).to(DEV)
        for l in range(32):
            Z = torch.bmm(A, Wb[l])
            A = torch.clamp(Z, min=0.0)
            s_z[l] += Z.sum(1)
            s_a[l] += A.sum(1)
        done += b
        ci += 1
        if ci % FLUSH_CHUNKS == 0 or done >= n_samp:
            for l in range(32):
                acc_z[l] += s_z[l].cpu().to(DT)
                acc_a[l] += s_a[l].cpu().to(DT)
                s_z[l].zero_()
                s_a[l].zero_()
    return ([s / n_samp for s in acc_z], [s / n_samp for s in acc_a])


def _pass_centred(Wb, M, mus, n_samp, seed, chunk):
    """Pass 2: centred moments of z at every layer. Contractions AND their
    accumulators live on the GPU; only a periodic flush crosses to CPU float64
    (see FLUSH_CHUNKS) -- this is what removes the per-layer sync stall."""
    dev2 = lambda: [torch.zeros(M, N, N, dtype=F32, device=DEV) for _ in range(32)]
    dev1 = lambda: [torch.zeros(M, N, dtype=F32, device=DEV) for _ in range(32)]
    cpu2 = lambda: [torch.zeros(M, N, N, dtype=DT) for _ in range(32)]
    cpu1 = lambda: [torch.zeros(M, N, dtype=DT) for _ in range(32)]
    g_M2, g_E21, g_E22, g_E31 = dev2(), dev2(), dev2(), dev2()
    g_m3d, g_m4d = dev1(), dev1()
    M2, E21, E22, E31 = cpu2(), cpu2(), cpu2(), cpu2()
    m3d, m4d = cpu1(), cpu1()
    mus_dev = [m.to(DEV, F32) for m in mus]          # (M, n) per layer
    done = 0
    ci = 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    while done < n_samp:
        b = min(chunk, n_samp - done)
        A = torch.randn(M, b, N, generator=g, dtype=F32).to(DEV)
        for l in range(32):
            Z = torch.bmm(A, Wb[l])
            A = torch.clamp(Z, min=0.0)
            d = Z - mus_dev[l][:, None, :]
            d2 = d * d
            d3 = d2 * d
            g_M2[l] += torch.bmm(d.transpose(1, 2), d)
            g_E21[l] += torch.bmm(d2.transpose(1, 2), d)
            g_E22[l] += torch.bmm(d2.transpose(1, 2), d2)
            g_E31[l] += torch.bmm(d3.transpose(1, 2), d)
            g_m3d[l] += d3.sum(1)
            g_m4d[l] += (d2 * d2).sum(1)
        done += b
        ci += 1
        if ci % FLUSH_CHUNKS == 0 or done >= n_samp:
            for l in range(32):
                for gacc, acc in ((g_M2, M2), (g_E21, E21), (g_E22, E22),
                                  (g_E31, E31), (g_m3d, m3d), (g_m4d, m4d)):
                    acc[l] += gacc[l].cpu().to(DT)
                    gacc[l].zero_()
    inv = 1.0 / n_samp
    return ([x * inv for x in M2], [x * inv for x in E21], [x * inv for x in E22],
            [x * inv for x in E31], [x * inv for x in m3d], [x * inv for x in m4d])


def _assemble(mus, amus, M2, E21, E22, E31, m3d, m4d, M):
    """Central moments -> cumulant slices, per MLP per layer (CPU float64)."""
    out = []
    for m in range(M):
        cums = []
        for l in range(32):
            C = M2[l][m]
            v = torch.diagonal(C).clone()
            cums.append({
                "mu": mus[l][m],
                "cov": C,
                "k3_iii": m3d[l][m],
                "k3_iij": E21[l][m],
                "k4_iiii": m4d[l][m] - 3.0 * v ** 2,
                "k4_iijj": E22[l][m] - torch.outer(v, v) - 2.0 * C ** 2,
                "k4_iiij": E31[l][m] - 3.0 * v[:, None] * C,
            })
        out.append(cums)
    return out, [amus[31][m] for m in range(M)]


def sampled_cums(W32_list, n_samp, seed, chunk=DEFAULT_CHUNK):
    """Cumulant slices for a batch of MLPs. W32_list[m] = 32 (n,n) DEV float32."""
    Wb, M = _stack_weights(W32_list)
    mus, amus = _pass_mu(Wb, M, n_samp, seed, chunk)
    M2, E21, E22, E31, m3d, m4d = _pass_centred(Wb, M, mus, n_samp, seed + 1, chunk)
    return _assemble(mus, amus, M2, E21, E22, E31, m3d, m4d, M)


def cached_cums(W64_list, seeds, n_samp, sample_seed=7000,
                chunk=DEFAULT_CHUNK, mbatch=DEFAULT_MBATCH):
    """Disk-cached slices for a list of MLPs (by corpus seed). Returns
    (cums_per_mlp, truth_per_mlp). Re-used by every gate."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cums_all, truth_all = [None] * len(seeds), [None] * len(seeds)
    todo = []
    for i, sd in enumerate(seeds):
        p = CACHE / f"s{sd}_n{n_samp}_x{sample_seed}.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False)
            cums_all[i], truth_all[i] = d["cums"], d["truth"]
        else:
            todo.append(i)
    for start in range(0, len(todo), mbatch):
        idx = todo[start:start + mbatch]
        t0 = time.time()
        W32 = [[W64_list[i][l].to(DEV, F32) for l in range(32)] for i in idx]
        cums, truth = sampled_cums(W32, n_samp, sample_seed + 17 * idx[0], chunk)
        for k, i in enumerate(idx):
            cums_all[i], truth_all[i] = cums[k], truth[k]
            torch.save({"cums": cums[k], "truth": truth[k]},
                       CACHE / f"s{seeds[i]}_n{n_samp}_x{sample_seed}.pt")
        print(f"  sampled MLPs {[seeds[i] for i in idx]} in {time.time() - t0:.0f}s "
              f"({(time.time() - t0) / len(idx):.0f}s/MLP)", flush=True)
    return cums_all, truth_all


# --------------------------------------------------------------- validate ----
def _validate(n_samp=500_000):
    """Same weights, same math, fast vs the old CPU-float64 path.

    Sample streams differ (the old path draws (b,n), the new (M,b,n)), so the
    comparison floor is SAMPLING noise at this N, not float precision. We
    therefore also run the fast path twice with different seeds to get that
    floor and check fast-vs-slow sits at it -- i.e. no BIAS was introduced.
    """
    from harness import load_corpus
    from e5_endtoend import true_slices, centred_slices

    rec = load_corpus(1)[0]
    W32 = [rec["W"][l].to(DEV, F32) for l in range(32)]

    t0 = time.time()
    S = true_slices(W32, n_samp, 4242)
    E22s, E31s = centred_slices(W32, n_samp, 4242, S)
    t_slow = time.time() - t0
    slow = []
    for l in range(32):
        v, C = S[l]["v"], S[l]["cov"]
        slow.append({"cov": C, "k3_iij": S[l]["k3_iij"], "k3_iii": S[l]["k3_iii"],
                     "k4_iiii": S[l]["k4_iiii"],
                     "k4_iijj": E22s[l] - torch.outer(v, v) - 2 * C ** 2,
                     "k4_iiij": E31s[l] - 3 * v[:, None] * C})

    t0 = time.time()
    fastA, _ = sampled_cums([W32], n_samp, 4242)
    t_fast = time.time() - t0
    fastB, _ = sampled_cums([W32], n_samp, 999)     # independent stream
    fastA, fastB = fastA[0], fastB[0]

    keys = ["cov", "k3_iij", "k3_iii", "k4_iiii", "k4_iijj", "k4_iiij"]
    print(f"\nvalidate | N={n_samp:,} | slow {t_slow:.0f}s vs fast {t_fast:.0f}s "
          f"({t_slow / t_fast:.0f}x)")
    print(f"{'':>10} " + " ".join(f"{'L' + str(l):>16}" for l in (1, 15, 31)))
    print("  (each cell: fast-vs-slow rel err  /  fast-vs-fast noise floor)")
    for k in keys:
        cells = []
        for l in (1, 15, 31):
            a, b, c = fastA[l][k], slow[l][k], fastB[l][k]
            e_fs = ((a - b).pow(2).sum() / b.pow(2).sum()).sqrt().item()
            e_ff = ((a - c).pow(2).sum() / c.pow(2).sum()).sqrt().item()
            cells.append(f"{e_fs * 100:6.2f}% /{e_ff * 100:6.2f}%")
        print(f"{k:>10} " + " ".join(f"{c:>16}" for c in cells))
    print("\n  PASS if fast-vs-slow ~= fast-vs-fast (same order): the fast path")
    print("  differs from the slow one only by its sample stream -- no bias.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--samples", type=int, default=500_000)
    a = ap.parse_args()
    if a.validate:
        _validate(a.samples)
