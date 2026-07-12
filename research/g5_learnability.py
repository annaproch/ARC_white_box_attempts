"""G5 — THE GATE: is the kappa3 PROPAGATION ERROR predictable from the state?

This decides whether to build the anchor at all, and it is designed to fail fast.

REFRAME (matters): `relu_step_full` -- the moment update -- reads ONLY the kappa3
SLICE, never the tensor. So a map that outputs the slice at each layer sets the
end-to-end MSE DIRECTLY; there is no sqrt(L) compounding in the quantity that
matters, and my earlier "geometric mean" law was the wrong frame. The tensor is
only a FEATURE (and the thing that generates the next slice).

So the whole question collapses to:

    can  f(propagated slice, t_i, t_j, rho_ij, star, layer)  beat the
    propagated slice's own ~25% error?

Warnings on the record (this is NOT a free bet):
  * E13 fitted an in-loop per-layer scalar gain: bought 2.7%, i.e. NOTHING.
    Its diagnosis: "the residual is a SHAPE error, which a scalar cannot touch."
  * E10: corr(propagated, true) = 0.97, but residual after an OPTIMAL GAIN is
    still 8-20%. So a gain-only correction caps there. The anchor must beat a
    scalar -- that is the unproven claim, and this script tests exactly it.
  * E12 (local features only, NO propagated kappa3): 27-40%. That is the
    "history-free" baseline the map must also beat.

CELLS (all per-layer ridge, HELD-OUT MLPs):
  0  propagated slice, as-is                       (baseline ~25%)
  1  propagated slice x optimal per-layer GAIN     (E10/E13's ceiling)
  2  local features only                           (E12 reproduction)
  3  local + star                                  (E12 best)
  4  local + star + PROPAGATED SLICE               <- THE ANCHOR
VERDICT: cell 4 >= 20% => the anchor adds nothing over a gain. DEAD, stop.
         cell 4 ~ 10-15% => end-to-end ~2-3.5e-6 => ~1.7-2.2x v4. BUILD.
         cell 4 <= 8%  => approaching leader territory.

Efficient by construction (memory/mac-gpu-perf-default): kappa3 tensor
propagation is 3 mode products (n^4) per layer on MPS float32 (~0.3 s/MLP);
generation uses the VERIFIED compiled diagram evaluator (one batched einsum,
not 156); only the tiny (P x P) Grams cross to CPU float64.
"""
from __future__ import annotations

import statistics as st
import sys
import time

import torch

from g2_k4_blocks import compile_diagrams, eval_compiled
from g1_k4_learn import basis_exponents, local_design, star_slice
from harness import CORPUS, DT, hermite_gains, relu_step

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256
DEG = 3
EXPS = basis_exponents(DEG)
LOW_IDX = [i for i, e in enumerate(EXPS) if sum(e) <= 2]
LAY = [1, 3, 7, 15, 23, 31]
CELLS = ["0 propagated (baseline)", "1 + optimal GAIN", "2 local only",
         "3 local + star", "4 local + star + PROPAGATED  <- ANCHOR"]


def propagate_with_tensor(rec, plan):
    """Self-consistent (mu, Sigma, kappa3-TENSOR) propagation with the complete
    diagram generation. Records, per layer, the propagated kappa3 slice and the
    features a shipped anchor could see. All n^3/n^4 work on MPS float32."""
    W32 = [rec["W"][l].to(DEV, F32) for l in range(32)]
    W64 = [rec["W"][l].to(DT) for l in range(32)]
    mu = torch.zeros(n, dtype=DT)
    C = torch.eye(n, dtype=DT)
    K3z = torch.zeros(n, n, n, device=DEV, dtype=F32)
    rows = []
    prev = None

    for l in range(32):
        zmu64 = mu @ W64[l]
        zcov64 = W64[l].T @ C @ W64[l]
        zmu = zmu64.to(DEV, F32)
        zcov = zcov64.to(DEV, F32)
        s = torch.sqrt(torch.clamp(torch.diagonal(zcov), min=1e-30))

        idx = torch.arange(n, device=DEV)
        prop_slice = K3z[idx, idx, :]                       # kappa3(z)_ppq
        stf = (star_slice(prev[0], W32[l], prev[1]) if prev is not None
               else torch.zeros(n, n, device=DEV, dtype=F32))
        rows.append({"zmu": zmu, "zcov": zcov, "s": s,
                     "prop": prop_slice.clone(), "star": stf,
                     "truth": rec["zk3"][l].to(DEV, F32)})

        # --- moment update from the propagated slices (+ corpus kappa4 slice,
        #     standing in for G1's learned kappa4 -- the shipped closure has it)
        mu, C = relu_step(mu, C, W64[l], 8,
                          prop_slice.cpu().to(DT),
                          rec["zk4"][l].to(DT))

        # --- generate kappa3(a) (3 sigma edges + pass-through) and transport
        gh, *_ = hermite_gains(zmu64, torch.diagonal(zcov64), kmax=12)
        gh = [x.to(DEV, F32) for x in gh]
        K3a = eval_compiled(plan, gh, zcov, K3z, None)
        prev = (zcov, zmu)
        if l < 31:
            Wn = W32[l + 1]
            T = torch.einsum("pqr,pa->aqr", K3a, Wn)
            T = torch.einsum("aqr,qb->abr", T, Wn)
            K3z = torch.einsum("abr,rc->abc", T, Wn)
    return rows


def design(r, cell):
    """(n^2, P) features for one layer, normalised. All on DEV float32."""
    s, zmu, zcov = r["s"], r["zmu"], r["zcov"]
    t = zmu / s
    rho = zcov / torch.outer(s, s)
    ti = t[:, None].expand(n, n)
    tj = t[None, :].expand(n, n)
    nrm = s[:, None] ** 2 * s[None, :]
    Xloc = local_design(ti, tj, rho, EXPS)
    if cell == 2:
        return Xloc, nrm
    blocks = [Xloc]
    extras = []
    if cell >= 3:
        extras.append(r["star"] / nrm)
    if cell >= 4:
        extras.append(r["prop"] / nrm)                    # THE anchor feature
    for ex in extras:
        ev = ex.reshape(-1, 1)
        blocks += [ev, ev * Xloc[:, LOW_IDX], ev * ev]
    return torch.cat(blocks, 1), nrm


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    n_train = n_mlps // 2
    plan = compile_diagrams(3, 1, 0)                       # 3 sigma + pass-through
    print(f"G5 | {n_mlps} MLPs ({n_train} train / {n_mlps - n_train} HELD OUT) | "
          f"device {DEV} | generation: {sum(len(d) for _, d in plan)} diagrams",
          flush=True)

    t0 = time.time()
    allrows = []
    for m in range(n_mlps):
        rec = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                         weights_only=False)
        allrows.append(propagate_with_tensor(rec, plan))
        if m == 0:
            print(f"  propagation: {time.time() - t0:.1f}s/MLP", flush=True)
    print(f"  all {n_mlps} propagated: {time.time() - t0:.0f}s", flush=True)

    off = ~torch.eye(n, dtype=torch.bool, device=DEV)
    res = {c: {l: [] for l in range(32)} for c in range(5)}

    for l in range(32):
        # cells 2-4: per-layer ridge
        coefs = {}
        for cell in (2, 3, 4):
            X0, _ = design(allrows[0][l], cell)
            P = X0.shape[1]
            G = torch.zeros(P, P, dtype=torch.float64)
            b = torch.zeros(P, dtype=torch.float64)
            for m in range(n_train):
                r = allrows[m][l]
                X, nrm = design(r, cell)
                y = (r["truth"] / nrm).reshape(-1)
                G += (X.T @ X).cpu().double()
                b += (X.T @ y).cpu().double()
            G += 1e-6 * torch.eye(P, dtype=torch.float64) * torch.diagonal(G).mean()
            coefs[cell] = torch.linalg.solve(G, b).to(DEV, F32)
        # cell 1: optimal per-layer scalar gain (fit on train)
        num = den = 0.0
        for m in range(n_train):
            r = allrows[m][l]
            num += (r["prop"][off] * r["truth"][off]).sum().item()
            den += (r["prop"][off] ** 2).sum().item()
        gain = num / max(den, 1e-30)

        for m in range(n_train, n_mlps):
            r = allrows[m][l]
            tr = r["truth"]
            dtr = tr[off].pow(2).sum().sqrt()
            res[0][l].append(((r["prop"] - tr)[off].pow(2).sum().sqrt() / dtr).item())
            res[1][l].append(((gain * r["prop"] - tr)[off].pow(2).sum().sqrt() / dtr).item())
            for cell in (2, 3, 4):
                X, nrm = design(r, cell)
                pred = (X @ coefs[cell]).reshape(n, n) * nrm
                res[cell][l].append(((pred - tr)[off].pow(2).sum().sqrt() / dtr).item())

    print(f"\n{'cell':<40} " + " ".join(f"{'L' + str(l):>7}" for l in LAY)
          + f" {'mean(32)':>9}")
    for c in range(5):
        row = " ".join(f"{st.mean(res[c][l]) * 100:>6.1f}%" for l in LAY)
        allm = st.mean([st.mean(res[c][l]) for l in range(32)])
        print(f"{CELLS[c]:<40} {row} {allm * 100:>8.1f}%")

    a4 = st.mean([st.mean(res[4][l]) for l in range(32)])
    print(f"\n  VERDICT: anchor error a = {a4*100:.1f}%")
    print("    >=20%  -> adds nothing over a gain. DEAD.")
    print("    10-15% -> end-to-end ~2-3.5e-6 -> fused ~1.7-2.2x v4. BUILD.")
    print("    <=8%   -> leader territory.")
    print(f"  (E12 local-only reference 27-40%; propagated-alone baseline above;")
    print("   E10 says gain-only caps at 8-20% residual, E13's in-loop gain gave 2.7%.)")


if __name__ == "__main__":
    main()
