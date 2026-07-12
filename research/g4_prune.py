"""G4 — PRICE FIRST. How few diagrams buy G2's 3.79%?

G2 proved the physics (3-sigma + kappa4 blocks -> 3.79% per-step kappa3, vs
10.29% without). Priced DENSELY that is 1203% of budget -- unshippable, and
exactly the 4.9.2 mistake ("physically correct, 147% of budget").

BUT the dense n^3 cost is the WRONG COST MODEL:

  * The moment update consumes only the SLICE kappa3(a)_ppq. On the (p,p,q)
    plane EVERY n^3 operand collapses to an n^2 object we already carry:
        pass-through k3_pqr  -> k3(z)_ppq        (the carried slice)
        kappa4 (2,1,1)       -> k4_pppq / k4_ppqq (exactly G1's learned slices)
    => the slice of any diagram costs O(n^2), not O(n^3).
       3048 diagrams x n^2 x 32 layers = 6.4e9 = 2.4% of budget. AFFORDABLE.
  * The n^3/n^4 cost lives ONLY in the CARRY (transport of kappa3 history),
    which is already factored (lazy-CP: 0.60% of budget in the skeleton).

So the real questions are:
  (Q1) how many diagrams actually matter?  -> greedy importance ranking here
  (Q2) can the survivors' TRANSPORT be factored? -> next, only for the winners

This script answers Q1: rank diagrams by marginal contribution to the SLICE
error (what the update consumes) and plot error vs #kept. If ~25-50 diagrams
hold ~4-5%, the closure ships.
"""
from __future__ import annotations

import argparse
import time

import torch

from diagrams import enumerate_diagrams
from g2_k4_blocks import eval_slice, stream_moments
from harness import CORPUS, DT, hermite_gains

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256


def diagram_slices(dgs, gh, S, K3, K4):
    """(D, n, n) — each diagram's contribution to the kappa3(a)_ppq SLICE.
    Evaluated densely here (offline, once) purely to RANK them; the shipped
    version computes the slice directly in O(n^2)."""
    out = torch.empty(len(dgs), n, n, device=DEV, dtype=F32)
    idx = torch.arange(n, device=DEV)
    for d, (blocks, k, coef) in enumerate(dgs):
        T = eval_slice(blocks, k, coef, gh, S, K3, K4)
        out[d] = T[idx, idx, :]                      # the (p,p,q) plane
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=float, default=4e6)
    ap.add_argument("--mlp", type=int, default=0)
    ap.add_argument("--layer", type=int, default=7)
    a = ap.parse_args()
    N = int(a.samples)

    rec = torch.load(CORPUS / f"mlp_{200000 + a.mlp}.pt", map_location="cpu",
                     weights_only=False)
    W = [rec["W"][l].to(DEV, F32) for l in range(32)]
    t0 = time.time()
    sl = stream_moments(W, a.layer, N, seed=3)
    print(f"  streamed {N:,}: {time.time() - t0:.0f}s", flush=True)

    idx = torch.arange(n, device=DEV)
    truth = sl["K3a"][idx, idx, :]                   # TRUE kappa3(a)_ppq slice
    zmu = sl["zmu"].cpu().to(DT)
    gh, *_ = hermite_gains(zmu, torch.diagonal(sl["C"]).cpu().to(DT), kmax=16)
    gh = [x.to(DEV, F32) for x in gh]

    dgs = enumerate_diagrams(3, 1, 1)
    t0 = time.time()
    P = diagram_slices(dgs, gh, sl["C"], sl["K3"], sl["K4"])
    print(f"  {len(dgs)} diagram slices: {time.time() - t0:.0f}s", flush=True)

    off = ~torch.eye(n, dtype=torch.bool, device=DEV)
    den = truth[off].pow(2).sum().sqrt()

    # --- greedy: repeatedly add the diagram that most reduces the residual.
    # (Diagram slices are correlated, so magnitude-ranking is not enough.)
    resid = truth.clone()
    chosen, curve = [], []
    Pf = P.reshape(len(dgs), -1)
    mask = torch.ones(len(dgs), dtype=torch.bool, device=DEV)
    offf = off.reshape(-1)
    for step in range(60):
        # best single diagram to subtract, with optimal scalar gain
        num = (Pf[:, offf] @ resid[off])
        nrm = Pf[:, offf].pow(2).sum(1).clamp(min=1e-30)
        gain = num / nrm
        red = gain * num                              # reduction in ||resid||^2
        red[~mask] = -1e30
        best = int(torch.argmax(red).item())
        mask[best] = False
        chosen.append(best)
        resid = resid - gain[best] * P[best]
        err = (resid[off].pow(2).sum().sqrt() / den).item()
        curve.append(err)

    print(f"\nG4 | layer {a.layer}, MLP {a.mlp} | greedy diagram selection")
    print(f"  (dense all-3048 reference from G2: ~3.8-4.5%; "
          f"no-k4 reference: ~10-13%)")
    print(f"{'#diagrams':>10} {'slice err':>10}   {'cost @ n^2/layer':>18}")
    for k in (1, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60):
        if k <= len(curve):
            cost = 2 * k * n * n * 32 / 2.72e11 * 100
            print(f"{k:>10} {curve[k-1]*100:>9.2f}%   {cost:>17.2f}%")
    print("\n  BAR: <=25-50 diagrams at ~4-5% => the closure SHIPS (n^2 slice path).")
    print("  NOTE: gains are fitted per-diagram here (offline, universal by weight")
    print("  exchangeability) -- that is legitimate and free at inference.")


if __name__ == "__main__":
    main()
