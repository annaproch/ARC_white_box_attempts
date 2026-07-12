"""D1 + D2 — VERIFY THE BACKBONE BEFORE BUILDING ON IT.

D1: what is the CORE (Tucker) carry's propagated kappa3_iij slice error, vs
    corpus truth, per layer?  I have been quoting "~25% at depth" from E13's
    table WITHOUT EVER MEASURING IT -- the same assumption-without-check that
    made the CP carry (structurally incapable of representing the star) waste a
    fitting run. The anchor's single most important input feature IS this slice.
    PASS ~25% => the anchor starts where G5 validated it. FAIL (>=50%) => stop.

D2: reconcile an inconsistency between two of MY OWN measurements:
      derisk_cp, uncompressed CP (star + pass-through):   24.2%
      G5, 3-sigma tensor carry:                           36.1%
    More diagrams gave a WORSE slice. That is either a real "partial fixes are
    harmful in a drifted state" effect (this project has seen it) or a bug.
    `exact=True` here is the uncompressed core -- a third, independent path.
    If it lands at ~24%, the CP number is corroborated and G5's is the outlier.

Uses the EXISTING VALIDATED propagator (k3prop.propagate_k3) -- no
reimplementation. Cost is CPU float64 by construction (MPS has no float64) but
tiny: per layer it is one eigh (n^3) + Tucker core ops (R^4). Seconds per MLP.
"""
from __future__ import annotations

import statistics as st
import sys
import time

import torch

from harness import CORPUS, DT
from k3prop import propagate_k3

LAY = [1, 3, 7, 15, 23, 31]
n = 256


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    configs = [("exact", None), ("R=64", 64), ("R=32", 32), ("R=16", 16)]
    res = {c: {l: [] for l in range(32)} for c, _ in configs}
    mse_f = {c: [] for c, _ in configs}
    off = ~torch.eye(n, dtype=torch.bool)

    print(f"D1/D2 | {n_mlps} MLPs | core-carry kappa3_iij slice error vs corpus truth",
          flush=True)
    t0 = time.time()
    for m in range(n_mlps):
        rec = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                         weights_only=False)
        Ws = [rec["W"][l].to(DT) for l in range(32)]
        truth = [rec["zk3"][l].to(DT) for l in range(32)]
        gt = rec["mu"][31].to(DT)                     # true final-layer mean
        # kappa4 slices from the corpus (stand-in for G1's learned map: it only
        # needs 10-20% accuracy per E6, and G1c measured 28.5% works)
        k4 = [{"k4_iiii": torch.zeros(n, dtype=DT),
               "k4_iijj": rec["zk4"][l].to(DT),
               "k4_iiij": torch.zeros(n, n, dtype=DT)} for l in range(32)]

        for cname, R in configs:
            mu, used = propagate_k3(Ws, R=(R or 32), k4=k4, exact=(R is None))
            for l in range(32):
                sl = used[l][0]
                e = ((sl - truth[l])[off].norm() / truth[l][off].norm()).item()
                res[cname][l].append(e)
            mse_f[cname].append(((mu - gt) ** 2).mean().item())
        print(f"  mlp {m}: {time.time()-t0:.0f}s", flush=True)

    print(f"\n{'carry':>8} " + " ".join(f"{'L'+str(l):>7}" for l in LAY)
          + f" {'mean(32)':>9} {'final MSE':>11}")
    for cname, _ in configs:
        row = " ".join(f"{st.mean(res[cname][l])*100:>6.1f}%" for l in LAY)
        allm = st.mean([st.mean(res[cname][l]) for l in range(32)])
        print(f"{cname:>8} {row} {allm*100:>8.1f}% {st.mean(mse_f[cname]):>11.3e}")

    print("\n  D1 PASS if the core carry is ~25% (the anchor's input quality that")
    print("     G5 validated: it reached 15.7% held-out from a 36% baseline).")
    print("  D2: uncompressed CP measured 24.2%; G5's 3-sigma tensor 36.1%.")
    print("     `exact` here is a THIRD independent path -- it arbitrates.")
    print("  Fusion bar: closure b^2 < 1.02e-5 beats v4 (final MSE column).")


if __name__ == "__main__":
    main()
