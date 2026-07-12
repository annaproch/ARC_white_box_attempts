"""E8 — validate the kappa3 PROPAGATOR (k3prop.py) against sampled truth.

Two questions, both decisive for what we can ship:
  (1) per-layer relative error of the propagated kappa3 slices vs sampled truth
      -- spec is <=5% (E6: 5% -> 8.8e-7, 10% -> 2.0e-6)
  (2) end-to-end final-layer MSE with the REAL (propagated) kappa3, using an
      ORACLE kappa4 (which E6 says only needs 10-20%, so it is not the binding
      constraint -- this isolates kappa3).

Baselines on the same MLPs: v4 = 3.4e-6 MSE (score 3.4e-7); leaders 8.2e-7;
oracle-slice ceiling 5.59e-7.
"""
from __future__ import annotations

import math
import statistics as st
import sys
import time

import torch

from e6_slice_spec import sample_slices
from harness import DT, load_corpus, mse
from k3prop import propagate_k3

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_TRUTH = 2_000_000


def rel(pred, truth, off):
    d = (pred - truth)[off]
    return (d.pow(2).sum() / truth[off].pow(2).sum()).sqrt().item()


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ranks = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                              else ["16", "32", "64"])]
    recs = load_corpus(n_mlps)
    Wb = torch.stack([r["W"] for r in recs]).to(DEV, torch.float32)
    t0 = time.time()
    oracle, truth = sample_slices(Wb, N_TRUTH, 7000, want_truth=True)
    print(f"  sampled truth: {time.time()-t0:.0f}s\n", flush=True)

    n = 256
    off = ~torch.eye(n, dtype=torch.bool)
    LAY = [1, 3, 7, 15, 23, 31]

    print("=== (1) propagated kappa3 slice error vs sampled truth ===")
    print(f"{'R':>4} " + " ".join(f"{'L'+str(l):>9}" for l in LAY) + "   end-to-end MSE")
    for R in ranks + ["EXACT"]:
        errs = {l: [] for l in LAY}
        mses = []
        for m in range(n_mlps):
            Ws = [recs[m]["W"][l] for l in range(32)]
            k4 = [{k: oracle[m][l][k] for k in
                   ("k4_iiii", "k4_iijj", "k4_iiij")} for l in range(32)]
            ex = (R == "EXACT")
            mu, used = propagate_k3(Ws, R=(32 if ex else R), k4=k4, exact=ex)
            mses.append(mse(mu, truth[m]))
            for l in LAY:
                errs[l].append(rel(used[l][0], oracle[m][l]["k3_iij"], off))
        row = f"{str(R):>5} " + " ".join(
            f"{st.mean(errs[l])*100:>8.1f}%" for l in LAY)
        sem = (st.stdev(mses) / math.sqrt(len(mses))) if len(mses) > 1 else 0
        print(f"{row}   {st.mean(mses):.3e} +- {sem:.1e}", flush=True)

    print("\n  spec: kappa3 <=5% -> MSE 8.8e-7 | 10% -> 2.0e-6")
    print("  refs: oracle 5.59e-7 | leaders 8.2e-7 | v4 3.4e-6 | MC@floor 6.5e-6")
    print("  (kappa4 is ORACLE here -- E6 says it only needs 10-20%, so this "
          "isolates kappa3)")


if __name__ == "__main__":
    main()
