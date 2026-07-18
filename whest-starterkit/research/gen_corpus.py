"""Corpus generator: training MLPs + sampled ground truth for the theta refit.

Each output file data/corpus/seed{S:05d}.npz is SELF-CONTAINED and ATOMIC
(written to .tmp then os.replace -> a kill at any moment never leaves a
corrupt or partial file). Contents:

  seed        ()        build_mlp seed -- weights are NOT stored; regenerate
                        with local_engine.build_mlp(width, depth, seed).
  w0_check    ()        float64 sum of W_0 as an RNG-drift tripwire: assert
                        against the regenerated weights before training.
  gt_mean     (d, n)    per-layer E[h]      float64  <- refit targets
  gt_m2       (d, n)    per-layer E[h^2]    float64  (variance targets/diags)
  zL_mean     (n,)      final-layer PRE-activation mean                }  slice /
  zL_m2       (n,)      ... second moment                              }  Edgeworth
  zL_m3       (n,)      ... third moment                               }  diagnostics
  n_samples   ()        MC sample count N (per-unit GT noise MSE ~ var/N)
  gt_seed     ()        RNG seed of the sampling stream (10_000_000 + seed)
  width, depth ()

Resume: existing files are skipped -> re-running continues where it stopped.
Stop:   touch data/corpus/STOP  (checked between MLPs; in-flight MLP finishes
        and is saved) -- or just kill the process (atomic writes make it safe).

Defaults: seeds 1000..1499 (disjoint from the 0..63 range used so far),
N=4e6 (per-unit GT noise ~7e-8, adequate for fitting toward raw ~1e-6).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from local_engine import build_mlp  # noqa: E402  (their weight convention: x @ W)


def gt_for_mlp(weights, n, depth, N, batch, gt_seed):
    """Sampled per-layer means/m2 + final-layer pre-activation moments.

    float32 matmuls (BLAS), float64 accumulators.
    """
    rng = np.random.default_rng(gt_seed)
    sum_h = np.zeros((depth, n)); sum_h2 = np.zeros((depth, n))
    sum_z = np.zeros(n); sum_z2 = np.zeros(n); sum_z3 = np.zeros(n)
    done = 0
    with np.errstate(all="ignore"):          # macOS Accelerate: spurious warnings
        while done < N:
            b = min(batch, N - done)
            x = rng.standard_normal((b, n)).astype(np.float32)
            for li, W in enumerate(weights):
                z = x @ W
                if li == depth - 1:
                    zd = z.astype(np.float64)
                    sum_z += zd.sum(0)
                    sum_z2 += (zd * zd).sum(0)
                    sum_z3 += (zd * zd * zd).sum(0)
                x = np.maximum(z, 0.0)
                xd = x.astype(np.float64)
                sum_h[li] += xd.sum(0)
                sum_h2[li] += (xd * xd).sum(0)
            done += b
    return (sum_h / N, sum_h2 / N, sum_z / N, sum_z2 / N, sum_z3 / N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--out", type=str, default=str(REPO / "data" / "corpus"))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"
    log = out / "progress.log"

    todo = [s for s in range(a.start, a.start + a.count)
            if not (out / f"seed{s:05d}.npz").exists()]
    print(f"[gen_corpus] {len(todo)}/{a.count} to do "
          f"(N={a.n:,}, out={out})", flush=True)

    t0 = time.time(); done_now = 0
    for s in todo:
        if stop_file.exists():
            print(f"[gen_corpus] STOP file found -- exiting cleanly "
                  f"({done_now} written this run).", flush=True)
            return
        t1 = time.time()
        mlp = build_mlp(a.width, a.depth, seed=s)
        weights = [np.asarray(w).astype(np.float32) for w in mlp.weights]
        gm, gm2, zm, zm2, zm3 = gt_for_mlp(
            weights, a.width, a.depth, a.n, a.batch, gt_seed=10_000_000 + s)
        # tmp must END in .npz or numpy appends the suffix behind our back
        tmp = out / f"seed{s:05d}.tmp.npz"
        final = out / f"seed{s:05d}.npz"
        np.savez_compressed(
            tmp, seed=s, width=a.width, depth=a.depth,
            w0_check=np.float64(weights[0].sum()),
            gt_mean=gm, gt_m2=gm2, zL_mean=zm, zL_m2=zm2, zL_m3=zm3,
            n_samples=a.n, gt_seed=10_000_000 + s)
        os.replace(tmp, final)               # atomic: never a partial file
        done_now += 1
        dt = time.time() - t1
        rate = (time.time() - t0) / done_now
        eta_h = rate * (len(todo) - done_now) / 3600
        line = (f"seed {s}  {dt:6.1f}s  done {done_now}/{len(todo)}  "
                f"ETA {eta_h:.1f}h")
        print("[gen_corpus] " + line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")
    print(f"[gen_corpus] complete: {done_now} MLPs in "
          f"{(time.time()-t0)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
