"""Sweep every v5 config on mini, one at a time, with clean artifacts.

WHY THIS EXISTS: v5.npz was written EIGHT times tonight, four of them by
np.savez round-tripping a file that flopscope.Module.save() had produced. Two
runs with (nominally) identical meta gave 9.17e-6 @10.13% util and 3.52e-4
@4.81% util -- i.e. the artifact state was NOT what I thought it was, and every
score attributed to a config is therefore unreliable.

Rules enforced here:
  * every config is written by flopscope.Module.save() ONLY -- never np.savez
  * one file per config, md5-logged, never edited in place
  * the estimator PRINTS what it loaded (md5 + flags) on every run
  * results are recorded against the md5 that actually ran

Usage:  uv run python sweep_configs.py --write     # build all config npz files
        uv run python sweep_configs.py --run       # run them all on mini
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import flopscope
import flopscope.numpy as fnp

import closure

HERE = Path(__file__).resolve().parent
COEFS = HERE.parent / "research" / "anchor_coefs.npy"

# name -> [rank, rank4, n_qmc, w_qmc, anchor_on, k4_on]
CONFIGS = {
    "A_closure_only":       [32, 8,    0, 0.0, 0, 1],   # pure closure, no anchor
    "B_closure_anchor":     [32, 8,    0, 0.0, 1, 1],   # pure closure + anchor
    "C_closure_qmc2500":    [32, 8, 2500, 0.5, 0, 1],   # closure + QMC fusion
    "D_closure_qmc4000":    [32, 8, 4000, 0.5, 1, 1],   # "staged"
    "E_closure_nok4":       [32, 8,    0, 0.0, 0, 0],   # no kappa4 core
    "F_qmc_only":           [32, 8, 6400, 1.0, 0, 1],   # QMC only (control!)
}


class V5Params(flopscope.Module):
    def __init__(self) -> None:
        self.theta = fnp.zeros((1, 1))
        self.anchor = fnp.zeros((1, 1))
        self.q0 = fnp.zeros((1, 1))
        self.meta = fnp.zeros((6,))


def write_all():
    C = np.load(COEFS).astype(np.float32)          # (32, 42) A1 anchor coefficients
    rng = np.random.default_rng(0)
    for name, meta in CONFIGS.items():
        p = V5Params()
        p.theta = fnp.asarray(np.zeros((32, closure.N_PARAMS), dtype=np.float32))
        p.anchor = fnp.asarray(C)
        p.q0 = fnp.asarray((rng.standard_normal((256, 32)) / 16).astype(np.float32))
        p.meta = fnp.asarray(np.asarray(meta, dtype=np.float32))
        f = HERE / f"cfg_{name}.npz"
        p.save(str(f))                              # Module.save ONLY
        h = hashlib.md5(f.read_bytes()).hexdigest()[:8]
        print(f"  {name:<22} md5={h}  meta={meta}")


def run_all():
    results = []
    for name in CONFIGS:
        src = HERE / f"cfg_{name}.npz"
        shutil.copy(src, HERE / "v5.npz")
        h = hashlib.md5((HERE / "v5.npz").read_bytes()).hexdigest()[:8]
        print(f"\n=== {name}  (md5 {h}) ===", flush=True)
        out = subprocess.run(
            ["uv", "run", "whest", "run", "--estimator", "estimator.py",
             "--dataset", "hf://aicrowd/arc-whestbench-public-2026@v1-phase1",
             "--split", "mini", "--runner", "subprocess"],
            cwd=HERE, capture_output=True, text=True, timeout=1800)
        txt = out.stdout + out.stderr
        loaded = re.search(r"\[v5\] .*", txt)
        score = re.search(r"Adjusted Final-Layer Score\s+([\d.e+-]+)", txt)
        raw = re.search(r"Raw Final-Layer MSE \[final_layer_mse\]\s+([\d.e+-]+)", txt)
        util = re.search(r"Mean Compute Utilization\s+([\d.]+)", txt)
        fail = re.search(r"Failed MLPs \[n_failed_mlps\]\s+(\d+)", txt)
        r = dict(name=name, md5=h,
                 loaded=(loaded.group(0) if loaded else "NO PROVENANCE LINE"),
                 score=score.group(1) if score else "?",
                 raw=raw.group(1) if raw else "?",
                 util=util.group(1) if util else "?",
                 failed=fail.group(1) if fail else "?")
        results.append(r)
        print(f"  loaded : {r['loaded']}")
        print(f"  score {r['score']} | raw {r['raw']} | util {r['util']} | "
              f"failed {r['failed']}", flush=True)

    print("\n\n=== SUMMARY (v4 reference: score 3.66e-7, raw 3.66e-6) ===")
    print(f"{'config':<22} {'md5':>8} {'score':>10} {'raw MSE':>10} {'util':>8} {'fail':>5}")
    for r in results:
        print(f"{r['name']:<22} {r['md5']:>8} {r['score']:>10} {r['raw']:>10} "
              f"{r['util']:>8} {r['failed']:>5}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.write:
        write_all()
    if a.run:
        run_all()
