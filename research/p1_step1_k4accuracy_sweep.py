"""P1 Step 1 — κ4-input-accuracy sensitivity of κ4→κ3 generation.

Question: the κ4→κ3 generation diagrams reduce postact-κ3-slice error ~80% when
fed TRUE κ4. A cheap 27-feature predictor gives κ4 at ~13.5% rel-err / cos 0.99.
Does that 13.5%-accurate κ4 preserve the gain, or does the generation amplify
the κ4 error (as the recurrent path did)?

Method: reuse the collapsed-diagram evaluator (exp_k4_to_k3_slice). For each
MLP/layer, degrade the true κ4 slices (k4_iijj, k4_iiij, k4_iiii) by adding
random noise sized to a target relative Frobenius error e, then measure the
resulting postact-κ3-slice off-diagonal error. cos-0.99 @ 13.5% is consistent
with a purely-perpendicular error of that size in high dim (cos≈1-e²/2), so
random additive noise is a faithful predictor model. Averaged over noise draws.

Arms per layer:
  true_k4   : e=0        (ceiling — reproduces the 80% reduction)
  e=0.135   : predictor-accuracy κ4
  e=0.25    : own-generation-quality κ4 (for reference)
  e=0.50
  no_k4     : k3_s3 plan (κ4 dropped entirely = the un-generated baseline)

Gate: at e=0.135, retain >=~85% of the (no_k4 - true_k4) gain, i.e. the learned
κ4 slice error must sit close to the true-κ4 floor and far below no-κ4.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ARC = Path("/Users/lisek/Projects/ARC_white_box")
sys.path.insert(0, str(ARC / "research"))
sys.path.insert(0, "/Users/lisek/Projects/whest-starterkit/research")
from exp_k4_to_k3_slice import (  # noqa: E402
    collapsed_plan, eval_plan, load_state, DEV, N, N_TRAIN,
)

SEEDS = list(range(200000, 200024))
LAYERS = (3, 7, 15, 23, 30)
K4_KEYS = ("K4m", "K4t", "k4d")     # the three κ4 slices the plan consumes
LEVELS = (0.0, 0.135, 0.25, 0.50)
NOISE_DRAWS = 4
torch.manual_seed(0)


def degrade(t, e, gen):
    """Return t + noise with ||noise||/||t|| == e (matched-norm random dir)."""
    if e == 0.0:
        return t
    g = torch.randn(t.shape, generator=gen, device=t.device, dtype=t.dtype)
    g = g * (t.norm() / g.norm().clamp_min(1e-30)) * e
    return t + g


def main():
    eye = torch.eye(N, dtype=torch.bool, device=DEV)
    plan_k3k4 = collapsed_plan(3, 1, 1)   # k3k4_s3: full generation incl. κ4
    plan_k3 = collapsed_plan(3, 1, 0)     # k3_s3: κ4 dropped (baseline)
    print(f"device {DEV}  plan_k3k4={len(plan_k3k4)} plan_k3={len(plan_k3)}")
    gen = torch.Generator(device=DEV)
    t0 = time.time()

    for layer in LAYERS:
        # results[level] = list over seeds of rel-err (noise-averaged)
        res = {lv: [] for lv in LEVELS}
        res["no_k4"] = []
        for seed in SEEDS:
            state, gains, target = load_state(seed, layer)
            den = target[~eye].pow(2).sum().sqrt().clamp_min(1e-30)

            # no-κ4 baseline (deterministic)
            pred0 = eval_plan(plan_k3, gains, state)
            res["no_k4"].append(
                float((pred0[~eye] - target[~eye]).pow(2).sum().sqrt() / den))

            for lv in LEVELS:
                errs = []
                draws = 1 if lv == 0.0 else NOISE_DRAWS
                for d in range(draws):
                    gen.manual_seed(1000 * seed + 7 * layer + d)
                    st = dict(state)
                    for k in K4_KEYS:
                        st[k] = degrade(state[k], lv, gen)
                    pred = eval_plan(plan_k3k4, gains, st)
                    errs.append(
                        float((pred[~eye] - target[~eye]).pow(2).sum().sqrt() / den))
                res[lv].append(float(np.mean(errs)))
        report(layer, res)
    print(f"\nelapsed {time.time()-t0:.1f}s")


def report(layer, res):
    def hold(v):
        return np.asarray(v)[N_TRAIN:]
    tk = hold(res[0.0]).mean()
    nk = hold(res["no_k4"]).mean()
    gain = nk - tk  # error reduction the true κ4 buys
    print(f"\nL{layer}  off-diag rel-err (hold 8 MLPs);  "
          f"true-κ4 {tk*100:.2f}%  no-κ4 {nk*100:.2f}%  "
          f"gain {gain*100:.2f}pp")
    print(f"  {'level':>10} {'rel_err':>9} {'retained_gain':>14}")
    for lv in LEVELS:
        h = hold(res[lv]).mean()
        retained = (nk - h) / gain if gain > 1e-9 else float('nan')
        tag = " <- predictor" if lv == 0.135 else ""
        print(f"  {lv:>10} {h*100:>8.2f}% {retained*100:>13.1f}%{tag}")
    print(f"  {'no_k4':>10} {nk*100:>8.2f}% {0.0:>13.1f}%")


if __name__ == "__main__":
    main()
