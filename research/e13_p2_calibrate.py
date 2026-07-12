"""E13 (P2) — per-layer gain calibration on the AFFORDABLE kappa3 propagator.

E10 showed the kappa3 error is a systematic MAGNITUDE bias, not a shape error
(corr 0.97-0.997; optimal gain climbing 0.93 -> 1.88, i.e. only ~1.02/layer,
COMPOUNDING). A single scalar per layer, applied INSIDE the recursion, attacks
it at source. Zero in-budget cost (32 numbers, shipped offline).

Run on the factored Tucker propagator (k3prop.K3State, R=64) -- the version that
actually fits the budget -- NOT the 147%-of-budget tensor one.

Fit: propagate all TRAIN MLPs in lockstep; at each layer fit ONE shared alpha_l
     minimising the kappa3-slice error, scale the state by it, continue.
Test: apply the fitted alphas to HELD-OUT MLPs. If they do not transfer, this is
      overfitting, not calibration.

kappa4 is ORACLE here (E6: 10-20% tolerance). Flagged: a shippable version still
needs an analytic kappa4 source.
"""
from __future__ import annotations

import math
import statistics as st
import sys
import time

import torch

from e6_slice_spec import sample_slices
from harness import DT, hermite_gains, load_corpus, mse, relu_step_full
from k3prop import K3State

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
n = 256
R = 64


def scale_state(stt, a):
    stt.G = stt.G * a
    if stt.gen is not None:
        h2, Sz, h1 = stt.gen
        stt.gen = (h2 * a, Sz, h1)


def run(recs, idx, k4, alphas=None, fit_truth=None):
    """Lockstep propagation over the given MLPs. If fit_truth is given, fit one
    shared alpha per layer (greedy, in-loop) and return them."""
    st_ = {m: K3State(n, R) for m in idx}
    mu = {m: torch.zeros(n, dtype=DT) for m in idx}
    cov = {m: torch.eye(n, dtype=DT) for m in idx}
    fitted, errs = [], {m: [] for m in idx}
    off = ~torch.eye(n, dtype=torch.bool)

    for l in range(32):
        sl = {}
        for m in idx:
            W = recs[m]["W"][l]
            sl[m] = st_[m].slices(W)

        if fit_truth is not None:
            num = den = 0.0
            for m in idx:
                p, t = sl[m][0][off], fit_truth[m][l][off]
                num += float((p * t).sum())
                den += float((p * p).sum())
            a = num / den if den > 0 else 1.0
            fitted.append(a)
        else:
            a = alphas[l] if alphas is not None else 1.0

        for m in idx:
            k3_iij = sl[m][0] * a
            k3_iii = sl[m][1] * a
            scale_state(st_[m], a)
            if fit_truth is None:
                t = k4[m][l]["_k3true"]
                errs[m].append(float(((k3_iij - t)[off].pow(2).sum()
                                      / t[off].pow(2).sum()).sqrt()))
            W = recs[m]["W"][l]
            cum = {"k3_iii": k3_iii, "k3_iij": k3_iij,
                   "k4_iiii": k4[m][l]["k4_iiii"],
                   "k4_iijj": k4[m][l]["k4_iijj"],
                   "k4_iiij": k4[m][l]["k4_iiij"]}
            zcov = W.T @ cov[m] @ W
            zmu = mu[m] @ W
            g, *_ = hermite_gains(zmu, torch.clamp(torch.diagonal(zcov),
                                                   min=1e-30), kmax=12)
            mu[m], cov[m] = relu_step_full(mu[m], cov[m], W, cum)
            ev, evec = torch.linalg.eigh(cov[m])
            st_[m].advance(W, g[1], g[2], zcov, evec[:, -R:].contiguous())
    return mu, fitted, errs


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    ntr = M // 2
    recs = load_corpus(M)
    Wb = torch.stack([r["W"] for r in recs]).to(DEV, torch.float32)
    t0 = time.time()
    oracle, truth = sample_slices(Wb, 2_000_000, 7000, want_truth=True)
    print(f"  oracle kappa4 + kappa3 truth: {time.time()-t0:.0f}s", flush=True)

    k4 = {m: [{**{k: oracle[m][l][k] for k in
                  ("k4_iiii", "k4_iijj", "k4_iiij")},
               "_k3true": oracle[m][l]["k3_iij"]} for l in range(32)]
          for m in range(M)}
    k3true = {m: [oracle[m][l]["k3_iij"] for l in range(32)] for m in range(M)}

    tr, te = list(range(ntr)), list(range(ntr, M))
    t0 = time.time()
    _, alphas, _ = run(recs, tr, k4, fit_truth=k3true)
    print(f"  fit {ntr} train MLPs: {time.time()-t0:.0f}s")
    print("  per-layer gains: " + " ".join(f"{a:.2f}" for a in alphas[:10])
          + f" ... (mean {st.mean(alphas):.2f})")

    LAY = [1, 3, 7, 15, 23, 31]
    print(f"\n=== HELD-OUT ({len(te)} MLPs) ===")
    print(f"{'':>12} " + " ".join(f"{'L'+str(l):>7}" for l in LAY) + "     MSE")
    for tag, al in (("uncalibrated", None), ("CALIBRATED", alphas)):
        mus, _, errs = run(recs, te, k4, alphas=al)
        ms = [mse(mus[m], truth[m]) for m in te]
        e = [st.mean(errs[m][l] for m in te) for l in LAY]
        sem = st.stdev(ms) / math.sqrt(len(ms)) if len(ms) > 1 else 0
        print(f"{tag:>12} " + " ".join(f"{x*100:>6.1f}%" for x in e)
              + f"   {st.mean(ms):.3e} +- {sem:.1e}", flush=True)

    print("\n  FUSION (STRATEGY 4.9.3): b^2 < 1.02e-5 beats v4 (3.4e-7);")
    print("     b^2 5e-6 -> 2.5e-7 (1.35x) | 3e-6 -> 1.9e-7 (1.8x) | 1e-6 -> 8.4e-8")
    print("  NB kappa4 is ORACLE here -- a shippable version still needs one.")


if __name__ == "__main__":
    main()
