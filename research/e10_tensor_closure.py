"""E10 — the complete closure, with kappa3 propagated as a FULL TENSOR.

Physics first, cost second. Offline we can afford what the grader cannot:
kappa3(a) as a full 256^3 tensor (67 MB fp32) and exact n^4 mode products on the
GPU. That measures what the COMPLETE diagram formula actually achieves
end-to-end, with no rank truncation and no contraction shortcuts to get wrong.

Only if this beats v4 (MSE 3.4e-6) do we invest in factoring the 156 diagram
contractions for flopscope.

Per layer:
    k3(z)  = k3(a) x1 W^T x2 W^T x3 W^T        (3 mode products, 3 n^4, GPU fp32)
    slices = k3(z)[i,i,j] and k3(z)[i,i,i]     (diagonal extractions)
    (mu,Sigma) <- relu_step_full                (complete r=2 formula, fp64)
    k3(a') = sum over the enumerated diagrams   (diagrams.py, verified)

kappa4 slices are ORACLE here (E6: they only need 10-20%, so they are not the
binding constraint -- this isolates kappa3).
"""
from __future__ import annotations

import math
import statistics as st
import sys
import time

import torch

from diagrams import enumerate_diagrams, eval_diagram
from e6_slice_spec import sample_slices
from harness import DT, hermite_gains, load_corpus, mse, relu_step_full

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256


def mode3(T, W):
    """T x1 W^T x2 W^T x3 W^T  -- three n^4 products, no n^6 intermediate."""
    T = torch.einsum("pqr,pi->iqr", T, W)
    T = torch.einsum("iqr,qj->ijr", T, W)
    return torch.einsum("ijr,rk->ijk", T, W)


def closure(Ws, k4, max_sigma=3, kmax=14, pmax=6):
    mu = torch.zeros(n, dtype=DT)
    cov = torch.eye(n, dtype=DT)
    K3a = torch.zeros(n, n, n, device=DEV, dtype=F32)   # kappa3 of the input = 0
    dgs = enumerate_diagrams(max_sigma, 1, 0)
    errs = []
    for l, W64 in enumerate(Ws):
        W = W64.to(DEV, F32)
        K3z = mode3(K3a, W)                              # exact, multilinear
        k3_iij = torch.einsum("iij->ij", K3z).cpu().to(DT)
        k3_iii = torch.einsum("iii->i", K3z).cpu().to(DT)

        cum = {"k3_iii": k3_iii, "k3_iij": k3_iij,
               "k4_iiii": k4[l]["k4_iiii"], "k4_iijj": k4[l]["k4_iijj"],
               "k4_iiij": k4[l]["k4_iiij"]}
        zcov = W64.T @ cov @ W64
        zmu = mu @ W64
        g, *_ = hermite_gains(zmu, torch.clamp(torch.diagonal(zcov), min=1e-30),
                              kmax=kmax)
        mu, cov = relu_step_full(mu, cov, W64, cum, pmax=pmax, kmax=kmax)

        # kappa3 of the new activations, from the enumerated diagram sum
        gd = [x.to(DEV, F32) for x in g]
        Sd = zcov.to(DEV, F32)
        K3a = torch.zeros(n, n, n, device=DEV, dtype=F32)
        for blocks, k, coef in dgs:
            K3a += eval_diagram(blocks, k, coef, gd, Sd, K3z, None)
        errs.append((k3_iij, K3z))
        del K3z
    return mu, errs


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sigmas = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                               else ["2", "3"])]
    recs = load_corpus(n_mlps)
    Wb = torch.stack([r["W"] for r in recs]).to(DEV, F32)
    t0 = time.time()
    oracle, truth = sample_slices(Wb, 2_000_000, 7000, want_truth=True)
    print(f"  oracle kappa4 + truth: {time.time()-t0:.0f}s\n", flush=True)

    off = ~torch.eye(n, dtype=torch.bool)
    LAY = [1, 3, 7, 15, 23, 31]
    print("=== complete-diagram closure, kappa3 as a full tensor ===")
    print(f"{'max_sig':>8} " + " ".join(f"{'L'+str(l):>8}" for l in LAY)
          + "   end-to-end MSE")
    for ms in sigmas:
        errs = {l: [] for l in LAY}
        cal = {l: [] for l in LAY}
        mses = []
        for m in range(n_mlps):
            Ws = [recs[m]["W"][l] for l in range(32)]
            k4 = [{k: oracle[m][l][k] for k in
                   ("k4_iiii", "k4_iijj", "k4_iiij")} for l in range(32)]
            t1 = time.time()
            mu, used = closure(Ws, k4, max_sigma=ms)
            mses.append(mse(mu, truth[m]))
            for l in LAY:
                p, t_ = used[l][0], oracle[m][l]["k3_iij"]
                pv, tv = p[off], t_[off]
                errs[l].append(((pv - tv).pow(2).sum()
                                / tv.pow(2).sum()).sqrt().item())
                # calibration headroom: optimal scalar gain + residual after it
                a = (pv * tv).sum() / (pv * pv).sum()
                corr = ((pv * tv).sum()
                        / (pv.norm() * tv.norm())).item()
                res = ((a * pv - tv).pow(2).sum() / tv.pow(2).sum()).sqrt().item()
                cal[l].append((a.item(), corr, res))
            print(f"    [mlp {m}: {time.time()-t1:.0f}s, "
                  f"MSE {mses[-1]:.2e}]", flush=True)
        sem = st.stdev(mses) / math.sqrt(len(mses)) if len(mses) > 1 else 0
        print(f"{ms:>8} " + " ".join(f"{st.mean(errs[l])*100:>7.1f}%" for l in LAY)
              + f"   {st.mean(mses):.3e} +- {sem:.1e}", flush=True)
        print(f"{'  gain':>8} " + " ".join(
            f"{st.mean(c[0] for c in cal[l]):>8.2f}" for l in LAY))
        print(f"{'  corr':>8} " + " ".join(
            f"{st.mean(c[1] for c in cal[l]):>8.4f}" for l in LAY))
        print(f"{'  resid':>8} " + " ".join(
            f"{st.mean(c[2] for c in cal[l])*100:>7.1f}%" for l in LAY)
            + "   <- rel err AFTER an optimal per-layer scalar gain", flush=True)

    print("\n  refs: v4 3.4e-6 | leaders 8.2e-7 | oracle-slice ceiling 5.59e-7")
    print("  E6 spec: kappa3 10% -> 2.07e-6 | 5% -> 8.8e-7")


if __name__ == "__main__":
    main()
