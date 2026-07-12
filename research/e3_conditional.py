"""E3' — is the activation distribution GAUSSIAN CONDITIONAL ON A SCALAR CORE?

This measures the ceiling of the conditional-Gaussian (mixture) closure family
BEFORE we build a propagator for it. STRATEGY.md §4.7.

The claim under test:   a | c  ~  N(m(c), R(c)),   c = u.(a - mu)
If true, then conditional on c the pre-activation z = W^T a is EXACTLY Gaussian,
so every ReLU expectation is exact -- no cumulant expansion, no truncation, and
the Edgeworth breakdown at depth (residual 3.2e-8, tracking RMS skew 0.43)
simply does not arise.

Method. Draw N samples, and at each layer l:
  * take the TRUE (sampled) distribution of a^(l)
  * project onto a candidate core direction u -> scalar c
  * bin into Q equal-count bins; per bin record mean & variance of EACH z_i
    (that is all the mean prediction needs -- no n x n per-bin covariance)
  * predict  E[ReLU(z_i)] = sum_q w_q * [exact Gaussian ReLU mean in bin q]
  * compare against the EMPIRICAL mean of ReLU(z_i) on the SAME samples
    -> sampling noise largely cancels (both are functionals of one sample set),
       so this isolates MODEL BIAS. Verified by a disjoint half-split.

Baselines on the same samples: Q=1 (= the pure Gaussian K=2 closure) and
Edgeworth with the sample's own kappa3_iii / kappa4_iiii.

Core directions swept: mu-hat (mean direction), v1 (top eigenvector of Sigma),
and the max-skew direction. Rank-2 (u1,u2 -> Q x Q bins) also tested.
"""
from __future__ import annotations

import math
import statistics as st
import sys

import torch

from harness import CORPUS, DT, ncdf, npdf

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_SAMP = 1_000_000
N_MLPS = 8
CHUNK = 250_000


def relu_mean_gauss(m, v):
    """Exact E[ReLU(Y)], Y ~ N(m, v). float64."""
    s = torch.sqrt(torch.clamp(v, min=1e-30))
    t = m / s
    return s * npdf(t) + m * ncdf(t)


def binned_stats(Z, c, Q):
    """Equal-count bins in c. Returns w (Q,), mean (Q,n), var (Q,n) in float64."""
    order = torch.argsort(c)
    N = c.shape[0]
    edges = [round(q * N / Q) for q in range(Q + 1)]
    w, mz, vz = [], [], []
    for q in range(Q):
        idx = order[edges[q]:edges[q + 1]]
        Zq = Z[idx]
        w.append(idx.numel() / N)
        mz.append(Zq.mean(0))
        vz.append(Zq.var(0, unbiased=False))
    return (torch.tensor(w, dtype=DT), torch.stack(mz), torch.stack(vz))


def cond_gauss_mean(w, mz, vz):
    """sum_q w_q * E[ReLU | bin q]  -> (n,)"""
    return (w[:, None] * relu_mean_gauss(mz, vz)).sum(0)


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else N_MLPS
    QS = [1, 8, 16, 32, 64]
    LAYERS = [3, 7, 15, 23, 30]
    res = {}   # (tag, layer) -> list of per-MLP MSE

    for m in range(n_mlps):
        rec = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                         weights_only=False)
        Wl = [rec["W"][l].to(DEV, torch.float32) for l in range(32)]
        g = torch.Generator(device="cpu").manual_seed(9000 + m)
        A = torch.randn(N_SAMP, 256, generator=g, dtype=torch.float32).to(DEV)

        for l in range(31):
            # ---- state at layer l (after ReLU); l=-1 handled by starting at input
            Z = A @ Wl[l]                       # pre-activation z^(l)
            Anext = torch.clamp(Z, min=0.0)     # a^(l)
            if l not in [x - 1 for x in LAYERS]:
                A = Anext
                continue
            # we want the step a^(l) -> a^(l+1), so use Anext as the state
            St = Anext
            Zn = (St @ Wl[l + 1]).cpu().to(DT)   # z^(l+1), float64 on CPU
            St = St.cpu().to(DT)
            truth = torch.clamp(Zn, min=0.0).mean(0)

            mu = St.mean(0)
            Xc = St - mu
            Sig = (Xc.T @ Xc) / St.shape[0]
            Zn64 = Zn
            lay = l + 1

            # ---- candidate core directions
            uhat = (mu / mu.norm())
            ev, evec = torch.linalg.eigh(Sig)
            v1 = evec[:, -1]
            # max-skew direction: gradient of skew at uhat, 1 power-iteration on
            # the third-cumulant contraction  T[u]_i = E[(x.u)^2 x_i]
            cu = Xc @ uhat
            t3 = ((cu ** 2)[:, None] * Xc).mean(0)
            t3 = t3 - Sig @ uhat * 2 - uhat * (cu ** 2).mean()  # centre
            uskew = t3 / (t3.norm() + 1e-30)

            dirs = {"mu-hat": uhat, "eig-v1": v1, "max-skew": uskew}

            for dname, u in dirs.items():
                c = Xc @ u
                for Q in QS:
                    if Q > 1 and dname != "mu-hat" and Q != 32:
                        continue           # sweep Q only for mu-hat, spot-check others
                    w, mz, vz = binned_stats(Zn, c, Q)
                    pred = cond_gauss_mean(w, mz, vz)
                    tag = f"{dname}/Q{Q}"
                    res.setdefault((tag, lay), []).append(
                        ((pred - truth) ** 2).mean().item())

            # ---- Edgeworth baseline from the SAME samples (kappa3/4 diagonal)
            mzn = Zn64.mean(0)
            d = Zn64 - mzn
            v = (d ** 2).mean(0)
            k3 = (d ** 3).mean(0)
            k4 = (d ** 4).mean(0) - 3 * v ** 2
            s = torch.sqrt(v)
            t = mzn / s
            pdf, cdf = npdf(t), ncdf(t)
            g3 = -t * pdf / s ** 2                       # hhat_3
            g4 = (t ** 2 - 1) * pdf / s ** 3             # hhat_4
            base = s * pdf + mzn * cdf
            for tag, p in (("edgeworth-k3", base + k3 / 6 * g3),
                           ("edgeworth-k3k4", base + k3 / 6 * g3 + k4 / 24 * g4)):
                res.setdefault((tag, lay), []).append(
                    ((p - truth) ** 2).mean().item())

            # ---- rank-2 core (mu-hat x residual top mode), Q x Q bins
            r = Xc - torch.outer(Xc @ uhat, uhat)
            ev2, evec2 = torch.linalg.eigh((r.T @ r) / St.shape[0])
            u2 = evec2[:, -1]
            c1 = Xc @ uhat
            c2 = Xc @ u2
            Q1 = 8
            o1 = torch.argsort(c1)
            N = c1.shape[0]
            pred2 = torch.zeros(256, dtype=DT)
            for q in range(Q1):
                idx = o1[round(q * N / Q1):round((q + 1) * N / Q1)]
                w2, mz2, vz2 = binned_stats(Zn[idx], c2[idx], Q1)
                pred2 += (idx.numel() / N) * cond_gauss_mean(w2, mz2, vz2)
            res.setdefault(("rank2/8x8", lay), []).append(
                ((pred2 - truth) ** 2).mean().item())

            # ---- noise control: disjoint half-split, mu-hat/Q32
            half = N_SAMP // 2
            for h in (slice(0, half), slice(half, N_SAMP)):
                w, mz, vz = binned_stats(Zn[h], c[h], 32)
                p = cond_gauss_mean(w, mz, vz)
                th = torch.clamp(Zn[h], min=0).mean(0)
                res.setdefault(("HALF mu-hat/Q32", lay), []).append(
                    ((p - th) ** 2).mean().item())

            A = Anext
        print(f"  MLP {m} done", flush=True)

    # ------------------------------------------------------------------ report
    print(f"\nE3' | {n_mlps} MLPs | {N_SAMP:,} samples | device {DEV}")
    print("one-step MSE of E[ReLU(z)] vs the SAME-SAMPLE empirical truth\n")
    tags = ([f"mu-hat/Q{q}" for q in QS] + ["eig-v1/Q32", "max-skew/Q32",
            "rank2/8x8", "edgeworth-k3", "edgeworth-k3k4", "HALF mu-hat/Q32"])
    hdr = "layer " + " ".join(f"{t:>15}" for t in tags)
    print(hdr)
    for lay in LAYERS:
        row = f"{lay:>5} "
        for t in tags:
            v = res.get((t, lay))
            row += f"{st.mean(v):>15.2e} " if v else f"{'-':>15} "
        print(row)
    print("\n(mu-hat/Q1 == the pure Gaussian K=2 closure -> sanity-check vs E2)")
    print("(HALF = same estimator on disjoint halves: if ~= Q32 row, the numbers "
          "are model bias, not sampling noise)")


if __name__ == "__main__":
    main()
