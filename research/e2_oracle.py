"""E2 — the decision experiment. Properly-posed oracle (STRATEGY.md §4.3).

Two fully-consistent measurements. Neither ever forms a mixed state (the old
oracle injected true Sigma with a propagated mu -- see the audit).

(A) RESTART CURVE  E_G(l): start from the TRUE state at layer l, run the pure
    Gaussian (K=2) closure for the remaining layers, MSE vs true final mean.
      E_G(-1) = plain K=2 baseline   (settles the 2.8x baseline drift)
      E_G(30) = ONE Gaussian step from truth = THE K=2 REPRESENTATION FLOOR
                -> the best any K=2-class method can EVER do.  Is it ~1e-6?
                   Is that where pluto (9.3e-7) lives?

(B) ONE-STEP ERRORS: from the true state at layer l, take ONE step with
    variant G / G+k3 / G+k3+k4 (true cumulant slices, so fully consistent)
    and compare mu,Sigma to the truth at l+1.  Prices what each correction
    removes per step, without building any propagator.

Also: noise sensitivity -- re-run (A) with the injected state perturbed by
synthetic noise of the corpus's own magnitude, to confirm the oracle numbers
are not artefacts of finite-sample truth.
"""
from __future__ import annotations

import math
import statistics as st

import torch

from harness import DT, load_corpus, mse, propagate, relu_step, self_test

N_MLPS = 24
RESTARTS = [-1, 0, 2, 4, 8, 12, 16, 20, 24, 26, 28, 29, 30]


def sem(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def restart_mse(rec, l, kmax=8):
    """Truth at layer l -> pure Gaussian closure to the end -> MSE vs truth."""
    W, mu, cov = rec["W"], rec["mu"], rec["cov"]
    n, L = mu.shape[1], mu.shape[0]
    if l < 0:
        m, C = torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT)   # exact input
    else:
        m, C = mu[l].clone(), cov[l].clone()
    if l >= L - 1:
        return mse(m, mu[L - 1])
    m, C = propagate(m, C, [W[j] for j in range(l + 1, L)], kmax)
    return mse(m, mu[L - 1])


def one_step(rec, l, use_k3, use_k4):
    """Truth at layer l -> ONE step -> compare mu,Sigma against truth at l+1."""
    W, mu, cov, zk3, zk4 = rec["W"], rec["mu"], rec["cov"], rec["zk3"], rec["zk4"]
    n = mu.shape[1]
    m0 = torch.zeros(n, dtype=DT) if l < 0 else mu[l]
    C0 = torch.eye(n, dtype=DT) if l < 0 else cov[l]
    k3 = zk3[l + 1] if use_k3 else None
    k4 = zk4[l + 1] if use_k4 else None
    m, C = relu_step(m0, C0, W[l + 1], 8, k3, k4)
    return mse(m, mu[l + 1]), mse(C, cov[l + 1])


def main():
    recs = load_corpus(N_MLPS)
    self_test(recs[0])
    L = recs[0]["mu"].shape[0]
    print(f"E2 | {len(recs)} MLPs | {recs[0]['n_samples']:,} GT samples | "
          f"depth {L} | self-tests PASSED\n")

    # ---------------- (A) restart curve
    print("=== (A) RESTART CURVE: truth at layer l -> Gaussian closure to end ===")
    print(f"{'start l':>8} {'steps':>6} {'final-layer MSE':>18} {'SEM':>10}  note")
    curve = {}
    for l in RESTARTS:
        v = [restart_mse(r, l) for r in recs]
        curve[l] = v
        note = ""
        if l == -1:
            note = "<- plain K=2 baseline (CANONICAL)"
        if l == 30:
            note = "<- K=2 REPRESENTATION FLOOR (1 step from truth)"
        print(f"{l:>8} {L-1-l:>6} {st.mean(v):>18.3e} {sem(v):>10.1e}  {note}")

    floor = st.mean(curve[30])
    base = st.mean(curve[-1])
    print(f"\n  plain K=2 / floor = {base/floor:.0f}x  -> headroom inside K=2")
    print(f"  pluto = 9.3e-7 ; floor = {floor:.2e} ; "
          f"pluto/floor = {9.3e-7/floor:.2f}")

    # ---------------- noise sensitivity of the injected truth
    print("\n=== noise check: perturb the injected state by the corpus's own "
          "sampling noise ===")
    g = torch.Generator().manual_seed(0)
    for l in (30, 24, 8):
        pert = []
        for r in recs:
            N = r["n_samples"]
            m, C = r["mu"][l].clone(), r["cov"][l].clone()
            d = torch.sqrt(torch.diagonal(C))
            m = m + torch.randn(m.shape, generator=g, dtype=DT) * d / math.sqrt(N)
            E = torch.randn(C.shape, generator=g, dtype=DT)
            E = (E + E.T) / math.sqrt(2)
            C = C + E * torch.outer(d, d) / math.sqrt(N)
            W = r["W"]
            if l < L - 1:
                m, C = propagate(m, C, [W[j] for j in range(l + 1, L)])
            pert.append(mse(m, r["mu"][L - 1]))
        print(f"  l={l:>2}: clean {st.mean(curve[l]):.3e} -> perturbed "
              f"{st.mean(pert):.3e}  ({st.mean(pert)/st.mean(curve[l]):.2f}x)")

    # ---------------- (C) THE CEILING: full propagation with ORACLE cumulants.
    # Start from the EXACT input (0, I) -- no injected state -- and run all 32
    # layers, feeding the TRUE k3/k4 slices of z into each ReLU step. This is
    # the ceiling of the whole third/fourth-order program: what we would score
    # if we could predict the cumulant slices perfectly. Caveat: the true slices
    # are paired with a (slightly) drifted mu,Sigma, so it is not exact -- but
    # the drift stays small precisely because the corrections work.
    print("\n=== (C) CEILING: propagate ALL 32 layers with oracle cumulants ===")
    n = recs[0]["mu"].shape[1]
    for name, u3, u4 in (("Gaussian (K=2)", False, False),
                         ("+ oracle k3", True, False),
                         ("+ oracle k3 + k4", True, True)):
        v = []
        for r in recs:
            m, C = torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT)
            for j in range(L):
                m, C = relu_step(m, C, r["W"][j], 8,
                                 r["zk3"][j] if u3 else None,
                                 r["zk4"][j] if u4 else None)
            v.append(mse(m, r["mu"][L - 1]))
        print(f"  {name:>20}: {st.mean(v):.3e} +- {sem(v):.1e}"
              f"   (pluto 9.3e-7, leader-beating target 5e-7)")

    # ---------------- (B) one-step errors
    print("\n=== (B) ONE-STEP from truth: does third/fourth-order info fix the "
          "step? ===")
    print(f"{'layer':>6} | {'mu: Gauss':>10} {'+k3':>10} {'+k3+k4':>10} | "
          f"{'cov: Gauss':>11} {'+k3':>10} {'+k3+k4':>10}")
    for l in (0, 3, 7, 15, 23, 29, 30):
        rows = [[one_step(r, l, a, b) for a, b in
                 ((False, False), (True, False), (True, True))] for r in recs]
        mus = [[x[i][0] for x in rows] for i in range(3)]
        cvs = [[x[i][1] for x in rows] for i in range(3)]
        print(f"{l:>6} | {st.mean(mus[0]):>10.2e} {st.mean(mus[1]):>10.2e} "
              f"{st.mean(mus[2]):>10.2e} | {st.mean(cvs[0]):>11.2e} "
              f"{st.mean(cvs[1]):>10.2e} {st.mean(cvs[2]):>10.2e}")

    # ---------------- S1: the skew bug (mean vs RMS over neurons)
    print("\n=== S1: per-neuron skew of z -- MEAN vs RMS (the suspected bug) ===")
    print(f"{'layer':>6} {'mean skew':>11} {'RMS skew':>10} {'|corr(gamma,t)|':>16} "
          f"{'<v1,mu_a>':>10} {'eig1frac':>9}")
    for l in (1, 7, 15, 23, 31):
        ms, rs, cr, al, ef = [], [], [], [], []
        for r in recs:
            zc, zk3 = r["zcov"][l], r["zk3"][l]
            s = torch.sqrt(torch.diagonal(zc))
            gam = torch.diagonal(zk3) / s ** 3
            t = r["zmu"][l] / s
            ms.append(gam.mean().item())
            rs.append(gam.pow(2).mean().sqrt().item())
            gc = gam - gam.mean()
            tc = t - t.mean()
            cr.append(abs((gc * tc).mean().item()
                          / (gc.std().item() * tc.std().item() + 1e-12)))
            Ca = r["cov"][l - 1] if l >= 1 else torch.eye(zc.shape[0], dtype=DT)
            ev, evec = torch.linalg.eigh(Ca)
            v1 = evec[:, -1]
            ma = r["mu"][l - 1] if l >= 1 else torch.zeros(zc.shape[0], dtype=DT)
            al.append(abs(torch.dot(v1, ma / (ma.norm() + 1e-12)).item()))
            ef.append((ev[-1] / ev.sum()).item())
        print(f"{l:>6} {st.mean(ms):>11.3f} {st.mean(rs):>10.3f} "
              f"{st.mean(cr):>16.3f} {st.mean(al):>10.3f} {st.mean(ef):>9.3f}")


if __name__ == "__main__":
    main()
