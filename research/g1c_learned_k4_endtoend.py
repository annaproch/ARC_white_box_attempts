"""G1c — END-TO-END value of LEARNED kappa4: oracle kappa3 + predicted kappa4
through the COMPLETE diagram formula (the E5 harness), on held-out MLPs.

G1 measured kappa4_iijj learnability at ~25-28% held-out (vs the 10-20% E6
spec) -- but E6's tolerance table used iid noise, and a regressor's error is
STRUCTURED. This measures what actually matters: plug the predictions into
the propagation and read the final-layer MSE.

Design (12 train / 12 held-out MLPs, slices sampled fresh at N=2e6 as in E5):
  models   per-layer ridge, G1 cell-C features (deg-4 local basis in
           (t_i,t_j,rho) + oracle-kappa3 features + star, with low-order
           interactions), one model per kappa4 family:
             iijj: y = k4_iijj/(s_i^2 s_j^2)     (n,n)
             iiij: y = k4_iiij/(s_i^3 s_j)       (n,n)
             iiii: y = k4_iiii/s_i^4             (n,)  small 1-D basis
  cells    1. plain K=2                          (baseline)
           2. oracle k3 + oracle k4   (FULL)     (ceiling, E5 ~5.6e-7)
           3. oracle k3 + learned iijj, oracle iiij/iiii   (attribution)
           4. oracle k3 + ALL kappa4 learned     <- THE G1c NUMBER
           5. oracle k3 + kappa4 dropped (zeros) (E6 says ~10x off ceiling)

Caveat carried: features come from the ORACLE z-state (consistent with the
oracle-kappa3 cell semantics). The fully-bootstrapped closure (features from
its own propagated state) is the next fidelity level, not this gate.
"""
from __future__ import annotations

import math
import statistics as st
import sys
import time

import torch

from harness import DT, load_corpus, mse, propagate, relu_step_full
from fastmoments import cached_cums
from g1_k4_learn import basis_exponents, star_slice, local_design

N_SAMP = int(__import__('os').environ.get('G1C_N', 8_000_000))

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256
DEG = 4
EXPS = basis_exponents(DEG)
LOW = [(a, b, c) for a, b, c in EXPS if a + b + c <= 2]
LOW_IDX = [EXPS.index(e) for e in LOW]
N_EXTRA = 6
P_PAIR = len(EXPS) + N_EXTRA * (1 + len(LOW))
P_DIAG = 11
LAM = 1e-6


# slices now come from fastmoments.cached_cums (GPU-resident, disk-cached):
# same math, ~50x faster, and every later gate reuses the same cache.


def pair_features(cums, Wl32, l):
    """(n^2, P_PAIR) design + norms, all on DEV float32. Oracle z-state."""
    zmu = cums[l]["mu"].to(DEV, F32)
    zcov = cums[l]["cov"].to(DEV, F32)
    zk3 = cums[l]["k3_iij"].to(DEV, F32)
    s = torch.sqrt(torch.clamp(torch.diagonal(zcov), min=1e-30))
    t = zmu / s
    rho = zcov / torch.outer(s, s)
    ti = t[:, None].expand(n, n)
    tj = t[None, :].expand(n, n)
    y3 = zk3 / (s[:, None] ** 2 * s[None, :])
    if l > 0:
        stf = star_slice(cums[l - 1]["cov"].to(DEV, F32), Wl32[l],
                         cums[l - 1]["mu"].to(DEV, F32))
        stf = stf / (s[:, None] ** 2 * s[None, :])
    else:
        stf = torch.zeros(n, n, device=DEV, dtype=F32)
    Xloc = local_design(ti, tj, rho, EXPS)
    low_cols = Xloc[:, LOW_IDX]
    blocks = [Xloc]
    y3t = y3.T
    for ex in (y3, y3t, y3 * y3t, y3 ** 2, y3t ** 2, stf):
        ev = ex.reshape(-1, 1)
        blocks.append(ev)
        blocks.append(ev * low_cols)
    X = torch.cat(blocks, 1)
    return X, s, t, y3


def diag_features(s, t, cums, l):
    """(n, P_DIAG) design for the kappa4_iiii model."""
    y3d = cums[l]["k3_iii"].to(DEV, F32) / s ** 3
    cols = [torch.ones_like(t), t, t ** 2, t ** 3, t ** 4,
            y3d, y3d ** 2, y3d * t, y3d * t ** 2, y3d ** 3, y3d ** 2 * t]
    return torch.stack(cols, 1)


def targets(cums, s, l):
    y_iijj = (cums[l]["k4_iijj"].to(DEV, F32)
              / (s[:, None] ** 2 * s[None, :] ** 2)).reshape(-1)
    y_iiij = (cums[l]["k4_iiij"].to(DEV, F32)
              / (s[:, None] ** 3 * s[None, :])).reshape(-1)
    y_iiii = cums[l]["k4_iiii"].to(DEV, F32) / s ** 4
    return y_iijj, y_iiij, y_iiii


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    torch.manual_seed(12345)
    print(f"G1c | {n_train} train / {n_test} HELD-OUT MLPs | N={N_SAMP:,} | "
          f"P_pair={P_PAIR} | device {DEV}", flush=True)

    n_mlps = n_train + n_test
    recs = load_corpus(n_mlps)
    W32 = [[r["W"][l].to(DEV, F32) for l in range(32)] for r in recs]
    W64 = [[r["W"][l] for l in range(32)] for r in recs]

    # one sampling pass for every MLP, cached to disk and shared by all cells
    CUMS, TRUTH = cached_cums(W64, [200_000 + m for m in range(n_mlps)], N_SAMP)

    # ---------------- phase 1: fit per-layer ridges on train MLPs ----------
    G = {k: [torch.zeros(p, p, dtype=torch.float64) for _ in range(32)]
         for k, p in (("iijj", P_PAIR), ("iiij", P_PAIR), ("iiii", P_DIAG))}
    b = {k: [torch.zeros(p, dtype=torch.float64) for _ in range(32)]
         for k, p in (("iijj", P_PAIR), ("iiij", P_PAIR), ("iiii", P_DIAG))}
    t0 = time.time()
    for m in range(n_train):
        cums = CUMS[m]
        for l in range(32):
            X, s, t, _ = pair_features(cums, W32[m], l)
            Xd = diag_features(s, t, cums, l)
            y_iijj, y_iiij, y_iiii = targets(cums, s, l)
            for key, Xk, yk in (("iijj", X, y_iijj), ("iiij", X, y_iiij),
                                ("iiii", Xd, y_iiii)):
                G[key][l] += (Xk.T @ Xk).cpu().double()
                b[key][l] += (Xk.T @ yk).cpu().double()
        print(f"  train mlp {m}: {time.time() - t0:.0f}s", flush=True)
    coefs = {k: [] for k in G}
    for k in G:
        for l in range(32):
            Gl = G[k][l] + LAM * torch.eye(G[k][l].shape[0], dtype=torch.float64) \
                 * torch.diagonal(G[k][l]).mean()
            coefs[k].append(torch.linalg.solve(Gl, b[k][l]))

    # ---------------- phase 2: held-out end-to-end ------------------------
    cells = {c: [] for c in ("k2", "oracle", "iijj_learned", "all_learned", "dropped")}
    sl_err = {k: [] for k in ("iijj", "iiij", "iiii")}
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)
    for m in range(n_train, n_train + n_test):
        cums, truth = CUMS[m], TRUTH[m]
        # predicted kappa4 slices per layer (+ held-out slice errors)
        preds = []
        for l in range(32):
            X, s, t, _ = pair_features(cums, W32[m], l)
            Xd = diag_features(s, t, cums, l)
            y_iijj, y_iiij, y_iiii = targets(cums, s, l)
            p_iijj = (X @ coefs["iijj"][l].to(DEV, F32)).reshape(n, n)
            p_iiij = (X @ coefs["iiij"][l].to(DEV, F32)).reshape(n, n)
            p_iiii = Xd @ coefs["iiii"][l].to(DEV, F32)
            for key, p, y in (("iijj", p_iijj, y_iijj.reshape(n, n)),
                              ("iiij", p_iiij, y_iiij.reshape(n, n))):
                sl_err[key].append(((p - y)[off].pow(2).sum()
                                    / y[off].pow(2).sum()).sqrt().item())
            sl_err["iiii"].append(((p_iiii - y_iiii).pow(2).sum()
                                   / y_iiii.pow(2).sum()).sqrt().item())
            preds.append({
                "k4_iijj": (p_iijj * (s[:, None] ** 2 * s[None, :] ** 2)).cpu().to(DT),
                "k4_iiij": (p_iiij * (s[:, None] ** 3 * s[None, :])).cpu().to(DT),
                "k4_iiii": (p_iiii * s ** 4).cpu().to(DT)})

        def run(cell):
            mu = torch.zeros(n, dtype=DT)
            C = torch.eye(n, dtype=DT)
            for l in range(32):
                c = dict(cums[l])
                if cell == "iijj_learned":
                    c["k4_iijj"] = preds[l]["k4_iijj"]
                elif cell == "all_learned":
                    c.update(preds[l])
                elif cell == "dropped":
                    c["k4_iijj"] = torch.zeros(n, n, dtype=DT)
                    c["k4_iiij"] = torch.zeros(n, n, dtype=DT)
                    c["k4_iiii"] = torch.zeros(n, dtype=DT)
                mu, C = relu_step_full(mu, C, W64[m][l], c)
            return mse(mu, truth)

        mu, _ = propagate(torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT), W64[m])
        cells["k2"].append(mse(mu, truth))
        for cell in ("oracle", "iijj_learned", "all_learned", "dropped"):
            cells[cell].append(run(cell))
        print(f"  test mlp {m}: K2 {cells['k2'][-1]:.2e} | "
              f"oracle {cells['oracle'][-1]:.2e} | "
              f"iijj-l {cells['iijj_learned'][-1]:.2e} | "
              f"ALL-l {cells['all_learned'][-1]:.2e} | "
              f"drop {cells['dropped'][-1]:.2e}   ({time.time() - t0:.0f}s)",
              flush=True)

    def sem(v):
        return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0

    print(f"\n=== G1c end-to-end final-layer MSE ({n_test} held-out MLPs) ===")
    for tag, key in (("plain K=2", "k2"),
                     ("oracle k3 + oracle k4 (ceiling)", "oracle"),
                     ("oracle k3 + LEARNED iijj (rest oracle)", "iijj_learned"),
                     ("oracle k3 + ALL kappa4 LEARNED", "all_learned"),
                     ("oracle k3 + kappa4 DROPPED", "dropped")):
        v = cells[key]
        print(f"  {tag:>42}: {st.mean(v):.3e} +- {sem(v):.1e}")
    print("\n  held-out slice errors (mean over layers): "
          + " | ".join(f"{k} {st.mean(v) * 100:.1f}%" for k, v in sl_err.items()))
    print("  score at the floor = MSE/10. Leaders' raw MSE ~8.2e-7; v4 fused "
          "needs closure b^2 < 1.02e-5.")


if __name__ == "__main__":
    main()
