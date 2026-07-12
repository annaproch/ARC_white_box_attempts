"""G3 — ANCHORED kappa3 propagation: break the sqrt(L) compounding.

THE PROBLEM (measured, E10): our kappa3 per-step generation error (eps ~ 7-10%
after noise deflation) COMPOUNDS incoherently over 32 layers into ~47% at L31.
E6's spec needs <=5%. No amount of per-step diagram work fixes compounding --
it is the sqrt(L) that kills us, not eps.

THE IDEA: propagation is open-loop. Make it a FILTER. Each layer, blend the
propagated kappa3 against a HISTORY-FREE anchor whose error does NOT compound
(the E12 learned-local regressor: ~27-34% at depth, but fresh every layer):

    k3_used(l) = (1-alpha_l) * k3_propagated(l) + alpha_l * k3_anchor(l)

THE LAW (derive before measuring, then check):
    e^2  <-  (1-a)^2 (e^2 + eps^2) + a^2 * anch^2
  fixed point, minimised over alpha  =>   alpha* = eps/anch,  e = sqrt(eps*anch)

  i.e. the depth error becomes the GEOMETRIC MEAN of the per-step error and the
  anchor error, NOT sqrt(L)*eps. With eps=8.5%, anch=30%: e ~ 16% (vs 47%).
  Note e depends SYMMETRICALLY on eps and anch => improving the ANCHOR (cheap,
  offline, unlimited compute) is worth exactly as much as improving the
  DIAGRAMS (hard). That is the whole reason to run this.

WHAT THIS SCRIPT DOES
  * propagates kappa3 as a FULL (n,n,n) tensor (67 MB -- affordable offline;
    the shippable version uses the lazy-CP carry, priced in the skeleton)
  * generation per layer via the machine-verified diagram evaluator (G2)
  * anchor = per-layer ridge on local state + star (the E12 map, refit here)
  * sweeps alpha (incl. alpha=0 = today's open-loop propagation)
  * reports kappa3 relative error vs corpus truth at every layer, and the
    end-to-end final-layer MSE through the complete moment formula.

PASS: anchored kappa3 error at depth <= ~20% AND end-to-end MSE < 1.02e-5
(the fusion bar that beats v4). Then the skeleton port is a day.

Efficient by construction (see memory/mac-gpu-perf-default): everything n^3
stays on MPS float32; no per-sample CPU work; the corpus supplies the truth.
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import torch

from diagrams import enumerate_diagrams
from g2_k4_blocks import eval_slice
from g1_k4_learn import basis_exponents, local_design, star_slice
from harness import CORPUS, DT, hermite_gains, relu_step_full, mse

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256
DEG = 4
EXPS = basis_exponents(DEG)
LOW_IDX = [EXPS.index(e) for e in EXPS if sum(e) <= 2]
LAY_REPORT = [1, 3, 7, 15, 23, 31]


# ---------------------------------------------------------------- anchor ----
def anchor_design(zmu, zcov, star):
    """(n^2, P) features for the history-free kappa3_iij anchor (E12 cell B)."""
    s = torch.sqrt(torch.clamp(torch.diagonal(zcov), min=1e-30))
    t = zmu / s
    rho = zcov / torch.outer(s, s)
    ti = t[:, None].expand(n, n)
    tj = t[None, :].expand(n, n)
    Xloc = local_design(ti, tj, rho, EXPS)
    sv = (star / (s[:, None] ** 2 * s[None, :])).reshape(-1, 1)
    return torch.cat([Xloc, sv, sv * Xloc[:, LOW_IDX]], 1), s


def fit_anchor(recs, n_train, lam=1e-6):
    """Per-layer ridge: (local state, star) -> kappa3_iij/(s_i^2 s_j). Offline."""
    P = None
    G = b = None
    coefs = []
    for l in range(32):
        for m in range(n_train):
            zmu = recs[m]["zmu"][l].to(DEV, F32)
            zcov = recs[m]["zcov"][l].to(DEV, F32)
            k3 = recs[m]["zk3"][l].to(DEV, F32)
            if l > 0:
                stf = star_slice(recs[m]["zcov"][l - 1].to(DEV, F32),
                                 recs[m]["W"][l].to(DEV, F32),
                                 recs[m]["zmu"][l - 1].to(DEV, F32))
            else:
                stf = torch.zeros(n, n, device=DEV, dtype=F32)
            X, s = anchor_design(zmu, zcov, stf)
            y = (k3 / (s[:, None] ** 2 * s[None, :])).reshape(-1)
            if P is None:
                P = X.shape[1]
            if m == 0:
                G = torch.zeros(P, P, dtype=torch.float64)
                b = torch.zeros(P, dtype=torch.float64)
            G += (X.T @ X).cpu().double()
            b += (X.T @ y).cpu().double()
        G += lam * torch.eye(P, dtype=torch.float64) * torch.diagonal(G).mean()
        coefs.append(torch.linalg.solve(G, b).to(DEV, F32))
    return coefs


def anchor_predict(coef, zmu, zcov, star):
    X, s = anchor_design(zmu, zcov, star)
    y = (X @ coef).reshape(n, n)
    return y * (s[:, None] ** 2 * s[None, :])          # un-normalise -> k3_iij


# ------------------------------------------------------- k3 tensor closure ---
def generate_k3(gh, S, K3prev_z, K4z, max_sigma=3):
    """kappa3(a) at this layer from (Sigma, kappa3(z), kappa4(z)) via the
    machine-verified diagram sum (G2's evaluator)."""
    out = torch.zeros(n, n, n, device=DEV, dtype=F32)
    for blocks, k, coef in enumerate_diagrams(max_sigma, 1, 0):
        out += eval_slice(blocks, k, coef, gh, S, K3prev_z, K4z)
    return out


def transport(K3a, W):
    """kappa3(z') = W^T-transform of kappa3(a): three mode products. n^3 * n."""
    T = torch.einsum("pqr,pa->aqr", K3a, W)
    T = torch.einsum("aqr,qb->abr", T, W)
    return torch.einsum("abr,rc->abc", T, W)


def run_mlp(rec, coefs, alpha, max_sigma=3, mode="slice", diag=None):
    """Propagate (mu, Sigma, kappa3-tensor) with anchoring weight `alpha`.
    Returns per-layer kappa3_iij relative error and the final-layer mean."""
    W32 = [rec["W"][l].to(DEV, F32) for l in range(32)]
    W64 = [rec["W"][l] for l in range(32)]
    mu = torch.zeros(n, dtype=DT)
    C = torch.eye(n, dtype=DT)
    K3z = torch.zeros(n, n, n, device=DEV, dtype=F32)      # kappa3 of z^(0) = 0
    errs = {}
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)

    for l in range(32):
        zmu64 = mu @ W64[l]
        zcov64 = W64[l].T @ C @ W64[l]
        zmu = zmu64.to(DEV, F32)
        zcov = zcov64.to(DEV, F32)

        truth = rec["zk3"][l].to(DEV, F32)
        cur = torch.diagonal(K3z, dim1=0, dim2=1).T          # k3_iij from tensor

        # --- ANCHOR: blend the carried kappa3 slice toward the history-free map
        if alpha > 0.0 or diag is not None:
            if l > 0:
                stf = star_slice(prev_zcov, W32[l], prev_zmu)
            else:
                stf = torch.zeros(n, n, device=DEV, dtype=F32)
            # NOTE: fed the PROPAGATED (drifted) state, while `coefs` were fit on
            # the CORPUS (true) state. That mismatch is exactly the bug that broke
            # the old oracle (true Sigma into a drifted state) and P1 -- so we
            # MEASURE the anchor's in-loop error instead of assuming E12's 27-34%.
            a_slice = anchor_predict(coefs[l], zmu, zcov, stf)

            if diag is not None:
                e_prop = cur - truth
                e_anch = a_slice - truth
                den_t = truth[off].pow(2).sum().sqrt()
                # rho: correlation of the two error fields. The anchoring LAW
                # assumes rho ~ 0; simulation says it survives to rho ~0.7 and
                # dies near 0.9. This is the assumption most likely to be wrong.
                ep, ea = e_prop[off], e_anch[off]
                rho = (ep @ ea / (ep.norm() * ea.norm() + 1e-30)).item()
                diag["anchor_err"][l].append((ea.norm() / den_t).item())
                diag["prop_err"][l].append((ep.norm() / den_t).item())
                diag["rho"][l].append(rho)

            if alpha > 0.0:
                if mode == "gain":
                    # E10: the tensor error is mostly a MAGNITUDE bias (corr 0.97,
                    # gain -> 1.88). Lift a state-PREDICTED gain from the slice to
                    # the whole tensor -- unlike E13's fixed scalar, which failed.
                    g = ((cur * a_slice).sum() / (cur * cur).sum().clamp(min=1e-30))
                    K3z = K3z * (1.0 + alpha * (g - 1.0))
                else:                                   # "slice": direct injection
                    delta = alpha * (a_slice - cur)
                    idx = torch.arange(n, device=DEV)
                    K3z[idx, idx, :] += delta
                cur = torch.diagonal(K3z, dim1=0, dim2=1).T

        # --- score the carried slice against corpus truth
        errs[l] = ((cur - truth)[off].pow(2).sum()
                   / truth[off].pow(2).sum()).sqrt().item()

        # --- moment update (complete formula) using the carried cumulants
        k3d = torch.diagonal(torch.diagonal(K3z, dim1=0, dim2=1), dim1=0, dim2=1)
        cum = {"k3_iii": k3d.cpu().to(DT),
               "k3_iij": cur.cpu().to(DT),
               "k4_iiii": torch.zeros(n, dtype=DT),
               "k4_iijj": torch.zeros(n, n, dtype=DT),
               "k4_iiij": torch.zeros(n, n, dtype=DT)}
        mu, C = relu_step_full(mu, C, W64[l], cum)

        # --- generate kappa3(a) and transport to the next z
        zvar = torch.diagonal(zcov64)
        gh, *_ = hermite_gains(zmu64, zvar, kmax=12)
        gh = [x.to(DEV, F32) for x in gh]
        K3a = generate_k3(gh, zcov, K3z, None, max_sigma)
        prev_zcov, prev_zmu = zcov, zmu
        if l < 31:
            K3z = transport(K3a, W32[l + 1])

    return errs, mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--test", type=int, default=8)
    ap.add_argument("--alphas", type=str, default="0,0.15,0.3,0.5")
    ap.add_argument("--sigma", type=int, default=3)
    ap.add_argument("--mode", type=str, default="slice", choices=["slice", "gain"])
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]

    recs = []
    for m in range(a.train + a.test):
        d = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                       weights_only=False)
        recs.append({k: d[k] for k in ("zmu", "zcov", "zk3", "W", "mu")})
    print(f"G3 | {a.train} train / {a.test} held-out | sigma<={a.sigma} | "
          f"alphas {alphas} | device {DEV}", flush=True)

    t0 = time.time()
    coefs = fit_anchor(recs, a.train)
    print(f"  anchor fitted: {time.time() - t0:.0f}s", flush=True)

    # --- diagnostic pass (alpha=0): the three assumptions the LAW rests on
    diag = {k: {l: [] for l in range(32)} for k in ("anchor_err", "prop_err", "rho")}
    for m in range(a.train, a.train + a.test):
        run_mlp(recs[m], coefs, 0.0, a.sigma, a.mode, diag)
    print("\n  ASSUMPTION CHECK (in-loop, on the PROPAGATED state, held-out):")
    print(f"{'':>14} " + " ".join(f"{'L' + str(l):>7}" for l in LAY_REPORT))
    for k, lab in (("prop_err", "propagated"), ("anchor_err", "ANCHOR err"),
                   ("rho", "rho(errors)")):
        sc = 1.0 if k == "rho" else 100.0
        unit = "" if k == "rho" else "%"
        print(f"{lab:>14} " + " ".join(
            f"{st.mean(diag[k][l]) * sc:>6.2f}{unit}" for l in LAY_REPORT))
    print("  (E12 measured the anchor at 27-34% on the TRUE state. If it is much")
    print("   worse here, the anchor must be REFIT on propagated states (DAgger).")
    print("   LAW assumes rho~0; simulation: survives to ~0.7, dies near 0.9.)")

    print(f"\n{'alpha':>6} " + " ".join(f"{'L' + str(l):>7}" for l in LAY_REPORT)
          + f" {'mean(32)':>9} {'final MSE':>11}")
    for al in alphas:
        per = {l: [] for l in range(32)}
        mses = []
        for m in range(a.train, a.train + a.test):
            errs, mu = run_mlp(recs[m], coefs, al, a.sigma, a.mode)
            for l in range(32):
                per[l].append(errs[l])
            mses.append(mse(mu, recs[m]["mu"][31]))
        row = " ".join(f"{st.mean(per[l]) * 100:>6.1f}%" for l in LAY_REPORT)
        allm = st.mean([st.mean(per[l]) for l in range(32)])
        print(f"{al:>6.2f} {row} {allm * 100:>8.1f}% {st.mean(mses):>11.3e}",
              flush=True)

    print("\n  LAW: e_depth = sqrt(eps * anchor_err) at alpha* = eps/anchor_err")
    print("  (vs sqrt(L)*eps unanchored). eps~8.5%, anchor~30% => predict ~16%.")
    print("  BAR: end-to-end < 1.02e-5 beats v4 via fusion; < 3e-6 => ~1.8x v4.")


if __name__ == "__main__":
    main()
