"""A1 — THE DECIDING RUN: fit the anchor on the SHIPPABLE (R=32 core) carry and
measure BOTH the held-out slice error AND the end-to-end final-layer MSE.

Why this and not another projection: E6's "kappa3 error -> MSE" table is noisy
and oracle-conditioned, so we do NOT extrapolate from slice error. We measure
b^2 directly. The fusion bar is b^2 < 1.02e-5 to beat v4.

Design (stopping rules fixed IN ADVANCE):
  * carry     = k3prop.K3State, R=32 -- the affordable rank (8.65e9 = 3.2% of B;
                r=64 was flopscope-measured at 10% of B => off the table).
  * anchor    = pointwise map on the n^2 grid, features
                [t_i, t_j, rho_ij, prop_slice, star, layer_frac, 1] (normalised),
                shared across layers and MLPs (weight exchangeability), fit by
                ridge on corpus truth. Same feature list the estimator computes.
  * DAgger    = fit open-loop, then RE-fit on the trajectory the anchor itself
                produces (the in-loop distribution shift is a real risk: the
                corrected slice changes mu/Sigma, which changes the features).
  * cells     : no anchor | anchor iter 1 | anchor iter 2   (all HELD OUT)

  STOP IF: held-out slice error > 25%  (anchor does not transfer)
  STOP IF: end-to-end b^2 > 1.02e-5    (no submission worth making)

Efficiency: propagation is n^2/n^3-small and float64-CPU by necessity (MPS has
no float64; the cumulant subtractions need it). ~1-2 s/MLP => a full 3-pass
DAgger over 24 MLPs is minutes. Gram accumulation for the ridge is on GPU.
"""
from __future__ import annotations

import statistics as st
import sys
import time

import torch

from harness import CORPUS, DT, hermite_gains, relu_step_full
from k3prop import K3State
from g1_k4_learn import basis_exponents

# G5's ACTUAL model class: per-layer, degree-3 polynomial in (t_i,t_j,rho) plus
# prop/star and their low-order interactions. The first A1 run used a single
# SHARED 7-feature LINEAR ridge -- a far weaker class than the one G5 validated
# at 15.7%, so it tested a strawman. Per-layer coefficients are free (npz);
# the poly features cost ~0.06% of budget on the n^2 grid.
EXPS = basis_exponents(3)
LOW = [i for i, e in enumerate(EXPS) if sum(e) <= 2]
NF = len(EXPS) + 2 * (1 + len(LOW))

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256
R = 32
LAY = [1, 3, 7, 15, 23, 31]


def feats_of(zmu, zcov, prop, star):
    """G5's feature map: deg-3 poly in (t_i,t_j,rho) + {prop, star} x low-order."""
    s = torch.sqrt(torch.clamp(torch.diagonal(zcov), min=1e-30))
    t = zmu / s
    rho = zcov / torch.outer(s, s)
    nrm3 = torch.outer(s * s, s)
    ti = t[:, None].expand(n, n)
    tj = t[None, :].expand(n, n)
    cols = [((ti ** a) * (tj ** b) * (rho ** c)).reshape(-1)
            for a, b, c in EXPS]
    X = torch.stack(cols, 1)
    blocks = [X]
    for ex in ((prop / nrm3), (star / nrm3)):
        ev = ex.reshape(-1, 1)
        blocks += [ev, ev * X[:, LOW]]
    return torch.cat(blocks, 1), nrm3


def run_closure(Ws, k4, coefs=None, truth=None):
    """One MLP through the R=32 core carry. If `coefs` given, the anchor is
    applied IN-LOOP (so mu/Sigma follow the corrected trajectory).
    Returns final mu, and per-layer (feats, prop, star, nrm3, slice_used)."""
    mu = torch.zeros(n, dtype=DT)
    cov = torch.eye(n, dtype=DT)
    stt = K3State(n, R)
    rec = []
    for l, W in enumerate(Ws):
        k3_iij, k3_iii = stt.slices(W)          # the CARRIED slice (the feature)
        zmu = mu @ W
        zcov = W.T @ cov @ W
        zvar = torch.clamp(torch.diagonal(zcov), min=1e-30)
        g, s, t, pdf, cdf = hermite_gains(zmu, zvar, kmax=12)

        # star (freshly generated at this layer) -- also a feature
        B = zcov @ (g[1][:, None] * W)
        Cm = (g[2][:, None] * W) * B
        Dm = g[2][:, None] * (B * B)
        star = 2.0 * (Cm.T @ B) + (Dm.T @ W)

        F, nrm3 = feats_of(zmu, zcov, k3_iij, star)
        used = k3_iij
        if coefs is not None:                                   # per-layer coefs
            used = k3_iij + (F @ coefs[l]).reshape(n, n) * nrm3
        rec.append({"F": F, "nrm3": nrm3, "prop": k3_iij, "star": star,
                    "used": used, "l": l})

        cum = {"k3_iii": k3_iii, "k3_iij": used,
               "k4_iiii": k4[l]["k4_iiii"], "k4_iijj": k4[l]["k4_iijj"],
               "k4_iiij": k4[l]["k4_iiij"]}
        mu, cov = relu_step_full(mu, cov, W, cum)
        ev, evec = torch.linalg.eigh(cov)
        stt.advance(W, g[1], g[2], zcov, evec[:, -R:].contiguous())
    return mu, rec


def fit(recs, truths, lam=1e-6):
    """PER-LAYER ridge (G5's class): predict (true - propagated)/nrm3."""
    out = []
    for l in range(32):
        G = torch.zeros(NF, NF, dtype=DT)
        b = torch.zeros(NF, dtype=DT)
        for rec, tr in zip(recs, truths):
            r = rec[l]
            y = ((tr[l] - r["prop"]) / r["nrm3"]).reshape(-1)
            G += r["F"].T @ r["F"]
            b += r["F"].T @ y
        G += lam * torch.eye(NF, dtype=DT) * torch.diagonal(G).mean()
        out.append(torch.linalg.solve(G, b))
    return out


def slice_err(rec, tr):
    off = ~torch.eye(n, dtype=torch.bool)
    return [((rec[l]["used"] - tr[l])[off].norm() / tr[l][off].norm()).item()
            for l in range(32)]


def main():
    n_mlps = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_tr = n_mlps // 2
    print(f"A1 | {n_mlps} MLPs ({n_tr} train / {n_mlps-n_tr} HELD OUT) | "
          f"R={R} core carry (the SHIPPABLE one)", flush=True)

    data = []
    t0 = time.time()
    for m in range(n_mlps):
        rc = torch.load(CORPUS / f"mlp_{200000+m}.pt", map_location="cpu",
                        weights_only=False)
        Ws = [rc["W"][l].to(DT) for l in range(32)]
        truth = [rc["zk3"][l].to(DT) for l in range(32)]
        # k4_iiii IS the diagonal of k4_iijj (kappa4(z_i,z_i,z_i,z_i)) -- it was
        # zeroed in the first A1 run, crippling the LARGEST kappa4 term in the
        # mean update ((k4d/24)*g4). k4_iiij is not in the corpus (needs a
        # 4th-order cross moment); left at zero, so this is still conservative.
        k4 = [{"k4_iiii": torch.diagonal(rc["zk4"][l]).clone().to(DT),
               "k4_iijj": rc["zk4"][l].to(DT),
               "k4_iiij": torch.zeros(n, n, dtype=DT)} for l in range(32)]
        data.append({"Ws": Ws, "truth": truth, "k4": k4,
                     "gt": rc["mu"][31].to(DT)})
    print(f"  corpus loaded: {time.time()-t0:.0f}s", flush=True)

    results = {}
    coefs = None
    for it in range(3):                       # it=0 -> no anchor; 1,2 -> DAgger
        t0 = time.time()
        recs, mus = [], []
        for d in data:
            mu, rec = run_closure(d["Ws"], d["k4"], coefs)
            recs.append(rec)
            mus.append(mu)
        errs = [slice_err(recs[m], data[m]["truth"]) for m in range(n_mlps)]
        mses = [((mus[m] - data[m]["gt"]) ** 2).mean().item() for m in range(n_mlps)]
        tag = "no anchor" if it == 0 else f"anchor iter {it}"
        ho = range(n_tr, n_mlps)
        results[tag] = (
            [st.mean([errs[m][l] for m in ho]) for l in range(32)],
            st.mean([mses[m] for m in ho]))
        print(f"  [{tag}] {time.time()-t0:.0f}s  "
              f"held-out slice {st.mean([st.mean(errs[m]) for m in ho])*100:.1f}%  "
              f"b^2 {results[tag][1]:.3e}", flush=True)
        # refit on THIS trajectory (DAgger: features come from the state the
        # anchor itself produced)
        coefs = fit([recs[m] for m in range(n_tr)],
                    [data[m]["truth"] for m in range(n_tr)])

    print(f"\n{'cell':<16} " + " ".join(f"{'L'+str(l):>7}" for l in LAY)
          + f" {'mean(32)':>9} {'b^2 (end-to-end)':>17}")
    for tag, (per, mse) in results.items():
        row = " ".join(f"{per[l]*100:>6.1f}%" for l in LAY)
        print(f"{tag:<16} {row} {st.mean(per)*100:>8.1f}% {mse:>17.3e}")

    best = min(v[1] for v in results.values())
    print(f"\n  FUSION BAR: b^2 < 1.02e-5 beats v4 (3.396e-7). best = {best:.3e}")
    print(f"  STOP RULES: slice > 25% => anchor does not transfer;")
    print(f"              b^2 > 1.02e-5 => no submission worth making.")
    print(f"  VERDICT: {'PROCEED' if best < 1.02e-5 else 'STOP -- above the bar'}")


if __name__ == "__main__":
    main()
