"""P1 diagnostic — where does the error actually enter, measured from the
mirror's OWN free-running roll (not teacher-forced).

For each corpus MLP, roll mirror_v2 (no_k4, shipped theta) from layer 0 and at
every layer record, against the true PRE-activation moments (moment_corpus
zcov/zk3):
  - variance rel-error   : |diag(S_own) - true_var| / true_var   (Phase 1's 6.7%)
  - carried κ3 slice err : off-diag rel-Frobenius of k3s_own vs true zk3 slice
  - mean rel-error       : |m_own - true_zmu| / rms(true_zmu)

Also an ORACLE probe: at each layer replace the carried preact κ3 slice with
the TRUE zk3 slice *only inside the covariance consumer* (not the carry push),
and record the resulting variance error — the from-own-roll version of Phase 1's
"true κ3 slice -> variance 1.4%". This isolates: is the carried-slice error the
variance bottleneck, and is the cov-consumer insertion the right lever?

No κ4, no predictor. Parity note printed: mirror final mean should match
mirror_v2.forward's final row (same code path).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ARC = Path("/Users/lisek/Projects/ARC_white_box")
CORPUS = ARC / "data/moment_corpus"
MIRROR = ARC / "mlp_cumulant_propagation"
ROOT = Path("/Users/lisek/Projects/whest-starterkit")
sys.path.insert(0, str(MIRROR))
import mirror_v2 as M  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N, R = 256, 48
SEEDS = list(range(200000, 200024))
LAYERS = (1, 3, 7, 15, 23, 30, 31)


def basis(S):
    u, _, _ = torch.linalg.svd(S.cpu(), full_matrices=False)
    return u[:, :R].to(DEV)


@torch.no_grad()
def roll(W, theta, true_zcov, true_zk3, oracle_slice_in_cov=False):
    """Free roll; return per-layer (var_err, slice_err) dicts + final mua row."""
    m = torch.zeros(N, device=DEV)
    S = W[0].T @ W[0]
    V = basis(S)
    T3 = torch.zeros((R,) * 3, device=DEV)
    star = None
    eye = torch.eye(N, device=DEV)
    var_err, slice_err = {}, {}
    rows = []

    for li in range(32):
        c = M._coeffs(theta[li])
        var = torch.diagonal(S).clamp_min(1e-8)
        s = var.sqrt()
        a = (m / s).clamp(-12, 12)
        pdf, cdf = M._norm_pdf(a), M._norm_cdf(a)
        h1, h2 = cdf, pdf / s
        ss = torch.outer(s, s)
        rho = (S / ss).clamp(-1, 1)

        k3s = None
        if li:
            G = (V[:, :, None] * V[:, None, :]).reshape(N, R * R)
            P = G @ T3.reshape(R * R, R)
            k3s, k3d = P @ V.T, (P * V).sum(1)
            h2s, Wr, Bs = star
            C, D = h2s[:, None] * Wr * Bs, h2s[:, None] * Bs * Bs
            k3s = k3s + 2 * C.T @ Bs + D.T @ Wr
            k3d = k3d + 3 * (h2s[:, None] * Wr * Bs * Bs).sum(0)

            # --- diagnostics vs true PRE-activation slice at this layer
            tv = torch.diagonal(true_zcov[li]).clamp_min(1e-12)
            var_err[li] = float((((var - tv).abs()) / tv).mean())
            tk3 = true_zk3[li]
            num = (k3s[~eye.bool()] - tk3[~eye.bool()]).pow(2).sum().sqrt()
            den = tk3[~eye.bool()].pow(2).sum().sqrt().clamp_min(1e-30)
            slice_err[li] = float(num / den)

        mua = s * pdf * (1 + c["p"]) + m * cdf * (1 + c["q"]) + c["moff"] * s
        ez2 = (m * m + var) * cdf + m * s * pdf
        if li:
            mua = mua - c["c3m"] * a * pdf * (k3d / var) / 6
            ez2 = ez2 + c["c3m"] * pdf * (k3d / s) / 3
        vara = (ez2 - mua * mua).clamp_min(1e-12) * (1 + c["s_dia"])
        Sig = M._wick_cov(a[None], pdf[None], cdf[None], s[None], rho[None])[0]
        Sig = Sig * (1 + c["s_off"])
        if li:
            slice_for_cov = true_zk3[li] if oracle_slice_in_cov else k3s
            q3 = slice_for_cov / s[:, None]
            ct = pdf[:, None] * h1[None, :] * q3
            Sig = Sig + c["c3c"] * (ct + ct.T) / 2
        Sig = Sig * (1 - eye) + torch.diag(vara)
        rows.append(mua)
        if li == 31:
            break

        Wn = W[li + 1]
        mnext = mua @ Wn
        Snext = Wn.T @ Sig @ Wn
        Vn = basis(Snext)
        Z = h1[:, None] * (Wn @ Vn)
        M3 = Z.T @ V
        T3n = c["c3p"] * torch.einsum("ia,jb,kc,abc->ijk", M3, M3, M3, T3)
        if star is not None:
            h2s, Wr, Bs = star
            A, B = Wr @ Z, Bs @ Z
            T3n = T3n + c["c3g"] * (
                torch.einsum("n,ni,nj,nk->ijk", h2s, A, B, B)
                + torch.einsum("n,ni,nj,nk->ijk", h2s, B, A, B)
                + torch.einsum("n,ni,nj,nk->ijk", h2s, B, B, A))
        star = (h2, Wn, S @ (h1[:, None] * Wn))
        m, S, V, T3 = mnext, Snext, Vn, T3n
    return var_err, slice_err, torch.stack(rows)


def run_theta(theta, tag):
    agg = {"var": {l: [] for l in LAYERS}, "slice": {l: [] for l in LAYERS},
           "mse_base": [], "mse_oracle": []}
    for seed in SEEDS:
        rec = torch.load(CORPUS / f"mlp_{seed}.pt", map_location="cpu",
                         weights_only=False)
        W = rec["W"].to(DEV, torch.float32)
        zcov = rec["zcov"].to(DEV, torch.float32)
        zk3 = rec["zk3"].to(DEV, torch.float32)
        tgt = rec["mu"][-1].to(DEV, torch.float32)
        ve, se, rows = roll(W, theta, zcov, zk3, oracle_slice_in_cov=False)
        _, _, rows_o = roll(W, theta, zcov, zk3, oracle_slice_in_cov=True)
        for l in LAYERS:
            if l in ve:
                agg["var"][l].append(ve[l]); agg["slice"][l].append(se[l])
        agg["mse_base"].append(float((rows[-1] - tgt).square().mean()))
        agg["mse_oracle"].append(float((rows_o[-1] - tgt).square().mean()))
    print(f"\n===== theta={tag} =====")
    print(f"{'layer':>6} {'var_err':>9} {'slice_err':>10}")
    for l in LAYERS:
        if agg["var"][l]:
            print(f"{l:>6} {np.mean(agg['var'][l])*100:>8.2f}% "
                  f"{np.mean(agg['slice'][l])*100:>9.2f}%")
    mb, mo = np.array(agg["mse_base"]), np.array(agg["mse_oracle"])
    print(f"final-mean MSE  base {mb.mean():.4e}  true-slice-in-cov "
          f"{mo.mean():.4e}  gain {100*(1-mo.mean()/mb.mean()):.1f}%  "
          f"wins {(mo<mb).sum()}/{len(mb)}")


def main():
    theta_ship = torch.from_numpy(np.load(ROOT / "v5.npz")["theta"]).to(DEV)
    theta_neutral = torch.zeros_like(theta_ship)
    run_theta(theta_neutral, "NEUTRAL (valid oracle — no pairing trap)")
    run_theta(theta_ship, "SHIPPED (oracle confounded by calibration)")


if __name__ == "__main__":
    main()
