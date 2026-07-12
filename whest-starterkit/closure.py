"""Analytic closure for the estimator: K=2 + kappa3 + kappa4, flopscope ops only.

Mirrors research/calibrate_k3k4.py exactly (same formulas, same coefficient
layout), so exported calibration params drop straight in.

State per layer: pre-activation (m, S); kappa3/kappa4 carried as rank-r Tucker
cores in a top-r basis of S; the star tensors born at the previous layer kept
at full fidelity via O(n^3) matmuls (never materialized).

Corrections (validated single-step: mean up to 1300x, cov up to 193x) are
applied to the mean AND the covariance together -- partial fixes are harmful,
because the Gaussian closure's errors cancel (see IMPROVEMENTS.md).

All sigma powers are folded into the cumulants BEFORE multiplying by the
bounded (phi, Phi) factors: h3 ~ 1/sigma^2 and h4 ~ 1/sigma^3 overflow float32
on dead neurons otherwise.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp

# per-layer coefficient order, matching research/calibrate_k3k4.py
IDX = dict(p=0, q=1, moff=2, s_dia=3, s_off=4,
           c3m=5, c3c=6, c3g=7, c3p=8, c4m=9, c4c=10, c4g=11, c4p=12)
N_PARAMS = 13
N_TERMS = 4          # Wick series (converged by k=4; measured)


def _coeffs(theta_row):
    """Same parametrisation as the trainer: bounded tanh / exp(1.5 tanh)."""
    t = fnp.tanh(theta_row)
    return {
        "p": 0.10 * t[IDX["p"]], "q": 0.10 * t[IDX["q"]], "moff": 0.10 * t[IDX["moff"]],
        "s_dia": 0.40 * t[IDX["s_dia"]], "s_off": 0.40 * t[IDX["s_off"]],
        "c3m": fnp.exp(1.5 * t[IDX["c3m"]]), "c3c": fnp.exp(1.5 * t[IDX["c3c"]]),
        "c3g": fnp.exp(1.5 * t[IDX["c3g"]]), "c3p": fnp.exp(1.5 * t[IDX["c3p"]]),
        "c4m": fnp.exp(1.5 * t[IDX["c4m"]]), "c4c": fnp.exp(1.5 * t[IDX["c4c"]]),
        "c4g": fnp.exp(1.5 * t[IDX["c4g"]]), "c4p": fnp.exp(1.5 * t[IDX["c4p"]]),
    }


def _top_basis(S, Q0, iters=2):
    """Top-r subspace of S by randomized subspace iteration, with a Rayleigh-Ritz
    step so the columns come out ORDERED by eigenvalue.

    Mirrors research/calibrate_k3k4.py exactly. Two things matter:
      * ordering: kappa4 uses the LEADING r4 columns, which is only the dominant
        sub-subspace if the columns are sorted;
      * the start block Q0 is SHIPPED, so the estimator and the trainer build the
        same basis and the calibrated coefficients are applied to what they were
        fitted on.
    QR + a small (r x r) eigh are flopscope primitives (and far cheaper than a
    Python Gram-Schmidt loop, which would have cost thousands of tiny ops per MLP
    in residual wall time).
    """
    Q = Q0
    for _ in range(iters):
        Q, _ = fnp.linalg.qr(fnp.matmul(S, Q))
    Sr = fnp.matmul(Q.T, fnp.matmul(S, Q))
    Sr = 0.5 * (Sr + Sr.T)
    _, U = fnp.linalg.eigh(Sr)          # ascending eigenvalues
    U = U[:, ::-1]                      # -> descending
    return fnp.matmul(Q, U)


def _wick_cov(a, pdf, cdf, s, rho):
    g = [cdf, pdf]
    he_prev, he_cur = fnp.ones_like(a), a
    for k in range(3, N_TERMS + 1):
        m = k - 2
        g.append(((-1) ** m) * he_cur * pdf)
        he_prev, he_cur = he_cur, a * he_cur - m * he_prev
    acc = fnp.zeros_like(rho)
    rho_k = rho
    sf = 1.0
    for k in range(1, N_TERMS + 1):
        sf *= float(k) ** 0.5
        gh = g[k - 1] / sf
        acc = acc + fnp.outer(gh, gh) * rho_k
        if k < N_TERMS:
            rho_k = rho_k * rho
    return fnp.outer(s, s) * acc


def predict_final_mean(mlp, theta, rank, rank4=8):
    """Final-layer mean via the calibrated K=2 + kappa3 + kappa4 closure.

    theta: (depth, 13) array of calibrated coefficients (zeros => uncalibrated).
    Returns (width,) or None if the shape is not covered by these params.
    """
    n = mlp.width
    depth = mlp.depth
    if theta is None or theta.shape[0] != depth or theta.shape[1] != N_PARAMS:
        return None
    r = int(rank)
    r4 = min(int(rank4), r)
    eye = fnp.eye(n, dtype=fnp.float32)

    W0 = mlp.weights[0]
    m = fnp.zeros(n, dtype=fnp.float32)
    S = fnp.matmul(W0.T, W0)                      # z_0 = W0^T x,  x ~ N(0, I)
    V = _top_basis(S, r)
    T3 = fnp.zeros((r, r, r), dtype=fnp.float32)
    T4 = fnp.zeros((r4, r4, r4, r4), dtype=fnp.float32)  # kappa4: smaller rank
    # kappa4's core costs O(n r^4): at r=32 it alone is 8.4% of budget, which
    # doubles the score multiplier. kappa4 is ~rank-1 at depth, so r4 << r.
    star = None                                   # (h2, h3, Wr, Bs) in z_l space

    for li in range(depth):
        c = _coeffs(theta[li])
        var = fnp.maximum(fnp.diag(S), 1e-8)
        s = fnp.sqrt(var)
        a = fnp.clip(m / s, -12.0, 12.0)
        pdf = flops.stats.norm.pdf(a)
        cdf = flops.stats.norm.cdf(a)
        h1 = cdf
        h2 = pdf / s
        h3 = -a * pdf / var
        ss = fnp.outer(s, s)
        rho = fnp.clip(S / ss, -1.0, 1.0)

        # ---- cumulant slices: carried cores + full-fidelity fresh star
        G2 = fnp.reshape(V[:, :, None] * V[:, None, :], (n, r * r))
        P3 = fnp.matmul(G2, fnp.reshape(T3, (r * r, r)))
        k3_slice = fnp.matmul(P3, V.T)
        k3_diag = fnp.sum(P3 * V, axis=1)
        V4 = V[:, :r4]
        G2b = fnp.reshape(V4[:, :, None] * V4[:, None, :], (n, r4 * r4))
        P4 = fnp.matmul(G2b, fnp.reshape(T4, (r4 * r4, r4 * r4)))
        k4_slice = fnp.matmul(P4, G2b.T)
        k4_diag = fnp.sum(P4 * G2b, axis=1)
        if star is not None:
            h2s, h3s, Wr, Bs = star
            C3 = h2s[:, None] * Wr * Bs
            D3 = h2s[:, None] * Bs * Bs
            k3_slice = k3_slice + 2.0 * fnp.matmul(C3.T, Bs) + fnp.matmul(D3.T, Wr)
            k3_diag = k3_diag + 3.0 * fnp.sum(h2s[:, None] * Wr * Bs * Bs, axis=0)
            C4 = h3s[:, None] * Wr * Bs
            D4 = h3s[:, None] * Bs * Bs
            k4_slice = (k4_slice + 2.0 * fnp.matmul(C4.T, Bs * Bs)
                        + 2.0 * fnp.matmul(D4.T, Wr * Bs))
            k4_diag = k4_diag + 4.0 * fnp.sum(h3s[:, None] * Wr * Bs * Bs * Bs, axis=0)

        # ---- corrections, dimensionless (no 1/sigma^k intermediate is formed)
        t3 = k3_diag / var
        t4 = k4_diag / (var * s)
        q3 = k3_slice / s[:, None]
        q4 = k4_slice / ss

        mu_a = (s * pdf * (1.0 + c["p"]) + m * cdf * (1.0 + c["q"]) + c["moff"] * s
                - c["c3m"] * (1.0 / 6.0) * a * pdf * t3
                + c["c4m"] * (1.0 / 24.0) * (a * a - 1.0) * pdf * t4)
        ez2 = ((m * m + var) * cdf + m * s * pdf
               + c["c3m"] * (1.0 / 3.0) * pdf * (k3_diag / s))
        var_a = fnp.maximum(ez2 - mu_a * mu_a, 1e-12) * (1.0 + c["s_dia"])

        Sig = _wick_cov(a, pdf, cdf, s, rho) * (1.0 + c["s_off"])
        c3_term = pdf[:, None] * h1[None, :] * q3
        Sig = Sig + c["c3c"] * 0.5 * (c3_term + c3_term.T)
        Sig = Sig + c["c4c"] * 0.25 * (pdf[:, None] * pdf[None, :] * q4)
        Sig = Sig * (1.0 - eye) + fnp.diag(var_a)

        if li == depth - 1:
            return mu_a

        # ---- push through the next linear layer
        W = mlp.weights[li + 1]
        S_pre = S
        m = fnp.matmul(mu_a, W)
        S = fnp.matmul(W.T, fnp.matmul(Sig, W))
        V_next = _top_basis(S, r)

        Z = h1[:, None] * fnp.matmul(W, V_next)
        M = fnp.matmul(Z.T, V)
        M4 = M[:r4, :r4]
        T3 = c["c3p"] * fnp.einsum("mkl,am,nk,ol->ano", T3, M, M, M)
        T4 = c["c4p"] * fnp.einsum("mklp,am,nk,ol,qp->anoq", T4, M4, M4, M4, M4)

        if star is not None:                        # compress the old star forward
            h2s, h3s, Wr, Bs = star
            A = fnp.matmul(Wr, Z)
            Bv = fnp.matmul(Bs, Z)
            U3 = fnp.einsum("i,im,ik,il->mkl", h2s, A, Bv, Bv)
            T3 = T3 + c["c3g"] * (U3 + fnp.transpose(U3, (1, 0, 2))
                                  + fnp.transpose(U3, (1, 2, 0)))
            A4, Bv4 = A[:, :r4], Bv[:, :r4]
            U4 = fnp.einsum("i,im,ik,il,ip->mklp", h3s, A4, Bv4, Bv4, Bv4)
            T4 = T4 + c["c4g"] * (U4 + fnp.transpose(U4, (1, 0, 2, 3))
                                  + fnp.transpose(U4, (1, 2, 0, 3))
                                  + fnp.transpose(U4, (1, 2, 3, 0)))

        star = (h2, h3, W, fnp.matmul(S_pre, h1[:, None] * W))
        V = V_next

    return None
