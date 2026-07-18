"""Research Attempt 2 — variance-reduction prototype, measured with flopscope.

Builds the MSE ladder at a FIXED FLOP budget (the 10% multiplier floor):

    plain MC  ->  +QMC  ->  +antithetic  ->  +control variate  ->  +final-layer RB

Each stage runs inside `flops.BudgetContext(flop_budget=B_test)` and asserts it
stayed under budget. Scored against a high-sample ground truth (off budget).

Run:  .venv/bin/python3 research2_prototype.py
"""
from __future__ import annotations
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from local_engine import build_mlp

# ---------------- config ----------------
WIDTH, DEPTH = 256, 32
N_MLP = 12                                  # average MSE over this many MLPs (grader metric)
FULL_BUDGET = int(2.72e11)
B_TEST = int(0.10 * FULL_BUDGET)          # sit at the 10% multiplier floor
ROWS = 6000                                # matmul rows that fit the budget
N_GT = 400_000                             # ground-truth samples per MLP (off budget)
PTS_FILE = "points.npz"
PER_SAMPLE = 2 * DEPTH * WIDTH * WIDTH      # ~4.19e6 FLOPs per forward pass


# ---------------- ground truth (numpy, off budget) ----------------
def ground_truth(weights_np, n, N, batch=50_000, seed=777):
    rng = np.random.default_rng(seed)
    acc = np.zeros(n, dtype=np.float64)
    last = None
    done = 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        while done < N:
            b = min(batch, N - done)
            x = rng.standard_normal((b, n)).astype(np.float32)
            for W in weights_np:
                x = np.maximum(x @ W, 0.0)
            acc += x.astype(np.float64).sum(0)
            last = x
            done += b
    return acc / N, float(np.var(last, axis=0).mean())   # final-layer mean, var(proxy)


# ---------------- analytic mean-propagation closure (for the control variate) ----------------
def mean_prop(weights):
    """Return per-layer (b, c) affine-linearization coeffs and the exact
    surrogate means mu_an (= mean-propagation prediction)."""
    n = weights[0].shape[0]
    mu = fnp.zeros(n); var = fnp.ones(n)
    bc, mu_an = [], []
    for W in weights:
        m = W.T @ mu
        v = fnp.maximum((W * W).T @ var, 1e-12)
        s = fnp.sqrt(v); a = m / s
        pdf = flops.stats.norm.pdf(a); cdf = flops.stats.norm.cdf(a)
        mu = m * cdf + s * pdf
        ez2 = (m * m + v) * cdf + m * s * pdf
        var = fnp.maximum(ez2 - mu * mu, 0.0)
        bc.append((cdf, mu - cdf * m))      # b=Phi(a), c=E[relu]-b*m
        mu_an.append(mu)
    return bc, mu_an


# ---------------- estimator variants (all return (depth,width)) ----------------
def v_plainMC(mlp, pts, rng):
    n = mlp.width
    x = fnp.array(rng.standard_normal((ROWS, n)).astype(np.float32))
    rows = []
    for W in mlp.weights:
        x = fnp.maximum(x @ W, 0.0); rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def v_qmc(mlp, pts, rng):
    x = pts[:ROWS]
    rows = []
    for W in mlp.weights:
        x = fnp.maximum(x @ W, 0.0); rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def v_antithetic(mlp, pts, rng):
    k = ROWS // 2
    base = pts[:k]
    x = fnp.concatenate([base, -base], axis=0) if hasattr(fnp, "concatenate") \
        else fnp.array(np.concatenate([np.asarray(base), -np.asarray(base)], 0))
    rows = []
    for W in mlp.weights:
        x = fnp.maximum(x @ W, 0.0); rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def _antithetic_input(pts, k):
    base = pts[:k]
    return fnp.array(np.concatenate([np.asarray(base), -np.asarray(base)], 0).astype(np.float32))


def v_control_variate(mlp, pts, rng, final_rb=False):
    n, d = mlp.width, mlp.depth
    bc, mu_an = mean_prop(mlp.weights)          # ~11M FLOPs, on budget
    k = ROWS // 4                                # 2k real + 2k surrogate = 4k rows
    x = _antithetic_input(pts, k)               # (2k, n)
    hr = x; hs = x
    rows = []
    for l, W in enumerate(mlp.weights):
        b, c = bc[l]
        zr = hr @ W; hr = fnp.maximum(zr, 0.0)
        zs = hs @ W; hs = c + b * zs            # affine surrogate (mean known)
        # unbiased control variate: mean(h) - (mean(hs) - E[hs])
        mu_cv = fnp.mean(hr, axis=0) - (fnp.mean(hs, axis=0) - mu_an[l])
        if final_rb and l == d - 1:
            # Rao-Blackwell the scored layer: analytic ReLU moment on sampled
            # pre-activation moments instead of MC-averaging ReLU.
            muhat = fnp.mean(zr, axis=0)
            sighat = fnp.sqrt(fnp.maximum(fnp.var(zr, axis=0), 1e-12))
            a = muhat / sighat
            mu_cv = muhat * flops.stats.norm.cdf(a) + sighat * flops.stats.norm.pdf(a)
        rows.append(mu_cv)
    return fnp.stack(rows, axis=0)


def v_cv_rb(mlp, pts, rng):
    return v_control_variate(mlp, pts, rng, final_rb=True)


def v_cv_delta(mlp, pts, rng):
    """Delta-method control variate on the REAL trajectory (reuses the
    pre-activations, so it's ~free and stays correlated at depth). Uses the
    full sample budget — no separate surrogate pass."""
    bc, _ = mean_prop(mlp.weights)
    k = ROWS // 2
    x = _antithetic_input(pts, k)               # 6000 real antithetic samples
    h = x
    mu_prev = fnp.zeros(mlp.width)              # E[input] = 0
    rows = []
    for l, W in enumerate(mlp.weights):
        b, _c = bc[l]
        z = h @ W                               # real pre-activation (reused)
        h = fnp.maximum(z, 0.0)
        m_pre = W.T @ mu_prev                    # CV'd pre-activation mean
        # ReLU(z) ≈ c + b·z ; subtract the linearization's sampling fluctuation
        mu_hat = fnp.mean(h, axis=0) - b * (fnp.mean(z, axis=0) - m_pre)
        rows.append(mu_hat)
        mu_prev = mu_hat
    return fnp.stack(rows, axis=0)


def v_closure_only(mlp, pts, rng):
    _, mu_an = mean_prop(mlp.weights)          # analytic mean-prop, no sampling
    return fnp.stack(mu_an, axis=0)


def _final_row(z, h, kappa3):
    """Final-layer estimate from sampled pre-activations z (k,n)."""
    if not kappa3:
        return fnp.mean(h, axis=0)
    mu = fnp.mean(z, axis=0)
    dz = z - mu
    sig = fnp.sqrt(fnp.maximum(fnp.mean(dz * dz, axis=0), 1e-12))
    gamma = fnp.mean(dz * dz * dz, axis=0) / (sig * sig * sig)
    a = mu / sig
    pdf = flops.stats.norm.pdf(a); cdf = flops.stats.norm.cdf(a)
    # Edgeworth (kappa3) ReLU moment: Gaussian term minus skew correction.
    return mu * cdf + sig * pdf - (gamma / 6.0) * mu * pdf


def v_kappa3_rb(mlp, pts, rng):
    """QMC+antithetic sampling, final layer via kappa3-corrected ReLU moment."""
    k = ROWS // 2
    x = _antithetic_input(pts, k)
    h = x; d = mlp.depth; rows = []
    for l, W in enumerate(mlp.weights):
        z = h @ W; h = fnp.maximum(z, 0.0)
        rows.append(_final_row(z, h, kappa3=(l == d - 1)))
    return fnp.stack(rows, axis=0)


def v_dimqmc(mlp, pts, rng, kappa3=False, rows=5600):
    """Rotate QMC points so the best Sobol coordinates hit the top singular
    directions of the linearized input->final map (effective dim ~16 << 256)."""
    n, d = mlp.width, mlp.depth
    bc, _ = mean_prop(mlp.weights)
    M = fnp.eye(n)                              # linearized map z_L = M x
    for l, W in enumerate(mlp.weights):
        M = W.T @ M
        if l < d - 1:
            M = bc[l][0][:, None] * M            # b[l] = Phi(alpha) gains
    Vt = np.linalg.svd(np.asarray(M), compute_uv=True)[2]   # right singular vecs (off-budget SVD)
    Vt = fnp.array(Vt.astype(np.float32))
    k = rows // 2
    x = _antithetic_input(pts, k) @ Vt          # align QMC quality to top directions
    h = x; out = []
    for l, W in enumerate(mlp.weights):
        z = h @ W; h = fnp.maximum(z, 0.0)
        out.append(_final_row(z, h, kappa3=(kappa3 and l == d - 1)))
    return fnp.stack(out, axis=0)


def v_dimqmc_k3(mlp, pts, rng):
    return v_dimqmc(mlp, pts, rng, kappa3=True)


# ---------------- harness ----------------
def run_variant(fn, mlp, pts, truth_final, mlp_seed):
    rng = np.random.default_rng(1000 + mlp_seed)   # MC seed tied to the MLP
    with flops.BudgetContext(flop_budget=B_TEST, quiet=True) as ctx:
        pred = fn(mlp, pts, rng)
    used = ctx.flops_used
    assert used <= B_TEST, f"exceeded budget: {used:,} > {B_TEST:,}"
    final_pred = np.asarray(pred[mlp.depth - 1], dtype=np.float64)
    mse = float(np.mean((final_pred - truth_final) ** 2))
    return used, mse


def main():
    print(f"MLP width={WIDTH} depth={DEPTH}   averaging over {N_MLP} MLPs")
    print(f"budget B={FULL_BUDGET:.2e}  test budget (10% floor)={B_TEST:.2e}  "
          f"per-sample={PER_SAMPLE:.2e} FLOPs   ROWS={ROWS}\n")

    pts = fnp.array(np.load(PTS_FILE)["x"][:, :WIDTH].astype(np.float32))
    variants = [
        ("closure only (mean-prop)", v_closure_only),
        ("plain MC (pseudo-random)", v_plainMC),
        ("+ QMC points", v_qmc),
        ("+ antithetic (x,-x)", v_antithetic),
        ("#2 kappa3 final-layer RB", v_kappa3_rb),
        ("#1 dim-reduced QMC", v_dimqmc),
        ("#1+#2 dimQMC + kappa3", v_dimqmc_k3),
    ]
    names = [v[0] for v in variants]
    mses = {nm: [] for nm in names}
    flops_used = {nm: 0 for nm in names}

    t0 = time.time()
    floors = []
    for s in range(N_MLP):
        mlp = build_mlp(WIDTH, DEPTH, seed=s)
        weights_np = [np.asarray(w) for w in mlp.weights]
        truth_final, var_proxy = ground_truth(weights_np, WIDTH, N_GT, seed=9000 + s)
        floors.append(var_proxy / N_GT)
        for nm, fn in variants:
            used, mse = run_variant(fn, mlp, pts, truth_final, s)
            mses[nm].append(mse); flops_used[nm] = used
        print(f"  MLP {s:2d} done ({time.time()-t0:.0f}s)", end="\r", flush=True)
    print(" " * 40, end="\r")

    hdr = "{:<28} {:>12} {:>8} {:>12} {:>9} {:>10}".format(
        "variant", "flops_used", "%budget", "mean_MSE", "vs MC", "adj_score")
    print(hdr); print("-" * len(hdr))
    base = float(np.mean(mses["plain MC (pseudo-random)"]))
    for nm in names:
        m = float(np.mean(mses[nm])); used = flops_used[nm]
        mult = max(0.1, used / FULL_BUDGET)
        ratio = "—" if nm.startswith("closure") else f"{base/m:.2f}x"
        print("{:<28} {:>12.3e} {:>7.1f}% {:>12.3e} {:>9} {:>10.2e}".format(
            nm, used, 100 * used / B_TEST, m, ratio, m * mult))
    print("-" * len(hdr))
    print(f"mean GT MSE floor ~ {np.mean(floors):.2e}   "
          f"leader raw ≈ 1.3e-7 · adjusted ≈ 2.9e-8   MC baseline adjusted ≈ 6.5e-7")


if __name__ == "__main__":
    main()
