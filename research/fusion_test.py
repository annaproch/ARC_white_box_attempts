"""Tune the MC + kprop-K2 fusion rule at the challenge shape (256x32).

Implements a self-contained K=2 cumulant propagation (Wick-series covariance
update, exact diagonal) matching what will be ported to flopscope, verifies it
against mlp_kprop k_max=2, then compares fusion rules over 10 seeds.
"""

import math
import time

import torch

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop

torch.set_grad_enabled(False)

N_W, DEPTH = 256, 32
GT_SAMPLES = 4_000_000
BATCH = 131_072
B = 2.72e11
FRACTION = 0.097
FPS = DEPTH * (2 * N_W**2 + 2 * N_W) + 16 * N_W  # per-sample cost incl. mean/var accum + rng
KPROP2_COST = 2.4e9
SQRT2PI = math.sqrt(2 * math.pi)


def norm_pdf(x):
    return torch.exp(-0.5 * x * x) / SQRT2PI


def norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def kprop2_numpy_style(Ws, n_terms=4):
    """K=2 cumulant propagation; Ws: list of (n,n) with x@W forward convention."""
    n = Ws[0].shape[0]
    mu = torch.zeros(n, dtype=torch.float64)
    cov = torch.eye(n, dtype=torch.float64)
    rows = []
    for W in Ws:
        Wd = W.double()
        mu_pre = Wd.T @ mu
        cov_pre = Wd.T @ cov @ Wd
        var = torch.clamp(torch.diag(cov_pre), min=1e-12)
        s = var.sqrt()
        a = mu_pre / s
        pdf, cdf = norm_pdf(a), norm_cdf(a)
        mu_post = s * pdf + mu_pre * cdf
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu_post**2, min=0.0)
        # Wick coefficients w_k = E[d^k ReLU(Z)]
        wick = [cdf, pdf / s, -a * pdf / var, (a * a - 1) * pdf / (var * s)]
        cov_post = torch.zeros_like(cov_pre)
        Ck = cov_pre.clone()
        for k in range(1, n_terms + 1):
            wk = wick[k - 1]
            cov_post += torch.outer(wk, wk) * Ck / math.factorial(k)
            if k < n_terms:
                Ck = Ck * cov_pre
        cov_post.fill_diagonal_(0.0)
        cov_post += torch.diag(var_post)
        mu, cov = mu_post, cov_post
        rows.append(mu_post)
    return torch.stack(rows)


results = {}
for seed in range(10):
    torch.manual_seed(seed)
    mlp = MLP(hidden_dim=N_W, num_layers=DEPTH + 1, nonlin="relu",
              init_kind="manual", w_var=2.0)
    for Wm in mlp.Ws:
        Wm.weight.copy_(torch.randn(N_W, N_W) * math.sqrt(2.0 / N_W))
        if Wm.bias is not None:
            Wm.bias.zero_()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    Ws_dev = [mlp.Ws[l].weight.to(device) for l in range(DEPTH)]
    Ws_np_style = [mlp.Ws[l].weight.T.cpu() for l in range(DEPTH)]  # x@W convention

    # ground truth
    sums = torch.zeros(N_W, dtype=torch.float64)
    done = 0
    while done < GT_SAMPLES:
        b = min(BATCH, GT_SAMPLES - done)
        x = torch.randn(b, N_W, device=device)
        for l in range(DEPTH):
            x = torch.relu(x @ Ws_dev[l].T)
        sums += x.sum(dim=0).cpu().double()
        done += b
    gt = sums / GT_SAMPLES

    # analytic: my K=2 port (and reference mlp_kprop on seed 0 only)
    t0 = time.time()
    mine = kprop2_numpy_style(Ws_np_style)[-1]
    t_mine = time.time() - t0
    if seed == 0:
        K_in = {1: torch.zeros(N_W), 2: torch.eye(N_W)}
        ref = mlp_kprop(mlp, K_in, k_max=2, output_all=True, output_d_max=1,
                        up_to_layer=f"act{DEPTH-1}")[f"act{DEPTH-1}"][1].to_tensor().double()
        print(f"seed 0 sanity: my K2 mse={float(((mine-gt)**2).mean()):.3e}  "
              f"mlp_kprop K2 mse={float(((ref-gt)**2).mean()):.3e}  ({t_mine:.1f}s)")

    # MC at reduced budget (paying for kprop)
    n_full = int(FRACTION * B / FPS)
    n_mc = int((FRACTION * B - KPROP2_COST) / FPS)
    est = {}
    for tag, n_s in [("mc_full", n_full), ("mc_reduced", n_mc)]:
        x = torch.randn(n_s, N_W, device=device)
        for l in range(DEPTH):
            x = torch.relu(x @ Ws_dev[l].T)
        m = x.mean(dim=0).cpu().double()
        v = x.var(dim=0).cpu().double() / n_s
        est[tag] = (m, v)

    m, v = est["mc_reduced"]
    a = mine
    d = m - a  # MC-minus-analytic discrepancy
    fused = {}
    # rule A: per-neuron shrinkage, b2 = max(d^2 - v, tiny)
    b2 = torch.clamp(d * d - v, min=1e-14)
    lam = v / (v + b2)
    fused["fuse_A"] = m - lam * d
    # rule B: layer-pooled floor on b2
    b2_pool = torch.clamp(d * d - v, min=0.0).mean()
    b2b = torch.clamp(d * d - v, min=0.25 * float(b2_pool))
    lamb = v / (v + b2b)
    fused["fuse_B"] = m - lamb * d
    # oracle per-neuron weights (upper bound)
    b2_or = (a - gt) ** 2
    lam_or = v / (v + b2_or)
    fused["oracle"] = m - lam_or * d

    row = {"kprop2_mine": float(((a - gt) ** 2).mean()),
           "mc_full": float(((est["mc_full"][0] - gt) ** 2).mean()),
           "mc_reduced": float(((m - gt) ** 2).mean())}
    row.update({k: float(((p - gt) ** 2).mean()) for k, p in fused.items()})
    results[seed] = row

keys = list(results[0].keys())
print(f"\n{'seed':>4s} " + " ".join(f"{k:>11s}" for k in keys))
for seed, row in results.items():
    print(f"{seed:4d} " + " ".join(f"{row[k]:11.3e}" for k in keys))
mean = {k: sum(r[k] for r in results.values()) / len(results) for k in keys}
print(f"{'MEAN':>4s} " + " ".join(f"{mean[k]:11.3e}" for k in keys))
print(f"\nn_full={int(FRACTION*B/FPS)}, n_reduced={int((FRACTION*B-KPROP2_COST)/FPS)}")
