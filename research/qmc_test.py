"""Do better sample points beat iid MC at depth 32?

Fixed point sets (shippable offline, 0-FLOP load) tested across 8 MLP seeds:
  iid          - fresh iid N(0,1) per MLP (current estimator)
  sobol        - scrambled Sobol -> Gaussian via inverse CDF (fixed set)
  sobol_mm     - sobol + exact moment matching (zero mean, identity cov via whitening)
  iid_mm       - fixed iid draw + exact moment matching
"""

import math

import torch

from mlp_kprop.mlp import MLP

torch.set_grad_enabled(False)

N_W, DEPTH, N_PTS = 256, 32, 6259
GT_SAMPLES = 4_000_000
BATCH = 131_072
device = "mps" if torch.backends.mps.is_available() else "cpu"


def moment_match(x):
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / x.shape[0]
    L = torch.linalg.cholesky(cov.double()).float()
    return x @ torch.linalg.inv(L).T  # exact zero mean, identity covariance


g = torch.Generator().manual_seed(12345)
sob = torch.quasirandom.SobolEngine(N_W, scramble=True, seed=12345)
u = sob.draw(N_PTS).clamp(1e-6, 1 - 1e-6)
sobol = math.sqrt(2.0) * torch.erfinv(2 * u - 1)
iid_fixed = torch.randn(N_PTS, N_W, generator=g)
point_sets = {
    "sobol": sobol,
    "sobol_mm": moment_match(sobol),
    "iid_mm": moment_match(iid_fixed),
}

tags = ["iid"] + list(point_sets.keys())
acc = dict.fromkeys(tags, 0.0)
print(f"{'seed':>4s} " + " ".join(f"{t:>10s}" for t in tags))
for seed in range(8):
    torch.manual_seed(seed)
    mlp = MLP(hidden_dim=N_W, num_layers=DEPTH + 1, nonlin="relu",
              init_kind="manual", w_var=2.0)
    for Wm in mlp.Ws:
        Wm.weight.copy_(torch.randn(N_W, N_W) * math.sqrt(2.0 / N_W))
        if Wm.bias is not None:
            Wm.bias.zero_()
    Ws = [mlp.Ws[l].weight.to(device) for l in range(DEPTH)]

    sums = torch.zeros(N_W, dtype=torch.float64)
    done = 0
    while done < GT_SAMPLES:
        b = min(BATCH, GT_SAMPLES - done)
        x = torch.randn(b, N_W, device=device)
        for l in range(DEPTH):
            x = torch.relu(x @ Ws[l].T)
        sums += x.sum(dim=0).cpu().double()
        done += b
    gt = sums / GT_SAMPLES

    def run(x0):
        x = x0.to(device)
        for l in range(DEPTH):
            x = torch.relu(x @ Ws[l].T)
        return x.mean(dim=0).cpu().double()

    row = {"iid": run(torch.randn(N_PTS, N_W, generator=torch.Generator().manual_seed(1000 + seed)))}
    row.update({t: run(p) for t, p in point_sets.items()})
    mses = {t: float(((m - gt) ** 2).mean()) for t, m in row.items()}
    for t in tags:
        acc[t] += mses[t] / 8
    print(f"{seed:4d} " + " ".join(f"{mses[t]:10.3e}" for t in tags))
print(f"{'MEAN':>4s} " + " ".join(f"{acc[t]:10.3e}" for t in tags))
