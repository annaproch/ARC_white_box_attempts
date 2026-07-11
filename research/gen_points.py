"""Generate the shipped QMC point set: scrambled Sobol -> Gaussian (inverse CDF),
then exact moment matching (zero mean, identity covariance). Offline = off-budget."""

import math

import numpy as np
import torch

N_PTS, DIM = 6272, 256  # ~0.096 * 2.72e11 budget at width 256, depth 32

torch.set_grad_enabled(False)
sob = torch.quasirandom.SobolEngine(DIM, scramble=True, seed=12345)
u = sob.draw(N_PTS, dtype=torch.float64).clamp(1e-9, 1 - 1e-9)
x = math.sqrt(2.0) * torch.erfinv(2 * u - 1)

x = x - x.mean(dim=0)
cov = (x.T @ x) / N_PTS
L = torch.linalg.cholesky(cov)
x = torch.linalg.solve_triangular(L, x.T, upper=False).T  # exact N(0, I) moments

chk_mean = float(x.mean(dim=0).abs().max())
chk_cov = float(((x.T @ x) / N_PTS - torch.eye(DIM)).abs().max())
print(f"moment check: max|mean|={chk_mean:.2e}  max|cov-I|={chk_cov:.2e}")

np.save("/private/tmp/claude-501/-Users-lisek-Projects-ARC-white-box/fffd79fe-57cb-423a-9b9c-e89e35bd048e/scratchpad/points_raw.npy",
        x.float().numpy())
print("saved points_raw.npy", x.shape)
