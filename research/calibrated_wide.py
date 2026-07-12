"""Wide-bound calibrated K=2 — BATCHED on the GPU.

Why wide bounds: 98% of K=2's final-layer error is propagated-moment error
(error_attribution.py), driven by a 27-43% UNDERESTIMATE of Sigma's top
eigenvalue / mean off-diagonal (layerwise_cov_drift.py). Our previous
calibration clamped every correction to 0.05*tanh = +-5% -- it could not
express a 30-40% covariance correction, yet still bought 3x. Here the closure
gets per-layer off-diagonal / diagonal scalings with +-60% bounds.

Why batched: every op is a 256x256 matmul -- tiny for this GPU. Propagating one
MLP at a time in float64 on the CPU was latency-bound. This version keeps ALL
MLPs in one (B, n, n) bmm on MPS in float32 (which is what the shipped
flopscope estimator uses anyway), and caches validation ground truth to disk.
~10-30x faster => more steps, more MLPs/step, richer closures.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/calibrated_wide.py
      flags: --steps 3000 --batch-mlps 16 --refresh-val
"""

import argparse
import math
import time
from pathlib import Path

import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
N_TERMS = 4
DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)
VAL_CACHE = DATA / "val_gt_24.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch-mlps", type=int, default=16)   # MLPs propagated per step
ap.add_argument("--x-batch", type=int, default=8192)    # MC samples for the target
ap.add_argument("--val-mlps", type=int, default=24)
ap.add_argument("--val-samples", type=int, default=2_000_000)
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--refresh-val", action="store_true")
args = ap.parse_args()


def make_ws(seeds):
    """(B, L, n, n) He-init weights on DEV, float32."""
    out = []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        out.append(torch.randn(DEPTH, N_W, N_W, generator=g) * math.sqrt(2.0 / N_W))
    return torch.stack(out).to(DEV)


def mc_final(Ws, n_samples, chunk=16384):
    """Batched sampled final-layer mean: Ws (B,L,n,n) -> (B,n) float32 on DEV."""
    B = Ws.shape[0]
    acc = torch.zeros(B, N_W, device=DEV, dtype=torch.float32)
    done = 0
    while done < n_samples:
        c = min(chunk, n_samples - done)
        x = torch.randn(B, c, N_W, device=DEV)
        for li in range(DEPTH):
            x = torch.relu(torch.bmm(x, Ws[:, li]))
        acc += x.sum(1)
        done += c
    return acc / n_samples


def k2_batched(Ws, theta):
    """Batched K=2 covariance propagation with per-layer wide-bound corrections.
    Ws (B,L,n,n); theta (L,6). Returns final-layer mean (B,n)."""
    B = Ws.shape[0]
    mu = torch.zeros(B, N_W, device=DEV)
    cov = torch.eye(N_W, device=DEV).expand(B, N_W, N_W).contiguous()
    eye = torch.eye(N_W, device=DEV)
    for li in range(DEPTH):
        W = Ws[:, li]                                    # (B,n,n)
        t = theta[li]
        p = 0.10 * torch.tanh(t[0])
        q = 0.10 * torch.tanh(t[1])
        m = 0.10 * torch.tanh(t[2])
        s_off = 0.60 * torch.tanh(t[3])                  # the correction the data demands
        s_dia = 0.60 * torch.tanh(t[4])
        g2s = 0.30 * torch.tanh(t[5])

        mu_pre = torch.bmm(mu.unsqueeze(1), W).squeeze(1)             # (B,n)
        cov_pre = torch.bmm(W.transpose(1, 2), torch.bmm(cov, W))     # (B,n,n)
        var = torch.clamp(torch.diagonal(cov_pre, dim1=1, dim2=2), min=1e-10)
        s = var.sqrt()
        a = mu_pre / s
        pdf = torch.exp(-0.5 * a * a) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(a / math.sqrt(2)))
        ss = s.unsqueeze(2) * s.unsqueeze(1)                          # (B,n,n)
        rho = torch.clamp(cov_pre / ss, -1.0, 1.0)

        mu_new = s * pdf * (1 + p) + mu_pre * cdf * (1 + q) + m * s
        ez2 = (mu_pre**2 + var) * cdf + mu_pre * s * pdf
        var_post = torch.clamp(ez2 - mu_new**2, min=1e-12) * (1 + s_dia)

        g = [cdf, pdf * (1 + g2s)]
        he_prev, he_cur = torch.ones_like(a), a
        for k in range(3, N_TERMS + 1):
            mm = k - 2
            g.append(((-1) ** mm) * he_cur * pdf)
            he_prev, he_cur = he_cur, a * he_cur - mm * he_prev
        acc = torch.zeros_like(cov_pre)
        rho_k, sf = rho, 1.0
        for k in range(1, N_TERMS + 1):
            sf *= math.sqrt(k)
            gh = g[k - 1] / sf
            acc = acc + gh.unsqueeze(2) * gh.unsqueeze(1) * rho_k
            if k < N_TERMS:
                rho_k = rho_k * rho
        cov_new = ss * acc * (1 + s_off)
        cov_new = cov_new * (1 - eye) + torch.diag_embed(var_post)
        mu, cov = mu_new, cov_new
    return mu


# ---- validation set (cached) ----------------------------------------------
if VAL_CACHE.exists() and not args.refresh_val:
    blob = torch.load(VAL_CACHE)
    val_seeds, val_gt = blob["seeds"], blob["gt"].to(DEV)
    print(f"loaded cached validation GT ({len(val_seeds)} MLPs, "
          f"{blob['n_samples']:.0e} samples each)")
else:
    val_seeds = list(range(80_000, 80_000 + args.val_mlps))
    t0 = time.time()
    Wv = make_ws(val_seeds)
    val_gt = mc_final(Wv, args.val_samples)
    torch.save({"seeds": val_seeds, "gt": val_gt.cpu(), "n_samples": args.val_samples},
               VAL_CACHE)
    del Wv
    print(f"built validation GT: {len(val_seeds)} MLPs x {args.val_samples:.0e} "
          f"samples in {time.time()-t0:.0f}s (cached to {VAL_CACHE.name})")

WV = make_ws(val_seeds)


@torch.no_grad()
def val_mse(theta):
    est = k2_batched(WV, theta)
    return float(((est - val_gt) ** 2).mean())


theta0 = torch.zeros(DEPTH, 6, device=DEV)
base = val_mse(theta0)
print(f"plain K=2 (theta=0): {base:.3e}")

theta = torch.zeros(DEPTH, 6, device=DEV, requires_grad=True)
opt = torch.optim.Adam([theta], lr=args.lr)
best = (base, theta0.clone())
t0 = time.time()
for step in range(1, args.steps + 1):
    seeds = torch.randint(3_000_000, 9_000_000, (args.batch_mlps,)).tolist()
    Ws = make_ws(seeds)
    with torch.no_grad():
        target = mc_final(Ws, args.x_batch)          # unbiased MC target (B,n)
    est = k2_batched(Ws, theta)
    loss = ((est * est).sum(1) - 2 * (est * target).sum(1)).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([theta], 1.0)
    opt.step()
    if step % args.eval_every == 0:
        v = val_mse(theta.detach())
        if v < best[0]:
            best = (v, theta.detach().clone())
        print(f"step {step:5d}  val {v:.3e}  best {best[0]:.3e}  "
              f"({time.time()-t0:.0f}s, {step/(time.time()-t0):.1f} steps/s)")

th = best[1]
print(f"\nplain K=2      : {base:.3e}")
print(f"wide-calibrated: {best[0]:.3e}   ({base/best[0]:.2f}x)")
print("(previous +-5%-clamped calibration reached ~2.4e-5, i.e. ~3x)")

s_off = (0.60 * torch.tanh(th[:, 3])).cpu()
s_dia = (0.60 * torch.tanh(th[:, 4])).cpu()
print("\nlearned covariance corrections (measured bias: -27..-43% on lambda1):")
for l in (0, 1, 2, 3, 7, 15, 23, 31):
    print(f"  L{l:2d}: off-diag {s_off[l]:+.3f}   diag {s_dia[l]:+.3f}")

torch.save(th.cpu(), DATA / "theta_wide.pt")
print(f"\nsaved {DATA / 'theta_wide.pt'}")
print("<=5e-6 => go pure-analytic at the floor (drop QMC).")
