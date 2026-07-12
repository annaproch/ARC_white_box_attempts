"""CALIBRATED K=3/K=4 CLOSURE — GPU-batched, checkpointed, resumable.

Closure (all measured, none guessed):
  state per layer: pre-activation (m, S), plus kappa3 / kappa4 carried as
  rank-r Tucker cores in a top-r basis V of S, plus the star tensors born at
  the previous layer kept at FULL fidelity (O(n^3) matmuls, never materialized).

  corrections (validated in step_unit_test: mean up to 1300x, cov up to 193x):
      mu_a  = gauss + c3m*(1/6) h3 k3_iii + c4m*(1/24) h4 k4_iiii
      Sig_a = wick  + c3c*(1/2)[h2_i h1_j k3_iij + h1_i h2_j k3_ijj]
                    + c4c*(1/4)  h2_i h2_j k4_iijj
  generation (star diagrams; degree-2 vertex -> h2, degree-3 -> h3):
      kappa3 <- c3g * sum_i h2_i sym(w_i (x) B_i (x) B_i)
      kappa4 <- c4g * sum_i h3_i sym(w_i (x) B_i (x) B_i (x) B_i)
      with B = S diag(h1) W'
  pass-through: cores transform as T x M x ... x M,  M = V'^T W'^T diag(h1) V

All coefficients (per layer: 5 K=2 params + 8 cumulant scales) are LEARNED
end-to-end against fresh random MLPs, initialised at their theoretical values.
This absorbs (a) the ~2x magnitude deficit from rank truncation, (b) the
diagram terms we do not compute (chains, triangles), (c) residual closure bias.

Efficiency: everything is batched over MLPs and stays on the GPU. The top-r
basis comes from randomized subspace iteration (matmul + Cholesky orth), not
eigh, so there is no CPU round-trip. Basis is detached (its gradient is a
second-order effect).

Checkpointing: state saved to data/calib_k3k4.pt after every eval. Ctrl-C any
time; re-running resumes. --export writes the best params for the estimator.

Run:  cd ../mlp_cumulant_propagation && uv run python ../research/calibrate_k3k4.py
      flags: --rank 16 --batch-mlps 8 --steps 100000 --export
"""

import argparse
import math
import signal
import time
from pathlib import Path

import torch

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_W, DEPTH = 256, 32
SQ2PI = math.sqrt(2 * math.pi)
N_TERMS = 4
DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)
CKPT = DATA / "calib_k3k4.pt"
VAL_CACHE = DATA / "val_gt_24.pt"

def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=32)      # kappa3 rank
    ap.add_argument("--rank4", type=int, default=8)      # kappa4 rank (cost ~ n r^4)
    ap.add_argument("--batch-mlps", type=int, default=8)
    # the MC target is an UNBIASED gradient signal at any batch size, so a small
    # x-batch just adds gradient noise -- and it dominates step cost. 2048 is ~4x
    # cheaper per step than 8192 and lets us take many more steps.
    ap.add_argument("--x-batch", type=int, default=2048)   # (unused: exact targets now)
    ap.add_argument("--train-mlps", type=int, default=128)
    ap.add_argument("--train-samples", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=50)
    # lower LR + cosine decay: the old 3e-3 with a noisy target oscillated
    # (best at step 600, then thrashing between 1.4e-5 and 3.5e-5).
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-mlps", type=int, default=24)
    ap.add_argument("--val-samples", type=int, default=2_000_000)
    ap.add_argument("--export", action="store_true", help="export best params and exit")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoint")
    return ap.parse_args()


# Module-level ranks so the closure can be imported and reused (export_closure.py).
# Call set_rank() to match a checkpoint before calling closure().
#
# kappa3's core costs O(n r^3), kappa4's costs O(n r^4) -- at r=32 the kappa4 core
# alone was 1.7e10 FLOPs (8.4% of budget), which doubled the score multiplier and
# ate a real 9% accuracy gain. kappa4 is nearly rank-1 at depth (measured), so it
# needs far less rank than kappa3. R4 indexes the LEADING columns of the (sorted)
# basis, i.e. kappa4 lives in kappa3's dominant sub-subspace.
R = 16      # kappa3 rank
R4 = 8      # kappa4 rank (<= R)

# 13 params/layer: [p, q, moff, s_dia, s_off, c3m, c3c, c3g, c3p, c4m, c4c, c4g, c4p]
NP = 13
IDX = dict(p=0, q=1, moff=2, s_dia=3, s_off=4,
           c3m=5, c3c=6, c3g=7, c3p=8, c4m=9, c4c=10, c4g=11, c4p=12)


def scales(theta):
    """theta (L, 13) -> dict of per-layer coefficients."""
    t = torch.tanh(theta)
    return {
        "p": 0.10 * t[:, IDX["p"]],
        "q": 0.10 * t[:, IDX["q"]],
        "moff": 0.10 * t[:, IDX["moff"]],
        "s_dia": 0.40 * t[:, IDX["s_dia"]],
        "s_off": 0.40 * t[:, IDX["s_off"]],
        # cumulant coefficients: exp(1.5 tanh) in [0.22, 4.48], init 1.0
        "c3m": torch.exp(1.5 * t[:, IDX["c3m"]]),
        "c3c": torch.exp(1.5 * t[:, IDX["c3c"]]),
        "c3g": torch.exp(1.5 * t[:, IDX["c3g"]]),
        "c3p": torch.exp(1.5 * t[:, IDX["c3p"]]),
        "c4m": torch.exp(1.5 * t[:, IDX["c4m"]]),
        "c4c": torch.exp(1.5 * t[:, IDX["c4c"]]),
        "c4g": torch.exp(1.5 * t[:, IDX["c4g"]]),
        "c4p": torch.exp(1.5 * t[:, IDX["c4p"]]),
    }


def make_ws(seeds):
    out = [torch.randn(DEPTH, N_W, N_W, generator=torch.Generator().manual_seed(int(s)))
           * math.sqrt(2.0 / N_W) for s in seeds]
    return torch.stack(out).to(DEV)


def mc_final(Ws, n_samples, chunk=16384):
    B = Ws.shape[0]
    acc = torch.zeros(B, N_W, device=DEV)
    done = 0
    while done < n_samples:
        c = min(chunk, n_samples - done)
        x = torch.randn(B, c, N_W, device=DEV)
        for li in range(DEPTH):
            x = torch.relu(torch.bmm(x, Ws[:, li]))
        acc += x.sum(1)
        done += c
    return acc / n_samples


_Q0 = torch.randn(N_W, R, generator=torch.Generator().manual_seed(7)).to(DEV)


def set_rank(r, r4=None):
    """Set the kappa3 rank (and kappa4's, defaulting to min(8, r))."""
    global R, R4, _Q0
    R = int(r)
    R4 = int(r4) if r4 is not None else min(8, R)
    _Q0 = torch.randn(N_W, R, generator=torch.Generator().manual_seed(7)).to(DEV)


def _qr(Y):
    """Batched orthonormal basis of Y's columns. Householder QR is stable when
    Sigma has a large eigenvalue gap (it does, at depth) -- Cholesky-QR is not,
    and collapsed to NaN there. Falls back to CPU if MPS lacks the kernel."""
    try:
        return torch.linalg.qr(Y).Q
    except Exception:
        return torch.linalg.qr(Y.cpu()).Q.to(Y.device)


@torch.no_grad()
def top_basis(S, iters=2):
    """Top-R subspace of S (B,n,n) by randomized subspace iteration, with a
    Rayleigh-Ritz step so the columns come out ORDERED by eigenvalue.

    Ordering matters: kappa4 uses a smaller rank than kappa3 and takes the first
    R4 columns of this basis, which is only the dominant sub-subspace if the
    columns are sorted. Detached (its gradient is a second-order effect)."""
    B = S.shape[0]
    Q = _Q0.expand(B, N_W, R).contiguous()
    for _ in range(iters):
        Q = _qr(torch.bmm(S, Q))
    Sr = torch.bmm(Q.transpose(1, 2), torch.bmm(S, Q))       # (B, R, R)
    Sr = 0.5 * (Sr + Sr.transpose(1, 2))
    try:
        w, U = torch.linalg.eigh(Sr)
    except Exception:
        w, U = torch.linalg.eigh(Sr.cpu())
        w, U = w.to(S.device), U.to(S.device)
    U = torch.flip(U, dims=[2])                              # descending eigenvalue
    return torch.bmm(Q, U)


def wick_cov(a, pdf, cdf, s, rho):
    g = [cdf, pdf]
    he_prev, he_cur = torch.ones_like(a), a
    for k in range(3, N_TERMS + 1):
        m = k - 2
        g.append(((-1) ** m) * he_cur * pdf)
        he_prev, he_cur = he_cur, a * he_cur - m * he_prev
    acc = torch.zeros_like(rho)
    rho_k, sf = rho, 1.0
    for k in range(1, N_TERMS + 1):
        sf *= math.sqrt(k)
        gh = g[k - 1] / sf
        acc = acc + gh.unsqueeze(2) * gh.unsqueeze(1) * rho_k
        if k < N_TERMS:
            rho_k = rho_k * rho
    return (s.unsqueeze(2) * s.unsqueeze(1)) * acc


def closure(Ws, theta):
    """Batched K=2 + kappa3 + kappa4 closure. Ws (B,L,n,n) -> final mean (B,n)."""
    B = Ws.shape[0]
    c = scales(theta)
    eye = torch.eye(N_W, device=DEV)

    m = torch.zeros(B, N_W, device=DEV)
    S = torch.bmm(Ws[:, 0].transpose(1, 2), Ws[:, 0])
    V = top_basis(S)
    T3 = torch.zeros(B, R, R, R, device=DEV)
    T4 = torch.zeros(B, R4, R4, R4, R4, device=DEV)   # kappa4: much smaller rank
    star = None                                   # (h2, h3, Wr, Bs) in z_l space

    for li in range(DEPTH):
        var = torch.clamp(torch.diagonal(S, dim1=1, dim2=2), min=1e-8)
        s = var.sqrt()
        al = torch.clamp(m / s, -12.0, 12.0)
        pdf = torch.exp(-0.5 * al * al) / SQ2PI
        cdf = 0.5 * (1 + torch.erf(al / math.sqrt(2)))
        h1 = cdf
        # star-diagram vertex weights (degree 2 -> h2, degree 3 -> h3). Bounded by
        # the var floor: |h2| <= 4e3, |h3| <= 5e8 -- safe in fp32, and they always
        # meet B-factors carrying compensating powers of sigma.
        h2 = pdf / s
        h3 = -al * pdf / var
        ss = s.unsqueeze(2) * s.unsqueeze(1)
        rho = torch.clamp(S / ss, -1.0, 1.0)

        # ---- cumulant slices: carried cores + full-fidelity fresh star
        G2 = (V.unsqueeze(3) * V.unsqueeze(2)).reshape(B, N_W, R * R)      # (B,n,R^2)
        P3 = torch.bmm(G2, T3.reshape(B, R * R, R))                        # (B,n,R)
        k3_slice = torch.bmm(P3, V.transpose(1, 2))                        # (B,n,n)
        k3_diag = (P3 * V).sum(2)
        V4 = V[:, :, :R4]                                                  # dominant sub-basis
        G2b = (V4.unsqueeze(3) * V4.unsqueeze(2)).reshape(B, N_W, R4 * R4)
        P4 = torch.bmm(G2b, T4.reshape(B, R4 * R4, R4 * R4))               # (B,n,R4^2)
        k4_slice = torch.bmm(P4, G2b.transpose(1, 2))                      # (B,n,n)
        k4_diag = (P4 * G2b).sum(2)
        if star is not None:
            h2s, h3s, Wr, Bs = star
            C3 = h2s.unsqueeze(2) * Wr * Bs
            D3 = h2s.unsqueeze(2) * Bs * Bs
            k3_slice = k3_slice + 2.0 * torch.bmm(C3.transpose(1, 2), Bs) \
                                + torch.bmm(D3.transpose(1, 2), Wr)
            k3_diag = k3_diag + 3.0 * (h2s.unsqueeze(2) * Wr * Bs * Bs).sum(1)
            C4 = h3s.unsqueeze(2) * Wr * Bs
            D4 = h3s.unsqueeze(2) * Bs * Bs
            k4_slice = k4_slice + 2.0 * torch.bmm(C4.transpose(1, 2), Bs * Bs) \
                                + 2.0 * torch.bmm(D4.transpose(1, 2), Wr * Bs)
            k4_diag = k4_diag + 4.0 * (h3s.unsqueeze(2) * Wr * Bs * Bs * Bs).sum(1)

        # ---- corrections in DIMENSIONLESS form: divide the cumulants by their
        # sigma powers FIRST, then multiply by the bounded (phi, Phi) factors.
        # (h3 ~ 1/sigma^2 and h4 ~ 1/sigma^3 overflow fp32 on dead neurons.)
        t3 = k3_diag / var                       # kappa3_iii / sigma^2  (~ sigma)
        t4 = k4_diag / (var * s)                 # kappa4_iiii / sigma^3 (~ sigma)
        q3 = k3_slice / s.unsqueeze(2)           # kappa3_iij / sigma_i
        q4 = k4_slice / ss                       # kappa4_iijj / (sigma_i sigma_j)

        cm3, cc3 = c["c3m"][li], c["c3c"][li]
        cm4, cc4 = c["c4m"][li], c["c4c"][li]
        mu_a = (s * pdf * (1 + c["p"][li]) + m * cdf * (1 + c["q"][li])
                + c["moff"][li] * s
                - cm3 * (1.0 / 6.0) * al * pdf * t3                    # h3 * k3_iii
                + cm4 * (1.0 / 24.0) * (al * al - 1) * pdf * t4)       # h4 * k4_iiii
        ez2 = ((m * m + var) * cdf + m * s * pdf
               + cm3 * (1.0 / 3.0) * pdf * (k3_diag / s))              # h2 * k3_iii
        var_a = torch.clamp(ez2 - mu_a * mu_a, min=1e-12) * (1 + c["s_dia"][li])

        Sig = wick_cov(al, pdf, cdf, s, rho) * (1 + c["s_off"][li])
        c3_term = pdf.unsqueeze(2) * h1.unsqueeze(1) * q3              # h2_i h1_j k3_iij
        Sig = Sig + cc3 * 0.5 * (c3_term + c3_term.transpose(1, 2))
        Sig = Sig + cc4 * 0.25 * (pdf.unsqueeze(2) * pdf.unsqueeze(1) * q4)
        Sig = Sig * (1 - eye) + torch.diag_embed(var_a)

        if li == DEPTH - 1:
            return mu_a

        # ---- push through the next linear layer
        W = Ws[:, li + 1]
        S_pre = S                                                  # this layer's z-cov
        m = torch.bmm(mu_a.unsqueeze(1), W).squeeze(1)
        S = torch.bmm(W.transpose(1, 2), torch.bmm(Sig, W))
        V_next = top_basis(S)

        Z = h1.unsqueeze(2) * torch.bmm(W, V_next)                 # (B,n,R)
        M = torch.bmm(Z.transpose(1, 2), V)                        # (B,R,R): new <- old
        M4 = M[:, :R4, :R4]                                        # kappa4's sub-basis map
        T3 = c["c3p"][li] * torch.einsum("bmkl,bam,bnk,bol->bano", T3, M, M, M)
        T4 = c["c4p"][li] * torch.einsum("bmklp,bam,bnk,bol,bqp->banoq",
                                         T4, M4, M4, M4, M4)

        if star is not None:                                       # compress the old star
            # The star tensor is sum_i h_i sym(A_i (x) B_i (x) B_i ...): the A-factor
            # must sit in EACH slot once. Written as explicit einsums -- doing it
            # with permutes put A in slot 1 twice and never in slot 2, so the core
            # was not symmetric (flopscope's symmetry check caught this) and every
            # kappa3/kappa4 slice was wrong.
            h2s, h3s, Wr, Bs = star
            A = torch.bmm(Wr, Z)
            Bv = torch.bmm(Bs, Z)
            T3 = T3 + c["c3g"][li] * (
                torch.einsum("bi,bim,bik,bil->bmkl", h2s, A, Bv, Bv)
                + torch.einsum("bi,bim,bik,bil->bmkl", h2s, Bv, A, Bv)
                + torch.einsum("bi,bim,bik,bil->bmkl", h2s, Bv, Bv, A))
            A4, Bv4 = A[:, :, :R4], Bv[:, :, :R4]
            T4 = T4 + c["c4g"][li] * (
                torch.einsum("bi,bim,bik,bil,bip->bmklp", h3s, A4, Bv4, Bv4, Bv4)
                + torch.einsum("bi,bim,bik,bil,bip->bmklp", h3s, Bv4, A4, Bv4, Bv4)
                + torch.einsum("bi,bim,bik,bil,bip->bmklp", h3s, Bv4, Bv4, A4, Bv4)
                + torch.einsum("bi,bim,bik,bil,bip->bmklp", h3s, Bv4, Bv4, Bv4, A4))

        # the new star is born from THIS layer's pre-activation stats (S, h1, h2, h3)
        star = (h2, h3, W, torch.bmm(S_pre, h1.unsqueeze(2) * W))
        V = V_next

    raise RuntimeError


def load_val(n_mlps=24, n_samples=2_000_000):
    """Cached validation ground truth."""
    if VAL_CACHE.exists():
        blob = torch.load(VAL_CACHE)
        return blob["seeds"], blob["gt"].to(DEV)
    seeds = list(range(80_000, 80_000 + n_mlps))
    t0 = time.time()
    gt = mc_final(make_ws(seeds), n_samples)
    torch.save({"seeds": seeds, "gt": gt.cpu(), "n_samples": n_samples}, VAL_CACHE)
    print(f"built validation GT in {time.time()-t0:.0f}s")
    return seeds, gt


def load_train(n_mlps, n_samples=1_000_000, chunk=16):
    """Fixed training corpus with ACCURATE targets.

    Drawing a fresh MC target every step (2048 samples) gives per-neuron noise
    ~sqrt(0.012/2048) = 2.4e-3, comparable to the error being minimised (~4e-3
    rms) -- the gradient was mostly noise, so training oscillated instead of
    converging. Precomputing accurate targets once fixes that AND removes the
    per-step sampling, which dominated step cost.

    416 params vs n_mlps*256 targets => overfitting is not a real risk, and the
    held-out val set would reveal it anyway.
    """
    cache = DATA / f"train_gt_{n_mlps}.pt"
    if cache.exists():
        blob = torch.load(cache)
        return blob["seeds"], blob["gt"].to(DEV)
    seeds = list(range(500_000, 500_000 + n_mlps))
    t0 = time.time()
    gts = []
    for i in range(0, n_mlps, chunk):
        gts.append(mc_final(make_ws(seeds[i:i + chunk]), n_samples).cpu())
        print(f"  train GT {min(i+chunk, n_mlps)}/{n_mlps} ({time.time()-t0:.0f}s)")
    gt = torch.cat(gts)
    torch.save({"seeds": seeds, "gt": gt, "n_samples": n_samples}, cache)
    print(f"built training corpus: {n_mlps} MLPs x {n_samples:.0e} samples "
          f"in {time.time()-t0:.0f}s")
    return seeds, gt.to(DEV)


def main():
    args = _parse()
    set_rank(args.rank, args.rank4)

    val_seeds, val_gt = load_val(args.val_mlps, args.val_samples)
    WV = make_ws(val_seeds)
    tr_seeds, tr_gt = load_train(args.train_mlps, args.train_samples)
    WT = make_ws(tr_seeds)     # (n_train, L, n, n) — kept on device, ~2 GB at 192

    @torch.no_grad()
    def val_mse(theta):
        tot, bs = 0.0, 8
        for i in range(0, len(val_seeds), bs):
            est = closure(WV[i:i + bs], theta)
            tot += float(((est - val_gt[i:i + bs]) ** 2).sum())
        return tot / (len(val_seeds) * N_W)

    theta = torch.zeros(DEPTH, NP, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=args.lr)
    start_step, best = 1, (float("inf"), theta.detach().clone())
    if CKPT.exists() and not args.fresh:
        st = torch.load(CKPT, map_location=DEV)
        if st.get("rank", R) != R or st.get("rank4", R4) != R4:
            print(f"checkpoint ranks {st.get('rank')}/{st.get('rank4')} != "
                  f"{R}/{R4}; use --fresh or match them")
            raise SystemExit(1)
        with torch.no_grad():
            theta.copy_(st["theta"])
        opt.load_state_dict(st["opt"])
        start_step = st["step"] + 1
        best = (st["best_val"], st["best_theta"].to(DEV))
        print(f"resumed from step {st['step']} (best val {st['best_val']:.3e})")

    if args.export:
        out = DATA / "theta_k3k4_best.pt"
        torch.save({"theta": best[1].cpu(), "rank": R, "val": best[0]}, out)
        print(f"exported best params (val {best[0]:.3e}) -> {out}")
        return

    base = val_mse(torch.zeros(DEPTH, NP, device=DEV))
    print(f"uncalibrated closure (theta=0): {base:.3e}   [K=2 ~1.3e-4, v4 raw 3.1e-6]")
    if best[0] == float("inf"):
        best = (base, theta.detach().clone())

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    def save(step):
        torch.save({"theta": theta.detach().cpu(), "opt": opt.state_dict(), "step": step,
                    "best_val": best[0], "best_theta": best[1].cpu(),
                    "rank": R, "rank4": R4}, CKPT)

    print(f"training on {len(tr_seeds)} MLPs with exact targets "
          f"(Ctrl-C to stop; checkpointed every eval)\n")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.steps - start_step, 1), eta_min=args.lr * 0.02)
    t0, step0 = time.time(), start_step
    for step in range(start_step, args.steps + 1):
        idx = torch.randperm(len(tr_seeds))[: args.batch_mlps]
        est = closure(WT[idx], theta)
        loss = ((est - tr_gt[idx]) ** 2).mean()      # exact target: clean gradient
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or stop["flag"]:
            v = val_mse(theta.detach())
            if v < best[0]:
                best = (v, theta.detach().clone())
            save(step)
            rate = (step - step0 + 1) / max(time.time() - t0, 1e-9)
            print(f"step {step:6d}  train {float(loss):.3e}  val {v:.3e}  "
                  f"best {best[0]:.3e}  lr {sched.get_last_lr()[0]:.1e}  "
                  f"({rate:.1f} steps/s)")
        if stop["flag"]:
            print("\nstopped; state saved. Re-run to resume, "
                  "or run research/export_closure.py to ship the best params.")
            break

    print(f"\nbest val MSE {best[0]:.3e}  (uncalibrated {base:.3e}, "
          f"gain {base/max(best[0],1e-30):.2f}x)")
    print("v4 raw MSE 3.1e-6 -> beat that and pure-analytic at the floor wins.")


if __name__ == "__main__":
    main()
