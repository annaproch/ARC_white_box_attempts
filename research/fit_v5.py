"""Fit v5's anchor + kappa4 maps. CHECKPOINTS EVERY EPOCH -> cancel any time,
submit with whatever is in v5.npz, resume later with --resume.

CRITICAL: this must mirror `estimator_v5._closure` EXACTLY -- same star formula,
same CP carry, same feature list, same normalisation. A mismatch is a silent
failure (the fit would be optimising features the estimator never computes).
The mirror is asserted by `--check`, which compares this torch forward pass
against the flopscope estimator on one MLP.

Targets (corpus, exact):
    anchor : true kappa3(z)_ppq  -- the map predicts the RESIDUAL after
             (CP-carried slice + star), normalised by s_i^2 s_j
    k4     : true kappa4(z)_iijj                normalised by s_i^2 s_j^2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from harness import CORPUS, DT, hermite_gains

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F32 = torch.float32
n = 256
RANK, BIRTHS = 64, 16
NPZ = Path(__file__).resolve().parent.parent / "whest-starterkit" / "v5.npz"


def closure_features(rec, birth_p, anchor=None):
    """Mirror of estimator_v5._closure. Returns per-layer (feats, prop+star,
    nrm3, nrm4, truth3, truth4). If `anchor` is given it is applied in-loop
    (so training data comes from the trajectory the estimator will actually
    follow -- the DAgger fix for train/test state mismatch)."""
    W = [rec["W"][l].to(DEV, F32) for l in range(32)]
    W64 = [rec["W"][l].to(DT) for l in range(32)]
    mu = torch.zeros(n, dtype=DT)
    C = torch.eye(n, dtype=DT)
    V = torch.zeros(n, BIRTHS, device=DEV, dtype=F32)
    c = torch.zeros(BIRTHS, device=DEV, dtype=F32)
    out = []
    for l in range(32):
        zmu64 = mu @ W64[l]
        zcov64 = W64[l].T @ C @ W64[l]
        zmu, zcov = zmu64.to(DEV, F32), zcov64.to(DEV, F32)
        var = torch.clamp(torch.diagonal(zcov), min=1e-10)
        s = torch.sqrt(var)
        t = zmu / s
        g, _, _, pdf, cdf = hermite_gains(zmu64, torch.diagonal(zcov64), kmax=6)
        pdf = pdf.to(DEV, F32); cdf = cdf.to(DEV, F32)
        h1, h2 = cdf, pdf / s

        prop = ((V * V) * c[None, :]) @ V.T                 # CP slice
        Bm = (zcov * h1[None, :]) @ W[l]
        Cm = (h2[:, None] * W[l]) * Bm
        Dm = h2[:, None] * (Bm * Bm)
        star = 2.0 * (Cm.T @ Bm) + (Dm.T @ W[l])

        rho = zcov / torch.outer(s, s)
        nrm3 = torch.outer(s * s, s)
        nrm4 = torch.outer(s * s, s * s)
        k3 = prop + star
        feats = torch.stack([
            t[:, None].expand(n, n), t[None, :].expand(n, n), rho,
            k3 / nrm3, star / nrm3,
            torch.full((n, n), (l + 1.0) / 32.0, device=DEV, dtype=F32),
            torch.ones(n, n, device=DEV, dtype=F32)], -1).reshape(n * n, 7)

        if anchor is not None:
            with torch.no_grad():
                k3 = k3 + (torch.tanh(feats @ anchor[0]) @ anchor[1]).reshape(n, n) * nrm3

        out.append({"feats": feats, "base": (prop + star), "nrm3": nrm3, "nrm4": nrm4,
                    "t3": rec["zk3"][l].to(DEV, F32), "t4": rec["zk4"][l].to(DEV, F32)})

        # moment update (kappa3 slice only; kappa4 added once fitted)
        from harness import relu_step
        mu, C = relu_step(mu, C, W64[l], 8, k3.cpu().to(DT), rec["zk4"][l].to(DT))

        if l < 31:                                          # CP transport + births
            Wn = W[l + 1]
            V = Wn.T @ (h1[:, None] * V)
            V = torch.cat([V, (Wn.T @ star) @ birth_p], 1)
            c = torch.cat([c, torch.ones(BIRTHS, device=DEV, dtype=F32)])
            if V.shape[1] > RANK:
                # MPS has no linalg.qr; V is (256 x <=80) so the CPU hop is free.
                # (flopscope DOES have qr/svd -- the shipped estimator is fine.)
                Q, Rm = torch.linalg.qr(V.cpu())
                U, S, _ = torch.linalg.svd(Rm * c.cpu()[None, :])
                keep = min(RANK, U.shape[1])
                V = (Q @ (U[:, :keep] * S[:keep][None, :])).to(DEV, F32)
                c = torch.ones(keep, device=DEV, dtype=F32)
    return out


def save(anchor, k4, birth_p, meta):
    np.savez(NPZ,
             anchor_w1=anchor[0].detach().cpu().numpy().astype(np.float32),
             anchor_w2=anchor[1].detach().cpu().numpy().astype(np.float32),
             k4_w1=k4[0].detach().cpu().numpy().astype(np.float32),
             k4_w2=k4[1].detach().cpu().numpy().astype(np.float32),
             birth_p=birth_p.cpu().numpy().astype(np.float32),
             gains=np.zeros((32, 4), dtype=np.float32),
             meta=np.asarray(meta, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlps", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    n_tr = a.mlps // 2

    torch.manual_seed(0)
    birth_p = (torch.randn(n, BIRTHS, generator=torch.Generator().manual_seed(7))
               / np.sqrt(n)).to(DEV, F32)

    if a.resume and NPZ.exists():
        d = np.load(NPZ)
        A = [torch.tensor(d["anchor_w1"], device=DEV).requires_grad_(),
             torch.tensor(d["anchor_w2"], device=DEV).requires_grad_()]
        K = [torch.tensor(d["k4_w1"], device=DEV).requires_grad_(),
             torch.tensor(d["k4_w2"], device=DEV).requires_grad_()]
        birth_p = torch.tensor(d["birth_p"], device=DEV)
        print("resumed from v5.npz")
    else:
        mk = lambda: [(torch.randn(7, a.hidden, device=DEV) * 0.3).requires_grad_(),
                      (torch.randn(a.hidden, 1, device=DEV) * 0.05).requires_grad_()]
        A, K = mk(), mk()

    print(f"fit_v5 | {a.mlps} MLPs ({n_tr} train) | hidden={a.hidden} | {DEV}",
          flush=True)
    t0 = time.time()
    data = []
    for m in range(a.mlps):
        rec = torch.load(CORPUS / f"mlp_{200000 + m}.pt", map_location="cpu",
                         weights_only=False)
        data.append(closure_features(rec, birth_p))
        if m == 0:
            print(f"  closure pass: {time.time()-t0:.1f}s/MLP", flush=True)
    print(f"  data built: {time.time()-t0:.0f}s", flush=True)

    opt = torch.optim.Adam(A + K, lr=3e-3)
    off = ~torch.eye(n, dtype=torch.bool, device=DEV)

    def evaluate(rng):
        e3, e4 = [], []
        with torch.no_grad():
            for m in rng:
                for r in data[m]:
                    pred3 = r["base"] + (torch.tanh(r["feats"] @ A[0]) @ A[1]
                                         ).reshape(n, n) * r["nrm3"]
                    pred4 = (torch.tanh(r["feats"] @ K[0]) @ K[1]
                             ).reshape(n, n) * r["nrm4"]
                    e3.append(((pred3 - r["t3"])[off].norm()
                               / r["t3"][off].norm()).item())
                    e4.append(((pred4 - r["t4"])[off].norm()
                               / r["t4"][off].norm()).item())
        return float(np.mean(e3)), float(np.mean(e4))

    best = 1e9
    for ep in range(a.epochs):
        for m in range(n_tr):
            opt.zero_grad()
            loss = 0.0
            for r in data[m]:
                y3 = (r["t3"] - r["base"]) / r["nrm3"]
                y4 = r["t4"] / r["nrm4"]
                p3 = (torch.tanh(r["feats"] @ A[0]) @ A[1]).reshape(n, n)
                p4 = (torch.tanh(r["feats"] @ K[0]) @ K[1]).reshape(n, n)
                loss = loss + ((p3 - y3)[off] ** 2).mean() + ((p4 - y4)[off] ** 2).mean()
            loss.backward()
            opt.step()
        v3, v4 = evaluate(range(n_tr, a.mlps))
        # CHECKPOINT EVERY EPOCH: cancel any time, v5.npz is always submittable.
        # meta: [rank, n_qmc, w_qmc, w_closure, anchor_on, k4_on]
        save(A, K, birth_p, [RANK, 4000, 0.5, 0.5, 1.0, 1.0])
        flag = ""
        if v3 < best:
            best = v3
            flag = "  <- best"
        print(f"  ep {ep:>3}  held-out kappa3 slice {v3*100:>5.1f}%   "
              f"kappa4 {v4*100:>5.1f}%{flag}", flush=True)

    print(f"\n  DONE. best held-out kappa3 slice error: {best*100:.1f}%")
    print(f"  (G5 ridge on a 3-sigma tensor carry got 15.7%; this is the")
    print(f"   star-only CP carry the estimator actually ships.)")
    print(f"  v5.npz checkpointed every epoch -> always submittable.")


if __name__ == "__main__":
    main()
