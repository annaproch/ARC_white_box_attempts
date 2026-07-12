"""ORDER-1 PASS-THROUGH DIAGRAMS for kappa3  (the third half-written formula).

E8: with a PERFECT carry the propagated kappa3 is still 22-27% off. Only the
leading pass-through diagram (0 extra Sigma edges) was implemented:

    order 0 :  h1(p) h1(q) h1(r) . k3(z)_pqr                      [HAVE]

Thm A.2 gives a SERIES. The order-1 families (one k3(z) block + one Sigma edge,
legs summing to 5) are:

 (i) SPANNING k3 block (1,1,1) + one Sigma edge.
     legs (2,2,1); count = 2*2*1 = 4; coeff 4/(2!2!1!) = 1:
        h2(p) h2(q) h1(r) k3_pqr Sigma_pq      + (Sigma_pr, Sigma_qr perms)

 (ii) NON-SPANNING k3 block (2,1,0) -- the third vertex is isolated, so the
      Sigma edge must connect it. Two sub-cases per assignment:
        Sigma_pr: legs (3,1,1); count = C(3,2) = 3; coeff 3/(3!1!1!) = 1/2:
            1/2 . h3(p) h1(q) h1(r) k3_ppq Sigma_pr
        Sigma_qr: legs (2,2,1); count = 2;         coeff 2/(2!2!1!) = 1/2:
            1/2 . h2(p) h2(q) h1(r) k3_ppq Sigma_qr
      over all 6 assignments of (doubled, single) among {p,q,r}.
 (k3_ppp needs two Sigma edges to connect q and r -> order 2, excluded.)

Family (ii) only needs the kappa3 SLICE k3_ppq -- which we already compute --
and contracts to n^3 matmuls. Family (i) needs the full tensor coupled through
Sigma, costing n^3 per carried term.

Contractions (k3(z')_iij = sum_pqr k3(a)_pqr W_pi W_qi W_rj):

 (ii), Sigma_pr case, doubled=p single=q:
     KW = K @ (h1 . W)        (K[p,q] = k3_ppq)          n^3
     B  = Sz @ (h1 . W)                                   (already computed)
     term = 1/2 * ((h3 . W . KW)^T @ B)                   n^3

 (i), for a carried term sym3(x,y,y) and edge Sigma_pq:
     A = (h2.x) . W ; Bm = (h2.y) . W ; f(i) = colsum(A . (Sz @ Bm))
     g = W^T (h1.y) ;  term = outer(f, g)                 n^3 per term
"""
from __future__ import annotations

import torch


def order1_from_slices(K_iij, Sz, W, h1, h2, h3):
    """Family (ii): all order-1 diagrams whose kappa3 block is NON-SPANNING.
    K_iij[p,q] = k3(z)_ppq. Pure matmuls; GPU-friendly."""
    Wh1 = h1[:, None] * W
    B = Sz @ Wh1                                   # B[p,j] = sum_r h1(r) Sz_pr W_rj
    KW = K_iij @ Wh1                               # KW[p,i] = sum_q h1(q) K_pq W_qi
    KWt = K_iij.T @ Wh1                            # KWt[q,i] = sum_p h1(p) K_pq W_pi

    out = torch.zeros_like(Sz)
    # doubled index on p (contracts to i), single on q (also -> i), edge to r (-> j)
    out += 0.5 * (((h3[:, None] * W) * KW).T @ B)          # Sigma_pr case
    out += 0.5 * (((h2[:, None] * W) * KW).T @ B)          # Sigma_qr case
    # doubled index on the j-side (r): k3_rrp / k3_rrq
    out += 0.5 * (((h2[:, None] * W) * B).T @ (h3[:, None] * W * KWt)).T * 0
    # symmetric partner: the doubled vertex sits on the j-leg
    out += 0.5 * ((h2[:, None] * W * KW).T @ B)
    return out


def order1_spanning(X, Y, lam, Sz, W, h1, h2, dev=None):
    """Family (i): SPANNING kappa3 block + one Sigma edge, for a carry held as
    sum_s lam_s sym3(x_s, y_s, y_s).  Batched over the carried terms on GPU.

    X, Y: (S, n) rows x_s, y_s.  Cost S n^3 -- run it on MPS in float32.
    """
    if lam.numel() == 0:
        return torch.zeros_like(Sz)
    dt = torch.float32
    dev = dev or (torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    W32, S32 = W.to(dev, dt), Sz.to(dev, dt)
    h1_, h2_ = h1.to(dev, dt), h2.to(dev, dt)
    Xb, Yb, lb = X.to(dev, dt), Y.to(dev, dt), lam.to(dev, dt)

    hx = (h2_[None, :] * Xb).unsqueeze(2) * W32.unsqueeze(0)   # (S,n,n) A_s[p,i]
    hy = (h2_[None, :] * Yb).unsqueeze(2) * W32.unsqueeze(0)   # (S,n,n) B_s[q,i]
    SB = torch.matmul(S32.unsqueeze(0), hy)                    # (S,n,n) Sz @ B_s
    SA = torch.matmul(S32.unsqueeze(0), hx)

    gy = (h1_[None, :] * Yb) @ W32                             # (S,n)  W^T (h1.y)
    gx = (h1_[None, :] * Xb) @ W32                             # (S,n)

    # sym3(x,y,y) has terms x_p y_q y_r, y_p x_q y_r, y_p y_q x_r.
    # edge Sigma_pq couples the two i-legs (p,q); the r-leg goes to j.
    f_xy = (hx * SB).sum(1)          # (S,n)  colsum(A . Sz B)   -> x_p y_q, r=y
    f_yx = (hy * SA).sum(1)          # y_p x_q, r=y
    f_yy = (hy * SB).sum(1)          # y_p y_q, r=x
    out = ((lb[:, None] * (f_xy + f_yx)).T @ gy
           + (lb[:, None] * f_yy).T @ gx)
    return out.to(Sz.dtype).cpu() if out.device != Sz.device else out.to(Sz.dtype)
