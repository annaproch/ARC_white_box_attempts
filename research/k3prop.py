"""kappa3 PROPAGATOR — the missing component between the oracle result and a
submittable estimator.  (STRATEGY.md 4.8)

Recursion.  With D1 = diag(h1), and W the layer's weights:

    k3(z^(l))   = k3(a^(l-1)) x_k W^T            (exact, multilinear)
    k3(a^(l))   = gen^(l)  +  pass^(l)
    pass^(l)    = k3(z^(l)) x_k D1               (leading pass-through diagram)
    gen^(l)_pqr = h2(p)h1(q)h1(r) Sz_pq Sz_pr + (2 perms)      [star, 2-edge]

Representation.  Carry ONLY the accumulated part C = pass in Tucker form
(V n x R, core G R^3).  The freshly generated gen is kept in its exact factored
form and contracted EXACTLY into the slice of the very next layer -- it is only
compressed when carried past that (the old code compressed it at birth and threw
away most of its energy).

Slices needed by the (mu,Sigma) update:
    gen part (exact, matmuls):   B = Sz diag(h1) W
                                 C_hi = h2(h) W_hi B_hi ;  D_hi = h2(h) B_hi^2
                                 k3(z')_iij = 2 (C^T B)_ij + (D^T W)_ij
    pass part (from Tucker):     Vt = W^T V
                                 k3(z')_iij = sum_c Vt_jc * [sum_ab G_abc Vt_ia Vt_ib]

Cost (flopscope-measured, budget_probe.py): the whole closure incl. R=256 runs
at 2.9% of the 2.72e11 budget, so R is essentially free -- buy fidelity.
"""
from __future__ import annotations

import torch

from harness import DT, hermite_gains, relu_step_full


def _tucker_slice(G, Vt):
    """k3_iij and k3_iii for k3 = G x1 Vt x2 Vt x3 Vt.  Cost n R^3 + n^2 R.

    Vectorised: one einsum, no Python loop over R (the loop version issued R
    tiny matmuls per layer -- the worst possible pattern on this hardware)."""
    inner = torch.einsum("ia,abc,ib->ic", Vt, G, Vt)      # (n, R)
    return inner @ Vt.T, (inner * Vt).sum(1)


def _sym3_core(Vb, X, Y, w):
    """Core of  sum_h w_h * sym3(x_h, y_h, y_h)  projected onto basis Vb.
    X, Y are (n, n) with column h = x_h, y_h.  Cost n R^3."""
    Xh = Vb.T @ X                       # (R, n)
    Yh = Vb.T @ Y
    Xw = Xh * w[None, :]
    G = (torch.einsum("ah,bh,ch->abc", Xw, Yh, Yh)
         + torch.einsum("ah,bh,ch->abc", Yh, Xw, Yh)
         + torch.einsum("ah,bh,ch->abc", Yh, Yh, Xw))
    return G


class K3State:
    """Accumulated (compressed) third cumulant of the activations, plus the
    freshly-generated part in exact factored form."""

    def __init__(self, n, R):
        self.R = R
        self.V = torch.zeros(n, R, dtype=DT)      # Tucker basis of the CARRY
        self.G = torch.zeros(R, R, R, dtype=DT)   # Tucker core of the CARRY
        self.gen = None                           # (h2, Sz, h1) of the last layer

    def slices(self, W):
        """k3(z')_iij, _iii at the next layer: exact gen + Tucker carry."""
        n = W.shape[0]
        iij = torch.zeros(n, n, dtype=DT)
        iii = torch.zeros(n, dtype=DT)
        if self.gen is not None:
            h2, Sz, h1 = self.gen
            B = Sz @ (h1[:, None] * W)                     # B_hi
            Cm = (h2[:, None] * W) * B                      # C_hi
            Dm = h2[:, None] * B * B                        # D_hi
            iij = iij + 2.0 * (Cm.T @ B) + (Dm.T @ W)
            iii = iii + 3.0 * (h2[:, None] * W * B * B).sum(0)
        if self.G.abs().sum() > 0:
            Vt = W.T @ self.V
            a, b = _tucker_slice(self.G, Vt)
            iij, iii = iij + a, iii + b
        return iij, iii

    def advance(self, W, h1, h2, Sz, Vnew):
        """Fold gen into the carry, push everything through this layer, and
        re-compress onto the new basis Vnew (n x R)."""
        R = self.R
        Gnew = torch.zeros(R, R, R, dtype=DT)
        F = h1[:, None] * (W.T)                 # F[p, a] : (D1 W^T)  -> maps a-index
        # 1) carry: C -> C x_k (D1 W^T),  factor V -> D1 W^T V
        if self.G.abs().sum() > 0:
            Vc = h1[:, None] * (W.T @ self.V)   # (n, R)
            A = Vnew.T @ Vc                     # (R, R) re-expression
            Gnew += torch.einsum("abc,ia,jb,kc->ijk", self.G, A, A, A)
        # 2) gen of the PREVIOUS layer also gets pushed through and compressed
        if self.gen is not None:
            h2o, Szo, h1o = self.gen
            Xo = h1[:, None] * W.T              # column h = h1 .* W[h,:]
            U = h1o[:, None] * Szo              # column h = u_h = h1o .* Sz[:,h]
            Yo = h1[:, None] * (W.T @ U)
            Gnew += _sym3_core(Vnew, Xo, Yo, h2o)
        self.G = Gnew
        self.V = Vnew
        self.gen = (h2, Sz, h1)                 # this layer's fresh gen, exact


class K3Exact:
    """NO COMPRESSION. Carries the accumulated kappa3 as an exact list of
    sym3(x_s, y_s, y_s) terms -- the form is closed under the pass-through
    transform (x,y -> D1 W^T x, D1 W^T y), and gen adds n new terms per layer.

    Diagnostic only (cost grows as n^3 L^2): it separates FORMULA error from
    RANK error. If this still misses the spec, more diagrams are needed; if it
    hits it, the problem is purely the compression.
    """

    def __init__(self, n):
        self.lam = torch.zeros(0, dtype=DT)
        self.X = torch.zeros(0, n, dtype=DT)      # rows x_s
        self.Y = torch.zeros(0, n, dtype=DT)      # rows y_s
        self.gen = None

    def slices(self, W):
        n = W.shape[0]
        iij = torch.zeros(n, n, dtype=DT)
        iii = torch.zeros(n, dtype=DT)
        if self.gen is not None:
            h2, Sz, h1 = self.gen
            B = Sz @ (h1[:, None] * W)
            Cm = (h2[:, None] * W) * B
            Dm = h2[:, None] * B * B
            iij = iij + 2.0 * (Cm.T @ B) + (Dm.T @ W)
            iii = iii + 3.0 * (h2[:, None] * W * B * B).sum(0)
        if self.lam.numel():
            # carried terms, contracted:  k3(z')_iij = sum_s lam_s
            #   [2 x_s(i) y_s(i) y_s(j) + y_s(i)^2 x_s(j)]   with x,y -> W^T x
            Xc = self.X @ W                        # (S, n)
            Yc = self.Y @ W
            lam = self.lam[:, None]
            iij = iij + 2.0 * ((Xc * Yc * lam).T @ Yc) + ((Yc * Yc * lam).T @ Xc)
            iii = iii + 3.0 * (lam * Xc * Yc * Yc).sum(0)
        return iij, iii

    def advance(self, W, h1, h2, Sz, _Vnew=None):
        # push every carried term through this layer:  x -> D1 W^T x
        if self.lam.numel():
            self.X = (self.X @ W) * h1[None, :]
            self.Y = (self.Y @ W) * h1[None, :]
        if self.gen is not None:
            h2o, Szo, h1o = self.gen
            # x_h = D1 W^T e_h = h1 .* W[h,:]   -> ROW convention: W * h1[None,:]
            # (W.T here multiplies h1 along the wrong axis -- that was the bug)
            Xo = W * h1[None, :]                   # row h = h1 .* W[h,:]
            U = h1o[:, None] * Szo                 # column h = u_h
            Yo = (U.T @ W) * h1[None, :]           # row h = D1 W^T u_h
            self.lam = torch.cat([self.lam, h2o])
            self.X = torch.cat([self.X, Xo]) if self.X.numel() else Xo
            self.Y = torch.cat([self.Y, Yo]) if self.Y.numel() else Yo
        self.gen = (h2, Sz, h1)


def propagate_k3(Ws, R=32, k4=None, kmax=12, pmax=6, exact=False):
    """Full closure: complete (mu,Sigma) diagram formula + propagated kappa3.
    k4[l] (optional) supplies the kappa4 slices for layer l; it only needs
    10-20% accuracy (E6), so an oracle/crude source is acceptable here.
    Returns mu at the final layer, and the kappa3 slices used (for validation)."""
    n = Ws[0].shape[0]
    mu = torch.zeros(n, dtype=DT)
    cov = torch.eye(n, dtype=DT)
    st = K3Exact(n) if exact else K3State(n, R)
    used = []
    for l, W in enumerate(Ws):
        k3_iij, k3_iii = st.slices(W)
        used.append((k3_iij, k3_iii))

        zmu = mu @ W
        zcov = W.T @ cov @ W
        zvar = torch.clamp(torch.diagonal(zcov), min=1e-30)
        g, s, t, pdf, cdf = hermite_gains(zmu, zvar, kmax=kmax)

        cum = {"k3_iii": k3_iii, "k3_iij": k3_iij}
        if k4 is not None:
            cum.update({k: k4[l][k] for k in ("k4_iiii", "k4_iijj", "k4_iiij")})
        else:
            z = torch.zeros(n, dtype=DT)
            cum.update({"k4_iiii": z, "k4_iijj": torch.zeros(n, n, dtype=DT),
                        "k4_iiij": torch.zeros(n, n, dtype=DT)})
        mu, cov = relu_step_full(mu, cov, W, cum, pmax=pmax, kmax=kmax)

        # new Tucker basis: top-R eigenvectors of the new activation covariance
        # (kappa3's accumulated part aligns with Sigma's leading modes at depth)
        if exact:
            st.advance(W, g[1], g[2], zcov)
        else:
            ev, evec = torch.linalg.eigh(cov)
            st.advance(W, g[1], g[2], zcov, evec[:, -R:].contiguous())
    return mu, used
