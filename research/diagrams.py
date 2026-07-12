"""PROGRAMMATIC DIAGRAM ENUMERATION for kappa3(ReLU(z)) — no hand algebra.

Hand-deriving these contractions produced three bugs today. Thm A.2 gives, for
r=3, a sum over diagrams; a diagram is a multiset of BLOCKS, each block B
assigning legs (b_p, b_q, b_r) to the three vertices:

    term = [prod_v h_{k_v}(v)] * [prod_B kappa_B]
           / [prod_B prod_v b_v(B)!] / [prod_{block types} mult!]

    k_v = sum_B b_v(B)
    constraints: the diagram is CONNECTED, and every size-2 block is MIXED
                 (spans two distinct vertices -- (2,0,0) is forbidden, its
                  variance is already absorbed into the Hermite coefficients)

Cross-checked against every hand derivation:
    star           blocks {(1,1,0),(1,0,1)}          -> h2 h1 h1 S_pq S_pr        OK
    (a,b)=(2,1)    blocks {(1,1,0)x2,(1,0,1)}        -> h3 h2 h1 S_pq^2 S_pr / 2  OK
    triangle       blocks {(1,1,0),(1,0,1),(0,1,1)}  -> h2 h2 h2 S_pq S_pr S_qr   OK
    pass-through   blocks {(1,1,1)}                  -> h1 h1 h1 k3_pqr           OK
    order-1 (i)    blocks {(1,1,1),(1,1,0)}          -> h2 h2 h1 k3_pqr S_pq      OK
    order-1 (ii)   blocks {(2,1,0),(1,0,1)}          -> h3 h1 h1 k3_ppq S_pr / 2  OK

`enumerate_diagrams` yields (blocks, coefficient). `eval_diagram` builds the
(n,n,n) tensor for one diagram from Sigma / kappa3(z) / kappa4(z). At the small
n used for verification these are tiny, so we can check the FORMULA exactly,
tensor-level, before deriving any contraction.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter

import torch

V = 3  # vertices p, q, r


def _blocks_of_size(k):
    """All (b_p,b_q,b_r) with sum k. Size-2 blocks must be MIXED."""
    out = []
    for b in itertools.product(range(k + 1), repeat=V):
        if sum(b) != k:
            continue
        if k == 2 and max(b) == 2:      # (2,0,0): not mixed -> forbidden
            continue
        out.append(b)
    return out


def _connected(blocks):
    seen = {0}
    changed = True
    while changed:
        changed = False
        for b in blocks:
            sup = {v for v in range(V) if b[v] > 0}
            if sup & seen and not sup <= seen:
                seen |= sup
                changed = True
    return seen == set(range(V))


def enumerate_diagrams(max_sigma=2, max_k3=1, max_k4=0):
    """All connected diagrams with <=max_sigma covariance edges, <=max_k3
    third-cumulant blocks and <=max_k4 fourth-cumulant blocks."""
    S2, S3, S4 = _blocks_of_size(2), _blocks_of_size(3), _blocks_of_size(4)
    out = []
    for n3 in range(max_k3 + 1):
        for n4 in range(max_k4 + 1):
            for ns in range(max_sigma + 1):
                if n3 + n4 + ns == 0:
                    continue
                for c3 in itertools.combinations_with_replacement(S3, n3):
                    for c4 in itertools.combinations_with_replacement(S4, n4):
                        for cs in itertools.combinations_with_replacement(S2, ns):
                            blocks = [(2, b) for b in cs] + [(3, b) for b in c3] \
                                     + [(4, b) for b in c4]
                            if not _connected([b for _, b in blocks]):
                                continue
                            k = [sum(b[v] for _, b in blocks) for v in range(V)]
                            denom = 1
                            for _, b in blocks:
                                for v in range(V):
                                    denom *= math.factorial(b[v])
                            for _, m in Counter(blocks).items():
                                denom *= math.factorial(m)
                            out.append((tuple(blocks), tuple(k), 1.0 / denom))
    return out


def eval_diagram(blocks, k, coef, g, S, K3, K4):
    """(n,n,n) tensor for one diagram. g[m] is the vector h_m(.) of Hermite gains."""
    n = S.shape[0]
    letters = "pqr"
    operands, subs = [], []
    for size, b in blocks:
        idx = "".join(letters[v] for v in range(V) for _ in range(b[v]))
        operands.append({2: S, 3: K3, 4: K4}[size])
        subs.append(idx)
    expr = ",".join(subs) + ",p,q,r->pqr"
    operands += [g[k[0]], g[k[1]], g[k[2]]]
    return coef * torch.einsum(expr, *operands)


def k3_activation(g, S, K3, K4, max_sigma=2, max_k3=1, max_k4=0):
    """kappa3 of ReLU(z) as a full (n,n,n) tensor, to the given truncation."""
    n = S.shape[0]
    out = torch.zeros(n, n, n, dtype=S.dtype)
    for blocks, k, coef in enumerate_diagrams(max_sigma, max_k3, max_k4):
        out = out + eval_diagram(blocks, k, coef, g, S, K3, K4)
    return out
