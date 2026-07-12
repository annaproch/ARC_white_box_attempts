# ARC WhiteBox Estimation Challenge 2026 — Estimator Improvements & Session Log

*2026-07-11/12 — research session with Claude Code*

---

# ⚠️ ROUND 7 — COMPETITOR ANALYSIS AND THE PIVOT (read this first)

Our v4 (QMC + calibrated K=2 fusion) scores **3.396e-7** public, 1.9× the MC
baseline. A top-10 competitor's per-MLP detail (submission tooltip,
`bruce-mckenzie · idx 6`) forces a strategy change.

## The evidence

| field | their value | ours (v4) |
|---|---|---|
| L0–L30 per-layer MSE | **0.5–0.7 (garbage; zeros baseline is 0.91)** | ~7e-6 (accurate) |
| L31 (only scored layer) | **3.24e-7** | ~3.1e-6 |
| FLOPs | 2.30e10 (**8.5% — at the floor**) | 2.96e10 (10.9%) |
| wall time | **20,947 ms** | ~300 ms |
| per-MLP score | **3.24e-8** | ~3.4e-7 |

## What it means

1. **They do not sample at all.** Sampling produces every intermediate layer
   mean for free; their intermediate rows are junk, so no forward pass over
   inputs is happening. Pure analytic propagation, final layer only.
2. **Wall-time signature = cumulant machinery.** 2.3e10 FLOPs in 21 s =
   **1.1 GFLOP/s**, ~100× below matmul throughput ⇒ thousands of small
   elementwise/transcendental ops. Exactly what the ARC paper reports for its
   own kprop code (App. I). This is not a fast-matmul (sampling/QMC) profile.
3. **21 s of wall time is allowed.** Only *residual* (non-flopscope) wall time
   is penalized at λ; time inside flopscope kernels is free. **We have been
   far too conservative about elementwise math.**
4. **They are ~10× better than us at the same (floor) compute**, and in
   variance-normalized terms (Var ≈ 0.012) their 3.24e-7 ≈ 2.7e-5 — which on
   the paper's depth law c_K(L/n)^K at L=32, n=256 is **K=3-class accuracy**.
   Our calibrated K=2 (2.4e-5 absolute ≈ 2e-3 normalized) is K=2-class.
   **They are propagating third-order information; we are not.**

## Our gaps (in order of cheapness to close)

- **G1 — Hermite/Wick series truncated far too early.** The exact off-diagonal
  ReLU covariance update is
  `Cov(ReLU(Zi),ReLU(Zj)) = Σ_{k≥1} ĥ_k(i)·ĥ_k(j)·C_ij^k / k!`.
  Our *shipped* flopscope estimator keeps **2 terms**; the torch experiments
  kept 4. Deep ReLU covariance is dominated by a strong common-mode component
  ⇒ pairwise |ρ| is **not** small ⇒ the series converges slowly and the
  truncation error compounds over 32 layers. Each extra term is one
  elementwise n×n multiply (~1e6 FLOPs/term/layer — negligible), and the
  wall-time finding says we can afford dozens.
- **G2 — no exact bivariate ReLU moment.** Closed forms exist: Cho–Saul
  arc-cosine kernel (zero mean); with means via Owen's T / bivariate normal
  CDF (Bibi et al. 2018; Wright et al. 2024 Thm 1 — both cited by the ARC
  paper). O(n²) elementwise per layer; removes truncation error entirely.
- **G3 — K=3 dismissed on a false premise (retraction).** We rejected K=3
  because the paper's *factorized* cost is 30n³L² — but that L² comes from the
  factor **rank growing with depth**. We asserted rank-compression "degrades,
  not improves" — that was an assumption, **never measured**. Fixed-rank-r
  compression each layer gives ~O(n²·r·L), which fits inside the floor. The
  leaderboard shows someone getting K=3 accuracy at 2.3e10 FLOPs; fixed-rank
  is the obvious mechanism.
- **G4 — budget misallocated.** Our analytic component uses **0.83%** of
  budget; samples eat the other ~10%. If the analytic side reaches
  b² ≈ 3e-7, sampling adds nothing (fused ≈ b² when b² ≪ v). The endgame is
  **pure analytic at the floor**, ~2.5e10 FLOPs of cumulant machinery.
- **G5 — wasted effort on unscored layers.** Only L31 is ranked. Intermediate
  accuracy is worth zero.

## G1 — TESTED AND DEAD (`wick_terms_test.py`)

Off-diagonal |ρ| **grows with depth** exactly as predicted (mean 0.05 → 0.26,
p95 0.60, **max 0.90** at L31) — but the Wick series is **fully converged by
k=4**: mean MSE 5.954e-5 (k=2), 5.959e-5 (k=4 … k=64), identical to 4 digits.
The ρ^k/k! weighting plus the decay of ReLU's Hermite coefficients kills the
tail even at ρ = 0.9. **Truncation was never our problem** — the whole ~6e-5
error of plain K=2 is the *Gaussian closure itself* (discarding 3rd+
cumulants), not its numerics.

## The sharper puzzle this creates

Our own measurements: paper's factorized K=3 at depth 32 → **2.3e-5**;
K=3-augmented (at 4× the entire budget) → **1.8e-6**. The competitor gets
**3.2e-7 at the floor**. So they are **not running textbook kprop at any
order** — no K in the paper's family reaches their accuracy at any budget we
measured. Something structurally different is happening, and we must locate
our error before building anything expensive.

## ERROR ATTRIBUTION — MEASURED (`error_attribution.py`, 6 MLPs)

| component | value | share |
|---|---|---|
| total final-layer MSE (plain K=2) | 6.10e-5 | 100% |
| **(A) propagated-moment error** | **5.98e-5** | **98%** |
| (B) Gaussian *shape* error at the final ReLU | 1.01e-6 | 1.7% |
| skew(z₃₁) | ±0.02 | ~Gaussian |
| ‖Δμ_z‖rms / ‖Δσ_z‖rms | 1.1e-2 / **4.3e-2** | σ is ~4× worse |
| eig1frac (top-mode share of last hidden layer variance) | **0.55–0.80** | — |

**Conclusions.**
1. The final pre-activation z is essentially Gaussian (CLT over 256 terms) —
   the ReLU integral is *fine*. Edgeworth/quadrature fixes to the 1-D marginal
   are worthless (which retro-explains why K=2.5 bought only 2.3%).
2. **All the error is in the propagated moments, and mostly in the variance**
   (~30% relative on σ_z vs ~3% on μ_z).
3. **Mechanism:** σ_z² = wᵀΣ_a w is dominated by Σ's top eigenvalue, and
   λ₁ ≈ n · (mean off-diagonal covariance). So a *small systematic bias in the
   off-diagonal ReLU covariance update is multiplied by n=256 into λ₁ and then
   compounds multiplicatively over 32 layers* (1%/layer → 1.01³² ≈ 1.37).
   That is the 30%.
4. This is precisely the term third cumulants correct — it lives in the
   **pairwise** structure, not the marginals. It also explains the pattern of
   our earlier results: a per-layer scalar covariance correction (learned θ)
   bought 3×, while α-dependent marginal tweaks bought nothing.

## LAYERWISE DRIFT — MEASURED (`layerwise_cov_drift.py`, 3 MLPs)

| layer | μ err | λ₁ err | mean-offdiag err | skew(c) | kurt(c) | eig1frac |
|---|---|---|---|---|---|---|
| L0 | 0.001 | −0.001 | −0.003 | 0.02 | 0.05 | 0.01 |
| L2 | 0.003 | −0.07 | −0.63 | **0.31** | 0.13 | 0.03 |
| L7 | 0.006 | −0.21 | −0.50 | **0.47** | 0.28 | 0.14 |
| L15 | 0.008 | −0.27 | −0.40 | **0.45** | 0.28 | 0.34 |
| L31 | 0.009 | **−0.27 … −0.39** | **−0.31 … −0.43** | **0.41–0.46** | 0.24–0.27 | **0.50–0.83** |

*(The `var err` column in the raw output is a metric artifact — dead neurons
have true variance ≈ 0, so the relative error explodes. Absolute diagonals are
fine, as the near-perfect marginals confirm.)*

**The mechanism, finally pinned down.** The failure is **one degree of
freedom**: the dominant eigenmode of the activation covariance is strongly
**non-Gaussian** (|skew| ≈ 0.3–0.5, kurt ≈ 0.1–0.3, established by layer 2 and
persistent) while carrying 50–83% of the variance by the deep layers. The
per-neuron *marginals* stay Gaussian (skew 0.02) — the non-Gaussianity is
concentrated almost entirely in that single common mode. A Gaussian closure
structurally cannot represent it, so it under-propagates the mode:
λ₁ is underestimated by 27–43%, hence σ_z by ~30%, hence 98% of our error.

**Consequence: we do NOT need full third-order tensors.** We need a *rank-1
non-Gaussian core*.

## THE BUILD: rank-1 non-Gaussian core closure (`core_closure.py`)

Model the activations as
`a = μ + √λ·u·c + ε`, with **c a scalar carrying its own skew/kurtosis**
(tracked, represented by a Q-node Gauss–Hermite quadrature warped by a cubic
Cornish–Fisher map) and **ε ~ N(0, R)**, R = Σ − λuuᵀ.

Per layer, conditional on c the pre-activation is *exactly* Gaussian, so every
ReLU expectation is a Q-node average of exact conditional formulas:
- `μ' = Σ_q w_q M(c_q)` (M = exact 1-D ReLU mean)
- `Σ' = Σ_q w_q [C(c_q) + M M ᵀ] − μ'μ'ᵀ` (law of total covariance; C from the
  converged k≤4 Wick series around the *conditional* Gaussian)
- re-extract the core (top eigenpair of Σ'), and push c forward:
  `c'|c ~ N(t(c), v(c))` with `t(c) = u'ᵀ(M(c) − μ')`, `v(c) = u'ᵀC(c)u'` —
  the new skew/kurt follow in closed form from the mixture moments.

The non-Gaussianity is **generated automatically** by the nonlinearity of
t(·), so we can start from a Gaussian c at layer 0 (true skew there: 0.02).

Cost at n=256, L=32, Q=16, K=4: ~2n³ + Q·K·n² per layer ≈ **1.5e9 FLOPs
(~0.6% of budget)** — pure analytic, comfortably at the floor, and a
many-small-ops profile that matches the competitor's 1.1 GFLOP/s signature.
This is ARC's deduction–projection / L²-sketching idea with the sketch chosen
by the data (the top eigenvector).

If it lands: layer the existing end-to-end calibration on top, drop sampling
entirely, and port to flopscope.

**Rank-1 core: TESTED, 1.06× — rejected.** With a stable particle
representation the recursion is healthy (λ₁ = 1.81 at L31, inside the true
1.5–2.5 range), but the gain is nil. Reason, visible in the drift table: the
λ₁ error accumulates over **layers 2–11, where eig1frac is only 0.03–0.27** —
the covariance is *not* rank-1 dominated where the error is actually made. The
top mode only dominates at the very end. Rank-1 is the wrong sketch.
(Also note: if c were Gaussian this closure is *identical* to K=2 — all its
value depends on the core's non-Gaussianity, and one mode isn't enough.)

**Scalar calibration is exhausted.** Widening the per-layer correction bounds
from ±5% to ±60% (the measured bias is −27…−43%) plateaus at **2.6e-5** —
statistically the same as the ±5% version's 2.4e-5. Two very different
parametrizations hitting the same wall means this is a *shape* limitation, not
an optimization one: the κ₃ correction
`ΔCov_ij ∝ g₂(i)g₁(j)·κ₃(i,i,j) + g₁(i)g₂(j)·κ₃(i,j,j)`
is **not proportional to Σ_ij**, so no uniform scaling can reproduce it.

## THE ORACLE TEST (prices the whole third/fourth-order program)

Key structural fact: **the closure only ever needs CONTRACTED cumulants** —
κ₃(i,i,j) and κ₄(i,i,j,j), which are **n² objects**, not n³/n⁴ tensors. They
are directly sampleable with one extra n×n matmul per batch
(`(a²)ᵀa`, `(a²)ᵀ(a²)`).

So `build_moment_corpus.py` caches, per MLP and per layer: μ, Σ, κ₃_iij,
κ₄_iijj (~25 MB/MLP). Then `oracle_cumulants.py` injects the **true** slices
into the covariance update and measures the final-layer MSE — the ceiling of
the entire κ₃/κ₄ program, obtained *before* building any propagator:

- **K2 + true κ₃ ≈ 1e-6** → third cumulants are the answer; then either build
  the rank-r κ₃ propagator, or (much cheaper) **learn the κ₃ slices from the
  K=2 state** (α_i, α_j, ρ_ij, layer) — n² cost, no tensor propagation at all.
- **needs κ₄ too** → add the fourth-order term.
- **even oracle-Σ (true Σ at every layer) is ≫ 1e-6** → the covariance is not
  the whole story and the mean path matters.

The corpus is also the substrate for **supervised layerwise training** (match
the true Σ at every layer: 32 × 65k targets per MLP, versus the 256 numbers of
final-layer MSE we have been squeezing gradients from — the likely cause of
every plateau so far).

## ORACLE RESULT — K=2 IS A SELF-CONSISTENT FIXED POINT (the day's key finding)

Corpus validated: `oracle-ALL` (true μ *and* Σ injected every layer) = **1.06e-6**,
matching the independently measured Gaussian-shape error (1.0e-6). So the
numbers below are real, not artifacts.

| variant | final-layer MSE |
|---|---|
| K=2 (baseline) | 1.30e-4 |
| K=2 + **true** κ₃(z) corrections (mean + covariance) | 1.72e-4 ❌ worse |
| K=2 + true κ₃ + κ₄ | 1.61e-4 ❌ worse |
| **oracle-cov** (true Σ every layer, propagated μ) | **1.89e-4** ❌ worse |
| oracle-ALL (true μ and Σ every layer) | 1.06e-6 ✅ |

**Interpretation.** With a *perfect* covariance the mean still accumulates
1.9e-4 of error — worse than K=2's total. So:

1. The per-layer **mean update carries a systematic shape bias that compounds
   over 32 layers**, and it — not the covariance — is the dominant error source.
2. **K=2's covariance underestimate partially cancels that mean bias.** The
   closure is a self-consistent fixed point. Improving *either* component in
   isolation — even to perfection — destroys the cancellation and makes the
   final answer worse.
3. This retro-explains every plateau: our calibrations (±5%, ±60%, rank-1 core)
   were being rewarded for *preserving the cancellation*, not for fixing physics.
   It also explains why a scalar off-diagonal boost couldn't help.

**Consequence for the build: mean and covariance must be corrected TOGETHER,
and the mean path is the priority.** Partial fixes are actively harmful.

## SINGLE-STEP UNIT TEST — THE FORMULAS ARE RIGHT (`step_unit_test.py`, 8 MLPs)

True pre-activation moments in, one ReLU step, compared against that layer's
true post-activation moments (no propagation, no compounding, no cancellation):

| layer | mean: Gaussian | +κ₃ | **+κ₃+κ₄** | cov: Gaussian | +κ₃ | **+κ₃+κ₄** |
|---|---|---|---|---|---|---|
| L1 | 2.6e-6 | 3.1e-7 | **3.8e-9** (692×) | 1.1e-5 | 3.1e-7 | **4.3e-8** (260×) |
| L3 | 4.9e-6 | 5.5e-7 | **3.8e-9** (1298×) | 1.4e-5 | 2.6e-7 | **7.5e-8** (193×) |
| L15 | 3.2e-6 | 4.2e-7 | **1.8e-8** (173×) | 8.2e-6 | 6.7e-8 | **6.2e-8** (133×) |
| L31 | 1.1e-6 | 2.1e-7 | **4.6e-8** (23×) | 1.5e-6 | 3.6e-8 | **3.6e-8** (42×) |

**Every correction reduces its step error at every depth ⇒ signs and
coefficients are correct.** And **κ₄ is not optional**: it buys another ~145×
on the mean *after* κ₃.

This also explains the paradoxical oracle: we injected **true** cumulants into
a **drifted** state (our propagated μ, Σ). A cumulant correction is only valid
for the distribution it belongs to — pairing truth with a drifted state is
inconsistent, which is worse than K=2's self-cancelling fixed point. Propagate
consistently and the L31 row projects a final-layer error of **~5e-8 — below
the competitor's 3.2e-7.**

---

# 📜 THE COMPLETE HISTORY — every idea, what happened, why

Chronological. Includes the mistakes, because most of the learning is in them.

## Phase 1 — Sampling (v1 → v3): finding the floor

| # | idea | outcome |
|---|---|---|
| 1 | Read the challenge paper (arXiv:2605.05179), port ARC's kprop | K≥3 **exceeds the budget** at depth 32 (114%/420%); K=2 gives 4.7e-5 — 10× worse than sampling at the floor. The paper's methods were tested at ≤12 layers; depth 32 changes everything. |
| 2 | **Monte Carlo at the compute floor** | The multiplier `max(0.1, C/B)` makes compute below 10% free. iid MC at the floor: **6.53e-7 = exactly the MC reference.** v1. |
| 3 | Antithetic pairs (x, −x) | **No gain.** After 32 ReLU layers the output has almost no odd dependence on the input. (This also killed the linearized `J·x` control variate before we built it.) |
| 4 | Scrambled **Sobol → Gaussian, moment-matched** | 3.2× better than iid. **v2: 3.78e-7 (1.7× vs MC).** |
| 5 | **Spherical projection** (no biases ⇒ 1-homogeneous ⇒ radial integral exact) | Free and exact. **v3: 3.69e-7.** |
| 6 | Learned quadrature weights (ridge/Bayesian) | **Collapse to uniform** (<2%). Sobol+mm is already ≈ BQ-optimal. |
| 7 | SGD-optimized point positions | **Noise-level** (+2%), then degrades. Gradient noise destroys the Sobol structure. |
| 8 | Trim points to duck under the 10% floor (v3.1) | **Made it worse** — and predicted the public score to 4 digits. **The floor is not a cliff**: above it the multiplier is linear and MSE is ~linear in 1/N, so the score surface is flat to first order. |
| 9 | Measure QMC's convergence exponent | **β ≈ 0.83 < 1** ⇒ never leave the floor. |

## Phase 2 — Fusion (v4): cumulants earn a place

| # | idea | outcome |
|---|---|---|
| 10 | Per-**neuron** fusion of MC + analytic | **Fails.** The per-neuron bias signal is `(mc − analytic)`, whose detection noise *is* the MC noise. Oracle weights give only 1.37×. |
| 11 | Per-**MLP** adaptive fusion weight | **Fails.** Analytic bias **co-scales** with sampling variance (corr(log b², log v̂) = +0.58) — both driven by activation scale — so the optimal weight is ~constant. |
| 12 | **Global offline-learned fusion weight** | Works. This is the sound level of adaptivity. |
| 13 | **End-to-end calibrated K=2 closure** (192 learned per-layer scalars, backprop through 32 layers) | Removes **66%** of the Gaussian-closure error (7.2e-5 → 2.4e-5). |
| 14 | Fuse QMC + calibrated-K2 (w = 0.137, errors independent: corr −0.04) | **v4: 3.396e-7 (1.9× vs MC).** Still the best submission. |

## Phase 3 — Competitor analysis: the pivot

| # | finding | evidence |
|---|---|---|
| 15 | A top-10 rival is **pure analytic** | Their intermediate layers are garbage (~0.5 MSE — sampling would give them free), 8.5% budget, **21 s wall time at 1.1 GFLOP/s** = thousands of small elementwise ops = cumulant machinery. |
| 16 | **21 s of wall time is allowed** | Only *residual* (non-flopscope) time is charged at λ. We had been far too conservative. |
| 17 | They achieve ~3.2e-7 raw, ~10× better than us at the same compute | And **no kprop variant in the paper reaches that at any budget** — so they are not running textbook cumulant propagation either. |

## Phase 4 — Diagnosis: where the error actually lives

| # | experiment | result |
|---|---|---|
| 18 | Wick-series truncation (k = 2 → 64) | **Converged by k=4** despite ρ reaching 0.90. Truncation was never the problem. |
| 19 | Error attribution at the final ReLU | **98% is propagated-moment error**, 1.7% is the Gaussian *shape* assumption. z₃₁ is Gaussian (skew 0.02) — Edgeworth on the marginal is worthless. |
| 20 | Layerwise covariance drift | λ₁ **underestimated 27–43%**; the dominant eigenmode is strongly **non-Gaussian** (skew 0.3–0.5) while carrying 50–83% of the variance. |
| 21 | Rank-1 non-Gaussian "core" closure | **1.06× — fails.** The drift accumulates over layers 2–11, where eig1frac is only 0.03–0.27: the covariance is *not* rank-1 dominated where the error is made. |
| 22 | Wide-bound scalar calibration (±5% → ±60%) | **Same plateau (2.6e-5).** A uniform scaling cannot reproduce a κ₃ correction that is *not* proportional to Σ_ij. Shape limitation, not capacity. |
| 23 | **Oracle: inject true Σ at every layer** | **WORSE than K=2** (1.9e-4 vs 1.3e-4). **K=2 is a self-consistent fixed point** — its covariance underestimate partially cancels the mean's accumulated shape bias. **Partial fixes are actively harmful.** This retro-explains every plateau. |
| 24 | Single-step unit test (true cumulants, one layer) | mean error 4.9e-6 → **3.8e-9** (1300×); covariance 1.4e-5 → 7.5e-8 (193×). **κ₄ is worth another 145× after κ₃.** The correction *formulas* are right. |

## Phase 5 — Building the cumulant closure

| # | idea | outcome |
|---|---|---|
| 25 | κ₃'s freshly-generated part has a **closed form** (star diagrams) | Exact: corr **0.996** at L1, computable in 3 matmuls — no n³ tensor. |
| 26 | But the star **decays 26×** by L31 while true κ₃ stays constant | The **accumulated** κ₃ dominates at depth — that is the term that forces ARC's factored tensor and its L² cost. |
| 27 | Is the accumulated κ₃ low-rank? | **Yes at depth** (top-1 SVD energy 0.98, aligned 0.998 with Σ's top eigenvector) — but *not* early (rank-1 corr 0.09 at L1). |
| 28 | **Hybrid: exact star + rank-r Tucker core** | Works. κ₃ corr 0.94–0.97 vs sampled truth. **K=2 → +κ₃ = 4.7× (1.30e-4 → 2.76e-5).** |
| 29 | (bug) compressing the star **at birth** | Threw away ~80% of its energy. Fixed: use it at full fidelity in the layer it lands, compress only when carried past. |
| 30 | End-to-end calibration of the whole closure | 4.0e-5 → **1.05e-5**. Plateaus there in *every* configuration (r=16, 32; r₄=1, 8, 32). |

## Phase 6 — The grader teaches us three lessons

| # | mistake | lesson |
|---|---|---|
| 31 | Shipped 3-way fusion → **100/100 MLPs failed** | flopscope **infers a symmetry tag from einsum's repeated operands** and rejects the result. An individual star term is symmetric only in (k,l); *only the sum of three* is fully symmetric. Fix: build with matmul + reshape. |
| 32 | I "fixed" the symmetrization first | **It was a no-op.** I misread torch's `permute` semantics. The uncalibrated value came back bit-identical, which is what caught it. |
| 33 | 3-way shipped → **score 4.74e-7, LOST to v4** | My analytic FLOP estimate was **2.7× too low** (4.16e9 vs measured **1.13e10**). I counted the Tucker cores and forgot the matmuls that dominate (~14n³/layer, not 6). **Never trust an analytic FLOP estimate — measure it.** |

## Phase 7 — The layerwise fit: the real diagnosis

| # | experiment | result |
|---|---|---|
| 34 | **Least-squares fit of the moment corrections** with true cumulants injected (the coefficients enter *linearly* — no SGD needed) | mean step → **2.5e-9** (1854×), covariance → 6.4e-8 (220×), and the fitted coefficients land at **c3m ≈ 1.01, c3c ≈ 0.98** — my theory was right. **The correction machinery is not the bottleneck.** |
| 35 | **Supervise the cumulant propagation** directly against sampled κ₃/κ₄ | κ₃ relative error **7.7% → 6.7%** (structural, not a scaling problem). **κ₄: 2.54 — worse than predicting zero**, and *worse at higher rank* (3.18 at r₄=32). |
| 36 | κ₄ chain diagram (same Σ-edge order as the star) | **Barely helps** (2.54 → 2.52). |
| 37 | **Why κ₄ is broken** | It omits the **κ₃-mixed diagrams** (`ĥ₂ĥ₁ĥ₁ĥ₁·κ₃(z)·Σ`), which are the **same order** as the star we do compute. Not a coefficient error — a structurally missing family. |
| 38 | κ₃ error vs rank | r=8: 24% · r=16: 14% · r=32: 7.7% · r=64: 5.7% · r=96: 4.7%. Rank-limited, diminishing. |

## Where that leaves the numbers

The closure's error decomposes cleanly: **corrections are exact; κ₃ is 6.7% wrong; κ₄ is worse than zero.** A 6.7%-wrong κ₃ leaves 6.7% of each layer's correction unapplied, compounding over 32 layers into ≈1e-5 — which is exactly the plateau we measured in every configuration.

**To beat v4 the closure needs MSE < ~4e-6 (≈3× better).** That 3× is precisely what a *working* κ₄ is worth (the unit test measured 145× on the mean step). Everything else — rank, coefficients, Wick terms — is already squeezed.

---

# ROUND 8 — THE CLOSURE SHIPS, AND WHAT IT COST US TO LEARN

## Leaderboard state

| version | public score | method |
|---|---|---|
| **v4 (best, on the board)** | **3.396e-7** | QMC(6400) + calibrated-K2, fused |
| leader | 9.16e-8 | pure analytic, 26% budget |
| top-10 rival | 3.24e-8 (per-MLP) | pure analytic, 8.5% budget, 21 s wall |

## Two configurations measured on the grader (mini split, 100 MLPs)

| estimator | raw MSE | utilisation | multiplier | **score** |
|---|---|---|---|---|
| v4 | 3.66e-6 | 10.9% | 0.109 | **3.66e-7** |
| 3-way (QMC5200 + K2 + closure) | 3.69e-6 | **12.8%** | 0.128 | **4.74e-7** ❌ |
| **pure analytic (closure only)** | 1.19e-5 | **4.95%** | 0.101 | **1.22e-6** |

Both closure runs: **0/100 failed** — the flopscope port works.

## The FLOP-model disaster (and the lesson)

The 3-way *improved accuracy* (3.69e-6 vs v4's 3.66e-6 at fewer QMC points, and
the held-out fit gave the closure a fusion weight of 0.31 while dropping K2
entirely) — but it **lost on score**, because my analytic FLOP estimate was
**2.7× too low**:

| | estimated | measured |
|---|---|---|
| closure cost | 4.16e9 (1.5%) | **1.13e10 (4.2%)** |

I counted the Tucker cores and forgot the matmuls that dominate: the **star
slice alone is four 2n³ matmuls** (`C3ᵀBs`, `D3ᵀWr`, plus two for κ₄), on top of
the covariance propagation (4n³) and the star's `B` matrix (2n³) — **~14n³ per
layer, not 6**. Residual wall time was fine (17 ms/MLP, 0.6%).

**Rule adopted: never trust an analytic FLOP estimate — measure it with
`whest run` before believing any sweep.** The export now uses measured costs and
warns loudly when extrapolating.

## THE INSIGHT THAT REFRAMES EVERYTHING

**Pure analytic uses only 4.95% of budget.** The multiplier floors at 10%, so a
sampling-free estimator has **~5% of budget sitting completely idle** — compute
is *free* until the floor.

That inverts the whole optimisation problem. For weeks (hours) I was shaving
FLOPs off the closure to fit it alongside QMC. In pure-analytic mode we should be
doing the **opposite**: spend the idle 5% on accuracy.

- κ₃ rank 32 → 64+ (cost O(n·r³), we can afford it)
- **κ₄ chain diagrams** (see below)
- more subspace-iteration passes, more Wick terms

Every one of these is blocked by *accuracy*, not budget — and we have budget to
burn. **Accuracy is now the only currency.**

## Why κ₄ has been useless (diagnosed)

Count Σ-edges per diagram — that is the order in the small parameter:

- **κ₃**: star = 2 edges, triangle = 3 → star dominates. Measured: star-only gives
  corr 0.996 at L1, and the triangle adds nothing. ✓
- **κ₄**: star = 3 edges, **chain = 3 edges** → *same order*. I implemented only
  the star, so we have been computing roughly **half** of κ₄'s leading generation.

This is why every κ₄ configuration measured the same: uncalibrated 3.13e-5
(r₄=8) vs **2.47e-5 (r₄=1, i.e. κ₄ mostly off)** — removing κ₄ made the closure
*better* — and calibrated val was identical (1.053e-5 vs 1.066e-5). The
calibration simply learns to switch κ₄ off.

The chain's slice is cheap once derived: with `P = diag(h₁)W`, `Q = diag(h₂)W`,
`R = Σ·P`, `U = Q ∘ R`, it collapses to `(Uᵀ Σ U)_pq` — two n³ matmuls.

## Two flopscope traps (cost us a failed submission)

1. **einsum infers symmetry from repeated operands.** `einsum("i,im,ik,il->mkl",
   h2, A, Bv, Bv)` sees the repeated `Bv` and tags the output symmetric across
   ALL axes, then validates and rejects — but an individual star term is symmetric
   only in (k,l); *only the sum of the three* is fully symmetric. Error surfaces as
   `Tensor not symmetric along axes (0,1,2): max deviation = inf`, which killed
   100/100 MLPs. **Fix: build these contractions with matmul + reshape, which
   carries no symmetry inference.** Same math, same FLOPs.
2. **flopscope has `linalg.qr` and `linalg.eigh`** — use them. My first port
   hand-rolled Gram-Schmidt in a Python loop (~32k tiny ops/MLP), which would have
   been a residual-wall-time disaster at λ = 1e11 FLOPs/s.

## A retraction (mine)

I "fixed" the star symmetrisation, claiming the permutes put the A-factor in slot
1 twice. **That was wrong** — I misread torch's `permute` semantics;
`U3.permute(0,2,3,1)[b,m,k,l] = U3[b,l,m,k]` *is* the slot-2 term. The permutes
were correct and my fix was a no-op (which is why the uncalibrated value came back
bit-identical). The real bug was flopscope's symmetry inference, above.

## THE PLAN: layerwise fitting (`fit_layerwise.py`)

Every end-to-end configuration plateaus at **~1.05e-5** (r=16, r=32/r₄=32,
r=32/r₄=8 — all the same), which is a *structural* limit, not a capacity one.
Cause: each layer's coefficients get their gradient from ONE distant loss (256
numbers, backpropagated through 32 layers).

The corpus (now 64 MLPs) has the TRUE μ, Σ, κ₃, κ₄ at **every layer**, so:

- **(A) Moment corrections are LINEAR in the coefficients** → closed-form least
  squares, per layer, with the true cumulants injected. No SGD at all. The
  *residual* of each fit says exactly what the correction form cannot express.
- **(B) Cumulant propagation** — the actual bottleneck — gets supervised directly
  against sampled κ₃/κ₄ (32 × 65k targets per MLP instead of 256), with the κ₄
  **chain diagram** fitted from a candidate dictionary rather than derived by hand
  (given today's track record, that is risk management, not humility).

## 📊 HONEST SCORECARD (as of the κ₃ closure, before κ₄ + calibration)

**Leaderboard reality:**

| entry | public score | raw final-layer MSE | budget used | method |
|---|---|---|---|---|
| **our v4 (submitted, best)** | **3.396e-7** | 3.12e-6 | 10.9% | QMC (6400 shipped pts) + calibrated K=2, fused |
| our v3 | 3.689e-7 | 3.67e-6 | 10.1% | QMC only |
| our v1 | 6.53e-7 | 6.5e-6 | 9.2% | iid MC — tied the MC reference exactly |
| MC reference | 6.47e-7 | — | — | organizers' baseline |
| top-10 rival (bruce-mckenzie) | 3.24e-8 (per-MLP) | 3.24e-7 | 8.5% | **pure analytic**, 21 s wall, no sampling |
| **leader** | **9.16e-8** | 3.55e-7 | 26% | unknown; analytic-dominant |

**Where the analytic closure actually stands:**

| closure | raw MSE | vs v4's raw (3.12e-6) |
|---|---|---|
| plain K=2 | 1.30e-4 | 42× worse |
| K=2 + κ₃, rank 16 | 3.92e-5 | 13× worse |
| K=2 + κ₃, **rank 48** | **2.76e-5** | **9× worse** |
| K=2 + κ₃ + κ₄, rank 8, *uncalibrated* | 4.61e-5 | 15× worse |
| *target to beat v4* | **< 3.1e-6** | — |

**Read this honestly: the analytic closure does NOT beat v4 yet — it is ~9×
short on raw MSE.** Being cheap (0.5–1% of budget vs 10%) buys nothing extra,
because the score multiplier floors at 10%; a pure-analytic estimator must win
on raw MSE alone.

What we have that justifies continuing:
- κ₃ propagation *works*: correlation with sampled truth 0.94–0.97 across depth,
  magnitude ~65% of truth (rank 48), and it already gives **4.7×** over K=2.
- The single-step unit test says **κ₄ is worth ~145× on the mean step after κ₃**
  — that is the missing factor, and it is not yet in any end-to-end number.
- Calibration bought 3× on plain K=2, and the closure now *contains* the right
  structural terms for it to correct (our κ₃ has the right shape but ~35% low —
  exactly what a learned per-layer scale fixes).
- Fallback: even at analytic b² ≈ 1e-5, **fusing** with the existing QMC gives
  ≈3e-6 → score ≈3e-7, i.e. roughly v4. So the work has a floor under it.

Risk, stated plainly: κ₄ + calibration must deliver ~9× for the pure-analytic
route to overtake v4, and ~30× more to reach the leader. Plausible from the
unit test, unproven end-to-end.

## RANK: a separate knob from κ₃/κ₄ (not a substitute)

- **κ₃/κ₄ = which physics is included** → sets the accuracy *ceiling*. No amount
  of rank reaches it (the unit test's 145× from κ₄ is unavailable to K=3).
- **rank = fidelity of the carried cores** → representation error only. It has
  built-in diminishing returns: the accumulated κ₃ is nearly rank-1 at depth
  (top-1 energy 0.98), and the early layers where it is genuinely high-rank are
  covered by the *uncompressed* star term.
- Measured: r=16 → 3.32×, r=48 → 4.72× over K=2. Real, but saturating.
- **Cost asymmetry matters:** κ₃'s core costs O(n·r³ + r⁴); κ₄'s costs
  O(n·r⁴ + r⁵). At r=48 the κ₄ star projection alone would be ~4.3e10 FLOPs =
  **16% of budget** — past the floor for nothing. So: **large rank for κ₃
  (32–48), small for κ₄ (8–16)**, which is exactly what the rank-1-at-depth
  evidence supports.

## THE BUILD: compute κ₃(z) analytically at O(n³) — no tensors

κ₃ of the activations is **structured**, not arbitrary. Leading order (star
diagrams):

    κ₃(a)_ijk = h₂(i)h₁(j)h₁(k)·Σ_ij·Σ_ik + (i↔j) + (i↔k)

and after the linear map the *slice we need* collapses to three matmuls:

    B = Σ·diag(h₁)·W
    C_ip = h₂(i)·W_ip·B_ip ,   D_ip = h₂(i)·B_ip²
    κ₃(z')_ppq = 2·(CᵀB)_pq + (DᵀW)_pq          ← O(n³), ~5e7 FLOPs/layer

i.e. ~1.6e9 FLOPs total — comfortably inside the floor, and **no n³ tensor is
ever materialized**. (My earlier "n² slices can't propagate" retraction stands
for arbitrary tensors; the point is that κ₃(a) is *generated* by a ReLU acting
on a near-Gaussian, so it has closed form.)

Two terms remain to be priced (`analytic_k3_test.py`):
- **triangle** `h₂h₂h₂·Σ_ij Σ_jk Σ_ki` — O(n⁴) exactly, but O(n²r²) with a
  rank-r truncation of Σ (justified: eig1frac 0.5–0.8).
- **pass-through** `h₁h₁h₁·κ₃(z)_ijk` — the incoming third cumulant carried
  forward. **This is the term that forces a factored tensor, and it is exactly
  why ARC's implementation costs 40× the budget.** If it turns out to be
  subleading at this depth, we get K=3-class accuracy at K=2 cost.

## The plan

1. **[done] Error attribution at the final ReLU** (`error_attribution.py`).
   The final estimate is g(μ̂,σ̂) = σ̂φ(α̂)+μ̂Φ(α̂) on the final pre-activation z.
   Exactly three error sources: (A) wrong propagated moments (μ̂,σ̂), (B) the
   Gaussian *shape* assumption about z (z is skewed), (C) interaction.
   Sampling gives true μ_z, σ_z, skew(z) and true E[ReLU(z)], so we price each:
   - **shape (B) dominates** → fix the 1-D marginal of z (Edgeworth /
     Gauss–Hermite quadrature on the tracked skew). Cheap, no tensor
     propagation — and it would explain how someone hits 3.2e-7 at the floor.
   - **moments (A) dominate** → the upstream moments are wrong → rank-r K=3.
   The script also reports the last hidden layer's covariance spectrum
   (top-eigenvalue share), which sizes the "non-Gaussian core + Gaussian
   remainder" closure below.

2. **Candidate closure if the core is low-rank** (deduction–projection made
   concrete): deep ReLU covariance is dominated by a **common mode**. Write
   z = μ + U c + ε with c ∈ R^r *non-Gaussian* (tracked by a small quadrature
   grid / its moments) and ε Gaussian. Then
   E[ReLU(z_i)] = E_c[ σ_ε,i φ(α_i(c)) + m_i(c) Φ(α_i(c)) ],
   evaluated by Q-node quadrature over c: O(Q·n²) per layer — cheap, and a
   *lot* of small elementwise ops (matching the competitor's 1.1 GFLOP/s
   wall-time signature). This captures the dominant non-Gaussianity (the
   radial/scale fluctuation) that a Gaussian closure structurally cannot.
2. **[hours] Fix G2.** Swap in the exact bivariate ReLU second moment
   (arc-cosine / Owen's T) — removes the series entirely.
3. **[days] Fix G3 — the real prize.** Fixed-rank factored third cumulant
   (rank r tuned to fill the floor), with power cumulants and the odd-K trace
   correction, i.e. the paper's K=3 made affordable by rank truncation.
4. **Keep the end-to-end calibration harness** — it composes with *any*
   closure (gave 3× on K=2) and is our differentiator. Re-fit after each of
   1–3.
5. **Re-derive the operating point (G4).** Once b² ≪ QMC variance, drop
   sampling and spend the whole floor on the analytic method — matching the
   competitor's profile (pure analytic, ~8.5–10% of budget, long wall time).

**Most promising single idea:** exact (or high-order) ReLU covariance closure
+ fixed-rank third-cumulant propagation, calibrated end-to-end, run
pure-analytic at the 10% floor.

---

## Context

- **Task (phase 1):** predict per-neuron mean activations of random He-init ReLU MLPs
  (width 256, depth 32, no bias, `N(0,1)` inputs) under a FLOP budget **B = 2.72e11**.
- **Leaderboard metric:** `final_layer_mse × max(0.1, C/B)`, where `C = flopscope FLOPs +
  λ·residual_wall_time` (λ = 1e11 FLOPs/s). Only the **final layer** is ranked.
  Compute below **10% of budget (2.72e10 FLOPs) is free** (multiplier floor 0.1).
  Exceeding the budget zeroes all predictions with multiplier 1.0.
- **Key resources reviewed:**
  - arXiv:2605.05179 — *Estimating the expected output of wide random MLPs more
    efficiently than sampling* (Wu, Lecomte, Winer, Robinson, Hilton, Christiano — ARC).
    Cumulant propagation ("kprop") to order K via Hermite/Wick expansions; error
    O(1/n^K); factorized variant for K≥3; needs power cumulants + odd-K trace corrections.
  - alignment.org: *Competing with sampling* (Matching Sampling Principle),
    *Estimating tail risk in neural networks*, *AlgZoo*.
  - `mlp_cumulant_propagation/` — ARC's official kprop implementation (torch,
    package `mlp_kprop`): `kprop_harmonic.mlp_kprop(...)` entry point, `wick.py`
    (closed-form ReLU Hermite/Wick coefficients), `factor_k3/k4.py`, `NamedFlopCounter`.

## Measured results (challenge shape: 256 × 32, He init)

Ground truth: 8e6-sample MC (noise floor ~1.5e-9). Scripts: session scratchpad
`kprop_depth32.py`, `hybrid_test.py`.

| method | final-layer MSE | FLOPs | % of budget |
|---|---|---|---|
| starterkit zeros (template) | ~0.91 | ~0 | 0% |
| starterkit mean propagation | ~9.5e-4 | 1.1e7 | ~0% |
| starterkit covariance prop (gain approx) | ~8.4e-5 | 1.6e9 | 0.6% |
| kprop K=1 | 2.2e-4 | 5e6 | ~0% |
| kprop K=2 (proper Wick covariance) | 1.7–7.3e-5 (3 seeds, avg ~4.7e-5) | 2.2e9 | 0.8% |
| kprop K=3 factorized | 2.3e-5 (no final-layer gain) | 3.1e11 | **114% — over budget** |
| kprop K=3 factorized augmented | 1.8e-6 | 1.15e12 | **421% — over budget** |
| **antithetic MC at the floor (~4.7k samples)** | **2.0–4.8e-6 (3 seeds)** | 2.7e10 | 10% (free) |

Key observations:

1. **K≥3 cumulant propagation is not viable at depth 32.** The factored rank grows
   with depth, so cost scales ~depth²·n³ — it exceeds the budget, and plain K=3
   doesn't even beat K=2 on the final layer at this depth. (The paper only tested
   ≤12 hidden layers.)
2. **Plain Monte Carlo at the free floor beats every in-budget analytic method
   ~10×** on the ranked (final-layer) metric, because activations decay with depth
   (final-layer variance ≈ 0.012, mean |μ| ≈ 0.23). Pure-MC score is flat in
   compute above the floor, so the optimum is exactly ~10% of budget.
3. **kprop K=2 is ~2× better than the starterkit covariance baseline** at the same
   cost — the fix is the Wick-coefficient ReLU covariance update (`wick.py`)
   instead of the `Φ(α_i)Φ(α_j)` gain approximation.
4. **Naive hybrid (K=2 + MC, scalar inverse-MSE weights) is worse than pure MC.**
   Oracle per-neuron weights give only ~1.5× over pure MC — smarter fusion needs
   reliable per-neuron bias estimates.

## Suggested improvements (ranked)

1. **Ship the floor-filling MC estimator** *(done — see log)*: antithetic pairs
   (x, −x), one batched matmul chain in `flopscope.numpy`, sized to ~9% of budget
   (margin for RNG at ~16 FLOPs/element and the residual wall-time bucket).
   Expected score ~3e-7 vs ~8.4e-6 for the covariance baseline (~25× better).
2. **Variance reduction on top of MC** (every 2× variance cut = 2× score):
   - input moment-matching (standardize the sample batch's empirical mean/var),
   - control variates with *exactly* known expectation — e.g. the linearized
     network `J·x` (mean exactly 0, no bias leakage),
   - quasi-MC / stratification on ‖x‖.
3. **Per-neuron-weighted fusion with kprop K=2**: port the K=2 Wick covariance
   update to flopscope (2.2e9 FLOPs, fits trivially); weight per neuron using the
   K=1↔K=2 discrepancy as a bias proxy with empirical-Bayes shrinkage toward MC.
   Oracle bound: ~1.5× over pure MC.
4. **Research direction — tame K=3's depth² cost**: rank-truncate the factored
   third cumulant each layer, or apply the augmented K=3 update to a few layers
   only. Measured cost curve says ~10× compression is needed to fit under the floor.
5. **Engineering:** keep Python overhead vectorized (λ·R counts toward C); seed
   from `mlp.seed` per contract; validate on the fixed mini split; watch
   `worst_mlp_adjusted_final_layer_score` (mean over 10 MLPs is outlier-sensitive).

## Session log

1. Read the starter kit (README, scoring model, problem setup, algorithm ideas,
   examples 01–04, flopscope primer).
2. Fetched and read arXiv:2605.05179 (the challenge's companion paper) and the
   alignment.org posts (*Competing with sampling*, *Estimating tail risk in neural
   networks*, *AlgZoo*).
3. Explored `mlp_cumulant_propagation` (ARC's official kprop repo): API, variants,
   FLOP counter; `uv sync` to install.
4. **Experiment 1** (`kprop_depth32.py`): kprop K=1/2/3/3-augmented at 256×32 vs
   8e6-sample MC ground truth, with the repo's FLOP counter → table above.
   Finding: K≥3 over budget at depth 32; K=2 ~2× better than starterkit baseline.
5. **Experiment 2** (`hybrid_test.py`, 3 seeds): antithetic MC at the floor
   (~4.7k samples) vs kprop K=2 vs hybrids → MC wins ~10×; scalar-weight hybrid
   worse than pure MC; oracle per-neuron weights ~1.5× over MC.
6. Checked flopscope RNG pricing (~16 FLOPs/element for `standard_normal` —
   negligible; ~1.2e7 total).
7. Implemented the antithetic-MC estimator in `whest-starterkit/estimator.py`
   (`_BUDGET_FRACTION = 0.09`).
8. Stage 1 (`python estimator.py`): runs; Stage-1 harness uses a small default
   budget so its MSE isn't representative.
9. Stage 2 (`whest validate`): **all checks pass**.
10. Stage 3 (`whest run --split mini` at the contest budget): see results below.

## Mini-split results (Stage 3)

`whest run --estimator estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner local`

| metric | value |
|---|---|
| **adjusted_final_layer_score** (primary) | **9.20e-07** |
| final_layer_mse (raw) | 9.20e-06 |
| all_layers_mse | 2.08e-05 |
| best / worst MLP score | 6.82e-08 / 4.73e-06 |
| mean score multiplier | 0.100 (floor) |
| mean compute utilization | 8.99% of budget |
| failed MLPs | 0 / 100 |
| residual wall time (total, 100 MLPs) | 0.028 s (λ·R ≈ 2.8e7 FLOPs-equiv per MLP — negligible) |

~9× better than the covariance-propagation baseline (8.4e-6) and ~100× better
than mean propagation (9.5e-5). The 100-MLP suite has more spread than the
3-seed experiment (mean final-layer MSE 9.2e-6 vs ~3.4e-6 on seeds 0–2);
per-MLP variance-reduction (suggestion 2) is the next lever.

11. Stage 5 (`whest package`): tarball built (`submission-20260711-174747.tar.gz`).
12. `whest submit --estimator estimator.py --watch`: first submission shipped to
    AIcrowd.

## First leaderboard result (public split, 50 MLPs)

- **Adjusted score 6.526e-7** — 0/50 failures, 9.16% budget utilization.
- **Exactly 1.0× the organizers' Monte Carlo reference (6.47e-7)** — as expected,
  since this estimator *is* floor-budget MC. Antithetic pairing bought ~nothing:
  after 32 ReLU layers the output has almost no odd (sign-linear) dependence on
  the input, which also predicts the linearized `J·x` control variate won't help.
- Spread: best MLP 9.75e-8, worst 2.58e-6, IQR 2.7e-7–7.4e-7 → the mean is
  dominated by a few high-variance MLPs; adaptive per-MLP fusion helps most there.
- Next levers, in order: (a) bump `_BUDGET_FRACTION` 0.09 → ~0.097 (~7% free);
  (b) per-neuron fusion with a flopscope port of kprop K=2, weights leaning on
  the analytic side for high-variance MLPs/neurons (oracle bound ~1.5×);
  (c) variance reduction that survives depth: exact-radius sampling (ReLU nets
  without bias are positively 1-homogeneous, so E[act] = E[r]·E[act(u)] with
  E[r] = √2·Γ((n+1)/2)/Γ(n/2) known exactly), orthogonal/stratified directions;
  (d) speculative: multilevel telescoping across depth with analytic tails.

## Round 2 experiments (same day)

**kprop K=2 fusion: tested and rejected** (`fusion_test.py`, 10 seeds).
A self-contained flopscope-portable K=2 port (Wick-series covariance to k=4,
exact diagonal) was verified against `mlp_kprop` (seed 0: 1.44e-5 vs 1.67e-5 —
port slightly better). But fusion with MC lost to pure MC: mean final-layer MSE
pure MC 6.49e-6 vs fuse-A 7.43e-6, fuse-B 7.05e-6; even oracle per-neuron
weights only reach 4.74e-6 (1.37×). The per-neuron bias signal `(mc − analytic)`
has detection noise equal to the MC noise itself, so realistic weighting is net
negative. Conclusion: no analytic component; spend everything on sampling.

**Quasi-Monte Carlo: tested and adopted** (`qmc_test.py`, 8 seeds, 6.3k points):

| point set | final-layer MSE (mean) |
|---|---|
| iid N(0,1) (previous submission) | 8.41e-6 |
| scrambled Sobol → Gaussian | 2.77e-6 |
| **Sobol + exact moment matching** | **2.65e-6 (3.2× better)** |
| iid + moment matching | 4.15e-6 |

The point set is input-distribution-only, so it is computed **offline**
(off-budget), moment-matched exactly (max |mean| 5e-17, max |cov−I| 1e-14),
and shipped as `points.npz` (`flopscope.Module`, 0-FLOP load in `setup()`).
This also eliminates the in-budget RNG cost. `estimator.py` v2: forward passes
over the shipped 6272-point set sized to ~9.6% of budget, iid-RNG fallback for
non-256-wide shapes. Antithetic dropped (measured ~no gain at depth 32).

**Mini-split, estimator v2:** adjusted score **4.28e-07** (was 9.20e-07 —
2.15× better), raw final-layer MSE 4.28e-06, all-layers 7.73e-06, utilization
9.59%, 0/100 failed. Expected public-split score ~3e-7 ≈ 0.5× the MC reference.

Packaging note: submissions must use **folder mode** (`whest package/submit
--estimator . --yes`) so `points.npz` ships; the kit's `.gitignore` had a
`*.npz` rule that silently excluded it (removed).

## v2 leaderboard result (public split, 50 MLPs)

- **Adjusted score 3.783e-7** — **1.7× better than the MC reference (6.47e-7)**;
  v1 → v2 = 6.53e-7 → 3.78e-7 (1.73×). 0/50 failures, utilization 9.75%.
- Tail improved more than the mean: worst MLP 2.58e-6 → 1.19e-6 (2.2×), IQR
  down ~30% — better-spread points help exactly the high-variance MLPs.
- Public raw final-layer MSE 3.78e-6 vs mini split 4.28e-6 vs local 8-seed
  2.65e-6: local seeds flatter systematically; the mini split is the honest
  predictor.
- Error no longer grows with depth (L0→L31 growth 0.85×) — sampling error
  tracks the decaying activation variance; steepest layer is L1, hinting
  early-layer point geometry is where uniformity matters most.
- Utilization margin is thin (9.75% vs 10% floor): trim `_BUDGET_FRACTION`
  slightly in v3 (≈0.0965) — ~50 fewer samples is negligible vs floor risk.

---

# Strategy: how to win (deep-dive, 2026-07-11 evening)

## Sources reviewed in full

- **arXiv:2605.05179 incl. appendices.** Decisive facts: (App. D) kprop MSE
  scales as c_K·(L/n)^K — error *grows* with depth as L^K, and the authors
  state the open problem: *"a sample-free algorithm whose error does not
  increase with depth… we leave this to future work"*. (Table 1, App. J) exact
  FLOP polynomials: K=2 basic 7n³L; K=2 augmented 11n³L (~5.9e9 — affordable,
  tested below); factorized K=3 30n³L² (~5.2e11 — the L² term is fatal at
  L=32). (App. E) power cumulants + odd-K trace are essential (ablation → O(1)
  error). (App. I) even the authors' wall-clock is dominated by Python-side
  elementwise work — mirrors the challenge's λ·R residual penalty.
- **mlp_cumulant_propagation repo.** TODOs are engineering-level. One
  strategic note (`symb/README.md`): whole-network tensor-network contraction
  can beat layerwise propagation asymptotically, "but constants too large to
  be practical".
- **alignment.org**: Competing with sampling (Matching Sampling Principle:
  beat O(1/ε²) sampling given structure), Estimating tail risk (layer-by-layer
  modeling loses cross-layer info), AlgZoo.
- **Dunbar & Aaronson (arXiv:2510.06527)**: deep random nets have nearly
  independent outputs *iff the activation has zero Gaussian mean*. ReLU has
  E[ReLU(Z)] > 0 ⇒ deep ReLU nets are the maximally-correlated case: final
  activations for any two inputs are strongly correlated (correlation → 1
  with depth). Deep-net fluctuations across inputs are low-effective-dimension.
- **Wu & Hilton LPE (ICLR 2025)**: importance sampling > activation
  extrapolation — tail-specific, not directly transferable to mean estimation.
- **Challenge announcement (alignment.org, June 2026)**: ARC states their
  methods "break down as the depth grows" and are "very confident" they can be
  significantly improved. **Any method is allowed, white- or black-box** —
  sampling-based approaches are explicitly legal; the only ban is
  circumventing the FLOP counter (our shipped-points pattern is the
  starterkit's own documented mechanism). Prize pool ≥ $100k with a **second
  prize track for algorithmic contributions in technical reports** — this
  document doubles as that report. LLM-assisted work is explicitly encouraged.
- **"Mechanistic estimation for expectations of random products"
  (alignment.org, May 2026)** — ARC's newest framework and their own answer to
  "keep more information only where it's valuable": **deduction–projection
  estimators** alternate exact computation steps with complexity-reducing
  projections, and **mechanistic L² sketching** projects the intermediate
  representation onto the top eigenfunctions of a (learned) kernel. Results so
  far: matches sampling up to log factors on random halfspace intersections /
  #3-SAT / permanents — but **only single-layer** so far; no depth analysis.
  For depth-32 MLPs this is the theoretically right replacement for cumulant
  truncation (project onto what matters for the *tail* of the computation,
  instead of keeping a fixed cumulant order everywhere) — a research-project-
  sized lift, not a weekend patch.
- **"A Mike's-Eye View of ARC's Research" (June 2026)**: formal MSP statement
  ("for any architecture and precision there is a mechanistic estimator that
  at least matches sampling over random instances"); confirms cumulant
  propagation is ARC's best current MLP result; structure-vs-randomness and
  training-trajectory advice are the longer-term picture.
- **"Obstacles in ARC's research agenda" (May 2025)**: the three named
  obstacles (finding explanations, mechanistic anomaly detection, low
  probability estimation) — background, no algorithmic content for this task.

## The scoring math (what actually matters)

Score = final-layer MSE × max(0.1, C/B), B = 2.72e11, C includes λ·wall-time.
Three regimes:

1. **Below 10% of B, compute is free** → the only question is: what is the
   most accurate estimator costing ≤ 2.72e10 FLOPs?
2. **Above the floor**, an estimator with MSE ∝ C^(−β) has score ∝ C^(1−β):
   worth exceeding the floor iff β > 1. *Measured*: iid MC β = 1 by
   definition (score flat); our moment-matched Sobol QMC has **β ≈ 0.83 < 1**
   (its edge decays with N; at full budget QMC ≈ iid). **The floor is the
   optimal operating point.** (Measured score curve: 2.69e-7 at 10%, 3.72e-7
   at 20%, 4.68e-7 at 100%.)
3. Ground truth is itself MC with N=1e9 ⇒ irreducible score floor ≈
   Var/1e9/10 ≈ 1e-12. Irrelevant for now (we're at ~3e-7).

## Why sampling wins here, mathematically

At depth 32 the kprop error c_K(L/n)^K is ~(1/8)^K per unit variance, while
final-layer activation variance has *decayed* to ~0.012 (mean |μ| ≈ 0.23) —
so sampling error Var/N at the free floor (~6.5k samples) is ~2e-6·(per-MLP
spread), an order of magnitude below any in-budget analytic method. Deep
correlation collapse (Dunbar–Aaronson) hurts analytics but *helps* sampling:
the integrand family is nearly constant + low-rank fluctuations.

## The kernel-quadrature framing (the key idea)

For a **fixed** point set P = {p_1..p_N} with weights w, the leaderboard
averages over random W, so the expected squared error is exactly a
**Bayesian-quadrature discrepancy** under the depth-32 NNGP kernel
κ(p,q) = E_W[act_i(p)·act_i(q)] (computable in closed form by the arccos/ReLU
kernel recursion — or empirically):

    E_W[err²] = wᵀ K w − 2 wᵀ κ̄ + E_{x,x'}[κ],   κ̄(p) = E_x[κ(p,x)]

Everything on the right is *independent of the test MLPs* — so both the
**weights** (convex problem, w* = K⁻¹κ̄) and the **points** (gradient descent)
can be optimized **offline and hardcoded**, at zero extra predict cost
(a weighted mean costs the same FLOPs as a plain mean). This directly answers
"find patterns empirically through sampling and hardcode them": the pattern
IS the kernel, and the offline-optimal quadrature rule is its exploitation.

Practical version (subsumes BQ, captures finite-width effects): generate
hundreds of MLPs offline with high-precision ground truth, and **ridge-regress
fixed weights w** (plus a global coefficient on cheap analytic features like
the mean-prop estimate) minimizing average ||Σ_j w_j·act(p_j) − truth||² —
validated on held-out MLPs and the public split. Overfitting is controlled by
ridge toward uniform 1/N and by the huge effective sample count
(MLPs × 256 neurons per MLP).

## Ranked plan

1. **Learned quadrature weights** (offline ridge on 128 MLPs, 32 held-out).
   *Result: rejected* — best learned weights 4.59e-6 vs uniform 4.64e-6 (<2%);
   the ridge collapses to uniform and the mean-prop coefficient regresses to
   exactly 0. Uniform weights on the Sobol-mm set are already ≈ the BQ optimum;
   the residual error is per-MLP idiosyncratic (posterior variance), not
   removable by static reweighting. Spherical projection: ~1% (kept anyway —
   exact and free).
2. **Learned/optimized point set** (in progress): SGD directly on the 6272×256
   point matrix against the *true ensemble* — per step draw a fresh random MLP
   and fresh Gaussian batch, unbiased loss ‖m_P‖² − 2⟨m_P, m_X⟩ (= quadrature
   MSE + const), rows re-projected to the exact-radius sphere. Fresh MLP every
   step ⇒ infinite training data, no overfitting channel; validated on 24
   fixed MLPs with 2e6-sample ground truth.
3. **Global-coefficient analytic fusion** inside the same regression (mean-prop
   / K=2 features with *offline-learned* fixed coefficients — per-MLP adaptive
   weighting already failed; global learned coefficients are the sound form).
4. **K=2-augmented** check (5.9e9 FLOPs): the only affordable untested kprop
   variant. *Result: rejected* — final MSE 6.3e-4 / 2.4e-3 / 7e+3 over 3 seeds
   (numerically unstable at depth 32; its own variance clamp fires). Every
   kprop variant is now measured at the challenge shape and out.
5. Out-of-scope after measurement: K≥3 (budget), antithetic & linearized
   control variates (no odd/linear signal at depth 32), per-MLP adaptive
   fusion (bias detection noise = MC noise), running above the floor (β<1).

## Research observations (full dump)

### Structure of the problem at 256×32

- **Signal decay:** final-layer activations have mean |μ| ≈ 0.23 and per-neuron
  variance ≈ 0.012 (it decays with depth). Small variance is exactly why
  sampling is strong at the final layer — and the final layer is all that's
  scored.
- **Per-MLP spread is huge:** final-layer variance varies ~25× across MLPs
  (leaderboard: best MLP 9.75e-8, worst 2.58e-6; wanda-sanders-type MLPs
  dominate the mean score). Any robustness gain on the worst tail moves the
  leaderboard more than uniform improvement.
- **No odd signal survives depth 32:** antithetic pairing gained ~nothing,
  which also implies any control variate linear in x (e.g. J·x) is dead. The
  output's dependence on the input is effectively even/nonlinear.
- **Correlation collapse (Dunbar–Aaronson):** deep-net outputs are nearly
  independent iff the activation has zero Gaussian mean; ReLU has positive
  mean, so activations for two independent inputs become strongly correlated
  with depth. Consequences: (a) the fluctuation across inputs is
  low-effective-dimensional; (b) analytic layerwise methods accumulate error
  (correlations are exactly what they truncate); (c) quadrature/QMC exploits
  it — but its edge *decays* with N (measured β ≈ 0.83), because the
  low-dimensional structure is exhausted by a few thousand well-spread points.
- **Positive 1-homogeneity** (no biases): act(λx) = λ·act(x) for λ>0 ⇒
  E[act] = E‖x‖ · E[act(u)] with E‖x‖ = √2·Γ(128.5)/Γ(128) exact. Radial
  variance is only ~1% of the total at n=256 (chi concentration), but the
  projection is free and exact, so we keep it.
- **Ground truth is itself MC (N=1e9):** irreducible score floor ~1e-12;
  irrelevant at our ~3e-7, but means "perfect" prediction is impossible and
  the last ~1e-11 of MSE is noise-fitting.

### Scoring-model exploits (all verified)

- Compute below 10% of budget is free (multiplier floor 0.1) — the contest is
  really "best estimator at 2.72e10 FLOPs". Exceeding the *full* budget zeroes
  you with multiplier 1.0 — a 10× cliff; never sail close to 100%.
- Only the final layer is ranked; all-layers MSE is a diagnostic.
- `setup()` and shipped `.npz` files (`flopscope.Module`) are **off-budget** —
  unlimited offline compute is legal and is the single most exploitable rule.
  50 MiB / 50 file caps apply; `*.npz` was silently gitignore-excluded.
- Residual wall-time counts: C = FLOPs + 1e11·R_m. Keep everything inside
  flopscope kernels (batched matmuls); our measured R ≈ 0.3 ms/MLP.
- RNG costs ~16 FLOPs/element — shipping fixed points avoids even that.
- flopscope credits symmetric einsums (`fnp.einsum("ij,ia,jb->ab", cov, w, w)`)
  — relevant only if analytic components return.

### The kprop landscape at depth 32 (all measured, all excluded)

- Depth scaling is the killer: MSE ~ c_K(L/n)^K grows as L^K (paper App. D);
  the authors themselves flag depth-robust sample-free estimation as open.
- Exact costs (paper Table 1): K=2 basic 7n³L ≈ 3.8e9 ✓ affordable, MSE
  ~4.7e-5 avg (7× worse than QMC at floor). K=2 augmented 11n³L ≈ 5.9e9 —
  numerically unstable at L=32 (MSE 6e-4 … 7e+3). Factorized K=3 30n³L²
  ≈ 5.2e11 — over budget purely from the L² term (factor rank grows per
  layer); plain K=3 also doesn't beat K=2's final-layer MSE at this depth.
- Power cumulants are essential (ablation degrades O(1/n^K) → O(1)); any
  reimplementation must keep them.
- Repo's `symb/` note: whole-network tensor-network contraction with optimal
  (non-layerwise) paths is asymptotically faster "but constants too large".

### Statistical no-go results (measured, with the reason)

- **Per-neuron fusion of MC + analytic cannot work at this budget:** the
  per-neuron bias signal is (mc − analytic), whose detection noise *is* the MC
  noise; even oracle per-neuron weights gave only 1.37×, realistic rules were
  net negative. General principle: any per-MLP adaptive scheme must extract
  its steering signal from the same sample budget, and at the floor that
  signal is at the noise level.
- **Learned static quadrature weights collapse to uniform** (<2% gain, ridge
  pushes |w − 1/N| → 0; mean-prop feature coefficient → exactly 0): the
  Sobol+moment-matched set is already ≈ BQ-optimal for the ensemble kernel.
  Remaining error is per-MLP posterior variance — only *better points* (or a
  fundamentally different information channel) can reduce it.
- **β < 1 for QMC** ⇒ never leave the floor; iid MC has β = 1 exactly (score
  flat in compute), so an estimator must have β > 1 before more compute helps.

### Ideas considered, not yet exhausted (future work)

- **Point-set optimization on the true ensemble** (running): SGD on the
  6272×256 matrix, fresh MLP + fresh Gaussian batch per step (unbiased loss
  ‖m_P‖² − 2⟨m_P, m_X⟩), spherical re-projection each step. No overfitting
  channel by construction.
- **Per-MLP point selection from a shipped dictionary:** ship k× more points,
  select N at predict time with a cheap fixed rule using W statistics (e.g.
  first-layer projections). Cheap (n² per point) but the deep composition, not
  the first layer, drives the integrand — speculative.
- **Multilevel depth telescoping:** E[f_L] = E[f_j] + E[f_L − f_j] with an
  analytic tail from layer j; needs the tail to be cheaper than the true
  continuation AND per-sample-correlated with it — no construction found yet
  (given x_j the continuation is deterministic; any cheap tail is a smoothed
  version whose per-sample residual is O(1)).
- **Robust aggregation for the worst-MLP tail:** trimmed/winsorized weighted
  means cost the same FLOPs; could trade a little average MSE for tail
  robustness. Untested.
- **Nonlinear readout:** a tiny offline-trained head g(sample statistics) →
  correction. The linear version (weights) regressed to uniform; nonlinear
  features (per-neuron sample variance, norms) might still carry signal —
  but the fusion no-go suggests limited headroom.
- **Deduction–projection / L² sketching for depth (ARC's May-2026 framework):**
  the principled fix for the L^K depth blow-up — after each exact layer step,
  project the activation-distribution representation onto the top
  eigenfunctions of a kernel matched to the *remaining* computation, instead
  of truncating at a fixed cumulant order. ARC has this working for 1 layer
  only. The offline-learnable ingredient (the kernel/eigenfunctions per layer
  index) fits this contest's ship-precomputation rule; the in-budget cost
  would be projection matmuls (O(n²k) per layer). Highest-upside, highest-
  effort direction on the board — the natural phase-2 project.

### The organizers' "Open directions" list, addressed

The starterkit's algorithm-ideas.md flags five untried directions. Status
against our measurements (key context: the ceiling of the analytic family —
full K=2 covariance, 1.4% of budget — loses to floor-sampling ~10×, so
refinements *within* that family are dominated):

| Direction | Verdict | Reason |
|---|---|---|
| Low-rank covariance (UUᵀ, rank k) | dominated | cheaper approximation of full cov; compute isn't binding, accuracy is; frees FLOPs we don't need (~900 MC samples' worth) |
| Layer-adaptive routing (full early → diagonal late) | dominated | full-cov-everywhere bounds this family's accuracy; the bias is joint-cumulant truncation over 32 layers, not budget placement |
| Spectral / SVD in `setup()` | premise wrong + dominated | `setup(ctx)` never sees the MLP — per-MLP SVD can't be off-budget; affordable in `predict` but lands in the mean-prop accuracy class; learned coefficient on analytic features regressed to 0 |
| Importance sampling | parked (untested, no identified edge) | no odd/linear input signal at depth 32, radial part exact, fluctuation spread (not rare-region-dominated — IS's win condition); right tool for tails, not means |
| Higher-order moments (skew/kurtosis) | measured & rejected | = cumulant prop K≥3: over budget at L=32, no final-layer gain; marginal-skew variants can't capture the joint off-diagonal cumulants where K=2's bias lives |

### Point-set optimization verdict (round 4)

SGD on the point matrix against the true ensemble (fresh MLP + fresh batch
per step, unbiased loss, spherical re-projection): **init 3.751e-6, best
3.644e-6 (+2.9%, within validation noise), then degrades** — gradient noise
destroys the Sobol structure faster than the tiny signal improves it. The
refinement run (12 MLPs/step, lr 3e-4) confirmed: best 3.681e-6 vs init
3.754e-6 (+1.9%, noise). Together with the weights result, the conclusion
stands: **uniform-weighted Sobol+mm points at the floor are ≈ the optimum of
the fixed-quadrature family.** Remaining levers are per-MLP information
channels (dictionary selection, robust aggregation) or the depth-robust
analytic frontier above.

---

# Cumulants revisited (round 5): the mathematics of a comeback

Prompted by the (fair) challenge: did we give up on cumulants too fast?
Re-derivation from first principles, all numbers at the 256×32 shape.

## The score exchange rate (the master inequality)

An estimator with per-MLP cost C and MSE m scores m·max(0.1, C/B).
Floor-QMC sits at m_q ≈ 4.0e-6, C = 0.1B → score 4.0e-7. Therefore **any
challenger must satisfy m ≤ 4.0e-6 · (0.1B / C)** — i.e. a method costing
20% of budget must *halve* the MSE just to tie. This inequality kills every
"where do we spend the K=3 budget" scheme before implementation:

- **Full factorized-augmented K=3** (the family's accuracy ceiling): measured
  1.8e-6 at seed 0 — genuinely better than sampling per FLOP-unconstrained
  accuracy! — but costs 4.2×B. It *cannot be submitted* (hard zero), and even
  rank-compressed to 50% of B it would need MSE ≤ 8e-7, a 2.3× improvement
  over its own uncompressed accuracy. Rank compression degrades, not
  improves. Dead by arithmetic.
- **Early-layer K=3-augment (layers 0..j) then K=2:** cost ≈ 72n³j²/2. At
  j=8: 3.9e10 (14% of B) and it corrects at most the first 8 layers' cumulant
  injection — bounded by ~(1−j/L)-type reduction of K=2's 4.7e-5, i.e.
  ~2-3e-5, needing ≤ 2.9e-6. Off by 10×. At j=4 it fits the floor but
  corrects ~1/8 of the injection: ~3.6e-5 vs needed 4e-6. Off by 9×.
- **Late-layer K=3:** the incoming third cumulants were already discarded by
  the K=2 prefix — the information the upgrade needs is gone. Strictly worse
  than the early variant per FLOP.
- General form: partial-K schemes buy a *linear* fraction of the correction
  at *quadratic* (rank-accumulation) cost, and the baseline gap is 10×.
  There is no allocation of K=3 FLOPs across layers that passes the exchange
  rate. This is why "be smart about where to use higher orders" fails *for
  in-budget propagation* — not because the idea is bad, but because the whole
  K=3 correction, even applied everywhere, is worth only ~parity with QMC
  while costing 42× the floor.

## Depth-telescoping: a clean negative theorem

Multilevel MC across depth (cheap surrogate g from layer-j state, many
samples of g + few of f−g) is *exactly break-even-or-worse* here: final
log-amplitude accumulates as a near-Brownian walk over layers, so the
correlation of any layer-j surrogate with the final output satisfies
Corr² ≈ j/L — proportional to its cost j/L. MLMC's gain condition
(√(V_diff·c_f) + √(V_g·c_g))² < V then evaluates to (√(1−ρ²) + ρ·√(c_g))²·V
with ρ² = c_g, which is ≥ V for all j (minimum at the endpoints). The
Brownian structure of deep ReLU amplitude makes depth-telescoping
information-flat. (Same structure explains QMC's β < 1.)

## The one live path: offline-calibrated closures + global fusion

The Gaussian closure is *systematically* wrong in a way that is **universal
across the MLP ensemble** (weights are exchangeable) — and universal errors
can be measured offline to arbitrary precision and shipped as calibration
tables at zero in-budget cost. This is the "learned closure" pattern from
turbulence modeling: keep the white-box propagation (μ, Σ, and cheap
diagonal-κ₃ at n² per layer), but correct each update — or regress the final
error on the closure state — using coefficients fitted on offline MLPs.

Why this reopens fusion: per-neuron fusion failed because K=2's bias
(4.7e-5) ≫ MC noise (4e-6), making the optimal analytic weight ~0.08 and the
weight estimate noise-dominated. But fusion weights can be **global,
learned offline** (no per-MLP noise), and if calibration pushes the residual
bias² b² down toward the QMC variance v, the fused MSE b²v/(b²+v) gives:
b² = 8e-6 → 2.8e-6 (score 2.8e-7, −30%); b² = 4e-6 → 2.1e-6 (score 2.1e-7,
−48%). **Win condition: calibrated K=2 residual bias² ≤ ~1e-5** (a 5×
calibration of the raw 4.7e-5), with K=2's 3.8e9 FLOPs costing only ~2% of
the sample budget.

The ceiling is measurable: decompose K=2's error on offline MLPs into the
part predictable from the K=2 internal state (ensemble-systematic ⇒
calibratable) vs the remainder (idiosyncratic ⇒ irreducible given that
state). Experiment `calibration_ceiling.py` (64 train + 24 held-out MLPs,
ridge on final-layer closure state incl. a propagated diagonal-κ₃ feature):

**Measured (held-out): 44% of K=2's error is ensemble-systematic.**
Raw 7.03e-5 → 5.59e-5 removing a *global scalar offset alone* (20.5% — the
Gaussian closure has a universal bias, confirming the premise) → 3.93e-5
with state-feature ridge. Fusion projection at that residual: 3.70e-6 →
score ~3.70e-7 vs v3's 3.99e-7 — only ~7%, because the residual is still
~10× QMC noise.

But post-hoc regression on the *final* state cannot undo 32 layers of error
compounding. The stronger form — **end-to-end calibration**, 6 learned
scalars per layer *inside* the closure update (192 universal parameters),
trained by autograd through the full K=2 propagation on a fresh random MLP
per step (unbiased loss, no overfitting channel) — corrects errors before
they compound. Experiment `calibrated_k2.py`:

**End-to-end calibration works: K=2 held-out MSE 7.18e-5 → 2.41e-5 (3×,
66% of the closure error removed).** (First attempt diverged — multiplicative
corrections compound as (1+p)³²; fixed with 0.05·tanh bounding, gradient
clipping, 4 MLPs/step, lr 2e-4. Converges within ~450 steps, stable.)
This is the largest analytic-side improvement measured all day and confirms
the premise: a large share of the Gaussian-closure error is universal and
shippable as 192 offline-learned scalars.

**But the exchange rate nets it to small change:** K=2 costs ~4e9 FLOPs ≈
950 samples ≈ 15% of the floor sample budget. Fusing calibrated-K2 (b² =
2.4e-5) with QMC (v ≈ 4.1e-6) gives fused MSE b²v/(b²+v) ≈ 3.5e-6 (−12.5%),
of which most is paid back either as fewer QMC points or as an over-floor
multiplier. Honest projected net: **−2 to −6% score** — verified empirically
in `fused_v4_check.py` (results below), including the error-independence
assumption and a train/val split for the global fusion weight.

## v4 (shipped): cumulants return via offline calibration

`fused_v4_check.py` verified the fusion honestly: calibrated-K2 and QMC
errors are independent (corr −0.04 held-out); global weight w = 0.137 fit on
24 offline MLPs; cutting QMC points to stay under the floor is a losing
trade (trimming the fixed set has local β ≈ 1.7), so v4 runs over the floor
instead. Estimator: 6400-point QMC + flopscope port of the calibrated K=2
(192 constants inlined; symmetric einsum ⇒ only ~0.6% of budget), fused at
the final layer only, with a finiteness guard falling back to pure QMC.

**Mini split (subprocess runner): 3.66e-07 vs v3 3.99e-07 (−8.3%);** raw MSE
3.45e-6 (−13.5%), utilization 10.59%, 0/100 failed. Projected public
≈ 3.4e-7 ≈ 1.9× the MC reference.

## Incorporating K=3/K=4 information (round 6, in progress)

K≥3 cannot *run* in-budget (L²-rank arithmetic), but its information enters
three ways, in increasing order of fidelity:

1. **Absorbed into the learned closure** (v4 does this implicitly): the
   ensemble-systematic component of (K=3 − K=2) that is a function of the
   K=2 state gets baked into the 192 calibration constants.
2. **"K=2.5" — Edgeworth-corrected propagation** (training now): track the
   *diagonal* third cumulant exactly through linear layers (κ₃ → (W∘³)ᵀκ₃,
   multilinearity, ~n² per layer — free, no rank explosion), and add the
   analytic Edgeworth correction terms to the mean/variance updates
   (Δμ = −κ₃·α·φ(α)/(6σ²) from E[∂³ReLU], ΔE[z²] = κ₃·φ(α)/(3σ)), each with
   an offline-learned scale, plus α-dependent basis corrections to the Wick
   coefficients (12 params/layer). This is a hand-built instance of ARC's
   deduction–projection idea: a bounded sketch of third-order structure with
   a learned projection.
3. **Rank-r factored third cumulant with learned ReLU gains** (future): the
   linear step is exact and cheap for a factor U ∈ R^{n×r} (U ← WᵀU, n²r);
   the expensive exact nonlinear update is replaced by a learned gain — the
   full L²-sketching program. Only worth it if (2) plateaus well above the
   K=3-augmented ceiling (1.8e-6).

Fusion economics (why this matters): with QMC v ≈ 4.2e-6 fixed, driving the
analytic bias² from 2.4e-5 → 1e-5 → 5e-6 moves the fused score −13% → −35%
relative to v3. The K=3-augmented ceiling (1.8e-6) is the theoretical limit
of what third-order information can contribute.

**K=2.5 verdict (measured): 2.352e-5 vs calibrated-K=2's 2.408e-5 — only
−2.3%.** The diagonal-κ₃/Edgeworth channel is nearly empty: the useful K=3
information lives in the *joint* (off-diagonal) third cumulants, exactly as
the earlier marginal-vs-joint analysis predicted. Tier 2 plateaus at
~2.4e-5; closing the gap to the 1.8e-6 ceiling requires the rank-r factored
sketch (tier 3) — a research-project lift, parked.

## v4 public result and the improvement roadmap for this architecture

**v4 public: 3.396e-7** (projected 3.4e-7 — model is well-calibrated),
1.9× ahead of the MC reference; raw MSE 3.12e-6 (−15% vs v3); utilization
10.89% (grader λ·R again ~0.3% above local). **The worst-MLP tail improved
most** (1.18e-6 → 9.67e-7): fusion helps precisely where sampling variance
is highest — which points at the next lever.

Ranked improvements for the fused architecture:

1. **Per-MLP adaptive fusion weight — tested and rejected, with a clean
   explanation.** Naive w_m = v̂_m/(v̂_m + b²_global) is much *worse* than
   global w (val mean 4.10e-6 vs 2.97e-6; worst MLP 2.85e-5 vs 6.9e-6):
   the analytic bias is not constant across MLPs — **it co-scales with the
   sampling variance** (measured corr(log b²_m, log v̂_m) = +0.58, log-log
   slope 0.61, median b²/v̂ ≈ 2.9), because both are driven by the network's
   overall activation scale. Co-scaling ⇒ the optimal MLP-level weight is
   ~constant ⇒ v4's global w is already the right rule, and adaptivity
   over-trusts an equally-degraded analytic estimate exactly on the tail
   MLPs. (A power-law-regularized w_m(v̂) from the slope-0.61 fit is the
   only surviving variant — projected worth ≤ 2–3%; parked.) This completes
   the trilogy: per-neuron adaptivity fails (noise), per-MLP adaptivity
   fails (co-scaling), global offline weights are the sound level.
2. **Lower b² compounds with #1** (all w_m rise). K=2.5 gave only −2.3%;
   next real step is tier 3.
3. **Train θ and w jointly against the fused objective** (θ currently
   minimizes analytic MSE; optimizing through the fusion makes the closure
   specialize in the component QMC lacks).
4. **Refit w/b² on ~200 offline MLPs** (current fit used 24; ~20% noise).
5. FLOPs (all minor, multiplier is linear): K2 costs 0.83% of budget
   (symmetric einsums already); merging the ~640 small fnp calls in the K2
   loop could shave ~0.2% of grader residual wall-time. Not worth it:
   cutting points to duck the floor (β_local ≈ 1.7, measured losing trade),
   fusing non-final layers (unscored).

### v3.1 post-mortem: the floor is not a cliff

The "free fix" (trim to 9.95% utilization) was a sign error, and the public
split confirmed it to four significant figures: 3.689e-7 × 1.011 (MSE of 70
fewer points) × 0.994 (multiplier refund) = 3.708e-7 predicted, 3.709e-7
measured. General lesson: above the floor the multiplier is linear in C and
trimmed-set MSE is ~linear in 1/N (local β ≈ 1), so **the score surface is
flat to first order around the floor** — only the 100%-budget line is a
cliff. Reverted to the v3 configuration; v3 (3.689e-7) stands as best.

## v3 (shipped)

6,400 spherical points (Sobol → Gauss → whiten → project to radius
E‖x‖ = √2·Γ(128.5)/Γ(128); exact radial integration by 1-homogeneity),
`_BUDGET_FRACTION = 0.0991` to fill the floor (~9.95% grader utilization —
the floor is soft, no cliff until 100%, so ~0.6% multiplier jitter is a good
trade for 2% more points). **Mini split: 3.99e-07** (v2: 4.28e-07, −6.8%),
0/100 failed. Expected public ≈ 3.4–3.6e-7 ≈ 1.8× better than the MC
reference.

**v3 public result: 3.689e-7** (v2: 3.783e-7 — only −2.5% vs −6.8% on mini).
Decomposition: grader utilization came in at **10.06%** (the predicted
over-floor: grader residual wall-time is ~4ms/MLP heavier than local), so the
mean multiplier was ~0.1006 → ~0.6% given back; the rest is public-set
variation (raw MSE −2.9% there vs −6.8% on mini). Not a failure — the
over-floor risk priced in advance materialized exactly. **Lesson for v4:
fraction ≈ 0.098 → multiplier pinned at 0.1000, reclaiming the 0.6% for
free.** Now 1.8× ahead of the MC reference; worst-MLP 1.18e-6 and best
5.17e-8 essentially unchanged from v2.
