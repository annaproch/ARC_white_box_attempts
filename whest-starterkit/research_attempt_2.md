# Research Attempt 2 — Variance-Reduced Estimation

**Goal:** close the ~60× *raw* final-layer MSE gap to the current leader
(raw ≈ 1.1–1.7e-7, adjusted ≈ 2.9e-8) while staying parked at the 10%
compute-multiplier floor. The gap is entirely raw accuracy — budget strategy
is already solved (everyone sits at ~10% util).

## Where we are vs the target

| | raw final-layer MSE | budget used | adjusted score |
|---|---|---|---|
| MC baseline | ~6.5e-6 | ~10% | 6.47e-7 |
| our v4 closure+QMC (blend) | ~7.8e-6 | ~12% | ~1e-6 |
| **leader** | **~1.3e-7** | ~9–13% | **2.9e-8** |

Reaching 1.3e-7 at ~6.5k sample-equivalents needs ~**100–200×** variance
reduction over plain MC (`MSE ≈ σ²/k`, σ²≈0.18). Plain QMC alone gives ~2–10×.
The rest comes from stacking the three levers below.

## The three levers (+ one)

1. **Antithetic inputs `(x, −x)`.** Free: draw `k/2` base points, evaluate both
   signs. For symmetric `N(0,1)` this cancels the odd part of the integrand.
   The layer-1 matmul `Wx` is even shared (`z` and `−z`). ~2× on its own.

2. **QMC / Sobol points** (`points.npz`, 6400×256, already generated). Same
   number of evaluations, better-placed. ~2–4×. FLOP-free; watch only the
   Python-side residual wall-time of point handling (keep it vectorized).

3. **Analytic control variate — the big lever, and *unbiased*.**
   Build a statistically-linearized surrogate network `ĥ` (each ReLU replaced
   by its optimal affine fit `ĥ_i = c_i + b_i·ẑ_i`, `b_i = Φ(α_i)`). Because `ĥ`
   is affine in `x`, its exact mean `s^(ℓ)` propagates in closed form
   (= the mean-propagation closure). The per-layer estimate is

   ```
   μ̂_cv^(ℓ) = mean_s h^(ℓ)(x_s)  −  β·( mean_s ĥ^(ℓ)(x_s) − s^(ℓ) )
   ```

   Since `s^(ℓ) = E[ĥ^(ℓ)]` *exactly*, this is **unbiased for any surrogate** —
   the closure only needs to *correlate* with `h`, not be accurate. This is
   strictly better than our current **convex blend** `w·mc + (1−w)·closure`,
   which is biased toward the closure. `Var(h − ĥ)` ≪ `Var(h)` because both are
   driven by the same `x`; correlation 0.9+ → 5–20× reduction.

   *(Upgrade path: swap the affine surrogate for the κ₃ closure as the control
   variate — higher correlation, same unbiased machinery.)*

4. **Final-layer Rao-Blackwellization.** Only the final layer is scored. Instead
   of MC-averaging `ReLU(z_L)`, estimate the final pre-activation moments
   `(μ̂_i, σ̂_i)` from samples and apply the **exact** ReLU moment
   `μ_i = μ̂_i Φ(μ̂_i/σ̂_i) + σ̂_i φ(μ̂_i/σ̂_i)`. Removes the ReLU-kink variance
   on the one layer that counts. ~2× there, ~free (one extra matmul).

## Budget accounting (n=256, d=32, B=2.72e11, target 10% = 2.72e10)

One forward pass = `2·d·n² ≈ 4.2M` FLOPs. Surrogate fuses into the batch (stack
real+surrogate rows → one matmul), so sampling cost ≈ `8·k·d·n²` for `k` base
QMC points (2k real antithetic + 2k surrogate rows):

```
8 · k · d · n²  =  2.72e10   →   k ≈ 1620 base points
                                  (2k=3240 real + 3240 surrogate; ≤ 6400 avail)
+ analytic closure (mean-prop)   O(d·n²) ≈ 11M   = 0.04% budget
+ CV combine + final-layer RB    O(k·n)          = negligible
```

**It does not blow the budget** — the levers are variance reduction on the
forward passes we already pay for, plus a ~0.04% one-time analytic pass. The
only ways to blow it (a per-sample *surrogate model* pass, or slow Python
residual) are avoided by fusing the surrogate into the batch matmul and keeping
everything vectorized.

## Prototype plan (`research2_prototype.py`)

Measure the MSE ladder at a **fixed** FLOP budget on one 256×32 MLP, each stage
wrapped in `flops.BudgetContext`, asserting `flops_used ≤ budget`:

```
plain MC  →  +QMC  →  +antithetic  →  +control variate  →  +final-layer RB
```

against a high-sample ground truth (off-budget). Expected: `~2.5e-5 → … →
~1e-6`, i.e. a clear, budget-respecting descent, with the CV step doing most of
the work. Landing at ~1e-6 in the prototype (affine surrogate, β=1, single MLP)
leaves the κ₃-closure-as-CV and adaptive β as the tuning that targets 1e-7.

## MEASURED RESULTS (research2_prototype.py, 12 MLPs, 256×32, 10% budget)

The hypothesis above is **largely falsified**. Measured mean final-layer MSE
(budget held at 92.7% of the 10% floor, all assertions passed — budget is fine):

| variant | mean_MSE | vs plain MC |
|---|---|---|
| closure only (mean-prop) | 1.37e-3 | — (analytic bias) |
| plain MC | 1.11e-5 | 1.00× |
| + QMC points | 5.12e-6 | **2.16×** |
| + antithetic | 5.48e-6 | 2.02× (does **not** stack with QMC) |
| CV (surrogate-pass) | 1.10e-5 | **1.00× — no help** |
| CV (delta, real trajectory) | 5.48e-6 | 2.02× — correction **telescopes to ~0** |
| final-layer RB | (worse) | **0.87× — hurts** (skewed pre-activation) |

**What actually happened, and why:**

- **QMC ≈ 2×, and that's the whole win.** Antithetic doesn't stack on top of it;
  Sobol already captures the low-order structure antithetic exploits.
- **Control variates do not help.** The surrogate-pass CV decorrelates from the
  real network by depth 32 (its own trajectory diverges) *and* burns half the
  samples. The delta CV reuses the real pre-activations but its known mean is
  `W_Lᵀ·μ̂_{L-1}` — the penultimate mean we're estimating — so the final-layer
  correction **telescopes to zero**. The final-layer mean's variance is
  dominated by uncertainty in `E[h_{L-1}]` recursively back to the input; a CV
  can't manufacture that knowledge.
- **Final-layer RB hurts.** It assumes `z_L` is Gaussian; it's skewed, so the
  analytic ReLU moment injects bias. (This is the κ₃ effect — a Gaussian RB is
  the wrong closure; a κ₃ RB might not be.)

**Bottom line:** the naive lever-stack tops out at ~2× (raw ≈ 5e-6, adjusted
≈ 5e-7) — roughly the MC baseline, and **~40× short of the leader** (raw ≈ 1.3e-7).
Budget was never the problem; *variance-reduction quality* is. The leader is
doing something fundamentally stronger than QMC + antithetic + affine-CV.

## Directions #1 and #2 — TESTED, both FAIL (12 MLPs, flopscope-verified)

| variant | mean_MSE | vs MC |
|---|---|---|
| + QMC (reference) | 5.12e-6 | 2.16× |
| **#2 κ₃ final-layer RB** | 5.59e-6 | 1.98× — **no gain over QMC** |
| **#1 dim-reduced QMC** | 6.44e-6 | 1.72× — **worse than QMC** |
| #1+#2 combined | 6.66e-6 | 1.66× |

- **#2 κ₃-RB.** The Edgeworth formula `−(γ/6)μφ(α)` is *correct* (validated: cuts
  the ReLU-moment bias ~10×). But RB never had only a bias problem — the estimate
  inherits the variance of the sampled `μ̂,σ̂,γ̂`, and the noisy `γ̂` cancels the
  gain. Direct averaging was already unbiased, so fixing bias buys nothing.
- **#1 dim-QMC.** The construction is sound (projection onto top direction =
  best Sobol coord), but it rests on a false premise. Diagnostic:
  the linearized map concentrates 80% in one direction, **yet those directions
  explain only ~20% of the *actual nonlinear* final-activation variance; the top
  32 cap at 24.4%, and empirical cross-cov directions < 3%.** So **~76% of the
  final-layer variance is genuinely high-dimensional and nonlinear** — there is
  no low-dim subspace to align QMC to.

## The real conclusion (7 techniques tested)

**Generic variance reduction caps at ~2× on this problem** (QMC), because the
final-layer mean is an intrinsically high-dimensional, high-order-nonlinear
integral — no low effective dimension, no low-order ANOVA structure, no
correlated-with-known-mean control variate. Antithetic, control variates
(surrogate + delta), Gaussian RB, κ₃ RB, and dimension-reduced QMC all add
nothing beyond QMC's 2×. The leader's ~40× is **not** in the sampling toolkit.

**The leverage is on the analytic-closure side, not the sampling side.** A strong
cumulant closure (κ₃/κ₄) is what can reach ~4e-7 (your own "oracle-slice"
estimate) and, refined, the leader's ~1.3e-7. This is exactly the `learn_cumulants`
direction you are already on — the prototype's value is negative but clarifying:
**stop tuning sampling tricks; invest in closure accuracy (the κ₃/κ₄ slice).**

## Dead / low-firing units — TESTED, no lever (analytic)

Observation: ~42% of final-layer units are low-firing (truth < 1e-2; 31% never
fire), so predicting them as ~0 seemed like free score. **It isn't.** (6 MLPs,
GT N=1e6, cov_prop):

- Low units are **0.1% of the total squared error** (MSE 1.03e-4).
- Oracle-perfect prediction of every low unit: **0.1% MSE improvement**.
- Best clamp-to-0 threshold: **0.0% improvement**. The closure already predicts
  them at 6.5e-4 vs truth 5.4e-4 (corr 0.97); their α=m/σ ≈ −5.2 (deep tail).

Why: MSE weights **absolute** error. A truth≈0 unit contributes ≤ pred² ≈ 1e-6;
an active unit (truth ~0.7) at 1% error contributes ~5e-5. The **58% active
units carry 99.9% of the error.**

**Sharpened target:** the error is entirely in the units that fire **~half the
time (α ≈ 0)** — where E[ReLU]=μΦ(α)+σφ(α) is most sensitive to distribution
*shape*. The κ₃ correction `−(γ/6)μφ(α)` **vanishes at α=0**, so the bulk units
need the **κ₄** term, not κ₃. Spend the cumulant budget on the half-firing units,
and prioritize the fourth cumulant there.

## Still-plausible sampling angle (single, narrow)

- **Cheaper per-sample forward → more samples.** Low-rank/sketched matmuls cost
  `O(n·r)`; at `r≪n`, many× more samples per budget. This attacks the *variance*
  head-on (more N) rather than the integrand's structure, so it's not subject to
  the high-dimensionality wall above — but it trades in matmul-approximation bias.
  Untested; the one sampling idea the diagnostics don't rule out.

## Next steps after the prototype

- Run on the mini split (100 MLPs) via `whest run`, compare adjusted score.
- Adaptive per-neuron `β_i = Cov(h_i,ĥ_i)/Var(ĥ_i)` (small bias, more reduction).
- Use the κ₃ closure (`closure.py`) as the control-variate surrogate.
- Fold the unbiased CV into the production estimator, replacing the convex blend.

---

# Research Attempt 2b — Analytic ladder, MEASURED 2026-07-18 (validate-yourself pass)

6 MLPs, 256×32, GT N=2e6 (floor ~4e-8). Mean **final-layer** MSE:

| method | MSE | note |
|---|---|---|
| M1 gain-cov (starterkit 03) | 1.05e-4 | baseline |
| M2 exact-bivariate cov | 1.03e-4 | **exact κ2 off-diag is NOT the bottleneck** |
| M3 + diagonal-κ3 Edgeworth | 9.5e-5 | diagonal slice worth only ~8% |
| M4 mixture-GH(9) × gain-cov | 6.6e-5 | conditioning on top linearized dir: **1.6×, cheap (≈5% budget), stacks** |
| D: Gaussian map + TRUE final (m,s) | 1.5e-6 | 70× below all propagation methods |
| D: + TRUE final κ3 | **1.78e-7** | **≤ leader-best bar** |

Bar (analytic, <1% budget ⇒ 0.1 multiplier): raw ≤ **2.9e-7** ties leader mean; ≤1.6e-7 ties best.

**Decomposition conclusion (the key fact):**
- ~99% of closure error is in the **propagated marginal moments (m_i, s_i, κ3_i)** of the
  final pre-activation — NOT in the final ReLU map.
- The final map itself is solved: Gaussian+κ3 Edgeworth with true moments hits 1.78e-7.
  Gaussian-only with true moments (1.5e-6) is NOT enough → κ3 must be carried at least
  into the final map.
- Required propagation accuracy for the win: per-unit |Δm| ≈ 4e-4 (since ∂map/∂m ≈ Φ ≈ 0.7).
  Current closures deliver ~1e-2 per-unit. Factor ~25 needed, entirely in marginal propagation.
- Since exact-κ2 (M2) and diagonal-κ3 (M3) don't close it, the missing terms are the
  **off-diagonal κ3/κ4 slices** feeding the (m,s) recursion — independently confirming the
  closure.py focus, now by measurement.
- Caveat: D shares samples between truth and moments, so sampling noise partially cancels;
  D is a shape-gap measure, ballpark reliable.

**Offline-precompute lever (rules allow shipped constants):** the MLP generator is public
(He-Gaussian 256×32) ⇒ unlimited synthetic training networks with near-exact GT offline;
fitted θ/anchor and precomputed 2-D kernel tables (e.g. small-ρ expansion kernels of
bivariate/trivariate rectified-Gaussian moments as functions of (α_i, α_j)) ship at zero
predict-time FLOPs.

## Mixture-conditioning experiment (2c) — MEASURED, 8 MLPs, GT N=2e6

| variant | mean_MSE | vs base | %budget | mult |
|---|---|---|---|---|
| base gain-cov | 9.19e-5 | 1.00× | 0.8% | 0.10 |
| mix K=3, 1 dir | 5.94e-5 | **1.55×** | 3.2% | 0.10 |
| mix K=9 / K=17 | 5.94e-5 | 1.55× | 7.9 / 14% | — |
| mix K=9 × exact-cov | 5.91e-5 | 1.56× | 22% | 0.22 ❌ |
| mix K=9 × diag-κ3 | 5.98e-5 | 1.54× | 7.9% | 0.10 |
| **mix 5×5, 2 dirs** | 4.92e-5 | **1.87×** | 20.5% | 0.21 ❌ as-is |
| CHEAP shared-σ | 5.09e-4 | **0.18× — fails** | 1.6% | 0.10 |
| no-gain direction | 8.93e-5 | 1.03× — no gain | 7.9% | 0.10 |

Findings:
- **Gain saturates completely at K=3** (K=3=9=17 to 3 digits): the integrand along the
  conditioned direction is ~cubic; 3-node GH suffices. → cheapest config wins.
- **2nd direction adds real gain** (1.55→1.87×). Not directly tested but implied by
  saturation: a 3×3 grid (9 nodes + base = 7.9% budget) should retain 1.87× at the 0.1 floor.
- **Direction choice is critical**: gain-weighted linearized top singular vector works;
  unweighted product direction gives NOTHING (1.03×). We have the right direction.
- **Exact-κ2 / diag-κ3 add nothing inside the mixture** (all ≈1.55×) — consistent with
  their standalone smallness. Overlap with the full off-diag-κ3 closure is UNTESTED.
- **Shared-σ cheap trick FAILS (5× worse than base)**: freezing σ from the base run is
  invalid — mean shifts along v change downstream variances substantially. The plausible
  fix (untested): per-node diagonal-variance mean-prop (~8M FLOPs/node, still ~free).

Verdict: mixture is a real, shippable ~1.55–1.87× at the 0.1 multiplier floor
(K=3 one-dir at 3.2%, or 3×3 two-dir at 7.9%). It is a cheap multiplier on top of the
main work — NOT a path to the 2.9e-7 bar by itself (still ~170× away).
