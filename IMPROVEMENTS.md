# ARC WhiteBox Estimation Challenge 2026 — Estimator Improvements & Session Log

*2026-07-11 — research session with Claude Code*

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

1. **Learned quadrature weights** (offline ridge on ~300 MLPs) — in progress.
2. **Learned/optimized point set** (SGD through the same objective, init at
   Sobol-mm) — second-order refinement once weights plateau.
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
