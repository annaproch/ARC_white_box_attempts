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
