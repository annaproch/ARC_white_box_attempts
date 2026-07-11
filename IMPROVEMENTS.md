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
    AIcrowd (see conversation for the returned score).
