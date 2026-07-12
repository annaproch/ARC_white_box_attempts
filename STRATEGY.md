# STRATEGY — ARC WhiteBox Challenge (source of truth)

*Started 2026-07-12. This document supersedes IMPROVEMENTS.md as the working
strategy. IMPROVEMENTS.md remains as the historical session log, but per the
audit below its conclusions are NOT to be relied on unless re-verified.
Append to the progress log at the bottom; never delete audit verdicts.*

---

## 0. Rules of evidence (adopted after the audit)

The old log drew strong conclusions from 3–8 MLPs while per-MLP final-layer
MSE spreads ~25×. Mean-over-6-MLPs has ~50% SEM. Half the "findings" are
noise. From now on:

1. **One canonical benchmark.** Headline numbers: the mini split (100 MLPs,
   N=1e9 baked GT). Iteration: one FIXED set of ≥48 local MLPs with ≥8e6-sample
   GT, never regenerated. Every number cited must name its benchmark and n.
2. **Paired comparisons only.** Same MLPs, same GT, report mean ± SEM *of the
   paired difference* and the per-MLP win rate. Differences < 2×SEM don't exist.
3. **No conclusion from < 24 MLPs.** Period. (The oracle test used 6; the
   drift table used 3.)
4. **Every derivation is machine-verified before use** — sympy or brute-force
   numeric checks at small n (n ≤ 8, where exact Gaussian expectations are
   computable by quadrature/1e8-sample MC). The log documents a mis-derived
   symmetrization, a misread `permute`, a 2.7× FLOP miscount, and a stale-file
   submission. Hand tensor algebra without a mechanical check is banned.
5. **FLOPs are measured (flopscope), never estimated. Residual wall-time is
   measured on the GRADER** (probe submission) before any fusion-economics
   decision. Grader residual was ~6× local for the closure.
6. **Artifact hygiene:** hash-log every exported .npz; delete artifacts on
   revert (a stale closure.npz shipped once).
7. **Public-split deltas < ~10% are noise for ranking purposes.** The private
   50 is a different draw; per-MLP score sd ≈ 0.7–1× the mean, so two
   submissions 3% apart on public are a coin flip on private. **Target
   margin: ≥ 1.5× over the current leader** (see §6.4).

---

## 1. Facts we trust (leaderboard-validated or pure arithmetic)

| fact | evidence |
|---|---|
| Score = final-layer MSE × max(0.1, C/B), B=2.72e11, C = flopscope FLOPs + 1e11·residual-sec; over-budget ⇒ zeros ×1.0 | starter-kit docs; v3.1 trim experiment predicted public score to 4 significant digits |
| Plain MC at the floor scores 6.5e-7 (MSE 6.5e-6) | our v1 = 6.526e-7 ≈ organizers' reference 6.47e-7 |
| Our v2/v3/v4 public: 3.783e-7 / 3.689e-7 / 3.396e-7 | leaderboard |
| Pure-analytic closure public: raw MSE 1.09e-5 (offline predicted 1.05–1.19e-5) | offline→public transfer of analytic methods is accurate |
| Leaders (2026-07-12): andrew_epstein 9.158e-8, pluto 9.268e-8, shiv_m 9.440e-8 — all ≈ 7× the MC reference | dashboards (pasted) |
| For any MSE ∝ 1/C method the score is flat in C above the floor | arithmetic |
| Antithetic ≈ no gain; no linear/odd input-dependence survives depth 32 | v1 public + local; kills *linear* CVs only |
| Grader wall-time inside flopscope kernels is effectively free (pluto: 20–42 s/MLP, residual charge ≈ 30 ms) and per-MLP wall caps are ≥ ~43 s | pluto's ledger |
| Off-budget levers are legal: `setup()` + shipped .npz ≤ 50 MiB; offline compute unlimited; `setup()` never sees the MLP (no per-MLP precompute) | starter-kit docs |
| Master constraint: a method costing C must satisfy MSE ≤ 6.5e-6 · (0.1B/C) just to tie MC; to beat the leader, MSE ≤ 9.2e-7 at the floor (or pro-rata above it) | arithmetic |

**The whole game in one line: score ≈ 6.5e-7 / VR, where VR is the
variance-reduction factor vs plain MC at your own compute. Leaders: VR ≈ 7.
Us: VR ≈ 1.9. Operating point and intermediate layers are second-order.**

---

## 2. Audit of IMPROVEMENTS.md (claim → verdict → action)

### 2.1 REJECTED — do not build on these

- **R7 pivot table ("rival 10× better at same compute; K=3-class; they
  propagate third-order information").** The comparison used the rival's BEST
  per-MLP score (bruce-mckenzie, 3.24e-8) against OUR MEAN (3.4e-7) — a
  cherry-pick. Real mean-vs-mean gap: 3.7×. "K=3-class" was numerology on top
  of it. The pivot's *direction* (analytic matters) may still be right, but
  its evidentiary basis was wrong.
- **The κ4 failure diagnoses.** Three mutually inconsistent stories in the
  log: (i) "missing κ3-mixed diagrams" (R7 end), (ii) "chain diagram never
  implemented" (R8), (iii) "chain diagram tested, barely helps: 2.54→2.52"
  (Phase-7 table). At most one is true. Verdict: κ4-propagation-as-built is
  harmful (that measurement is consistent across configs and trusted), but
  the *reason* is unknown. Action: machine-verified re-derivation or
  corpus-supervised learning — no more hand diagram algebra.
- **"Rank compression degrades, not improves" (early R5)** — asserted, never
  measured, already self-retracted once. Strike entirely.
- **"Sobol+mm ≈ the fixed-quadrature optimum."** What was shown: ridge weights
  collapse to uniform and *SGD* on points is noise-bound. That proves SGD
  can't improve it, not that the optimum is reached. Low priority anyway
  (β<1 caps the upside), but don't quote it as a theorem.

### 2.2 RETEST — load-bearing but unsafe

- **The oracle "self-consistent fixed point" (R7's centerpiece: true-Σ
  injection is WORSE than K=2; "mean path is the priority; partial fixes
  harmful").** Problems: (a) n=6 MLPs; (b) its K=2 baseline is 1.30e-4 while
  every other measurement of plain K=2 is 4.7–7.2e-5 — the harness itself is
  suspect; (c) it contradicts the error-attribution finding ("98% of error is
  propagated moments, *mostly variance*, σ 4× worse than μ") unless carefully
  reconciled; (d) the 2×2 is incomplete (true-μ-only was never run).
  **Action → E2 (clean 2×2), the single most important retest.** If the clean
  run shows K2 + true κ3(+κ4 slices) ≤ ~1.5e-6, the pure-analytic route to
  9e-7 is real (and pluto plausibly lives there). If it reproduces the
  cancellation, the mean-update becomes the target. Either outcome redirects
  everything.
- **Plain-K2 baseline drift (4.7e-5 → 1.30e-4 across five experiments).**
  Either multiple inconsistent implementations or seed noise; both
  disqualify cross-table comparisons. Action: one canonical K2, one canonical
  benchmark, re-baseline everything in one table (E0).
- **Single-step unit test (κ3 → κ4 corrections give 100–1300× per-step; "κ4
  worth 145× after κ3").** Probably right (clean design), but 8 MLPs and it
  feeds the κ4 decision. Rerun under E0/E2 harness. Also quantify the
  sampling noise of the corpus κ3/κ4 slice targets before supervising on them.
- **Local-vs-public QMC inflation.** QMC measured 3.17× locally but delivered
  1.72× on public (v1→v2). Cause never found. Any locally-measured VR must be
  ×0.6 until shown otherwise, or measured on mini directly. One diagnostic
  run: same point set, mini-split GT vs local GT.
- **"Fluctuations are low-dim / eig1frac 0.55–0.8"** — measured on 3 MLPs.
  This number sizes every conditional/mixture closure and every sketch-CV.
  Re-measure on ≥24 (E3a).

### 2.3 TRUSTED (scope-limited)

- Wick series converged by k=4; truncation was never the K2 bottleneck.
- Estimator-level fusion no-gos: per-neuron (noise), per-MLP (bias co-scales
  with variance, corr 0.58). **Scope: estimator-level averaging only — says
  nothing about per-sample control variates or multilevel estimators.**
- β ≈ 0.83 for OUR QMC family ⇒ floor-optimal *for us*. Not a law for other
  methods (andrew operates at 26% happily; flat frontier makes it free).
- K=2-augmented unstable at depth 32; K≥3 textbook kprop over budget
  (Table 1 arithmetic: 30n³L² = 190% of B). These are solid.
- End-to-end calibration of K2: 7.2e-5 → 2.4e-5 held-out (48 MLPs, decent n).
- Depth-telescoping no-go: the Brownian Corr²≈j/L argument is hand-wavy but
  the conclusion is plausible. **Scope: telescoping in DEPTH. It does not
  cover multilevel-in-WIDTH (cheap sketch surrogate + few full passes) —
  that door is open and untested (E4).**

---

## 3. First-principles picture (what actually determines VR)

Write the fixed-network output as f_i(x) over x~N(0,I). Per-MLP quenched
variance v_m = mean_i Var_x f_i. MC at N points: MSE = v_m/N. Every mechanism
is one of:

1. **Better integration of the same f** (QMC/quadrature): bounded by how much
   of f's variance lies in "smooth" (low effective-dimension / low
   spherical-harmonic degree) components. Our measured ceiling so far: ~1.9×.
2. **Per-sample surrogate** (control variate / multilevel): VR = 1/(1−ρ²·adj)
   where ρ = per-sample correlation of a cheap surrogate g with f, and adj
   accounts for g's cost and its own mean-estimation. Linear g is dead
   (no odd signal). Nonlinear cheap g (rank-r sketch of the layer maps,
   amplitude trackers) — never tested (E4).
3. **Analytic bias reduction** (closures): pure-analytic needs MSE ≤ 9.2e-7
   outright; fusion needs b² ≈ v to matter. Current closure: 9.6e-6.
4. **Offline-learned corrections** (ensemble-systematic error is shippable):
   calibration bought 3× on K2; ceiling measured at ~44% of K2's error being
   ensemble-systematic *post-hoc* (more end-to-end).

**Theory tool (unverified — verify in E1):** for the depth-L arccos-kernel
composite ρ_L = φ∘…∘φ, the fraction of quenched-ensemble output power at
spherical-harmonic degree d equals the d-th Taylor coefficient of ρ_L at 0.
Quick derivation gives: degree-1 share ~4% at L=32 (odd signal dead ✓,
consistent with antithetic), degrees 1+2 ≈ 5% ⇒ **polynomial-in-x CVs and
moment-matched points cap at ~1.05× VR**; the variance mass sits at degrees
d ≳ L²/44 ≈ 23 (α_d ~ d^(−5/2) tail). If E1 confirms this, it explains why
QMC's edge exhausts (β<1) and *rules out* whole families cheaply. Caveat:
this is the ENSEMBLE (annealed) spectrum; a FIXED network's fluctuation may
concentrate differently (quenched) — E3a measures that side.

---

## 4. Leader forensics (from the pasted dashboards, 2026-07-12)

| | andrew_epstein (#1) | pluto (#2) | shiv_m (#3) |
|---|---|---|---|
| public score | 9.158e-8 | 9.268e-8 | 9.440e-8 |
| final-layer MSE | 3.55e-7 | 9.27e-7 | 8.16e-7 |
| budget used | 26.34% (bimodal: 43 MLPs ~5.8e10, 7 MLPs ~9.5e10) | 9.57% (floor) | 11.59% |
| all-layers MSE | 2.82e-6 (all layers good) | 0.7537 (zeros for L0–L30) | 0.7537 (zeros for L0–L30) |
| wall/MLP | 4.3–13.5 s | 20–43 s | 26–40 s |
| per-layer profile | error GROWS L0→L30 (3.8e-7→4.7e-6), then L31 drops 18× (2.6e-7) | n/a | n/a |

Deductions (confidence-tagged):

- **These are (at least) two different methods** (user's observation,
  confirmed): pluto/shiv predict ONLY L31; andrew predicts everything.
  pluto and shiv share the exact all-layers constant 0.7537 (both emit the
  same placeholder) and similar wall profiles — plausibly the same family.
  But per-MLP they differ up to 5× (kimberly-henderson: pluto 2.88e-7,
  shiv 6.03e-8) — different failure modes.
- **pluto = pure analytic at the floor (high confidence).** 2.3e10 FLOPs,
  final-only, 40 s of slow elementwise-heavy kernels, MSE 9.3e-7.
  **Existence proof: a ~2.3e10-FLOP sample-free method reaches 9.3e-7 —
  i.e., someone has solved (to VR≈7) what ARC calls an open problem.**
  Cost decodes: 2.3e10 ≈ 40n³L ≈ Q≈10 conditional-Gaussian (mixture)
  covariance propagations — or ≈ our own κ3-closure at rank ~56 (measured
  2.7e10 at r=64). Two concrete architecture guesses, both testable offline.
- **andrew = hybrid (medium confidence).** Intermediate errors grow with
  depth — NOT a plain-sampling signature at first glance, BUT: per-layer
  sampling MSE ∝ Var_l/N would *fall* if Var_l falls, and his L0 (3.8e-7) is
  ~100× better than 13k-sample MC at L0 — so intermediates are not plain MC
  either. Consistent decode: QMC-quality points (early layers nearly exact,
  edge decaying with depth-roughness) + a final-layer-only mechanism worth
  ~5× (L30 4.7e-6 → L31 2.6e-7), + a fallback heavy-sampling path on 7/50
  MLPs which SAVED him on MLPs where pluto/shiv blew up (joshua-aguilar:
  andrew 1.07e-7 vs pluto 2.72e-7 / shiv 3.17e-7). Get our own v4 per-layer
  ledger to calibrate what QMC-at-N profiles look like (E5a).
- **Shared difficulty axis:** all three leaders' best/worst MLPs overlap
  heavily (bruce-mckenzie/angela-brown easy; justin-johnston/joshua-aguilar/
  sheila-palmer hard) — per-MLP difficulty is intrinsic (likely amplitude/
  variance), not method-specific. Robustness on ~5 tail MLPs moves the mean
  as much as uniform gains.
- The top-3 spread (3%) is far inside private-split noise — they are
  effectively tied. Beating them *robustly* needs ≥1.5×.

### 4.1 E5a RESULTS (2026-07-12, paired per-MLP analysis on the named 50)

Our pure-analytic closure ledger (1.094e-6 submission) joined against all
three leaders. Script: scratchpad `e5a_ledger_analysis.py`. Findings:

1. **Per-MLP errors are mostly method-idiosyncratic.** All pairwise Spearman
   correlations of per-MLP final MSE are only 0.20–0.37 (log-Pearson ≤ 0.43)
   — including pluto vs shiv (0.365), despite their identical L0–L30
   placeholder. The earlier "shared difficulty axis" claim (§4 above) was an
   eyeball overread of tail overlap: there IS a weak common axis (consensus-
   hard trio justin-johnston / sheila-palmer / joshua-aguilar, mean rank
   ≥ 44/50 for everyone), but most of each method's error is its own.
   **Consequence: no intrinsic per-MLP floor near 1e-6, and ensembling two
   decorrelated analytic closures is worth up to ~1.6–1.8×.**
2. **pluto is NOT simply "our closure done right."** ours/pluto per-MLP ratio:
   geo-mean 12.1×, but gsd 1.71× with range 3.2× (kimberly-henderson) to
   36.7× (monica-brown) — different error structure, not a constant-factor
   gap. The κ3-rank-56 decode of pluto's cost stays plausible but weaker.
3. **andrew's edge = better primary (~3.7× geometric over pluto/shiv) PLUS a
   router.** On his 7 fallback MLPs (heavy-sampling path) he scores a normal
   1.8–3.3e-7 while pluto/shiv/we blow up on most of them (joshua-aguilar:
   andrew 3.1e-7 vs pluto 2.7e-6, shiv 2.7e-6, ours 2.9e-5). The fallback set
   overlaps 5 of our closure's 9 worst MLPs. **A per-MLP analytic-health
   check that reroutes budget to sampling is measured to be worth ~8× on
   catastrophe MLPs.** His router is imperfect — it missed his own worst 3
   (justin-johnston 1.28e-6, maria-mcintosh, chad-yates, all standard-path),
   so there is headroom over andrew via better routing.
4. **Tail arithmetic at the top:** pluto's 5 worst MLPs contribute ~25% of his
   mean; with andrew's router he'd score ~7.5e-8 and take #1. Tail robustness
   is the cheapest known lever at the frontier — and it composes with any
   primary method.

### 4.2 E5b RESULTS — v4's ledger, and the RETRACTION of the router idea

v4 is QMC-dominated ⇒ its per-MLP MSE is a proxy for intrinsic quenched
variance v_m. Script: scratchpad `e5b_v4_axis.py`. This **retracts item 3 of
§4.1 and build item B-R**.

**Retraction.** §4.1 claimed andrew's fallback path was "worth ~8× on
catastrophe MLPs." That compared his fallback MLPs against *other people's*
analytic methods — the wrong counterfactual. Correct test = oracle per-MLP
routing between each analytic method and v4-sampling:

| method | analytic alone | + ORACLE routing to sampling | MLPs where sampling wins |
|---|---|---|---|
| andrew | 3.54e-7 | 3.54e-7 (**+0%**) | 1/50 |
| pluto | 9.27e-7 | 9.04e-7 (+2.5%) | 4/50 |
| shiv | 8.16e-7 | 8.06e-7 (+1.2%) | 2/50 |
| ours | 1.09e-5 | 3.10e-6 | **49/50** |

Routing buys ~nothing for the leaders because their analytic already beats
sampling almost everywhere. Oracle inverse-variance FUSION is also only
1.2–1.4× (pluto 6.5e-7, shiv 5.9e-7, andrew 2.9e-7) — and the extra compute
eats most of it (this is v4's own already-measured exchange-rate problem).
**B-R demoted: keep a cheap divergence guard as a safety net, not a lever.**

**The variance axis (the real finding).** Correlation of each method's
per-MLP error with v_m (Spearman): **ours 0.465, pluto 0.371, shiv 0.348,
andrew 0.028.** So (a) our closure's error co-scales with variance (the old
corr-0.58 claim is directionally right for us); (b) **andrew's error is
INDEPENDENT of intrinsic variance** — his method is not variance-limited at
all, and is structurally different from pluto/shiv, which do inherit the
axis; (c) sampling error rises with v_m while good-analytic error doesn't, so
**high-variance MLPs are EASY for a good closure** — e.g. bruce-mckenzie is
v4's WORST (7.63e-6) yet pluto's BEST (3.24e-7) and andrew's 2nd-best.

**⇒ THE WHOLE GAP IS THE PRIMARY CLOSURE.** Our closure loses to plain
sampling on 49/50 MLPs; pluto's beats sampling on 46/50 at 2× our FLOP spend
(2.3e10 vs 1.14e10) and is **11.8× more accurate**. No routing, fusion
weight, point set, or operating point closes 11.8× in the primary estimator.
**Single objective: closure MSE 1.09e-5 → ≤9e-7. Everything else is noise.**
We even have FLOP headroom (pluto spends 2× what we do; kernel wall-time is
free). E2 is therefore THE experiment; E1/E3/E4 serve it; E5 is closed.

**Still open (cheap):** does the public-50 name set match the published
mini-100? If so, per-MLP forensics gets much sharper.

---

## 4.3 THE ORACLE, PROPERLY POSED (design + the reframing number)

### Where error can possibly come from

Per layer there are exactly two steps. **Linear (`z = Wᵀa`): EXACT** — cumulants
transform by multilinearity, no approximation. **Nonlinear (`a' = ReLU(z)`):
the only source of error**, and it has two independent parts:

- **Representation error** — we carry (μ, Σ); the truth has κ₃, κ₄, …. We
  cannot represent the true distribution at all.
- **Formula error** — given what we carry, is the update right? (Wick series
  converges by k=4 ⇒ **this is not our problem**; that result is trusted.)

Both then compound over 32 layers.

### Why the old oracle was broken

It injected **true Σ with propagated (wrong) μ** — a state whose mean and
covariance come from different worlds. The ReLU update is *jointly* nonlinear
in (μ,Σ), so a mismatched pair can easily be worse than a self-consistent but
biased one. That is almost certainly the entire content of the "true Σ is
WORSE than K=2" paradox, which was then promoted to physics ("K=2 is a
self-consistent fixed point; partial fixes are actively harmful") and used to
steer the whole program. **Treat as an artifact of a badly-posed experiment
until E2 says otherwise.**

### The right design: telescoping error decomposition

Let `S*_l` = the **fully consistent** true state at layer l (every component
from sampled truth). Define `mu_hat(l)` = start from `S*_l`, run OUR closure
for the remaining layers. Then `mu_hat(L) = mu*` exactly, `mu_hat(0)` is our
real prediction, and

    total error = mu_hat(0) − mu*  =  Σ_l [ mu_hat(l) − mu_hat(l+1) ]   (exact)

Each `e_l` is *the error injected by layer l+1's update, amplified by every
downstream layer*. No inconsistent state is ever formed. Gives a per-layer
error budget AND the downstream amplification factor — i.e. it says **where**
the error is born and **how much the tail multiplies it**.

Run it at each representation level: state = (μ,Σ) / (μ,Σ,κ₃) / (μ,Σ,κ₃,κ₄).
**The level at which the final error reaches ~1e-6 is exactly what we must
carry.** That is the decision E2 exists to make.

### The reframing number: the K=2 representation floor

Take the **true** (μ,Σ) at the last hidden layer, do the Gaussian ReLU
integral. That is the best ANY K=2-class method can ever do. The old (suspect)
harness measured it two independent ways — oracle **1.06e-6** and direct
"Gaussian shape error" **1.0e-6** — which agree.

**Pluto scores 9.27e-7.**

⇒ **Leading hypothesis: pluto is a near-exact (μ,Σ) propagator sitting on the
K=2 floor.** He may be doing nothing exotic with higher cumulants at all.

| | final-layer MSE |
|---|---|
| plain K=2 | 1.3e-4 |
| calibrated K=2 | 2.4e-5 |
| + κ₃ (rank 32–48) | 9.6e-6 |
| **K=2 oracle floor** | **~1.1e-6** |
| pluto | 9.3e-7 |

**There is ~9× of headroom INSIDE the K=2 representation.** If E2 confirms the
floor, we have a well-posed engineering target — propagate μ and Σ accurately
— and NO open research problem to solve. This is the best news in the project.

## 4.4 A LIKELY BUG THAT REOPENS THE FINAL-LAYER SHAPE CORRECTION

Two logged claims cannot both be true: (a) "skew(z₃₁) = ±0.02 ⇒ z is Gaussian
⇒ Edgeworth on the marginal is worthless"; (b) "the top eigenmode of a carries
50–83% of the variance with skew 0.3–0.5".

Derivation. With `v₁` the top eigenmode (skew γ₁) and
`h_i = (w_i·v₁)/√(2/n) ~ N(0,1)` iid across neurons:

    skew(z_i) = γ₁ · eig1frac^{3/2} · h_i³

**Mean over neurons is EXACTLY ZERO** (E[h³]=0, sign symmetry of the random
weights). RMS is `γ₁·eig1frac^{3/2}·√15 ≈ 2.3·γ₁` (≈0.9 at γ₁=0.4), with a
heavy tail (median only ~0.07). So "mean skew ≈ 0.02" is **precisely what you
measure if you average a symmetric ± quantity over neurons** — a mean-vs-RMS
bug. The verdict "z is Gaussian, Edgeworth is dead" is unsupported, and it is
load-bearing (it closed the door on final-layer shape correction).

**Honest caveat.** Pushing further: `Δμ_i = −γ_i·σ·t_i·φ(t_i)/6`, and IF
`v₁ ∥ μ_a` then the same `h_i` sets both `t_i ≈ 4.66·h_i` and `γ_i ∝ h_i³`, so
skew and kink-proximity ANTI-correlate (neurons near the kink, where shape
matters, have tiny skew) and my integral gives a shape error ~2e-9, **not**
1e-6. So either the measured 1e-6 is wrong, or v₁ is not aligned with μ_a and
several non-Gaussian modes matter. Cheap to measure; do not argue it.
**Side-check S1: RMS (not mean) per-neuron skew of z at each layer, plus
corr(γ_i, t_i) and the alignment ⟨v₁, μ_a/‖μ_a‖⟩.** One hour, reopens or
permanently closes the shape correction.

Also: **why "K=2.5" only bought 2.3% is now explained** — it propagated the
DIAGONAL κ₃ via `(W∘³)ᵀκ₃`, which is valid only if a's components are
independent. They are strongly correlated (that is the whole problem), so its
κ₃(z) was garbage. That experiment tested nothing; it is not evidence against
third-order information.

## 4.5 ROUTES TO THE CLOSURE (what we build after E2)

**Route A (primary bet) — near-exact (μ,Σ) propagation via a LEARNED update.**
The Σ-update error per layer IS the non-Gaussianity of z, and it compounds
(1%/layer → 1.37× over 32 layers; λ₁ measured 27–43% low). Correcting it needs
contracted κ₃(z) slices. Our propagated κ₃ is 6.7% off and rank-limited with
brutal diminishing returns (r=32→7.7%, r=96→4.7%, cost O(n·r³)). So **do not
propagate κ₃ — LEARN the correction**: fit (local K=2 state) → (true μ,Σ
update), supervised against the corpus at EVERY layer (32×65k targets/MLP vs
the 256 numbers we have been backpropagating through 32 layers). Weight
exchangeability ⇒ the map is universal. Unlimited offline compute, n² at
inference, and **it cannot fail silently** (fit against sampled truth). This is
the turbulence-modelling playbook and my primary bet.

**Route B — scale-mixture closure.** `z | s ~ N(μ(s), Σ(s))` with a scalar
latent s on a Q-node quadrature. Conditional on s everything is Gaussian ⇒
every ReLU expectation exact; skew and excess kurtosis come free, no tensor.
Cost `Q·4n³L ≈ 2.1e10` at Q=10 — matches **pluto's 2.3e10 and his 40 s wall
time**. The old rank-1 "core" attempt (1.06×) was 3 MLPs with a fragile
particle scheme and a diagnosis in the RETEST bucket; not trusted.

**Route C — final-layer shape correction.** Cheap, additive to A/B, needs only
the n-vector κ₃(z_L)_iii. Gated entirely by side-check S1.

## 4.6 E2 RESULTS (2026-07-12) — measured, 24 MLPs, canonical harness

`research/harness.py` (self-tests: linear step exact to float32 eps; Hermite
gains verified to 3e-8 against their integral definition; layer-0 corpus
deviation exactly = MC noise) and `research/e2_oracle.py`. Injected-state noise
perturbation changes every number by ≤2% ⇒ **not artefacts of finite-sample
truth.**

### The ladder (final-layer MSE, mean ± SEM over 24 MLPs)

| | MSE | note |
|---|---|---|
| plain K=2 (CANONICAL) | **9.54e-5 ± 1.1e-5** | settles the 2.8× baseline drift |
| K=2 + **oracle κ₃** every layer | **9.20e-6 ± 6.7e-7** | 10.4× |
| *our existing κ₃ closure (prior measurement)* | *9.6e-6* | **= THE κ₃ CEILING** |
| K=2 + **oracle κ₃+κ₄** every layer | **1.26e-6 ± 1.2e-7** | 76× |
| K=2 representation floor (1 step from truth) | **1.205e-6 ± 1.3e-7** | old 1.06e-6 replicates |
| **pluto** | 9.3e-7 | **0.77× the K=2 floor** |
| target to beat the leaders | ≤5e-7 | |

### Four verdicts

1. **My "pluto sits on the K=2 floor" hypothesis is REFUTED.** He is *below*
   it (0.77×). Pluto is **not** a K=2 propagator — he carries genuinely
   non-Gaussian information. Good: the floor is not a wall.
2. **κ₃ FIDELITY IS A DEAD END.** Oracle-κ₃ = 9.20e-6; our shipped κ₃ closure
   already measures 9.6e-6 — *statistically at the ceiling*. The 6.7% κ₃
   error costs essentially nothing. **Rank, better κ₃ propagation, star/core
   fidelity: all worthless. Stop touching them.** (This retro-explains the
   plateau at ~1e-5 in *every* configuration — it was the κ₃ ceiling, not an
   optimisation failure.)
3. **κ₄ IS THE ENTIRE REMAINING PRIZE: worth exactly 7.3×** (9.20e-6 →
   1.26e-6). That lands at score ≈1.26e-7 — 2.7× better than v4, and within
   1.4× of the leaders. It is the only lever with a measured payoff.
4. **Error is injected EARLY.** Per-layer injection from the restart curve:
   ~8e-6/layer at l=2–4, ~6e-6 at l=4–8, decaying to ~1.5e-6/layer at depth.
   Layer 0 injects almost nothing (input is exactly Gaussian). Amplification
   ≈ 1 (errors add incoherently; final MSE ≈ Σ_l one-step injections).
   **⇒ any correction must work at layers 2–8, where eig1frac is only
   0.01–0.11.** This is the real objection to a rank-1 mixture closure, and it
   survives (unlike the old 3-MLP version of it).

### S1 CONFIRMED — the skew bug was real

| layer | mean skew | **RMS skew** | \|corr(γ,t)\| | ⟨v₁,μ_a⟩ | eig1frac |
|---|---|---|---|---|---|
| 1 | 0.000 | 0.038 | 0.71 | 0.04 | 0.012 |
| 7 | 0.002 | 0.185 | 0.92 | 0.93 | 0.109 |
| 15 | −0.002 | 0.311 | 0.93 | 0.98 | 0.314 |
| 23 | −0.005 | 0.385 | 0.92 | 0.99 | 0.504 |
| 31 | 0.006 | **0.427** | 0.91 | 0.994 | 0.611 |

The logged "skew(z₃₁)=±0.02 ⇒ z is Gaussian ⇒ Edgeworth worthless" was the
**mean over neurons, which is zero by sign symmetry** — exactly as predicted by
`skew(z_i) = γ₁·eig1frac^{3/2}·h_i³`, E[h³]=0. True RMS skew is **0.427**; z is
strongly non-Gaussian per-neuron. Every structural prediction confirmed:
⟨v₁,μ_a⟩→0.994 (top mode aligns with the mean direction), corr(γ,t)≈0.92.
(My *pessimistic* corollary — that this anti-correlation suppresses the shape
error to ~2e-9 — was WRONG: measured one-step gain from κ₃ at L30 is 6.5×.
Measurement beat the back-of-envelope, as it should.)

### What limits the oracle-κ₃κ₄ run at 1.26e-6

Residual one-step μ error with κ₃+κ₄ is ~3e-8, so Σ over 32 layers ≈ 6e-7 —
we measure 1.26e-6. The gap is (a) Edgeworth truncated at 4th order, and
(b) **our κ₄ covariance correction is INCOMPLETE**: the corpus has only
κ₄_iijj, so terms needing κ₄_iiij are missing (flagged in `harness.py`).
So 1.26e-6 is the ceiling of *our correction structure*, not of the
information in κ₃+κ₄. Completing the formulas may buy more.

## 4.7 THE 1.26e-6 CEILING — DIAGNOSIS, A PROPOSED PIVOT, AND ITS REFUTATION

### 4.7.1 Diagnosis: Edgeworth breaks down with depth (this part holds)

The κ₃+κ₄ **residual** one-step error, next to the RMS skew measured in S1:

| layer | residual μ err (κ₃+κ₄) | RMS skew of z |
|---|---|---|
| 0 | 2.7e-9 | ~0.02 |
| 3 | 4.4e-9 | ~0.10 |
| 7 | 9.0e-9 | 0.185 |
| 15 | 1.6e-8 | 0.311 |
| 23 | 2.4e-8 | 0.385 |
| 30 | **3.2e-8** | **0.427** |

The residual grows **12× with depth, in lockstep with the skew**. Edgeworth is
an *asymptotic expansion in small non-Gaussianity*: at layer 3 (skew 0.10) it is
essentially exact (1100× better than Gaussian); at layer 30 (skew 0.43) it is
running out of road. **We are not information-limited** (we feed it *perfect*
κ₃ and κ₄) — **we are reconstruction-limited.** Adding κ₅/κ₆ chases a
slowly-converging series that is expensive to propagate: a bad trade.

### 4.7.2 The pivot I proposed (and why it looked right)

Model the state as **Gaussian conditional on a scalar core**:

    a | c  ~  N(m(c), R(c)),    c = û·(a − μ),   û = μ/‖μ‖

Then z = Wᵀa is *exactly* Gaussian conditional on c, so every ReLU expectation
is exact — no series, no truncation, all orders of the core captured. Cost
~2–5e9 FLOPs (one WᵀRW per layer + Q·kmax·n² for the conditional Wick series).

The supporting evidence looked strong:
- `corr(γ_i, t_i) = 0.92` at **every** depth (even layer 7) ⇒ per-neuron skew is
  a near-deterministic function of that neuron's overlap with the mean
  direction ⇒ "the non-Gaussianity is rank-1".
- `⟨v₁, μ_a⟩ = 0.93 → 0.994` ⇒ the non-Gaussian direction **is** the mean
  direction (physically sensible: ReLU rectifies, pumping positive skew into
  the "all-on" direction, which is where μ points).
- `eig1frac` is only 0.012–0.109 early ⇒ **the covariance is high-rank but the
  non-Gaussianity is rank-1**. This also explained why the old rank-1 core
  closure failed: it used the **top eigenvector of Σ** — at layer 1 that has
  `⟨v₁,μ⟩ = 0.043`, i.e. *orthogonal to the skew*. Right idea, wrong axis.

### 4.7.3 ⛔ E3′ REFUTES IT — the conditional-Gaussian closure is DEAD

`research/e3_conditional.py`, **8 MLPs**, 1e6 samples, one-step MSE of
E[ReLU(z)] against the same-sample empirical truth (so sampling noise cancels;
verified by a disjoint half-split — the HALF column reproduces the Q32 column
to 3 s.f. ⇒ these are **model bias**, not noise).

| layer | Gauss (Q=1) | cond-Gauss Q=8 | Q=32 | Q=64 | eig-v1 Q32 | rank-2 8×8 | **Edgeworth κ₃+κ₄** |
|---|---|---|---|---|---|---|---|
| 3 | 4.90e-6 | 1.42e-6 | 1.45e-6 | 1.45e-6 | 3.29e-6 | 1.24e-6 | **5.1e-9** |
| 7 | 4.59e-6 | 1.65e-6 | 1.69e-6 | 1.70e-6 | 2.12e-6 | 1.20e-6 | **9.0e-9** |
| 15 | 3.17e-6 | 1.40e-6 | 1.42e-6 | 1.42e-6 | 1.53e-6 | 8.24e-7 | **1.9e-8** |
| 23 | 1.68e-6 | 9.72e-7 | 9.76e-7 | 9.76e-7 | 9.76e-7 | 4.41e-7 | **2.3e-8** |
| 30 | 1.22e-6 | 6.95e-7 | 6.90e-7 | 6.90e-7 | 6.92e-7 | 2.92e-7 | **3.6e-8** |

**⭐ CROSS-VALIDATION (important):** E3′ (direct sampling) and E2 (moment
propagation from the corpus) are *completely independent* codepaths, and they
agree: Gaussian one-step at L30 **1.22e-6 vs 1.21e-6**, at L23 1.68e-6 vs
1.64e-6, at L3 4.90e-6 vs 4.95e-6; Edgeworth κ₃+κ₄ at L30 **3.6e-8 vs 3.2e-8**.
**The harness is validated.** Cite these numbers with confidence.

**The conditional-Gaussian closure is 20–280× WORSE than Edgeworth κ₃+κ₄,**
and it **saturates in Q** (Q=8 ≈ Q=32 ≈ Q=64 to 3 s.f.) — so this is a
model-class limit, not a quadrature-resolution limit. Rank-2 buys ~2× and is
still ~8–24× worse than Edgeworth. (Sub-finding that DID hold: û *is* a better
core than v₁ — 1.45e-6 vs 3.29e-6 at layer 3 — so the *diagnosis* of the old
rank-1 failure was right; the *cure* was not.)

**Why the reasoning was wrong.** `corr(γ_i, t_i) = 0.92` says the per-neuron
skew is predictable from the neuron's overlap with μ. It does **not** say the
joint law is Gaussian once you condition on one scalar. Conditioning on
c = û·(a−μ) removes only the component of the non-Gaussianity *aligned with û*;
the measurement says most of it survives. Non-Gaussianity being *directionally
predictable* ≠ *rank-1*. I conflated the two.

**Lesson (process):** the whole pivot was argued from three correlations and
would have cost a week. It was killed by a 10-minute measurement that had a
built-in falsification criterion. Keep doing exactly this.

### 4.7.4 What this leaves us with (and the new prime suspect)

Edgeworth κ₃+κ₄ is per-step **excellent**: residual 5e-9 → 2e-8. If those added
up with amplification ≈1 the end-to-end would be ~3–6e-7, not the **1.26e-6**
E2 measured. So the gap is **not** in the mean update. Look at the *covariance*
one-step residual (E2 table):

| layer | cov: Gauss | +κ₃ | +κ₃+κ₄ |
|---|---|---|---|
| 0 | 1.13e-5 | 2.99e-7 | **3.91e-8** |
| 15 | 6.01e-6 | 5.00e-8 | 4.47e-8 |
| 23 | 3.09e-6 | 3.15e-8 | **3.07e-8** |
| 30 | 1.59e-6 | 2.63e-8 | **2.58e-8** |

**κ₄'s covariance correction helps at layer 0 (7.6×) and does NOTHING at depth**
(3.15e-8 → 3.07e-8). That is exactly the fingerprint of our **incomplete κ₄
covariance formula**: the corpus stores only κ₄_iijj, so every term needing
κ₄_iiij is missing. The covariance residual stalls at ~2.6e-8 and then
*compounds over 32 layers* — and a drifted Σ is what makes the true-cumulant
corrections mismatched downstream, inflating the end-to-end number.

**⇒ New prime suspect: the ceiling is set by the COVARIANCE update, not the
mean update, and specifically by the missing κ₄ covariance terms.**

## 4.8 ⭐ THE FORMULA WAS HALF-WRITTEN (E4/E5/E6) — and the build spec

### 4.8.1 Five missing diagram families

Re-derived the covariance update from the Hermite diagram summation formula
(2605.05179 Thm A.2) instead of copying the existing code. For r=2, blocks of
size ≤2 must be MIXED (span both neurons); blocks ≥3 need not be. **Every
cumulant block therefore generates a SERIES in Σ_ij, not one term:**

    Gaussian :       sum_p (1/p!) h[p](i)   h[p](j)   S^p       p>=1  ✓ had it
    k3_iij   :  1/2  sum_p (1/p!) h[p+2](i) h[p+1](j) S^p k3    p>=0  ✗ p=0 only
    k3_iii   :  1/6  sum_p (1/p!) h[p+3](i) h[p](j)   S^p k3    p>=1  ✗ MISSING
    k4_iijj  :  1/4  sum_p (1/p!) h[p+2](i) h[p+2](j) S^p k4    p>=0  ✗ p=0 only
    k4_iiij  :  1/6  sum_p (1/p!) h[p+3](i) h[p+1](j) S^p k4    p>=0  ✗ MISSING
    k4_iiii  : 1/24  sum_p (1/p!) h[p+4](i) h[p](j)   S^p k4    p>=1  ✗ MISSING

The p=0 truncation is only valid if Σ_ij is small. **Measured (E4): off-diag
|ρ| grows to mean 0.361 / p95 0.765 / max 0.946 by L30.** It is not small.
The MEAN was also incomplete: Gram-Charlier has a **κ₃² power-cumulant term**
`(1/72)·ĥ₆·κ₃_iii²` at order 6 — absent from our code (and it is exactly the
"power cumulant" device the paper says is *essential*).

**⇒ THIS EXPLAINS THE ENTIRE κ₄ SAGA.** IMPROVEMENTS.md logged "κ₄ rel error
2.54 — worse than predicting zero, and *worse at higher rank*." That is the
signature of implementing ONE κ₄ family (iijj) out of THREE. E4 shows adding
families **one at a time makes it worse before it gets better** (8.06e-8 →
2.78e-7 → … → 6.72e-9) because they are the SAME ORDER and only work together.
**κ₄ was never broken — it was half-written.** Propagating a half-written
formula more faithfully (higher rank) makes it worse, precisely as observed.

E4 one-step covariance residual, complete vs shipped: **2.6–12× better**.

### 4.8.2 The ceiling moved: 1.26e-6 → 5.59e-7 (E5, 8 MLPs)

| | final-layer MSE |
|---|---|
| plain K=2 | 1.279e-4 |
| oracle κ₃+κ₄, **truncated** formula | 1.201e-6 ← reproduces E2 ✓ cross-validated |
| oracle κ₃+κ₄, **COMPLETE** formula | **5.591e-7 ± 1.5e-7** |

At the floor that is **score 5.6e-8**. Leaders (2026-07-12): 8.22e-8 / 9.16e-8 /
9.27e-8 — and their raw MSE is stuck at ~8.2e-7. **5.59e-7 beats all of them**,
with no κ₅ and no new physics. (shiv_m's jump to #1 at 8.22e-8 was NOT an
accuracy gain — identical per-MLP MSEs, he just trimmed compute 11.59%→9.94%
to sit exactly on the floor. Worth 1.15× and confirms the flat-frontier maths.)

### 4.8.3 ⭐ THE BUILD SPEC (E6, measured — 103 s batched on MPS)

The 5 cumulant slices are now the ONLY missing piece. How accurate must they be?

**Sampling hybrid: DEAD (measured, not argued).** Estimating slices from N real
samples costs 4× a forward pass (3 extra n²/sample matmuls). Best achievable
score 9.8e-7 (N=1e4, 32% budget) vs v4's 3.4e-7. At large N plain MC overtakes
it (the closure has a bias floor). Closed.

**SPLIT SPEC — κ₃ and κ₄ have VERY different tolerances** (MSE; rows κ₃ err,
cols κ₄ err):

| κ₃\κ₄ | 0% | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|
| **0%** | 5.81e-7 | 5.82e-7 | **6.65e-7** | 8.58e-7 | 2.48e-6 |
| **2%** | **5.73e-7** | 7.12e-7 | **6.58e-7** | 1.04e-6 | 2.36e-6 |
| **5%** | 8.82e-7 | 1.04e-6 | 1.27e-6 | 1.36e-6 | 3.20e-6 |
| **10%** | 1.99e-6 | 2.03e-6 | 3.04e-6 | 2.19e-6 | 3.75e-6 |
| **20%** | 7.07e-6 | 9.02e-6 | 6.34e-6 | 7.38e-6 | 9.58e-6 |

κ₄ 0→20% costs **1.5×**. κ₃ 0→20% costs **12×**.
**κ₃ accuracy is worth ~10× κ₄ accuracy.** κ₄ dropped entirely = 5.70e-6
(necessary, but crude is fine).

    ┌──────────────────────────────────────────────────────────────┐
    │ BUILD SPEC:  kappa3 slices <= 2-5% relative   (the hard part) │
    │              kappa4 slices <= 10-20%          (easy)          │
    │              complete diagram formula          (DONE)         │
    │ => MSE 5.7-6.6e-7 => score 5.7-6.6e-8 => #1 by 1.25-1.44x     │
    └──────────────────────────────────────────────────────────────┘

**This inverts the old priority.** IMPROVEMENTS.md's parting instruction was
"THE NEXT SESSION HAS ONE JOB: FIX κ₄" — but κ₄ barely matters, while κ₃ was
left at 6.7%, which is *exactly* the regime where the error explodes (5%→8.8e-7
but 10%→2.0e-6). **All engineering goes into κ₃ fidelity.**

Caveat to carry: the spec was measured with iid multiplicative noise. A rank-R
truncation error is *structured*, so 2-5% is indicative, not exact — validate
against the real propagation, not against this number.

## 4.9 STEP BACK (2026-07-12 evening) — what went right, what I missed

### 4.9.1 What went right (keep doing)

1. **Measure the ceiling and the tolerance BEFORE building.** E2 (oracle) → E5
   (complete-formula ceiling 5.59e-7) → E6 (spec: κ₃ ≤5%, κ₄ 10-20%). Every
   architecture decision since has been made against a measured number instead
   of an argument. This is the single most valuable habit adopted.
2. **Falsifiable experiments kill bad ideas in minutes, not weeks.** The
   conditional-Gaussian closure (a week of work) died in 10 minutes because the
   test had a built-in refutation criterion. Same for the router and the
   sampling hybrid.
3. **Programmatic diagram enumeration** (`diagrams.py`) after three hand-derivation
   bugs in one day. It reproduces every hand derivation and generates families we
   would never have found by hand.
4. **Cross-validating independent codepaths** (E2 vs E3′ vs E5 agree to ~3%).
   In a project whose history is littered with numbers that don't replicate,
   this is what makes a result citable.
5. **Verifying at the REAL shape.** n=12/32 said star-only was 62.8% wrong;
   n=256 says 9.5% at L1 and 69% at L3. Small-n is a different physics regime
   (1/n expansion) — it would have sent us somewhere useless.

### 4.9.2 What I missed (do not repeat)

1. **⛔ I forgot the competition is ESTIMATION UNDER BUDGET, not exact
   computation.** I built a complete-diagram closure that is physically correct
   and **147% of budget** (n⁴ mode products) — unshippable by construction. I
   validated the *physics* without checking whether *that implementation* could
   ever run in-budget. Same class of error as the logged 2.7× FLOP miscount I
   had criticised. **Rule: every closure design must be priced (flopscope)
   BEFORE it is validated, not after.**
2. **⛔ I under-weighted FUSION** (see §4.9.3). I treated the closure as a
   standalone competitor that had to beat MC, when it is a *fusion component*
   that only has to be independent and comparable. This changes the required
   accuracy by ~5×, and therefore changes which architectures are viable.
3. **I implemented cumulant formulas to "leading order" three times** (r=2
   update, κ₃ generation, κ₃ pass-through) and only tested end-to-end, where
   formula error and rank error alias. Every site where a cumulant enters is a
   diagram *series*.
4. **I cited IMPROVEMENTS.md's "our κ₃ closure = 9.6e-6, therefore κ₃ is at its
   ceiling" without measuring it**, and built a whole decision on it — the same
   sin the audit was written to prevent.
5. **I chased rank** (R=16…256 sweeps) before checking whether rank was the
   binding variable. It wasn't (R=128 ≈ exact carry).

### 4.9.3 ⭐ THE FUSION ARITHMETIC (the reframing that matters)

Two independent estimators fuse to `MSE = b²v/(b²+v)`. The closure costs ~3% of
budget; QMC takes the rest of the floor (~7% → ~4,600 points → v ≈ 5.1e-6;
v3 measured 6,400 points → 3.67e-6). To beat v4 (score 3.4e-7):

    b²·v/(b²+v) < 3.4e-6   =>   b² < 1.02e-5

| closure b² | fused MSE | score | vs v4 |
|---|---|---|---|
| 1.0e-5 | 3.4e-6 | 3.4e-7 | tie |
| 5e-6 | 2.5e-6 | 2.5e-7 | **1.35×** |
| 3e-6 | 1.9e-6 | 1.9e-7 | **1.8×** |
| 2e-6 | 1.4e-6 | 1.4e-7 | **2.4×** |
| 1e-6 | 8.4e-7 | 8.4e-8 | **beats the leaders** |

**The closure does NOT need to beat sampling. It needs b² ≲ 3e-6 and
independence.** v4 already fuses — but its analytic component (calibrated K=2,
b² = 2.4e-5) is so weak it earns only w = 0.137. A closure at b² ≈ 3e-6 earns
w ≈ 0.5 and transforms the score. **This is the cheapest lever on the board and
I nearly walked past it.**

### 4.9.4 Where the analytic κ₃ actually stands (E10, measured)

Complete-diagram closure, κ₃ as a FULL TENSOR (correct physics, 147% of budget):

| max σ edges | L1 | L3 | L7 | L15 | L23 | L31 | end-to-end MSE |
|---|---|---|---|---|---|---|---|
| 2 | 10.3% | 20.6% | 19.8% | 28.9% | 34.0% | 41.5% | 6.75e-6 |
| 3 | 11.0% | 15.9% | 22.0% | 36.3% | 41.3% | 47.5% | 6.23e-6 |

**κ₃ error ACCUMULATES with depth** (~10%/step compounding incoherently over 31
layers → √31·10% ≈ 56%). The one-step floor with κ₃-blocks-only is 10.6%
(E9b) — the residual is κ₄(z)-BLOCK-limited, not σ-edge-limited.

**But the error is a systematic MAGNITUDE bias, not a shape error:**

| | L1 | L3 | L7 | L15 | L23 | L31 |
|---|---|---|---|---|---|---|
| optimal gain | 0.93 | 1.07 | 1.18 | 1.44 | 1.58 | **1.88** |
| correlation | 0.9966 | 0.9895 | 0.9868 | 0.9789 | 0.9768 | **0.9719** |
| resid after gain | 8.2% | 14.4% | 16.2% | 20.4% | 20.4% | 19.9% |

corr stays 0.97-0.997; the gain climbs monotonically to 1.88 — i.e. our κ₃ is
systematically **too small at depth**, and it COMPOUNDS (1.88 over 31 layers is
only ~1.02/layer). **A per-layer scalar gain, applied INSIDE the recursion and
learned offline (the ensemble is weight-exchangeable ⇒ it must be universal),
attacks the error at source.** Zero in-budget cost.

## 4.10 THE PLAN TO BEAT v4 — SUPERSEDED by §4.12 (kept for the reasoning)

> **P1 and P2 below BOTH FAILED — see §4.11. The live plan is §4.12
> (depth-truncated sampling). Do not restart P1/P2/P3 without reading §4.11.**


**Target: closure b² ≲ 3e-6 at ≤3% of budget, fused with QMC at ~7%.**
Projected score 1.9e-7 (1.8× better than v4). Nothing here requires exact κ₃.

**P1 — LEARNED κ₃ SLICES ("precompute and memorise"). Test first: cheapest,
biggest payoff, no tensor propagation at all.**
By mean-field universality `κ₃_iij/(σ_i²σ_j)` should be a function of
`(ρ_ij, t_i, t_j, layer)`. Evidence: S1 measured `corr(γ_i,t_i)=0.92`. Fit the
map OFFLINE on the corpus to arbitrary precision, ship as a table (0-FLOP load
via `flopscope.Module`), evaluate at **n² per layer ⇒ ~1% of budget**. That
leaves ~9% for QMC (v ≈ 4.05e-6). At κ₃≈10% (b²≈2.07e-6) → fused 1.37e-6 →
**score 1.37e-7 = 2.5× v4.**
*Gate: regress κ₃_iij on the corpus and report R² / relative error. Hours.*

**P2 — per-layer gain calibration** on the factored (affordable) propagator,
fitted in-loop and TRANSFERRED to held-out MLPs. Zero in-budget cost. Expect
κ₃ 20-47% → ~15%. Compose with P1.

**P3 — factored contraction of the ≤2-σ-edge diagrams** (64 families) — only if
P1/P2 miss. Cost is S·n³/layer, so S ≲ 35 to stay in budget. This is the
expensive path; do not start it before P1 is measured.

**P4 — κ₄(z) blocks** (needed for the one-step floor below 10.6%): a full κ₄
tensor is out (n⁴ = 17 GB, n⁵ mode products). Only viable as learned slices —
i.e. P1 applied to κ₄ as well.

**Deferred: the (μ,Σ) complete formula is DONE and cheap (0.81% of budget) —
it goes into any submission regardless.**

## 4.11 P1 AND P2 BOTH FAILED (measured) — and what that proves

### P1 — learned κ₃ slices from local state: DEAD (E12, 4 s on MPS)

Fit `κ₃_iij/(σ_i²σ_j) = f(t_i, t_j, ρ_ij, layer)` offline on 12 MLPs, evaluated
on 12 **held-out**:

| features | L1 | L3 | L7 | L15 | L23 | L31 | mean(32) |
|---|---|---|---|---|---|---|---|
| local only | **83.9%** | 61.2% | 46.1% | 33.9% | 27.4% | 26.9% | 39.8% |
| **+ star feature** | **5.7%** | 47.9% | 44.2% | 33.6% | 27.3% | 26.8% | 35.2% |
| *(propagated tensor, ref)* | *11.0%* | *15.9%* | *22.0%* | *36.3%* | *41.3%* | *47.5%* | — |

1. **The slice is NOT a local function of pairwise statistics.** Obvious in
   hindsight: the star is `2(CᵀB)_ij + (DᵀW)_ij` with `B = Σ·diag(ĥ₁)·W` — a
   contraction over p, i.e. **non-local in W and Σ**. My mean-field-universality
   argument was wrong; `corr(γ_i,t_i)=0.92` only ever supported the DIAGONAL.
2. Handing the regression the **star** fixes layer 1 completely (83.9% → **5.7%**,
   inside spec and better than the raw star's 9.5%) — because at L1 z⁰ is exactly
   Gaussian, so there IS no pass-through and the star is the whole answer.
3. **Depth is unchanged.** κ₃ at depth depends on ACCUMULATED HISTORY; no function
   of the current-layer state can recover information that is not in it.

**⇒ κ₃ MUST be propagated. Learned slices cannot substitute — only correct.**

### P2 — per-layer gain calibration: DEAD (E13, 2.7%)

Fitted one shared scalar per layer, in-loop, on the AFFORDABLE (factored, R=64)
propagator; transferred to 8 held-out MLPs:

| | L1 | L3 | L7 | L15 | L23 | L31 | end-to-end MSE |
|---|---|---|---|---|---|---|---|
| uncalibrated | 10.4% | 34.1% | 26.8% | 21.7% | 23.3% | 26.0% | 1.320e-5 |
| calibrated | 8.7% | 31.8% | 25.9% | 23.7% | 24.4% | 25.4% | **1.284e-5** |

Fitted gains come out ~1.01/layer — because applying the gain **in-loop stops the
compounding**, so almost none is needed. And once compounding is gone the residual
is a **shape** error, which a scalar cannot touch.

**The subtlety:** E10's dramatic gain (0.93 → 1.88) was measured on the **tensor**
closure with **complete** diagrams — that one has a genuine magnitude bias. The
**affordable factored** closure's error is dominated by leading-diagram
truncation instead. **Calibration only fixes the error the closure we cannot
afford has.**

### Score-keeping: four architectures priced and killed in one day

| route | verdict |
|---|---|
| conditional-Gaussian / mixture closure | dead (20–280× worse than Edgeworth; saturates in Q) |
| sampling the cumulant slices | dead (costs 4× a forward pass) |
| learned local slices (P1) | dead (κ₃ depends on history, not local state) |
| scalar gain calibration (P2) | dead (2.7%) |

Fusion bar: **b² < 1.02e-5** to beat v4. Affordable closure = **1.284e-5** →
fused score 3.6e-7, *slightly worse* than v4's 3.396e-7. **We are just on the
wrong side of the line, and every cheap route to move is now closed.**

## 4.12 ⭐⭐ THE HOLISTIC RETHINK — DEPTH-TRUNCATED SAMPLING (untested, top priority)

### The leaders' numbers fit NEITHER pure story

| | raw MSE | budget | wall | intermediates |
|---|---|---|---|---|
| shiv_m (#1) | 8.16e-7 | 9.94% | 27–40 s | zeros |
| pluto | 9.27e-7 | 9.57% | 20–43 s | zeros |
| andrew | 3.55e-7 | 26.3% | 4–13 s | all accurate |
| plain MC @ floor | 6.5e-6 | 10% | — | free |
| our QMC (v3) | 3.67e-6 | 10% | — | free |
| our best affordable closure | 1.28e-5 | ~3% | — | free |

Pure sampling cannot get there (our QMC family is exhausted at 3.67e-6). Pure
kprop cannot either: reaching 8.16e-7 via `c_K(L/n)^K` needs **K ≈ 5**, wildly out
of budget. **⇒ they are running a HYBRID. I have been treating the two halves as
competitors instead of composing them.**

### The idea

Everything we have tried samples from **x**, costing `32·2n²` per sample. **We do
not have to start at the input.**

> **Propagate the distribution analytically to layer k, then SAMPLE from it and
> run only the remaining (32−k) layers exactly.**

- **Cost/sample collapses to `(32−k)·2n² + n²`** (the `+n²` generates
  `z = μ + Lξ`). k=24 → 3.5× cheaper; k=28 → 7×; k=30 → 13×. Same budget ⇒ that
  many more samples ⇒ variance down by the same factor.
- **The deep ReLUs are handled EXACTLY** — zero closure error in layers k…31,
  which is precisely where our formulas are worst and where error accumulates.
- **The bias is the closure's error at layer k**, far smaller than at layer 32
  because it has not compounded yet.

### The arithmetic lands exactly on the leaders

| k | cost/sample | N at floor | variance term |
|---|---|---|---|
| 24 | 1.2e6 | 22,700 | 1.85e-6 |
| 28 | 5.9e5 | 46,200 | **9.1e-7** |
| 30 | 3.3e5 | 83,000 | **5.1e-7** |
| 31 | 2.0e5 | 136,000 | 3.1e-7 |

k = 28–30 lands **exactly** in the leaders' range (8.2–9.3e-7) — and it does so
with a **K=2-quality closure, which we already have**.

### It also explains both ledgers

- **pluto/shiv:** elementwise-heavy wall (the closure), at the floor,
  intermediate layers junk because the sampled part only covers the tail.
- **andrew:** error growing with depth (analytic signature) then an **18× drop at
  L31 only** — which matches our own E4 measurement that the final-layer cumulant
  correction is worth 8–30×.

### Why I missed it (and why the logged no-go does NOT apply)

IMPROVEMENTS.md has a "clean negative theorem" killing **depth telescoping**
(MLMC across layers) and I let it stand. **This is not telescoping.** MLMC needs a
cheap *surrogate correlated with the full network*. This replaces the *head* with
an analytic distribution and samples only the *tail* — a biased estimator with a
much cheaper per-sample cost. Different object, different economics. The no-go
does not cover it and I never checked.

### The test (cheap, decisive, next thing to run)

For each k:  `MSE(k) = Var_k/N(k) + bias(k)²`, where
- `N(k) = budget / ((32−k)·2n² + n²)` — pure arithmetic;
- `bias(k)` = sample from `N(μ̂_k, Σ̂_k)` (**the closure's own state, errors and
  all**), propagate exactly to the end, compare to truth.

Gives the optimal k and the achievable MSE directly. **Composes with QMC** (use a
QMC point set at layer k, not iid) **and with the final-layer Edgeworth
correction** — both already built and validated. Nothing new has to be derived.

**Upside estimate:** k≈28–30 with QMC and a κ₃-corrected final ReLU should land
below 8e-7 ⇒ score < 8e-8 ⇒ first place. Even a crude version beats v4.

## 5. Gating experiments (cheap first; each has an unambiguous verdict)

**E0 — Re-baseline everything (½ day, CPU/GPU).** One canonical
implementation each of: plain K2, calibrated K2, κ3 closure (r=32), QMC-6400.
Run all on the SAME fixed 48-MLP local set AND the mini split. One table,
mean ± SEM, paired. Kills the 2.8× baseline drift; all later work cites this.

**E1 — Verify the harmonic-spectrum theory (hours, CPU).** Compute the Taylor
spectrum of ρ_L numerically (mpmath, the composite-series script from
2026-07-12, unrun); verify against direct MC on a width-4096 depth-32 net
(estimate degree-d power via Gegenbauer projections on sampled pairs).
Verdict: if confirmed, permanently close all polynomial-CV/moment-matching
ideas and quantify QMC's ceiling; if refuted, my spectrum math was wrong.

**E2 — Clean oracle 2×2(×κ) (1 day, GPU; THE decision experiment).** One
canonical propagator, ≥24 corpus MLPs, N≥8e6 GT: inject each subset of
{true μ, true Σ, true κ3-slices, true κ4-slices} per layer and measure
final-layer MSE. Includes the never-run true-μ-only cell, and quantifies
corpus slice noise first.
- If K2+κ3(+κ4 slices) ≤ ~1.5e-6 → third-order info suffices → pure-analytic
  route is real; priority = propagate κ3 correctly (machine-verified) or
  learn slices from state.
- If the R7 "cancellation" reproduces → mean-update bias is the physics;
  mixture/conditional closures move up (they fix mean and cov jointly).
- If neither → analytic ceiling is high; hybrid/CV becomes primary.

**E3 — Quenched low-dimensionality (½ day, GPU).**
(a) eig1frac and top-k spectrum of the fluctuation across ≥24 MLPs (was: 3).
(b) Oracle conditioning: how much of the per-layer moment drift is explained
by conditioning on the top-k modes (k=1,2,4,8)? Prices the Q-node
mixture-of-Gaussians closure (pluto-shaped: cost ≈ 4Q·n³L, Q=10 ≈ 2.2e10 ✓)
before building it. The old rank-1 failure (1.06×, 3 MLPs) is not trusted.

**E4 — Per-sample surrogate correlation (½ day, GPU; the hybrid gate).** For
≥24 MLPs: run 2k full passes, log per-sample final activations; test cheap
surrogates: rank-r sketched propagation (project each W to top-r×top-r,
r=8/16/32; cost ~r²/n² of full), per-layer amplitude trackers (JL-projected
norms). Measure per-sample ρ² with f and plug into the multilevel formula
VR = 1/[√(1−ρ²)+ρ√(c_g/c_f)]². Verdict: ρ² ≥ 0.8 at c_g/c_f ≤ 5% → build
(projected ≥ 3.5×, composable with QMC and with any analytic mean for g).
Note: the depth-telescoping no-go does NOT cover this (different axis).

**E5 — Cheap intelligence (minutes).**
(a) Pull OUR v4 submission dashboard (user pastes): per-MLP + per-layer
ledger → paired per-MLP comparison vs all three leaders on the named 50;
calibrate the QMC per-layer profile against andrew's.
(b) Check whether the graded-100 names match the published mini-100 (if the
weights of the public 50 are local, per-MLP forensics gets 10× sharper; if
GT were also published for them it would be a design flaw — verify and, if
so, flag to organizers rather than exploit).
(c) Check submission-rate limits + per-MLP wall cap in docs (pluto survived
43 s/MLP — find the actual limit; long-wall analytic methods are safe).

---

## 6. Decisions

### 6.0 SUPERSEDING DECISION (after E5a/E5b, 2026-07-12)

The per-MLP forensics settle §6.1 early. Sampling-side work (point sets,
routing, fusion weights, operating point) is **capped at ~1.4× oracle** and is
not where the 3.7× leaderboard gap lives. The gap is 11.8× of raw accuracy in
the **primary analytic closure**. Therefore:

> **PRIMARY OBJECTIVE: analytic closure MSE 1.09e-5 → ≤ 9e-7 (11.8×), at
> ≤ 2.5e10 FLOPs. Pure-analytic, final-layer-only. Everything else is
> deprioritised until this moves.**

Keep v4 on the board as the fallback submission. Keep a divergence guard.
Do not spend more time on QMC, routing, or fusion economics.

### 6.1 Fully analytic vs hybrid? (superseded by §6.0 — kept for rationale)
**Both stay live for exactly one gated week; the default endgame is HYBRID.**
Reasoning: (i) hybrid has an unbiasedness floor — it can't be worse than
sampling; (ii) every analytic improvement also strengthens the hybrid (as
fusion component or CV mean); (iii) the #1 spot is hybrid-shaped. BUT pluto
is an existence proof that pure-analytic 9.3e-7 at the floor exists, and two
concrete architectures fit its cost signature (Q≈10 Gaussian mixture; κ3-rank
~56 done right). E2+E3 decide within the week: if the analytic gates go
green, pure analytic becomes primary (it also generalizes better to a
phase-2 shape change — no shipped point set, no depth-baked tables).
If they stay red, we stop sinking time into closures and go all-in on
E4-style per-sample variance reduction plus tail robustness.

### 6.2 κ4 or not?
**Not in its current form — the measurement that it's harmful is trusted.**
The program survives only as:
(a) machine-verified re-derivation (sympy + small-n brute force — the three
contradictory diagnoses mean we don't even know which diagram family is
missing), or
(b) corpus-supervised learned slices κ4_iijj = g(local K2/κ3 state), fit
against sampled truth at every layer (cannot fail silently), or
(c) implicitly via a mixture closure (a scale-mixture of Gaussians carries
excess kurtosis for free — if E3b gates green, explicit κ4 tensors may be
unnecessary).
Gate: E2's κ4-slice cell shows what κ4 is actually worth end-to-end; only
then choose (a)/(b)/(c).

### 6.3 Operating point
Floor by default; flat frontier means this is a non-decision for β≈1
methods. Never approach 100% (hard cliff). Long wall-time inside fnp kernels
is free — stop optimizing it below ~10 s/MLP.

### 6.4 Success criteria & phase-2 posture
- To *beat* 9.16e-8 robustly (private-split coin-flip margin): target
  **final-layer MSE ≤ 5e-7 at the floor (VR ≈ 13×)**. Interim milestone:
  ≤ 9e-7 (parity with the leaders' cluster).
- Phase 2 (Aug 1) may change the shape. Prefer shape-generic machinery
  (closures parameterized by (n, L), surrogates, per-layer-index-free
  calibrations) over shape-baked artifacts (the 6400×256 point set, the
  32-row θ table). Budget one day post-announcement for re-tuning.
- The $20k report track is real: this document + honest negative results are
  an asset. Keep the audit trail.

---

## 6.5 POST-E2 DECISIONS (2026-07-12)

- **STOP all κ₃ work** (rank, star/core fidelity, propagation). Measured at
  ceiling. This closes the single biggest time sink of the last two days.
- **κ₄ is the whole program.** Measured payoff 7.3× → 1.26e-6 (score ~1.26e-7).
  Two admissible routes (hand tensor algebra remains BANNED):
  - **K4-LEARN (primary):** regress the κ₄ slice from the local K=2/κ₃ state
    (α_i, α_j, ρ_ij, κ₃ slice, layer index) against the corpus's exact
    per-layer targets. Cannot fail silently; n² at inference; unlimited offline
    compute. We now know the exact payoff *and* have exact targets.
  - **K4-DERIVE:** sympy + brute-force small-n (n≤8) verification of the
    missing diagram families. Only if K4-LEARN plateaus.
- **Next experiment E2b (cheap, do first): κ₄ SLICE-NOISE SENSITIVITY.**
  Corrupt the oracle κ₄ with r% relative noise, re-run the (C) propagation,
  and plot MSE vs r. This gives the accuracy spec K4-LEARN must hit *before we
  build it* — and if the tolerance is, say, <5%, K4-LEARN is likely infeasible
  and we pivot to the mixture closure immediately.
- **E2c: complete the κ₄ correction formulas** (the κ₄_iiij covariance terms
  are missing from both the corpus and `harness.py`). Re-derive with machine
  verification and re-measure the ceiling — 1.26e-6 may not be the true limit.
- **⛔ Mixture/conditional-Gaussian closure: KILLED by E3′** (§4.7.3). It is
  100–300× worse than Edgeworth κ₃+κ₄ and saturates in Q. Do not revisit in
  any rank. The whole "scale-mixture / non-Gaussian core / particle closure"
  family (including pluto's presumed architecture) is off the table for us.
- **Edgeworth κ₃+κ₄ is the right reconstruction after all** — per-step μ
  residual 5e-9…2e-8. Keep it; do not replace it.

### POST-E3′ PRIORITY ORDER (the covariance is the bottleneck)

The mean update is *not* what limits us. Per-step μ residual (5e-9–2e-8) would
sum to ~3–6e-7 end-to-end, but E2 measured 1.26e-6. The gap is the **covariance
update**, which stalls at ~2.6e-8/step and then compounds — and a drifted Σ is
what makes downstream cumulant corrections mismatched.

1. **E2c — COMPLETE THE κ₄ COVARIANCE FORMULA (top priority).** κ₄'s covariance
   correction gives 7.6× at layer 0 and **exactly nothing at depth** (3.15e-8 →
   3.07e-8) — the fingerprint of the missing κ₄_iiij terms (the corpus stores
   only κ₄_iijj). Machine-verified derivation (sympy + brute-force n≤8) — hand
   tensor algebra remains BANNED. Then re-run E2's ceiling: **1.26e-6 may not
   be the real limit**, and this is the cheapest way to find out.
2. **E2d — where does the end-to-end error actually accumulate?** Run the E2
   restart curve *with* oracle κ₃+κ₄ to get the per-layer injection profile of
   the corrected closure. Confirms (or refutes) that Σ-drift is the culprit and
   says which layers to spend on.
3. **E2b — κ₃/κ₄ slice-noise sensitivity.** How accurate must a *predicted*
   slice be? Sets the spec for K4-LEARN before we build it.
4. **K4-LEARN** — regress the κ₄ slice from the local K=2/κ₃ state against the
   corpus's exact per-layer targets. Only after (1)–(3).
5. Then re-derive the achievable end-to-end target. **Do not plan past this
   point on projections** — every projection made in this project so far
   (κ₃-rank, oracle "fixed point", mixture closure) has been wrong; the
   measurements have been right.

## 7. Progress log (append-only)

- **2026-07-12 (later)** — E5a/E5b done with all four ledgers (ours-closure,
  v4, pluto, shiv, andrew) on the named public 50. **Router idea RETRACTED**
  (oracle routing to sampling: +0% for andrew, +2.5% pluto — their analytic
  already beats sampling 46-49/50). Oracle fusion only 1.2-1.4× and compute
  eats it. Analytic error vs intrinsic variance: ours 0.465, pluto 0.371,
  shiv 0.348, **andrew 0.028** (variance-independent ⇒ structurally different
  from pluto/shiv). High-variance MLPs are EASY for good closures.
  **Verdict: the entire 3.7× gap is 11.8× of raw accuracy in the primary
  closure (§6.0). All sampling-side work deprioritised. E2 is the only thing
  that matters; E1/E3/E4 serve it.** My own E5a router claim was a
  wrong-counterfactual error — logged as a process failure, same class as the
  ones found in the audit.
- **2026-07-12** — Audit of IMPROVEMENTS.md completed (this doc). Key
  reversals: R7 pivot evidence was a cherry-pick (best-vs-mean); oracle
  fixed-point demoted to RETEST; κ4 diagnoses rejected as self-contradictory;
  "never leave the floor" scope-limited. Leader forensics from pasted
  dashboards: pluto = pure-analytic existence proof (9.3e-7 @ 2.3e10 FLOPs,
  final-layer-only, 40 s wall); andrew = hybrid-shaped with a fallback path
  that rescues hard MLPs; top-3 within private-split noise of each other.
  Decisions: hybrid-default with a one-week analytic gate (E2/E3); κ4 only
  machine-verified or learned; polynomial-CV family likely capped ~1.05×
  pending E1. Next: E0 re-baseline, then E2.
