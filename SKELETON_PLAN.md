# SKELETON PLAN — shippability-first build of the κ₃/κ₄ closure

*2026-07-12. Scope: derisk items 1 (walking skeleton) and 2 (torch→flopscope
gap / op coverage) ONLY. Distribution-matching, probe submissions, and
submission-hygiene items are deliberately out of scope for now and tracked in
the conversation log, not here. This document is subordinate to STRATEGY.md's
rules of evidence.*

---

## 0. Why this exists

Three past failures shared one root cause: **the physics was validated in
torch first and priced/ported last** (147%-of-budget complete closure; 2.7×
FLOP miscount; einsum-symmetry rejection killing 100/100 MLPs). The fix is to
invert the build order:

> **Build the shippable flopscope implementation of the target architecture
> FIRST, with crude/random coefficients. Validate cost, ops, and failure
> handling. Only then pour physics into it.**

An accuracy number without a flopscope-measured cost attached is a projection.
Projections lose here; this plan makes them structurally impossible.

## 1. The target architecture the skeleton must exercise

From the κ₃/κ₄ plan (see conversation 2026-07-12), the shippable estimator is:

1. **(μ, Σ) propagation** — complete-formula K=2 update (einsum with symmetric
   tag, per shipped `estimator.py` idiom).
2. **Lazy-CP κ₃ carry** — carried direction matrix `V (n × R)` + coefficients
   `c (R,)`; per-layer transport `V ← Wᵀ(ĥ₁ ∘ V)` (one n×n·n×R matmul);
   new source directions born each layer; periodic QR+SVD recompression
   capping R at R_max.
3. **Source-family dictionary** — the star slice exactly (4 matmuls, built
   with matmul+reshape, never einsum-with-repeated-operands), plus S
   additional n³-shaped family contractions (skeleton: placeholders with
   random coefficient tables; later: the offline-pruned renormalized
   dictionary).
4. **Slice extraction** — κ₃(z)_ppq from the carry: `(V∘V)·diag(c)·Vᵀ`.
5. **Learned pointwise correction (stub)** — tiny elementwise MLP on per-pair
   features (t_p, t_q, ρ_pq, slice, star), implemented as (n², F)@(F, h)
   matmuls + tanh; PLUS a 1-D table-interp variant (`fnp.interp`) to prove
   table lookups price correctly. Random weights for now (shipped via
   `flopscope.Module` npz later — 0-FLOP load).
6. **κ₄ learned-local slice (stub)** — same pointwise machinery, second head.
7. **Guard + fallback** — per-layer cheap invariant checks; on trip, bail out
   immediately and rerun plain calibrated-K2 from scratch (§3).

## 2. Item-2 audit: flopscope op coverage (DONE, 2026-07-12)

Read from `flopscope/src/flopscope` source (0.8.0rc5):

| need | flopscope support | verdict |
|---|---|---|
| table interp / gather | `interp`, `take`, `take_along_axis`, `searchsorted`, `digitize`, `where`, `piecewise` in `_array_ops`/`_pointwise` | ✅ covered |
| elementwise MLP (tanh/exp/erf) | counted unary ufuncs + `flops.stats.norm.pdf/cdf`, `stats/_erf` | ✅ covered |
| QR / SVD / eigh for recompression | `numpy/linalg/_decompositions.py`, `_svd.py` | ✅ covered (price it, don't assume) |
| Python branching on computed values (guards) | `FlopscopeArray` is a plain `numpy.ndarray` subclass — `bool()`, `float()`, `.item()` work; only *executed* ops are charged. Shipped `estimator.py` already does `bool(fnp.all(fnp.isfinite(...)))` and it has run 0/100-failure on the grader | ✅ proven |
| 0-FLOP artifact load | `flopscope.Module.from_file` (shipped code uses it) | ✅ proven |
| einsum on repeated operands | **KNOWN TRAP** — symmetry inference rejects partially-symmetric outputs | ⛔ rule: matmul+reshape for all κ₃/κ₄ contractions; einsum only for the symmetric covariance update (proven idiom) |

Residual unknowns (accepted for now, out of scope per user): grader residual
wall-time multiple (~6× local, historical) for elementwise-heavy kernels —
measured only by a probe submission, deferred.

## 3. Fallback design (the "could fallbacks bite us later?" answer)

Mechanically fallbacks are safe (branching is proven, only executed ops are
charged). The three real bite modes, and how the skeleton designs each away:

1. **Double-spend past the floor (the real danger).** If the primary spends
   ~9% and *then* trips, running a QMC fallback (~10%) totals ~19% → the
   multiplier doubles on that MLP; past 100% the MLP zeroes at ×1.0.
   **Design rule: primary budget ≤ ~7.5% of B, fallback = calibrated-K2
   (0.8%), so worst case ≈ 8.3% — still under the free 10% floor. The
   fallback is then literally free.** Never make full-budget QMC the trip
   target; QMC is a *fusion component* decided up front, not a panic path.
2. **Silent masking.** A fallback that trips often turns "the closure is
   broken" into "the score is mysteriously mediocre" — invisible unless
   counted. **Design rule: the trip decision must be loggable offline (mini
   split runs report trip rate) and the trip rate on mini must be ≈ 0–2%.
   A trip rate above 5% is a red build, not a working safety net.** (On the
   grader we can't log, but per-MLP score + utilization signatures in the
   ledger reveal trips: a tripped MLP shows K2-magnitude MSE ~1e-4.)
3. **False positives eating accuracy.** Over-tight envelopes trip on healthy
   but unusual MLPs — exactly the high-variance MLPs where a good closure
   SHINES (E5b: bruce-mckenzie is sampling's worst but pluto's best).
   **Design rule: guards check only physics-violating states (non-finite,
   Σ diag ≤ 0, magnitude > envelope with envelope set at ≥ 10× the corpus
   p99), fitted offline on ≥ 1000 generated MLPs, never hand-tuned.**

Also: guards run every layer but cost ~n² comparisons (~1e5 FLOPs — noise),
and bail-early means a trip at layer k wastes only k/32 of the primary's
spend. Conclusion: **fallbacks don't bite if (a) primary + fallback fits
under the 10% floor, (b) trip rate is measured on mini, (c) envelopes are
generous and offline-fitted.** All three are structural properties of the
skeleton, not promises.

## 4. Skeleton acceptance gates (all must pass before ANY physics work)

- **SK-1 (cost):** measured total FLOPs (BudgetContext) ≤ 7.5% of B=2.72e11
  with the full op profile running (carry R_max=128, star + S≥8 families,
  both pointwise stubs, guards). Report a per-component ledger.
- **SK-2 (ops):** every op in §1 executes under flopscope with no symmetry
  rejections and no silent numpy fall-through warnings.
- **SK-3 (fallback):** force-trip the guard mid-run; the K2 fallback completes,
  total cost stays < 10% of B, output is finite.
- **SK-4 (contract):** wired into `whest-starterkit/estimator.py` shape:
  `whest validate` passes; mini-split run completes 0/100 failed. (Score is
  expected to be BAD — coefficients are random/crude. Score is not a gate
  here; only shippability is.)
- **SK-5 (wall):** local wall time per MLP recorded (grader multiple deferred,
  but > a few seconds local on numpy is a flag to investigate early).

Status: SK-1/SK-2/SK-3 exercised by `research/skeleton_v0.py`; SK-4 is the
next step after SK-1..3 are green.

### Measured results (2026-07-12, `research/skeleton_v0.py` + `whest-starterkit/estimator_skeleton.py`)

- **SK-1 PASS — 6.66% of B** (1.81e10 FLOPs), full op profile. Ledger:
  families 3.95%, star 0.79%, (μ,Σ) 0.61%, carry 0.60%, recompress 0.32%,
  slice extract 0.19%, pointwise stubs 0.19%, guards ~0, table load 0.
  Wall 0.42 s/MLP local. Headroom to the 7.5% cap: 0.84%; to the floor: 3.3%.
  - First run measured **9.22%**: the placeholder families used 2 matmuls each
    instead of sharing the `Σ·diag(h)·W` bases. CSE alone bought 2.6% of B —
    the plan's CSE rule, demonstrated by the ledger before any physics exists.
  - Re-tagging cov with `flops.as_symmetric` after the update reclaimed the
    symmetric-einsum credit: (μ,Σ) cost 0.79% → 0.61%. Rule: re-tag cov every
    layer (symmetrize first — `as_symmetric` VALIDATES and raises otherwise).
- **SK-2 PASS** — every §1 op runs under flopscope: `interp`, QR+SVD,
  pointwise tanh-MLP on (n²,F), `stats.norm.pdf/cdf`, Python-branch guards,
  matmul+reshape star. No symmetry rejections. Symmetry-loss warnings audited:
  cov re-tagged (credit-bearing); rho/feats/dmu/corr losses are elementwise
  consumers with no credit at stake (suppressed with a comment; re-enable when
  porting new contractions).
- **SK-3 PASS** — forced NaN at layer 16: guard tripped at layer 17, bail-early
  + K2 fallback (0.60%), total **4.05% of B**, finite output. The fallback is
  free under the floor, as designed.
- **SK-4 PASS** — `whest validate` passes (its tiny 4×2 probe MLP exercised
  the shape-fallback path). Mini split (100 MLPs, subprocess runner):
  - **0/100 failed**; mean utilization **6.995%**, multiplier pinned at 0.10.
  - No guard trips (a tripped MLP would read ~1.5% utilization; mean ≈ the
    full-path cost on every MLP).
  - Residual wall time 0.91 s TOTAL / 100 MLPs ≈ 9 ms/MLP local; even at the
    historical ~6× grader multiple that is ~0.2% of B. Flopscope backend time
    1.07 s/MLP (free). Effective compute 1.90e12 vs raw 1.81e12 across the run.
  - Raw final-layer MSE 7.83e-5 ≈ plain-K2 level — exactly as expected with
    random coefficients and EPS-scaled corrections; NOT a gate.

**ALL GATES GREEN. The shell is shippable. Physics work (pruned dictionary,
κ₄ regressor, real coefficients) now pours into a container whose cost, ops,
guards, and harness contract are already proven.** Next build steps live in
the κ₃/κ₄ plan (gates G1–G4), not here.

## 4b. Compute hygiene on this machine (M-series, 128 GB unified)

Added after the sampler turned out to be the wall (see STRATEGY §7, 2026-07-12
tooling entry). Rules for every offline experiment from now on:

1. **Contract on the GPU, accumulate in CPU float64.** Never move per-sample
   (chunk × n) tensors to the CPU to contract them there. Move the (n,n)
   *result*. The old path burned 1.7e13 CPU float64 FLOPs per MLP with the GPU
   idle (142 s/MLP → 4 s/MLP at the same N).
2. **Keep accumulators on-device; flush periodically.** A `.cpu()` inside the
   per-layer loop is a pipeline stall. Per-layer × per-chunk transfers cost 5k
   stalls per batch (69 s/MLP even after fix 1). Flush every ~4 chunks.
3. **Use the memory.** 128 GB unified ⇒ batch MLPs through `bmm` (M=6,
   chunk=500k ≈ 15 GB live). Bigger batches = larger matmuls = a saturated GPU.
4. **Cache sampled artifacts to disk** (`data/slice_cache/`), keyed by
   (mlp seed, N, sample seed). Gates share them; nothing is sampled twice.
5. **Any speed rewrite must be validated for BIAS, not just speed**: rerun the
   old path on the same weights and check the new-vs-old difference sits at the
   new-vs-new sampling-noise floor (`fastmoments.py --validate`). Speed
   rewrites that quietly change numbers are exactly the class of error the
   rules of evidence exist to catch.
6. **Quote the target-noise floor with every learned-slice number** (G1b). At
   N=2e6 the `k4_iiij` target is 80% noise at L1 — a "45% held-out error" there
   was mostly measuring the corpus, not the model.

## 5. What deliberately stays OUT of the skeleton

- Real diagram coefficients / the pruned dictionary (G4, offline).
- κ₄ regressor training, pointwise-MLP training (offline; slots into the
  proven stubs).
- Distribution-matched training MLP generation (out of scope per user).
- Probe submission / grader wall measurement (out of scope per user).
- Any accuracy claims. The skeleton's only claims are cost and shippability.
