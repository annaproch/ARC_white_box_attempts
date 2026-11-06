# ARC White-Box Estimation: approaches and experiments

Research code for estimating the expected activations of a random ReLU MLP under Gaussian input, given its weights and a limited compute budget. Most experiments use the Phase 1 architecture: width 256, depth 32, He initialization, and no biases. The challenge rewards both low final-layer mean-squared error and low compute cost.

## Main approach

The current implementation propagates **means, full covariances, and a compressed third-order cumulant** through the network. Means and covariances provide a Gaussian approximation; third-order cumulants capture departures from Gaussianity that affect the next ReLU update.

We use a **Tucker representation** for the carried cumulant: a small core and a shared basis approximating the leading covariance eigenspace. For width `n` and rank `R`, its storage is `O(nR + R³)`, rather than `O(n³)` for the full third-order tensor. Newly generated cumulant contributions are evaluated through matrix contractions without constructing the full tensor.

The analytical update is **calibrated with learned per-layer coefficients**, trained through a differentiable version of the propagation to minimize final-layer mean error against Monte Carlo targets from random networks. This jointly adjusts the mean, covariance, and cumulant updates. A central finding was that improving an intermediate cumulant in isolation can make the final prediction worse: errors in the approximate updates can cancel, so component changes need end-to-end validation and joint refitting.

The checked-in `v5.npz` has third-order rank **48**, the separate learned slice correction ("anchor") **off**, fourth-order propagation **off**, and QMC blending **off**. Fourth-order support remains in the code, with a stored rank of 8 that is inactive in this configuration. The normal path is analytical; the estimator retains a QMC fallback if the closure is unavailable or fails its output guard.

This configuration is verified from the artifact's metadata and matches [its provenance](whest-starterkit/v5.npz.PROVENANCE). The older header in `estimator.py` describes a previous κ₄-enabled artifact. The current artifact's provenance explicitly says it was not yet grader-validated; this repository does not establish a final competition score for it.

## Where to start

| File | Purpose |
| --- | --- |
| [whest-starterkit/estimator.py](whest-starterkit/estimator.py) | Current estimator entry point, parameter loading, guard, and sampling fallback. |
| [whest-starterkit/closure.py](whest-starterkit/closure.py) | Analytical moment updates, Tucker propagation, and calibration coefficients. |
| [whest-starterkit/v5.npz](whest-starterkit/v5.npz) | Learned coefficients, basis initialization, and configuration. |
| [whest-starterkit/points.npz](whest-starterkit/points.npz) | Stored quadrature points for sampling and fallback. |
| [research/](research/) | Experimental implementations, fitting scripts, and diagnostic comparisons. |
| [IMPROVEMENTS.md](IMPROVEMENTS.md), [STRATEGY.md](STRATEGY.md) | Chronological research notes, including unsuccessful experiments and superseded plans. |

## Experiments we tried

| Approach | What we tested and learned | Code |
| --- | --- | --- |
| Gaussian moment propagation (K=2) | Track means and covariances analytically. Cheap, but approximation errors accumulate through deep networks. | [Baseline study](research/kprop_depth32.py) |
| Quasi-Monte Carlo | Scrambled Sobol points transformed to Gaussian inputs, with moment matching and spherical normalization. A strong sampling approach; early local tests reported about 3.2× lower error than iid MC. | [QMC comparison](research/qmc_test.py), [point generation](research/gen_points.py) |
| Learned quadrature weights and points | Ridge/Bayesian quadrature weights and gradient-based point optimization. Learned weights stayed close to uniform; point optimization gave little reliable improvement. | [Weights](research/learn_weights.py), [points](research/optimize_points_v2.py) |
| Calibrated K=2 and QMC fusion | Learn bounded corrections to Gaussian propagation, then combine its estimate with QMC. The archived v4 was the strongest submitted variant recorded in the research notes; the later analytical v5 did not establish a better score. | [Calibration](research/calibrated_k2_v2.py), [fusion check](research/fused_v4_check.py), [v4](whest-starterkit/_archive/estimator_v4.py) |
| Tucker cumulant propagation | Carry higher-order structure in a low-dimensional covariance basis. Tested rank, accuracy, and compute tradeoffs; this supplies the current analytical backbone. | [κ₃ propagation](research/k3prop.py), [κ₃/κ₄ calibration](research/calibrate_k3k4.py) |
| CP cumulant representations | Represent cumulants as sums of factored terms. Dense checks exposed algebra and layer-indexing errors in the early implementation; corrected transport worked, but pruning terms lost accuracy and retaining all terms was expensive. These are limitations of the tested implementations and compression rules, not an impossibility result for CP. | [CP verification](research/derisk_cp.py), [pruning](research/g4_prune.py) |
| Learned cumulant corrections | Predict normalized cumulant slices or residuals from standardized means, correlations, propagated cumulants, and fresh contributions. Tried small `7 → 8 → 1` tanh networks and per-layer polynomial ridge models, including refitting on corrected trajectories. Better slice prediction did not reliably translate into better final-layer error; the separate anchor is disabled in the current artifact. | [Neural fitting](research/fit_v5.py), [learnability](research/g5_learnability.py), [anchor fitting](research/a1_anchor.py) |
| Fourth-order corrections | Tested analytical κ₄ carry, learned slices, and their influence on mean/covariance updates. Added complexity and cost did not justify keeping κ₄ active in the current artifact. | [Learned κ₄](research/g1_k4_learn.py), [end-to-end test](research/g1c_learned_k4_endtoend.py) |
| Oracle and diagram diagnostics | Substitute sampled cumulants, verify contraction formulas, and measure which approximation errors matter. These distinguish one-step improvements from improvements after a full rollout. | [Oracle study](research/oracle_cumulants.py), [diagram checks](research/e9_verify_diagrams.py), [error attribution](research/error_attribution.py) |

Reported results above are historical observations on particular local or public evaluations, not a fresh benchmark or a final leaderboard claim. See the notes for the corresponding settings and limitations.

## Running and reproducing

The estimator depends on the competition's `flopscope` and `whestbench` packages. Dependency specifications and a lockfile are in `whest-starterkit/`:

```bash
cd whest-starterkit
uv sync --frozen
```

Use the harness with the **directory containing all four runtime files**: `estimator.py`, `closure.py`, `v5.npz`, and `points.npz`. Passing only the Python file can omit the learned parameters and change the behavior. The bundled [starter-kit documentation](whest-starterkit/README.md) explains the original harness, but some standalone-demo instructions predate our replacement estimator.

Research scripts additionally use PyTorch and generated moment corpora. [build_moment_corpus.py](research/build_moment_corpus.py) and [gen_corpus.py](whest-starterkit/research/gen_corpus.py) generate different corpora used by different experiments. The corpora are intentionally excluded from Git.

**Training reproduction is incomplete in this checkout.** The local `mlp_cumulant_propagation/` directory is a separate, ignored repository. It contains the differentiable `mirror.py` and `fit_theta*.py` training scripts used for the current calibration; those files are not included when cloning this repository. Some research scripts also retain local paths or depend on external experimental modules. The scripts here are a research record, not a single reproducible training pipeline.

## Attribution and licensing

The competition starter kit is from [AIcrowd](https://github.com/AIcrowd/whest-starterkit); its [MIT license and copyright notice](whest-starterkit/LICENSE) are retained. The research builds on the Alignment Research Center's [work on mechanistic estimation](https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/). The separate reference checkout and generated datasets are not bundled here.

There is currently no root-level license specifying reuse terms for the original research files outside the starter-kit subtree. The upstream MIT notice should not be interpreted as automatically licensing every file in this repository.
