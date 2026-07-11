# Concepts — Why this challenge exists

> [← Documentation](../README.md)

Background reading. These three docs explain the problem framing, the scoring metric, and how ground truth is generated. Helpful before you start tuning; essential before debating leaderboard outcomes.

| Doc | What it covers |
|---|---|
| [problem-setup.md](problem-setup.md) | The MLP architecture, He initialization, and the research framing — why "competing with sampling" is the milestone this challenge targets. Includes a "Further reading" pointer to the relevant ARC posts and papers. |
| [scoring-model.md](scoring-model.md) | How the leaderboard score is computed: ASCII pipeline diagram, explicit equation block for `adjusted_final_layer_score` and `all_layers_mse`, behavior when the FLOP budget is exceeded, and a calibration table from the bundled examples. |
| [ground-truth.md](ground-truth.md) | How the evaluator generates the reference values you're scored against — Monte-Carlo sampling, sample counts, the inherent noise floor. |

Read in order if you want the full picture. **At minimum, skim `scoring-model.md`** — it's what drives every number you'll obsess over.

## ➡️ Where to look next

- Ready to write code? → [Tutorial: Stage 1](../getting-started/stage-1-standalone.md), [How-to: write-an-estimator](../how-to/write-an-estimator.md).
- Want the exact API contract? → [Reference: estimator-contract](../reference/estimator-contract.md).
