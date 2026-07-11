# Releasing whest-starterkit

> [← Documentation](README.md)

Single source of truth for release operations. Stay current.

## Cadence

- **Patch** (`v1.0.0` → `v1.0.1`): doc fixes, typo fixes, example improvements.
- **Minor** (`v1.0.0` → `v1.1.0`): when whestbench bumps to a new minor.
- **Major** (`v1.0.0` → `v2.0.0`): when participant-visible workflow changes.

## Routine release

1. **Verify CI is green.** All Stage 1-3 smokes pass before release.
2. **Refresh and review dependencies.** Run `uv lock` after dependency updates and inspect release notes when either `whestbench` or `flopscope` has semantically significant changes.
3. **Run `python estimator.py` locally** to sanity-check the default template output after dependency refresh.
4. **Banner sweep:** if the *previous* release added a "removed example" banner to README, delete it now (retire-after-1-release rule).
5. **Banner add:** if this release removes or renames any example or doc, add a banner block at the top of README:

   ```markdown
   > ⚠️ `examples/02_mean_propagation.py` was renamed to `examples/02_propagation_v2.py` in vX.Y. Update bookmarks.
   ```

   This banner stays for ONE release, then is deleted by the *next* release.
6. **Merge.**
7. **Tag and push:**

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
8. **Cut a GitHub Release** with the changelog entry.

## Surface-contract changes (major bump)

If this PR changes any of:

- `BaseEstimator.predict` signature
- `MLP` dataclass field names
- `local_engine.{build_mlp, monte_carlo_layer_means, compare_against_monte_carlo}` signatures

→ This is a **major bump**. Coordinate any new contract through [docs/reference/estimator-contract.md](reference/estimator-contract.md) until a separate contributing guide lands.

## v1 launch sequence (one-time)

See spec §9.3 in `whestbench-public/.aicrowd/superpowers/specs/2026-04-24-whest-starterkit-design.md`.
