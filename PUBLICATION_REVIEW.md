# Public-repository review — 2026-11-06

**Assessment:** no apparent credentials or private datasets were found in the scanned tracked files and reachable Git history. Publishing the source looks reasonable from that perspective, subject to the privacy and rights limitations below. This is a targeted scan, not a guarantee that every possible secret format has been detected.

## Scope and method

- Inspected all 168 files in the existing Git index and the corresponding working files, plus the new documentation.
- Traversed all locally available refs and HEAD: 62 commits and 256 distinct historical file blobs. Checked commit messages as well as file contents.
- Git could not run because the system developer tools were unavailable. A temporary read-only Ruby inspector decoded loose and packed Git objects, resolved deltas, and verified reconstructed object hashes; it also read the Git index. No history was rewritten.
- Searched for private-key blocks, common provider tokens, credential assignments, credential-bearing URLs, and JWTs. Reviewed matches for local paths, email addresses, confidentiality wording, and credential-like filenames. No credential-pattern matches were found.
- Inspected the numerical archives' contents without executing them or loading pickles. Checked the active artifact's metadata and its matching provenance hash.
- Reviewed the starter-kit MIT license, dependency configuration, nested CI workflow, ignore rules, research corpus generators, and runtime file requirements.
- Did not read protected `.env` files. No such paths appeared in the scanned Git trees or index. Ignored virtual environments, separate repositories, generated datasets, and local tooling are outside the publication assessment; do not force-add or upload the entire working directory.
- Remote-only branches, deleted/unreachable Git objects, GitHub-side artifacts, and account settings were not inspected. Normal branch publication uses reachable history, but this review does not cover a raw backup of `.git` or the entire workspace.

## Findings

| Finding | Implication |
| --- | --- |
| Commit author/committer identities contain a username, local hostname, and private LAN address. | Publishing existing history exposes these. They are not credentials. Use a sanitized history or a fresh public snapshot if you do not want those details public; changing current Git configuration does not alter old commits. |
| Two research scripts contain user-specific absolute paths; upstream documentation also includes example local paths. | Minor privacy exposure and a portability problem. See `research/p1_diag_carry.py` and `research/p1_step1_k4accuracy_sweep.py`. Removing them in a new commit would not remove historical copies. |
| No tracked credential files, logs, sampled corpora, or separate reference checkout were found, including the inspected historical paths. | Keep the existing exclusions. `.gitignore` was extended with `.env`, `.env.*`, private-key extensions, `.netrc`, and `.pypirc`. Ignore rules do not remove already committed content. |
| No current tracked file exceeds 10 MB. | No obvious large-file publication obstacle in the current index. |
| The starter kit includes an MIT license and AIcrowd copyright notice. | Preserve that notice when publishing the copied/modified starter kit. |
| No root-level license covers the original research files outside the starter-kit subtree. | This does not itself prevent publishing your own work, but public visibility does not grant an open-source reuse license. Select explicit reuse terms before presenting the entire project as open source. |
| The separate ignored `mlp_cumulant_propagation/` checkout contains the current differentiable mirror and fitting scripts. | A public clone will not reproduce current training by itself. The README now states this limitation. Adding that code later requires its own review and preservation of applicable upstream notices. |
| The active `v5.npz` disables κ₄; the estimator header describes an older κ₄-enabled configuration. | The new README follows the actual metadata and provenance, and does not claim the older public score for this artifact. |

The user reports that the competition has finished. The official [challenge rules page](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules) was checked, but its retrieved content did not include the rules themselves. Consequently, this review does **not** verify post-competition redistribution conditions or any separate agreement accepted by participants. No embargo/confidentiality wording was found in the scanned source history; that is not evidence about an external agreement.

## Changes and validation

Added a project README covering the current algorithm, experimental approaches, artifact configuration, limitations, and attribution. Added pointers from the starter-kit and research READMEs, and credential exclusions to `.gitignore`. No estimator behavior, learned weights, Git history, or repository visibility was changed.

Validated relative documentation links and the artifact metadata. Runtime tests were not run: these are documentation/ignore-rule changes, and the local Python runtime could not load its library in this environment. No repository was published or pushed.
