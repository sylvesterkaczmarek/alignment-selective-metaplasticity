# Research integrity update

## Candidate and audit scope

Candidate 0.2.0rc1, prepared on 5 November 2026. Base commit 439c379.
This implementation pass prepares a draft pull request. Independent review,
merge and publication remain separate. The baseline passed 142 tests.
Fifteen separately written synthetic checks reproduced failures before repair.
They are retained as regression cases, with the original failing log in the local
review evidence. This is an implementation audit, not outside certification.

## Confirmed corrections

| Area | Reproduced problem | Correction |
| --- | --- | --- |
| Protection maps | Missing or misspelled parameters silently lost protection; broadcasting and negative scales were accepted. | Require complete, exact-shape, finite parameter mappings, bounded scales and binary masks before any update; own detached snapshots. |
| EWC inputs | One-sided penalty arguments and a negative coefficient silently disabled the intended penalty. | Validate the coefficient and both full mappings before training. |
| Importance measurement | Existing gradients and mixed module modes were overwritten; the loop fetched an extra batch; empty evidence became zeros. | Use autograd gradients in evaluation mode, preserve individual modes, bound iteration before fetching and reject empty evidence. |
| Evaluation | Measurement changed model modes and could silently score nonfinite logits or malformed labels. | Restore every module mode even on errors and validate logits/labels. |
| Checkpoints | A malformed late RNG field could reject a load after overwriting live weights; trainability changes were unnoticed. | Schema 2 records trainability, stages model/optimizer and RNG validation, checks moment shapes and supports rollback of commit failures. |
| Trial selection | Repeated copies of one development cell could satisfy the repetition count. | Require unique declared cells and matching result identities; recompute selection from its bound journal before test access. |
| Reporting | Mean loss gave a short final batch the same weight as a full batch. | Weight loss diagnostics by examples. Invalid JSON results cannot replace existing artifacts. |
| Study lifecycle | Reusing a StudyRun repeated training and appended another trajectory. | Single-use lifecycle and detached result/configuration snapshots. |

## Compatibility

Complete valid built-in protection maps keep their update rule. Partial maps,
unknown keys, broadcasting masks and nonbinary freeze masks are now rejected.
Use explicit ones for trainable entries and zeros for frozen entries.
The EWC coefficient still multiplies half the weighted squared displacement.

Batch-gradient importance still squares a minibatch mean before averaging and
remains distinct from a per-example empirical Fisher. It now measures the model
in evaluation mode, as the per-example estimator already does. For custom models
with dropout or batch normalisation this is an intentional measurement change.
The included MLP and transformer do not use dropout. The original batch proxy's
equal batch weighting and quantile normalisation remain explicit.

Checkpoint schema 1 omitted trainability and is rejected by the new exact-restart
loader. Preserve old checkpoints and use their original source environment to
read them. Do not relabel a schema-1 file as schema 2. The contract remains CPU,
standard TensorDataset/DataLoader, epoch boundaries, explicit independent loader
generators, standard SGD or AdamW, and complete caller-supplied provenance.
Do not load untrusted checkpoint files. Restricted torch loading does not prevent
all denial-of-service or memory hazards. Neither these snapshots nor atomic JSON
replacement establish crash durability of an entire experiment.

Historical numerical results are preserved. Twelve newly executed reference runs
at the base commit and twelve at the candidate, in the same environment, match
exactly in behaviour, drift, histories and bypass metrics. Loss diagnostics differ
intentionally because their averaging was corrected. This is within-environment
parity, not a claim that different PyTorch builds produce identical arithmetic.

## Research follow-up

The calibrated transformer design was recorded before development and test
execution. Settings were selected only from development outcomes. All four test
cells remain in every complete comparison. The same design was subsequently
replayed with finalized candidate source, without changing settings or choosing
new cells after seeing results. That replay is reproducibility evidence, not four
additional independent test cells. See calibrated-transfer.md.

## Verification boundaries

The suite covers protection, failed checkpoint loads, exact SGD/AdamW restart,
unique-cell selection, paired comparisons, invalid inputs and report consistency.
CI checks Linux Python 3.10 through 3.14, macOS Python 3.13, dependency advisories
for the recorded reference versions, and a freshly installed wheel outside the
checkout. The reference requirements record actual versions; they are not a
hash-pinned cross-platform supply-chain lock. The static audit does not cover
advisories unknown to its database or every independently resolved environment.

Primary references for the implementation choices:

- PyTorch reproducibility guidance: https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch restricted serialization: https://docs.pytorch.org/docs/stable/notes/serialization.html
- Fisher-estimation distinctions: https://arxiv.org/abs/2502.11756
