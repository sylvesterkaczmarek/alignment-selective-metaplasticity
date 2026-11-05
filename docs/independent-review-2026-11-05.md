# Independent verification

## Reviewed candidate

Reviewed source 2393c326f822be6b0cc5fc90221d384d5b8d3937, version 0.2.0rc1,
from a fresh clone and fresh Python 3.13.3 environment with recorded reference
packages. Its original 210 tests passed. A newly authored suite initially contained
43 cases: 22 passed and 21 exposed the findings below. The original failing tests
and output are retained in the local review evidence, unchanged.

This is a separately executed review and independent test design within this
workflow. It is not external third-party certification.

## Findings and repairs

| Finding | Reproduction | Repair |
| --- | --- | --- |
| Incomplete optimizer validation | Missing AMSGrad state, wrong moment dtype, invalid group settings and missing fields could be accepted, sometimes failing only on a subsequent update. | Validate raw tensors and each group's actual native settings before state loading/casting. |
| Incomplete exact-restart model and loader identity | Nonpersistent buffers were lost; explicit or altered batch samplers could pass with misleading loader metadata. | Checkpoint schema 3 preserves all registered buffers and rejects unsupported/changed effective batching. |
| Unchecked precision conversion | Finite float64 protection values could become infinity in float32. | Validate converted tensors before returning a protection mapping. |
| Incomplete evidence-file verification | Corrupted provenance or a plot could pass because the supplied checksum manifest was not checked; nonfinite journal extras were accepted. | Enforce finite duplicate-free JSON, exact manifest membership, artifact hashes and provenance agreement. |

The malformed-container cases also now raise defined ValueError failures before
changing model, optimizer or RNG state. Five additional tests cover the complete
valid AMSGrad/buffer restart, missing integrity records, an independent EWC formula
and an independent NumPy recalculation of reported held-out statistics.

## Verification executed

The repaired suite contains 258 tests, including 48 independently authored review
cases. Valid SGD/AdamW restart was checked with dropout, shuffling and protection
coefficients 0, 0.35 and 1. A native-softmax formula independently verifies the
model Fisher estimator. Positive controls preserve valid state and updates.

The complete calibrated experiment was rerun in the fresh environment before
repair: all 20 development and 20 held-out records matched the candidate in every
non-timing field, including data hashes, final weights, costs and histories.
A final-source reproduction is recorded separately in the release evidence.
The test cells remain the original four; repeated executions are not new samples.

The repairs do not change valid built-in training updates. Prior result files
remain byte-identical. No setting, seed or outcome was replaced in response to
an unfavourable result. The selective transformer's near-total loss for model
seed 601 remains visible; the checked alignment score is approximately 0.00778.
Paired intervals versus shuffled and uniform controls include zero. EWC retains
more alignment while falling below the held-out acquisition floor.

## Reproduction and release gates

Run the complete suite and tests/test_independent_release_review.py in a fresh
checkout. Run scripts/verify_calibration.py against the checked journals, then
rerun both phases of experiments.calibrated_transfer into an absent directory.
The artifact verifier checks recorded evidence; it does not itself train models.

Release gates cover Linux Python 3.10 through 3.14, macOS Python 3.13, lint,
dependency advisories, built-wheel/source equality, a fresh non-editable package
installation and verification of public GitHub release downloads. Exact results
belong to the tested source/runtime; cross-version bitwise reproducibility is not
promised. The release distributes via GitHub; no PyPI ownership or upload is assumed.

See migration-0.2.md for the schema transition and remaining restart boundaries.
