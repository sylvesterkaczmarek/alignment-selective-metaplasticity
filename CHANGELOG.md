# Changelog

## 0.2.0

5 November 2026. Independently reviewed research-integrity release.

- Validate raw optimizer state and per-group settings before loading or implicit casting.
- Preserve nonpersistent model buffers and their alias topology in checkpoint schema 3.
- Reject changed/custom batch-sampler semantics outside the exact CPU restart contract.
- Reject protection tensors that become nonfinite when converted to parameter precision.
- Validate finite, duplicate-free JSON and enforce packaged experiment checksums and provenance.
- Retain 48 separately authored review cases, bringing the complete suite to 258 tests.
- Reproduce all 40 recorded calibration runs without changing the original four evaluation cells.
- Publish verified GitHub distributions and verify public downloads without altering historical evidence.


## 0.2.0rc1

Unpublished research-integrity candidate, 5 November 2026.

- Reject incomplete, misshapen, nonfinite and out-of-range protection arguments before updates.
- Preserve model modes and gradients during importance estimation; enforce exact probe budgets.
- Validate evaluation outputs and sample-weight loss diagnostics.
- Add schema-2 checkpoint validation, trainability binding and staged restore.
- Validate unique experimental cells and journal-bound development selection.
- Add paired-cell uncertainty, a calibrated transformer follow-up and a one-seed demonstration.
- Add recorded reference dependencies, pinned CI actions and isolated distribution verification.
- Preserve all previous checked studies and explicit negative results.

Separate independent review is required before merge or release.
