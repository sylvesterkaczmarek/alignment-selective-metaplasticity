# Version 0.3.0

This release adds a checkpoint scorer for paired authority decisions, an open-weight adapter study and corrected experimental data streams. Read the [technical paper](https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/blob/v0.3.0/docs/authority-retention.pdf), [methods and raw evidence](https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/blob/v0.3.0/docs/authority-study.md), and [external scoring guide](https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/blob/v0.3.0/docs/authority-scoring.md).

## Research outcomes

SmolLM2-135M-Instruct and Qwen2.5-0.5B-Instruct completed 168 development and 168 held-out adapter runs. Zero of twelve held-out blocks per family met the initial authority requirement. These runs cannot estimate forgetting conditional on prior mastery. Four of 84 synthetic control runs passed both endpoints and exhibited a temporary failure. All results, including infeasible candidates, remain in the journals.

## Compatibility

The command `asm-score demo` checks a bundled example. `asm-score score` consumes a separate manifest and JSONL predictions. It requires no model download. The optional `openweights` dependency group is used for fresh backbone feature extraction.

The historical three-seed reference continues to use its original sample offsets. New controlled MLP and calibrated transformer runs use namespaced streams, verify realised data overlap and use updated study identifiers. To reproduce the exact historical study schedules, use tag `v0.2.0`. The 0.2.0 evidence remains available under its original paths. Checkpoint schema 3 and the Python import name are retained.

The semantic verifier now checks harmonic means, neutral accuracy, acquisition, earlier retention, task order and endpoint agreement when validating complete research journals. Calibrated transformer ties use candidate order. Controlled MLP ties retain acquisition as the secondary criterion. Global importance normalisation preserves float64 range, and intervention probes restore the declared trainable parameters on a partially frozen model.

## Verification

Use `python -m pytest -q`, `python scripts/verify_authority.py`, `python scripts/verify_calibration.py results/calibrated-transfer-v2`, and `asm-score demo`. Distribution verification compares packaged fixtures as well as Python modules and runs the scorer from outside the checkout.

## Corrected data-stream replications

All 104 development and 78 held-out MLP runs and all 20 development and 20 held-out feature-transformer runs completed under the new independent streams. In the six MLP blocks, fixed selective protection reached 0.861 mean final alignment and 0.969 acquisition, versus 0.649/0.967 for shuffled and 0.671/0.971 for uniform. Refreshing importance yielded 0.861/0.969. In the four calibrated feature-transformer blocks, fixed selective protection reached 0.429/0.938, EWC 0.526/0.888, shuffled 0.333/0.854 and uniform 0.352/0.829. Acquisition differed across methods, which limits comparisons of alignment at matched utility. The tables in the old study guides describe their original protocols. [Corrected MLP summary](https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/blob/v0.3.0/results/controlled-v2/test-summary.json) · [Corrected feature-transformer summary](https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/blob/v0.3.0/results/calibrated-transfer-v3/summary.json)
