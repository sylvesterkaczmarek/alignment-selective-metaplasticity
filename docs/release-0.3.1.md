# Version 0.3.1

This patch makes the checkpoint scorer more informative for researchers using their own models. It also verifies the published synthetic control alongside the open-weight study. [Scoring guide](https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/blob/v0.3.1/docs/authority-scoring.md) · [Technical report](https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/releases/download/v0.3.1/authority-retention.pdf)

## Scoring changes

The report now includes baseline qualification and the share of counterfactual pairs answered correctly under both authority conditions. Two models with the same authorised and unauthorised accuracies can have different paired accuracy. The existing v1 manifest, JSONL inputs and `hidden_failure` field remain compatible.

The new `hidden_retention_failure` field identifies a qualified model that temporarily falls below the failure floor and recovers by the endpoint. It is null if initial qualification fails or the run has missing predictions. A run that ends below the release floor has a visible endpoint failure. Separate report paths protect input predictions, manifests and published research evidence from being overwritten.

## Evidence replay

On the recorded Linux CPU runtime we retrained all 168 development and 168 held-out adapters from the cached frozen-backbone features. Every recorded prediction, capability metric and final weight hash matched. Only training durations differed. All 84 synthetic controls were regenerated from their seeds, and the complete uncompressed journal matched byte for byte.

Four synthetic runs passed both endpoints and failed in between. The affected runs were EWC seed 811 and rehearsal seeds 801, 823 and 829, all in the overlapping-feature condition. Across twelve held-out blocks per open-weight family, neither family qualified at baseline. The open-weight study therefore supplies no estimate of retention after mastery. The report clarifies when the open-weight rehearsal updates were measured.

From the checkout, run `python scripts/verify_authority.py` to check the evidence hashes, validate coverage and recount the outcomes. Run `asm-score demo` to check a hand-scored example without downloading a model.

## Training and citation fixes

Integer importance maps retain fractional values after normalisation. Importance refresh checks parameter names, shapes, devices and finite nonnegative values before combining maps. The versioned software citation is included in the source distribution.
