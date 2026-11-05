# Alignment-Selective Metaplasticity 0.2.0

This release improves the integrity of a controlled PyTorch continual-learning
experiment. It includes complete protection-map validation, non-destructive
measurement, strict experiment-cell identity, journal-bound selection, paired-cell
uncertainty and a development-calibrated transformer follow-up.

A separate verification pass reproduced all 40 recorded study runs and found
additional gaps in optimizer-state validation, buffer/batching restart identity,
precision conversion and artifact verification. Those findings are repaired and
covered by 48 independent checks. The complete suite contains 258 passing tests.

## Research interpretation

The calibrated transformer remains inconsistent across four held-out cells.
Selective metaplasticity has alignment 0.683 ± 0.462 and acquisition 0.942 ± 0.014.
EWC retains alignment 0.892 ± 0.034 with acquisition 0.878 ± 0.067. One selective
cell loses nearly all alignment, and descriptive paired intervals against shuffled
and uniform controls include zero. The results do not establish reliable transfer
or superiority at matched utility. Every original outcome remains in the evidence.

## Compatibility

Checkpoint schema 3 preserves nonpersistent buffers and binds effective default
batching. Earlier checkpoint schemas require their original code. Partial or
unrepresentable protection mappings are rejected. Read docs/migration-0.2.md.
The Python import remains alignment_metaplasticity.

## Release contents

Verified wheel and source distribution, exact release-source archive, checksums,
distribution verification and review/reproduction evidence. The public release
files are checked again after publication. GitHub is the distribution channel for
this release; package-index publication has not been configured.

This is a small synthetic classifier study. No human participants, language models,
external model service or paid inference are used. No general safety, tamper
resistance, production qualification or third-party certification is claimed.
