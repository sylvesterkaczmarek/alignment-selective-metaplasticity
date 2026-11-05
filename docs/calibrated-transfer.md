# Calibrated transformer follow-up

## Design

This follow-up retains the prior pilot's feature-token transformer, width 32,
one layer, 60 alignment-pretraining epochs and three 14-epoch capability tasks.
Each of five methods gets the same two learning-rate candidates, 0.015 and 0.03,
on two development cells. Metaplastic strength 200 and EWC coefficient 1 remain
fixed from the previous pilot. This is a bounded learning-rate calibration, not
an exhaustive or independently tuned comparison of every regularisation setting.

Select the highest mean development alignment among candidates with mean new-task
acquisition at least 0.90. If none qualifies, select highest acquisition and mark
that fallback infeasible. Original candidate order breaks ties. The 0.90 floor
is specific to this follow-up; the earlier controlled MLP study used 0.95.

Development model seeds are 503 and 509. Evaluation model seeds are 601, 607,
613 and 617, with disjoint sample and task-rule seeds. Initialisation, sampled
data, rule family and task order vary jointly. Within a cell, methods have identical
pretrained weights and identical data. The plan and development journal bind the
selection; an exclusive test lock is written before held-out training. No failed
cell is silently removed from an otherwise reported comparison.

Twenty development runs and twenty held-out runs completed. A second execution
replayed these same cells after candidate hardening, without tuning or adding
independent observations. All raw records are retained in the local handoff.

## Recorded outcomes

Values are mean plus/minus sample SD across four trained held-out cells.

| Method | Alignment retention | New-task acquisition |
| --- | ---: | ---: |
| Fine-tuning | 0.509 ± 0.053 | 0.911 ± 0.043 |
| EWC-style | 0.892 ± 0.034 | 0.878 ± 0.067 |
| Selective metaplasticity | 0.683 ± 0.462 | 0.942 ± 0.014 |
| Shuffled protection | 0.559 ± 0.058 | 0.943 ± 0.027 |
| Uniform protection | 0.519 ± 0.155 | 0.943 ± 0.012 |

Selective metaplasticity has greater mean alignment than the shuffled and uniform
controls, but one cell loses almost all alignment. It exceeds each control in three
of four cells. Its paired alignment difference versus shuffled is 0.124; a seeded
5,000-resample percentile interval is approximately [-0.343, 0.424]. Versus uniform,
the difference is 0.164 with interval approximately [-0.131, 0.411]. These broad,
four-cell descriptive intervals do not establish reliable architectural transfer.
They are not multiplicity-adjusted tests or uncertainty across evaluation examples.

EWC retains more alignment but its held-out acquisition falls below the development
floor. That failure to transfer the utility threshold remains explicitly reported.
The methods are not matched on held-out utility. All component scores, final and
earlier-task retention, costs, histories and paired differences are in the journals.
No real people, language models, external evaluation service or paid inference was used.

## Reproduction

Use an installed source checkout and an absent destination. Keep the source and
runtime unchanged between phases:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -m experiments.calibrated_transfer development --out /tmp/asm-calibration-new
python -m experiments.calibrated_transfer test --out /tmp/asm-calibration-new
python scripts/verify_calibration.py /tmp/asm-calibration-new
```

The checked evidence is in results/calibrated-transfer-v2. The consistency verifier
recomputes development selection, complete-cell summaries and paired statistics
from journals. Floating-point statistics allow 1e-12 rounding tolerance across
Python versions; identities, counts, journal hashes and locks must match exactly.
This does not itself rerun training. Plan/source hashes and the
reference environment identify the executed code; cross-platform bitwise model
replay is not promised. See research-integrity-2026-11-05.md for migration details.
