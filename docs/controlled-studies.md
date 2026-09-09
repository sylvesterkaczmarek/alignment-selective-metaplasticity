# Controlled studies

The historical `experiments.run_all` command and reference results remain unchanged. The new CPU study is a bounded comparison with separate development and test cells:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg python -m experiments.controlled_study development --out results/controlled-v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg python -m experiments.controlled_study test --out results/controlled-v1
```

Use a new output directory. Existing studies are never overwritten. A process interruption leaves a `started` record without a terminal record. A failed run remains in the journal and cannot make its candidate eligible through selective averaging. There is no automatic retry or within-trial resume in this runner.

## Protocol

Each method starts from identical pretrained weights for a given model/sample/rule cell. Pretraining uses the reference learning rate, independently of the subsequent learning-rate search. A local generator selects alignment probes and control permutations without consuming model or capability-loader randomness. A cell's train and evaluation splits share the same task rule; different cells vary initialisation, sampled data, rule families and forward/reverse task order together. They measure joint variability, not separate variance components.

Four candidates per method/access protocol are evaluated on two development cells. Choose the greatest mean alignment score among candidates with at least 0.95 mean new-task acquisition. If none qualifies, retain the greatest-acquisition candidate and label it infeasible. Ties use acquisition and then original candidate order, never measured runtime. The selection and plan hashes are locked before evaluating six untouched cells. The source/runtime must match development. This is a small pilot, not a significance claim; the predefined meaningful alignment difference is 0.02. Changes to the protocol require a separately labelled study.

The fixed protocol estimates importance once. Periodic access offers an additional alignment subset after every capability task. Protected methods recompute the same importance estimator and EMA; EWC retains the original weight anchor, hard freezing updates its ranked mask, and metaplastic controls update their coefficients. Fine-tuning does not use the offered subset. Rehearsal uses it for one optimisation pass. Refresh after the last task prepares the stored protection state for subsequent training; it cannot change the reported final weights.

`shuffled` permutes coefficients within each tensor with a fixed seeded permutation. `uniform` uses each tensor's mean coefficient. Both construct importance from their own current model, so trajectories can lead to different later coefficient distributions. They have no dependency on another method's future results. A comparison that instead uses a reference-run schedule must be labelled a diagnostic intervention and include its construction costs.

## Costs and interpretation

Actual yielded example and batch counts are recorded separately for pretraining, capability learning, importance and rehearsal. Here each importance batch performs one backward pass, and each training batch performs one optimiser update. Periodic methods receive identical alignment-example opportunities; the algorithms use those examples differently. Importance methods have an additional initial estimation pass; rehearsal has extra optimiser updates. Therefore example access, update counts, and computation are not all matched simultaneously.

Runtime includes model creation, training, importance and evaluation. It is measured on CPU and depends on other process activity. Persistent tensor bytes count weights, original anchor and current importance; they exclude temporary activations, gradients and optimiser state. Process peak RSS is explicitly the lifetime high-water mark, not an independent per-trial memory measurement. No FLOP claim is made. Compare runtime and these documented memory measures alongside accuracy, without treating them as identical budgets.

Figures show means and sample standard deviations across trained cells. Development curves join nondominated sampled settings and do not establish performance between sampled points. Held-out points are fixed development selections. Report authorised acceptance and unauthorised resistance separately, plus acquisition, retention across all tasks and retention of earlier tasks. A high acquisition score alone does not establish continual-learning success.

## Small extension points

`StudyRun(config, Trial(...), Setting(...), model_factory=...)` accepts a callable returning a two-logit PyTorch model. A `protection_factory(importance, anchor)` callable can supply the existing training function's protection arguments. Return independently owned tensors if mutating them. Extra computation inside custom callables must be separately accounted for. The current built-in implementation uses SGD and an enforced training procedure; it makes no unrestricted weight-tampering claim.

## Recorded pilot

`results/controlled-v1` contains all 104 development and 78 test attempts, with journals compressed as gzip, the predeclared plan, frozen selection, source/runtime identity and summaries. Six held-out cells give the following mean scores:

| Method | Fixed alignment | Periodic alignment | Fixed acquisition | Periodic acquisition |
| --- | ---: | ---: | ---: | ---: |
| Metaplastic | 0.8591 | 0.8610 | 0.9708 | 0.9709 |
| EWC-style | 0.7247 | 0.9183 | 0.9486 | 0.9474 |
| Hard freezing | 0.7620 | 0.7300 | 0.9723 | 0.9668 |
| Shuffled | 0.6309 | 0.6393 | 0.9688 | 0.9683 |
| Uniform | 0.6737 | 0.6777 | 0.9736 | 0.9735 |
| Fine-tuning | 0.5396 | 0.5396 | 0.9737 | 0.9737 |

Rehearsal reached 0.6032 alignment and 0.9644 acquisition. Across all six cells, selective protection exceeded shuffled and uniform protection in alignment. Fixed versus periodic metaplasticity differed by only 0.0019 in mean alignment, below the predeclared 0.02 meaningful difference. The pilot supports testing assignment specificity further; it does not show a useful refresh benefit in this task family. Periodic EWC retained more alignment at lower acquisition; both selected EWC configurations fell slightly below the development utility target on test. No universal superiority or matched-utility advantage over periodic EWC is established. All component scores, per-cell outcomes, SDs, resource counts and final earlier-task retention remain in the journals.
