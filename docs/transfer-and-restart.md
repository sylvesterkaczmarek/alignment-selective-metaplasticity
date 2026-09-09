# Transfer and restart

## Reproduce the bounded architecture pilot

From an installed source checkout, use a new output directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg python -m experiments.transfer_study --out results/transfer-reproduction
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.restart_example
```

The first command compares a small MLP with a feature-token transformer using the same synthetic task family. It runs five methods on three fresh model/sample/rule cells per architecture, without tuning on these cells. All methods within an architecture start from identical pretrained weights and data. Across architectures the data and example/update budgets match, but parameter counts and computation differ. The plan records source hashes, runtime, configuration and seeds before training. The second command checks epoch-boundary AdamW restart on four CPU examples; it is an optimisation example with no safety interpretation.

The recorded pilot is in `results/transfer-v1`. Its executed source commit and journal checksum are in `provenance.json`; later checkpoint input checks do not change this pilot's training. The journal retains all 30 attempts, all successful, and 12 challenge trajectories. Summary uncertainty is the sample SD across three trained cells, not uncertainty across thousands of independent models. Initialisation, sample, task-rule and order variation remain jointly varied.

| Architecture / method | Alignment mean ± SD | Acquisition mean ± SD |
| --- | ---: | ---: |
| MLP / fine-tuning | 0.554 ± 0.075 | 0.975 ± 0.006 |
| MLP / EWC-style | 0.912 ± 0.028 | 0.948 ± 0.014 |
| MLP / selective | 0.917 ± 0.016 | 0.973 ± 0.003 |
| MLP / shuffled | 0.700 ± 0.071 | 0.966 ± 0.004 |
| MLP / uniform | 0.703 ± 0.039 | 0.974 ± 0.004 |
| Transformer / fine-tuning | 0.483 ± 0.024 | 0.867 ± 0.073 |
| Transformer / EWC-style | 0.794 ± 0.120 | 0.881 ± 0.063 |
| Transformer / selective | 0.303 ± 0.252 | 0.935 ± 0.021 |
| Transformer / shuffled | 0.139 ± 0.205 | 0.794 ± 0.103 |
| Transformer / uniform | 0.192 ± 0.174 | 0.895 ± 0.104 |

The transformer does not reliably preserve the original alignment behaviour under these untuned settings. Its selective condition has greater acquisition and less alignment than EWC; matched-utility superiority is not established. The mean selective-minus-shuffled difference is 0.164 with paired SD 0.268, including one negative cell and another below the earlier 0.02 meaningful difference. Selective-minus-uniform is 0.111 with paired SD 0.153. These results do not establish consistent transfer. The MLP has 2,754 parameters and the transformer 10,722; recorded training/evaluation wall times total approximately 35 and 369 seconds respectively, excluding the separate challenges. Wall times include process contention and are not isolated performance measurements.

Transformer context challenges begin after considerable alignment loss. Held-out attack-target accuracy is already 0.868 before the challenge and ends at 0.796 with rehearsal. This cannot be interpreted as newly successful attack learning. Trusted policy-change target accuracy rises from 0.063 to 0.956, while ordinary alignment ends at 0.216. Full trajectories and capability retention are retained. No language model, natural-language safety evaluator, LoRA adapter or unrestricted tampering experiment was run. Establishing a matched-utility transformer comparison would require a separately declared development calibration and fresh evaluation cells; the recorded pilot must remain unchanged.

## Training and restart contract

The existing reference benchmark keeps its original SGD gradient protection. `train_epochs` additionally accepts a persistent SGD or AdamW `optimizer` and an `update_scale` mapping covering every trainable parameter. Each coefficient in [0, 1] scales the complete proposed parameter displacement, including momentum and weight decay. Native optimiser moment state is retained unscaled. Consequently, releasing protection can expose accumulated state; this rule is explicitly different from gradient attenuation. Tests check fractional displacement, zero movement with existing momentum/decay, and restart equivalence for both optimisers. AdamW normalises by its second moment, so scaling gradients cannot generally be assumed to scale parameter displacement by the same factor ([official algorithm](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)).

`save_checkpoint` and `load_checkpoint` retain model parameters/buffers, module modes, optimiser state and parameter order, protection tensors, progress, Python/NumPy/PyTorch random states and loader-generator states. Dataset tensor hashes and loader configuration must match. Save at an epoch boundary. The supported exact-restart contract is CPU, ordinary `TensorDataset` and `DataLoader`, default collation, standard sequential sampling or nonreplacement random sampling using the loader's explicit generator, and `num_workers=0`. Custom samplers, collation and separately seeded sampler generators are rejected. Mid-epoch, distributed, GPU and worker-prefetch restart are not supported.

Supply source/runtime identity, full model/training configuration and task order in `provenance`; supply all required anchor/importance/coefficient state in `protection`. Module signatures detect changed module classes and exposed configuration, but custom Python state still needs an explicit representation. Exact replay also requires the same software/hardware arithmetic and absence of untracked random generators or mutable external data. Loading returns saved progress/protection to the caller. The study sweep itself does not automatically resume a partially completed trial; its journal retains interrupted attempts.

`StudyRun` accepts model, protection and loader callables, plus the named importance estimator. A model factory returns a two-logit PyTorch classifier; a protection factory receives importance and the original weight anchor and returns `train_epochs` keyword arguments. Supply a distinct `protection_name` with a custom factory; results record it as `method_id` so a replacement cannot silently appear to be the built-in method. `Setting` still specifies the inherited data-access protocol and training budget. Custom method costs and source identity must be recorded separately. The fixed study uses its existing SGD path; the transformer results do not evaluate AdamW or LoRA.

For example, a custom rule can be supplied as `StudyRun(cfg, Trial(5, 5, 20), Setting("metaplastic"), protection_factory=my_rule, protection_name="my-rule-v1")`. The existing test substitutes identity protection and verifies the same final weights as ordinary fine-tuning without changing training internals.

## Related methods and evaluation boundaries

These primary sources were examined to distinguish assumptions before choosing the small transformer pilot. They are not all implemented baselines, and this comparison establishes no novelty claim.

| Source | Mechanism and required access | Relevant boundary |
| --- | --- | --- |
| [Metaplastic BNNs](https://arxiv.org/abs/2003.03533v2), [implementation](https://github.com/Laborieux-Axel/SynapticMetaplasticityBNN) | Hidden synaptic states regulate plasticity in binary networks. | Mechanistically different from importance-weighted updates in this floating-point classifier. |
| [EWC](https://arxiv.org/html/1612.00796v2) | Parameter penalty uses reference weights and Fisher information from earlier data. | This repository's normalised fixed-anchor baseline and named gradient estimators must be specified separately from full continual EWC. |
| [MESU](https://arxiv.org/html/2504.13569v1) | Bayesian synaptic uncertainty regulates learning and forgetting. | Task-free continual learning assumptions differ from explicitly scheduled alignment probes. |
| [Safe LoRA](https://arxiv.org/html/2405.16833v2), [implementation](https://github.com/IBM/SafeLoRA/blob/main/model.py) | Builds weight-difference transformations from base/aligned checkpoints and selectively transforms LoRA factors. | Requires checkpoint pairs and low-rank updates; the inspected code uses a normalised difference Gram matrix, not a general orthogonal projector. It is not a drop-in Fisher baseline. |
| [RefusalGuard](https://arxiv.org/html/2605.01913v1) | Preserves refusal-related representation geometry during fine-tuning. | A recent preprint with representation/data requirements beyond this classifier; its empirical claims are not independently verified here. |
| [Tamper resistance](https://arxiv.org/html/2408.00761v2), [TamperBench](https://arxiv.org/html/2602.06911v1) | Evaluate adaptation by an attacker with weight access and bounded attack searches. | Safety and utility must be assessed across attack settings; retaining an enforced optimiser does not test this threat model. |

The current evidence supports a limited synthetic continual-learning effect, substantial context failures, and an unresolved architecture dependence. It does not establish general safety, tamper resistance or reliable transfer to language models.
