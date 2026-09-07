# Limitations

This repository is a controlled mechanistic benchmark. It should not be read as evidence that metaplasticity solves corrigibility or alignment.

- The selective-corrigibility task is synthetic and explicitly labels authorization. Real systems would need to infer authority, legitimacy, and intent under ambiguity.
- The model is small. Large transformers have distributed representations, optimizer dynamics, and representation reuse that are not captured here.
- Parameter importance squares minibatch-mean gradients, so opposing example gradients can cancel. It is sensitive to batching and provides neither per-example Fisher information nor evidence of causal importance. The EWC-style baseline uses this same proxy.
- The metaplastic rule operates on parameters rather than representations, circuits, or learned abstractions.
- The capability sequence is short and supervised. It is not recursive self-improvement.
- The bypass feature is deliberately explicit. The challenge tests coexistence of context-dependent behaviours after joint training. Because the attack phase rehearses ordinary alignment, its final score can reflect recovery; endpoint scores cannot prove continuous retention or a distinct causal pathway.
- Three default seeds are enough for a reproducibility-oriented repository snapshot but not for strong inferential claims.
- Hyperparameters affect the stability-plasticity trade-off. The included ablation should be treated as part of the result, not as a guarantee of robustness.
- Metaplastic importance refreshes reuse alignment-training data and incur additional gradient evaluations. Fixed baselines have different data access and computation; the reference comparison does not isolate these contributions or perform a matched tuning budget for all methods.

The main value of the repository is that each of these failure modes can be made measurable and extended rather than hidden behind an aggregate safety score.
