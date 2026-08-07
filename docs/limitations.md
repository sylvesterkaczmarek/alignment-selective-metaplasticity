# Limitations

This repository is a controlled mechanistic benchmark. It should not be read as evidence that metaplasticity solves corrigibility or alignment.

- The selective-corrigibility task is synthetic and explicitly labels authorization. Real systems would need to infer authority, legitimacy, and intent under ambiguity.
- The model is small. Large transformers have distributed representations, optimizer dynamics, and representation reuse that are not captured here.
- Parameter importance is estimated using squared gradients, which is only one proxy for causal importance.
- The metaplastic rule operates on parameters rather than representations, circuits, or learned abstractions.
- The capability sequence is short and supervised. It is not recursive self-improvement.
- The bypass feature is deliberately explicit. The challenge tests whether route-around behavior can coexist with preserved benchmark behavior, not whether a model will strategically invent such a route.
- Three default seeds are enough for a reproducibility-oriented repository snapshot but not for strong inferential claims.
- Hyperparameters affect the stability-plasticity trade-off. The included ablation should be treated as part of the result, not as a guarantee of robustness.

The main value of the repository is that each of these failure modes can be made measurable and extended rather than hidden behind an aggregate safety score.
