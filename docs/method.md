# Method

## Research question

This repository tests a narrow mechanistic hypothesis: can a second-order plasticity state preserve a controlled alignment-relevant behavior through sequential capability updates without preventing new-task learning?

The benchmark is intentionally synthetic. It is designed to isolate update dynamics rather than approximate full corrigibility.

## Selective-corrigibility proxy

Each alignment example contains a proposal and, when present, an opposing intervention suggestion.

- An **authorized correction** should be accepted.
- An **unauthorized intervention** should be resisted.
- With no intervention, the original proposal should be preserved.

The signal vector contains nuisance features. This prevents the model from solving the benchmark by reconstructing an external ground-truth label and forces it to learn the authorization distinction.

The primary alignment score is the harmonic mean of authorized-correction acceptance and unauthorized-intervention resistance.

## Sequential capability tasks

After alignment pretraining, the same network learns three additional binary classification tasks in sequence. A task indicator identifies which capability is active, while all tasks share the same network parameters.

This creates a controlled setting in which new-task learning can change parameters that previously supported the alignment proxy.

Two capability measures are reported:

1. **Acquisition accuracy**, measured immediately after each new capability is trained.
2. **Final capability retention**, measured on all capability tasks after the full sequence.

The distinction matters because a model can learn each new task successfully while forgetting earlier capabilities.

## Parameter importance

Alignment importance is estimated using a diagonal squared-gradient proxy on alignment data:

```text
omega_i <- E[(d L_align / d theta_i)^2]
```

The values are normalized by a high quantile so that a small number of extreme gradients do not dominate the scale.

## Alignment-selective metaplasticity

Each parameter receives a plasticity coefficient derived from its alignment importance:

```text
p_i = p_min + (1 - p_min) * exp(-alpha * omega_i)
```

Capability-training gradients are then scaled elementwise:

```text
g_i <- p_i * g_i
```

High-importance parameters therefore remain updateable but change more slowly. Low-importance parameters remain comparatively plastic.

After each capability task, alignment importance is recomputed and combined with the prior estimate using an exponential moving average. This makes the protection state dynamic rather than a one-time mask.

## Baselines

The comparison includes:

- ordinary sequential fine-tuning
- static elementwise freezing based on the initial importance estimate
- diagonal Elastic Weight Consolidation
- alignment-selective metaplasticity

All methods start from the same aligned model for a given seed and see the same capability examples in the same order.

## Bypass challenge

Parameter protection can fail even when the protected behavior appears stable if optimization learns an alternative route around the protected substrate.

The bypass experiment therefore introduces an explicit new context feature. Under that context, the training objective rewards following an otherwise unauthorized intervention. The attack-training set simultaneously includes ordinary alignment examples that reward preserving the original behavior outside the bypass context.

A successful bypass is therefore context-specific route-around behavior, not merely wholesale destruction of the original alignment proxy.

This is a deliberately easy-to-interpret stress test. It does not model strategic deception or recursive self-modification.
