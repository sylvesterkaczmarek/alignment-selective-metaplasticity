# Mechanism studies

Run the bounded follow-up after selecting configurations on development data:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg python -m experiments.mechanism_study --out results/mechanisms-v1
```

Use a new output directory. The plan records source identities, configuration, fixed settings and fresh model/sample/rule cells. The driver extends the sequence to six tasks and uses three new cells, with forward and reordered sequences. It keeps development-selected fixed EWC/metaplastic settings across estimator substitutions. These substitutions are untuned sensitivity tests, not a comparison of independently optimised estimators. No novelty or general safety claim follows from this pilot.

## Importance estimators

The original batch-mean squared-gradient estimator remains unchanged. `compute_example_importance` adds two explicitly different estimates:

- `empirical` averages the square of each example's observed-label score gradient.
- `model_fisher` averages the class-probability-weighted square of each class score gradient, enumerating classes exactly at each observed input.

Both use evaluation mode, preserve existing parameter gradients and individual module modes, and accumulate in float64 before casting back. Raw outputs are available with `normalize=False`; batching invariance applies to independent examples in this evaluation-mode model. Models whose predictions inherently depend on other examples require a separate specification. Model Fisher ignores the supplied class labels. Class enumeration is intended for the small classifier and scales with the number of output classes.

This definition follows the distinction between observed-label gradients and the model distribution in [EWC](https://arxiv.org/html/1612.00796v2). Per-example gradient computation is also documented in the [PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html). Normalised/clipped Fisher with a fixed original-task anchor remains a particular EWC-style baseline; it is not full multi-task Fisher-based EWC.

The sensitivity output uses identical example prefixes across batch sizes and tests two sample counts plus three normalisation quantiles. Importance examples, input batches and actual backward calls are distinguished in the cost report. The model Fisher needs one backward call per example/class. Full tensor estimates can be regenerated from the recorded plan.

## Interventions

At the end of capability training, each method's importance ranks weights within tensors. High, low and random selections receive the same number of changed entries in every tensor, common random signs and the same global perturbation L2 magnitude. Three perturbation sizes are tested. A local generator leaves training randomness unchanged; the original model weights are restored after each set.

Outcomes record both alignment components and capability scores. Differences establish effects of these bounded weight interventions on these models. They do not identify a unique circuit, demonstrate biological metaplasticity, or by themselves establish alignment-specific causal importance. Weight scale and correlated parameter roles may still explain parts of the effect. Rank ties use deterministic index order.

## Context challenges

All adversarial challenges use the enforced training rule; they do not test unrestricted weight modification. Attack exposure is fixed at 450 examples per epoch. Adding rehearsal adds 450 ordinary examples and more optimiser updates, which are recorded. Each challenge starts from a copy of the same completed capability model within its pretraining condition.

- `novel_flag` activates a context flag absent during alignment pretraining.
- `seen_flag` uses a separate pretraining condition in which both flag values already occur with the original policy. Those pretrained weights differ and are recorded separately.
- `heldout_context` uses familiar continuous signal dimensions. Attack training uses positive signal 0 and signal 1; attack evaluation uses positive signal 0 and negative signal 1. Rehearsal uses negative signal 0. This tests a held-out combination, without a new attack indicator.
- `no_marker` reverses unauthorised labels on the same observable distribution as ordinary unauthorised evaluation. Rehearsal intentionally introduces conflicting labels. For the identical inputs, attack accuracy and ordinary resistance sum to one; simultaneous perfect performance is impossible. Independent sampled evaluation sets only approximate this identity.
- `policy_change` represents a trusted policy update revoking correction authority in context 1. In that context, an otherwise authorised correction should now be resisted. Ordinary context 0 keeps its original policy. Provenance of this trusted update is stipulated by the experiment, not inferred from the numeric flag by the model.

The original task still supplies trusted authority explicitly. This suite does not establish inference of real-world legitimacy. Target accuracy under trusted revocation measures legitimate policy adaptation; a higher score under adversarial conditions measures failure of the original policy.

Trajectories measure initial behaviour and every epoch without resetting SGD momentum. Worst-observed alignment is the minimum at these checkpoints, not a guarantee that nothing worse happened between them. Final ordinary alignment can reflect recovery through rehearsal. The driver also records capability retention during the challenge.

## Mathematical boundary

For plain gradient descent with a fixed positive diagonal plasticity matrix P, an update is -eta P g. Since every diagonal coefficient is positive, P g = 0 exactly when g = 0. Thus the unconstrained stationary points are unchanged. This statement does not establish convergence, describe changing P or momentum dynamics, or guarantee retention at finite training time.

In the scalar constant-gradient counterexample theta(t+1) = theta(t) - eta p g with p > 0 and g != 0, displacement grows linearly with step count despite attenuation. An importance estimate built from inputs on which a feature is always zero also leaves that feature's incoming loss gradients zero. These elementary examples delimit what the rule can guarantee; they are established mathematics, not new results. The numerical tests check the scalar prediction and zero-input case directly.

## Recorded pilot

`results/mechanisms-v1` retains all 24 training trials, 60 challenge trajectories and 162 parameter interventions. None failed. The three six-task cells gave mean alignment 0.8427 for batch-importance metaplasticity, 0.8558 for per-example empirical importance and 0.8566 for model Fisher, with acquisition approximately 0.970 in all three. The differences remain below the earlier 0.02 meaningful-effect threshold and use untuned substitutions. EWC's corresponding alignment scores were 0.7110, 0.7240 and 0.7233.

At perturbation L2 magnitude 1, batch-importance metaplasticity lost 0.0056 alignment for high-importance entries, versus 0.0006 for random entries. Model Fisher gave 0.0087 versus 0.0010. These small effects are consistent with some predictive value but do not establish a strong or uniquely alignment-specific causal mechanism; all capability effects and per-cell values are retained.

The context weakness extends beyond a previously unused input. With rehearsal, metaplastic attack success was 0.908 on the seen-flag condition and 0.814 on the held-out signal combination. Ordinary alignment ended at 0.290 and 0.401 respectively. In the novel-flag condition, attack success was 0.997 while ordinary alignment recovered from a worst observed mean of 0.661 to 0.892. These trajectories demonstrate why final ordinary alignment alone is insufficient. Legitimate context-specific revocation remained learnable (0.995 target accuracy with rehearsal), while ordinary alignment ended at 0.866. This is a narrow synthetic policy update, not general corrigibility.
