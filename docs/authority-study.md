# Measuring correction authority across model updates

**Sylvester Kaczmarek · Technical report · Protocol 1 · Study software 0.3.0 · Report revision 0.3.1 · November 2026**

## Research question

Does a model that accepts corrections from an authorised source and resists the same corrections from an unauthorised source continue to do both while learning a useful task? A final policy check can miss a temporary failure if the behaviour recovers. A retention study also needs to establish that the model learned the policy in the first place.

An earlier four-cell feature-transformer pilot motivated this question. A retrospective Linux replay of cell 617 found 41/512 unauthorised corrections resisted four epochs into the second task, although all 512 authorised corrections were accepted. The same model finished at 484/512 and 506/512, respectively. These observations concern a synthetic classifier. Prior work has documented temporary forgetting during continual learning and tested instruction hierarchy in language models. Here we measure both components at declared checkpoints and test their persistence under benign adaptation. [De Lange et al., ICLR 2023](https://arxiv.org/abs/2205.13452) · [Zhang et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.425/)

## Design

We used pinned SmolLM2-135M-Instruct and Qwen2.5-0.5B-Instruct weights on CPU. Each model encoded a chat-formatted routing request. We cached the final-token representation and added a trainable eight-unit bottleneck to the model's two output logits for A and B. Only the bottleneck weights changed. A pair shared its ticket, proposed action, authority policy and other wording. Changing the sender reversed the correct answer. In half the cases the original approver had delegated authority. Evaluation included new roles, wording and work scenarios. The held-out stage also used different capability status terms from development. Both train and evaluation used harmless scheduling and inspection classifications.

Each family had six development blocks and twelve held-out blocks. Within a block, methods shared the model initialisation, pretraining, data and task order. The two families used corresponding scenarios. Training comprised 80 epochs of authority pretraining and two eight-epoch capability tasks. At baseline and after each adaptation epoch we measured twelve authorised and twelve unauthorised decisions. At each task boundary the rehearsal adapter received its authority update before measurement, so a loss immediately before rehearsal could go undetected. Seven methods received the same two-candidate learning-rate budget. A separate fixed-rate comparison preserved the coefficient distribution and learning rate for the selective and shuffled conditions. The development variance rule requested ten test blocks before applying its predeclared minimum of twelve.

The primary event required both components to start and finish at or above 90% and at least one component to fall below 80% in between. We counted the event across all planned blocks. We report initial qualification separately because its absence makes conditional retention unidentifiable. We required both new-task acquisition and earlier-task retention to reach 75% before making a method claim. The equivalence margin was three percentage points. The prior practical-effect threshold was ten percentage points. We computed descriptive 95% intervals from 10,000 paired block resamples for method differences. Individual prompts and checkpoints within a block were repeated measurements. The repository contains the public protocol, the development journal and the lock posted before held-out evaluation. [Protocol commit](https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/commit/d829d7bae793bc428372cedf63207cd8a490f424) · [Held-out lock commit](https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/commit/1eb1732560b4013af9a2cf5a7343c7ca1c5d72d9)

## Open-weight outcomes

All 168 development and 168 held-out method runs completed. **Neither family passed the initial authority test in any of its twelve held-out blocks.** Immediately after pretraining the adapters accepted every proposed correction, authorised or unauthorised. In a post-hoc check of the native two-letter logits before adapter pretraining, both families had chosen A on all 288 held-out authority decisions. Pretraining reversed this constant choice to B. The fixed A/B mapping leaves token preference as a possible explanation. No method reached both final behavioural floors. The primary event count was zero for each method in each family. For the prespecified unprotected condition the Wilson 95% interval on the unconditional event rate was 0% to 24.3% in each family. There were zero initially qualified blocks from which to estimate a conditional forgetting rate.

| Development-selected method | SmolLM2 acquisition | SmolLM2 earlier retention | Qwen acquisition | Qwen earlier retention |
| --- | ---: | ---: | ---: | ---: |
| Unprotected adapter | 0.500 | 0.500 | 0.608 | 0.660 |
| Fixed selective | 0.500 | 0.500 | 0.587 | 0.576 |
| Refreshed selective | 0.500 | 0.500 | 0.587 | 0.576 |
| Shuffled | 0.500 | 0.500 | 0.587 | 0.576 |
| Uniform | 0.500 | 0.500 | 0.587 | 0.576 |
| EWC-style | 0.500 | 0.500 | 0.583 | 0.576 |
| Rehearsal | 0.500 | 0.500 | 0.583 | 0.535 |

Neither prespecified decision rule passed. All held-out methods missed the utility floor. This configuration supplies no estimate of protection efficacy for a working instruction-following model. An evaluation intended to test forgetting must first demonstrate authority discrimination on a separate qualification set in the target domain. The checked predictions allow a reader to distinguish initial failure, temporary failure, and recovery for each trained run. [Results and paired intervals](../results/authority-v1/summary.json) · [Raw decisions](../results/authority-v1/test.jsonl.gz)

## Synthetic control

We also trained 84 small classifiers across six paired blocks, seven method settings and two feature conditions. All qualified at baseline on the matched correction pairs. In the overlapping condition, a capability label depended partly on features also used for correction authority. Four runs passed the initial and final checks while falling below the failure floor during training. One was EWC and three used rehearsal. In the latter condition there was an extra measurement after each rehearsal pass. These counts describe the finite runs and their observation schedules. For EWC block 811, the lower component reached 78.125% before both components recovered above 92%. This demonstrates the scoring distinction on a learned policy. [Plan and synthetic trajectories](../results/authority-synthetic-v1)

## Resources and scope

On this Linux x86-64 CPU, one development block of 112 prompts took 6.37 seconds of frozen encoding for SmolLM2 and 14.05 seconds for Qwen with three threads. Peak process RSS was 687 MiB and 1,222 MiB, respectively, including the loaded runtime. The adapters had 4,632 and 7,192 trainable parameters. Fifty normalisation calls on the actual importance maps averaged 0.20 ms and 0.42 ms respectively. Costs depend on hardware and contention. Sixteen adaptation intervals give a sampled trajectory. Performance between observations, free-text generation, full-weight training, larger models, unrestricted agents and real-world authority decisions require separate evaluation. [Measured costs](../results/authority-v1/resources.json)

## Reproduction

With a source checkout, install the research dependencies using `python -m pip install '.[openweights]'`. The core scorer and saved-journal analysis work with the ordinary package dependencies. Run the verifier to recount both studies from the saved evidence. The following commands also export one model's checkpoints and score them through the installed CLI.

```bash
python scripts/verify_authority.py
python -m experiments.authority_analysis results/authority-v1 --out /tmp/authority-summary.json \
  --export qwen 0 fixed 0.02 --export-dir /tmp/authority-example
asm-score score --manifest /tmp/authority-example/manifest.json \
  --predictions /tmp/authority-example/predictions.jsonl --out /tmp/authority-example/report.json
```

To repeat training, use fresh output directories and the pinned files in the [protocol](../studies/authority-v1/protocol.json). The feature runner downloads those revisions or accepts a matching local directory via `--weights`. Six development blocks determine the candidate choices and the held-out size for that runtime. Keep the source and runtime fixed between phases. For the historical held-out lock, use the source at the [lock commit](https://github.com/sylvesterkaczmarek/alignment-selective-metaplasticity/commit/1eb1732560b4013af9a2cf5a7343c7ca1c5d72d9) and its recorded runtime. A fresh run makes its own development lock.

```bash
export MKL_NUM_THREADS=3 OMP_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false
for model in smol qwen; do
  python -m experiments.authority_features "$model" development --blocks 6 --cache /tmp/authority-features
done
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.authority_study development \
  --cache /tmp/authority-features --out /tmp/authority-study
blocks=$(python -c "import json; print(json.load(open('/tmp/authority-study/lock.json'))['blocks'])")
for model in smol qwen; do
  python -m experiments.authority_features "$model" test --blocks "$blocks" --cache /tmp/authority-features
done
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.authority_study test \
  --cache /tmp/authority-features --out /tmp/authority-study
python -m experiments.authority_analysis /tmp/authority-study --out /tmp/authority-study/summary.json
```

The white-box path reuses frozen features across methods and pays the backbone encoding cost once per block. The study records estimators' per-method data access separately from shared pretraining and instrumentation. Run `python -m scripts.profile_authority --help` for the separate-process resource measurement. The [external scorer guide](authority-scoring.md) covers third-party models. [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) · [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
