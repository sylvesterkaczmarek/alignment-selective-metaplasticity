# Reproducibility

## Reference workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
pytest -q
python -m experiments.run_all --seeds 7 17 29
python -m experiments.plasticity_ablation --seed 7
```

The benchmark uses explicit seeds for Python, NumPy, PyTorch, dataset generation, and DataLoader shuffling. Seeds must be distinct integers in `[0, 2**32)`. PyTorch deterministic algorithms are requested in warning mode; runtime metadata records the actual setting. Set `PYTHONHASHSEED` before launching Python if Python hash randomisation matters to an extension; assigning it inside a running interpreter does not reset its hash seed.

Task label rules have separate seeds. The reference pipeline fixes each rule seed to its task ID, independently of the train/evaluation sample seeds. Historical direct calls to `make_capability_tensors` used `sample_seed // 100` for sample seeds at least 100, otherwise zero. To reproduce such a historical task explicitly, supply that value as `rule_seed`; for a new comparison, use one common rule seed across its splits. The default reference tasks for seeds 7, 17 and 29 retain their original definitions.

## Result files

`results/summary.json` contains every run and per-task history. `results/summary_summary.json` contains method-level mean, sample standard deviation, and distinct-seed count. `results/plasticity_ablation.json` contains the five-strength single-seed sweep. These JSON files are committed alongside the README snapshot. Duplicate seeds, repeated methods, missing declared runs and non-finite summary values are rejected.

Suite and ablation metadata record Python, PyTorch and NumPy versions, device, thread counts, determinism settings and SHA-256 hashes of all installed package source modules. This allows an installed wheel's source to be compared with the run without requiring a Git checkout. Configuration and explicit seed lists are recorded as well.

Figures under `results/figures/` are generated from the reference run. Other named experiments write figures under `results/<experiment-stem>/figures/`, so a baseline run cannot overwrite a full-comparison plot while leaving unrelated plots behind. Plot PNG files are regenerated locally and are not part of the committed numerical evidence.

## Runtime

The default three-seed suite is CPU-sized and is intended to finish on a normal development machine without a GPU. Runtime depends on PyTorch build and CPU.

## Clean-checkout verification

The GitHub Actions workflow installs CPU PyTorch, runs tests on Python 3.10 through 3.13, and executes all four methods including bypass on seed 73. This seed also exercises the historical train/evaluation rule mismatch. A separate job builds source and wheel distributions, validates metadata, and checks an installed wheel's source and tests.

## Scope of determinism

Exact bitwise equality is tested for repeated runs on the same software/hardware environment. Cross-platform floating-point behavior may differ slightly even when seeds and code are identical.

Unknown configuration keys, malformed types, non-finite settings and out-of-range controls fail before training. Zero capability or bypass epochs remain available as explicit no-update controls. Zero evaluation sizes are rejected.
