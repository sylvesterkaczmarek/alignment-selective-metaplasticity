# Reproducibility

## Reference workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m experiments.run_all --seeds 7 17 29
```

The benchmark uses explicit seeds for Python, NumPy, PyTorch, dataset generation, and DataLoader shuffling. PyTorch deterministic algorithms are requested where available.

## Result files

`results/summary.json` contains every run and per-task history. `results/summary_summary.json` contains method-level mean, standard deviation, and sample count. Figures under `results/figures/` are generated directly from the summary file.

## Runtime

The default three-seed suite is CPU-sized and is intended to finish on a normal development machine without a GPU. Runtime depends on PyTorch build and CPU.

## Clean-checkout verification

The GitHub Actions workflow installs the package, runs the test suite, and executes a one-seed smoke experiment on Python 3.11. The test matrix also covers Python 3.12 and 3.13.

## Scope of determinism

Exact bitwise equality is tested for repeated runs on the same software/hardware environment. Cross-platform floating-point behavior may differ slightly even when seeds and code are identical.
