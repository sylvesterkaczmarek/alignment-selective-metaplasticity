#!/usr/bin/env bash
set -euo pipefail

python -m experiments.run_all --seeds 7 17 29 --out results
python -m experiments.plasticity_ablation --seed 7 --out results/plasticity_ablation.json
pytest -q
