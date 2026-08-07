.PHONY: install test baseline compare ablation bypass all

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

baseline:
	python -m experiments.baseline_forgetting

compare:
	python -m experiments.metaplastic_retention

ablation:
	python -m experiments.plasticity_ablation

bypass:
	python -m experiments.bypass_test

all:
	python -m experiments.run_all
