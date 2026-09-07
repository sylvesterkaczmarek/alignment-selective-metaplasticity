from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import pytest
import torch

import alignment_metaplasticity.experiment as experiment
from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.summary import summarize_suite
from experiments import _common


def _run(seed=7, method="fine_tune"):
    alignment = {
        "selective_corrigibility": 0.8,
        "authorized_acceptance": 0.8,
        "unauthorized_resistance": 0.8,
    }
    return {
        "seed": seed,
        "method": method,
        "initial_alignment": {**alignment, "selective_corrigibility": 1.0},
        "final_alignment": alignment,
        "forgetting": 0.2,
        "mean_capability_accuracy": 0.7,
        "mean_acquisition_accuracy": 0.9,
        "latest_capability_accuracy": 0.9,
        "parameter_drift": 0.1,
    }


def test_unknown_method_is_rejected_before_pretraining(monkeypatch):
    def unexpected_pretraining(*args):
        pytest.fail("An invalid method must not start model training")

    monkeypatch.setattr(experiment, "prepare_seed", unexpected_pretraining)
    with pytest.raises(ValueError, match="Unknown method"):
        experiment.run_method(load_config(), 7, "metaplatsic")


@pytest.mark.parametrize(
    "seeds, methods",
    [
        ([], ("fine_tune",)),
        ([7, 7], ("fine_tune",)),
        ([7], ()),
        ([7], ("fine_tune", "fine_tune")),
        ([7], ("ewcc",)),
        ([True], ("fine_tune",)),
        ([-1], ("fine_tune",)),
    ],
)
def test_invalid_suite_is_rejected_before_any_run(monkeypatch, seeds, methods):
    def unexpected_run(*args, **kwargs):
        pytest.fail("An invalid suite must not start model training")

    monkeypatch.setattr(experiment, "run_method", unexpected_run)
    with pytest.raises(ValueError):
        experiment.run_suite(None, seeds, methods)


def test_repeated_seed_cannot_inflate_summary_sample_count():
    run = _run()
    with pytest.raises(ValueError, match="Duplicate seed/method"):
        summarize_suite({"runs": [run, copy.deepcopy(run)]})


def test_missing_declared_run_cannot_produce_partial_comparison():
    with pytest.raises(ValueError, match="every declared seed/method pair"):
        summarize_suite({"seeds": [7, 17], "methods": ["fine_tune"], "runs": [_run()]})


def test_mixed_bypass_results_cannot_be_silently_omitted():
    run = _run(seed=17)
    run["bypass"] = {"bypass_success_rate": 1.0, "alignment_after_bypass": run["final_alignment"]}
    with pytest.raises(ValueError, match="whether the bypass challenge"):
        summarize_suite({"runs": [_run(), run]})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.8"])
def test_invalid_observations_cannot_be_reported_as_statistics(value):
    run = _run()
    run["final_alignment"]["selective_corrigibility"] = value
    with pytest.raises(ValueError, match="finite numeric observations"):
        summarize_suite({"runs": [run]})


def test_summary_counts_distinct_seeds_and_uses_sample_standard_deviation():
    first, second = _run(), _run(seed=17)
    first["parameter_drift"], second["parameter_drift"] = 1.0, 3.0
    summary = summarize_suite({"runs": [first, second]})
    assert summary["fine_tune"]["parameter_drift"] == {
        "mean": 2.0,
        "std": pytest.approx(2**0.5),
        "n": 2,
    }


def test_individual_experiment_preserves_existing_reference_figures(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _common, "parse_args", lambda config: argparse.Namespace(config=config, seeds=[7], out=tmp_path)
    )
    monkeypatch.setattr(_common, "run_suite", lambda *args, **kwargs: {})
    monkeypatch.setattr(_common, "summarize_suite", lambda suite: {})
    calls = []

    def write_figure(summary, directory):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "alignment_retention.png"
        path.write_text(f"figure {len(calls)}", encoding="utf-8")
        calls.append(path)

    monkeypatch.setattr(_common, "plot_summary", write_figure)
    _common.run_named("unused", "summary")
    _common.run_named("unused", "baseline_forgetting", methods=("fine_tune",), include_bypass=False)
    assert calls[0] != calls[1]
    assert calls[0].read_text(encoding="utf-8") == "figure 0"
    assert calls[1].read_text(encoding="utf-8") == "figure 1"


def test_suite_records_runtime_and_exact_installed_source(monkeypatch):
    monkeypatch.setattr(experiment, "run_method", lambda *args, **kwargs: _run())
    suite = experiment.run_suite(None, [7], ("fine_tune",), include_bypass=False)
    metadata = suite["metadata"]
    assert metadata["device"] == "cpu"
    assert metadata["torch"] == torch.__version__
    assert metadata["torch_num_threads"] == torch.get_num_threads()
    source = Path(experiment.__file__)
    assert metadata["source_files_sha256"][source.name] == hashlib.sha256(source.read_bytes()).hexdigest()
