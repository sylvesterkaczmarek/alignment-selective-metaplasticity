from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, stdev
from typing import Any

from .config import validate_seed
from .experiment import METHODS


def _stats(values: list[float]) -> dict[str, float]:
    if not values or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("Summary statistics require nonempty, finite numeric observations")
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def summarize_suite(suite: dict[str, Any]) -> dict[str, Any]:
    if not suite["runs"]:
        raise ValueError("Cannot summarize an empty suite")
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    bypass_presence = set()
    for run in suite["runs"]:
        validate_seed(run["seed"])
        if run["method"] not in METHODS:
            raise ValueError(f"Unknown method {run['method']!r}")
        key = (run["seed"], run["method"])
        if key in seen:
            raise ValueError(f"Duplicate seed/method run {key!r}; repetitions are not independent samples")
        seen.add(key)
        bypass_presence.add("bypass" in run)
        by_method[run["method"]].append(run)
    if len(bypass_presence) > 1:
        raise ValueError("All runs must agree on whether the bypass challenge was included")
    if "seeds" in suite and "methods" in suite:
        seeds, methods = suite["seeds"], suite["methods"]
        if not seeds or not methods:
            raise ValueError("Declared seeds and methods must not be empty")
        for seed in seeds:
            validate_seed(seed)
        if len(set(seeds)) != len(seeds) or len(set(methods)) != len(methods):
            raise ValueError("Declared seeds and methods must be unique")
        expected = {(seed, method) for seed in seeds for method in methods}
        if seen != expected:
            raise ValueError("Runs must match every declared seed/method pair exactly once")

    summary: dict[str, Any] = {}
    for method, runs in by_method.items():
        summary[method] = {
            "initial_selective_corrigibility": _stats(
                [r["initial_alignment"]["selective_corrigibility"] for r in runs]
            ),
            "final_selective_corrigibility": _stats(
                [r["final_alignment"]["selective_corrigibility"] for r in runs]
            ),
            "authorized_acceptance": _stats(
                [r["final_alignment"]["authorized_acceptance"] for r in runs]
            ),
            "unauthorized_resistance": _stats(
                [r["final_alignment"]["unauthorized_resistance"] for r in runs]
            ),
            "forgetting": _stats([r["forgetting"] for r in runs]),
            "mean_capability_accuracy": _stats([r["mean_capability_accuracy"] for r in runs]),
            "mean_acquisition_accuracy": _stats([r["mean_acquisition_accuracy"] for r in runs]),
            "latest_capability_accuracy": _stats([r["latest_capability_accuracy"] for r in runs]),
            "parameter_drift": _stats([r["parameter_drift"] for r in runs]),
        }
        if "bypass" in runs[0]:
            summary[method]["bypass_success_rate"] = _stats(
                [r["bypass"]["bypass_success_rate"] for r in runs]
            )
            summary[method]["alignment_after_bypass"] = _stats(
                [r["bypass"]["alignment_after_bypass"]["selective_corrigibility"] for r in runs]
            )
    return summary
