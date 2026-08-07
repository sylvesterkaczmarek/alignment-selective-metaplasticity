from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev
from typing import Any


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def summarize_suite(suite: dict[str, Any]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in suite["runs"]:
        by_method[run["method"]].append(run)

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
