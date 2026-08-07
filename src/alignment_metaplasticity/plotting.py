from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_summary(summary: dict[str, Any], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(summary)
    labels = [m.replace("_", " ").title() for m in methods]

    outputs: list[Path] = []

    values = [summary[m]["final_selective_corrigibility"]["mean"] for m in methods]
    errors = [summary[m]["final_selective_corrigibility"]["std"] for m in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(labels, values, yerr=errors, capsize=4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Selective-corrigibility score")
    ax.set_title("Alignment proxy after sequential capability updates")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    path = out_dir / "alignment_retention.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    values = [summary[m]["mean_capability_accuracy"]["mean"] for m in methods]
    errors = [summary[m]["mean_capability_accuracy"]["std"] for m in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(labels, values, yerr=errors, capsize=4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean final capability accuracy")
    ax.set_title("Final capability retention")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    path = out_dir / "capability_accuracy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    values = [summary[m]["mean_acquisition_accuracy"]["mean"] for m in methods]
    errors = [summary[m]["mean_acquisition_accuracy"]["std"] for m in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(labels, values, yerr=errors, capsize=4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean acquisition accuracy")
    ax.set_title("New capability acquisition at time of training")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    path = out_dir / "capability_acquisition.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(path)

    if all("bypass_success_rate" in summary[m] for m in methods):
        values = [summary[m]["bypass_success_rate"]["mean"] for m in methods]
        errors = [summary[m]["bypass_success_rate"]["std"] for m in methods]
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        ax.bar(labels, values, yerr=errors, capsize=4)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Bypass success rate")
        ax.set_title("Route-around challenge")
        ax.tick_params(axis="x", rotation=18)
        fig.tight_layout()
        path = out_dir / "bypass_success.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

        values = [summary[m]["alignment_after_bypass"]["mean"] for m in methods]
        errors = [summary[m]["alignment_after_bypass"]["std"] for m in methods]
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        ax.bar(labels, values, yerr=errors, capsize=4)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Selective-corrigibility score")
        ax.set_title("Original alignment proxy after route-around challenge")
        ax.tick_params(axis="x", rotation=18)
        fig.tight_layout()
        path = out_dir / "alignment_after_bypass.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    return outputs


def plot_ablation(payload: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strengths = [float(r["metaplastic_strength"]) for r in payload["runs"]]
    retention = [float(r["final_alignment"]["selective_corrigibility"]) for r in payload["runs"]]
    acquisition = [float(r["mean_acquisition_accuracy"]) for r in payload["runs"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(strengths, retention, marker="o", label="Alignment retention")
    ax.plot(strengths, acquisition, marker="o", label="Capability acquisition")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Metaplastic strength")
    ax.set_ylabel("Score")
    ax.set_title("Metaplastic-strength ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
