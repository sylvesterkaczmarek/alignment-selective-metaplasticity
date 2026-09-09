from __future__ import annotations

import copy
import hashlib
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .config import load_config, validate_config, validate_seed
from .evaluation import accuracy, evaluate_alignment, parameter_drift
from .importance import (
    clone_parameters,
    compute_gradient_importance,
    ema_importance,
    plasticity_from_importance,
    static_train_masks,
)
from .model import SelectiveCorrigibilityNet
from .repro import set_seed
from .tasks import BenchmarkSpec, make_alignment_tensors, make_capability_tensors, make_loader
from .training import train_epochs


Method = Literal["fine_tune", "static_freeze", "ewc", "metaplastic"]
METHODS: tuple[Method, ...] = ("fine_tune", "static_freeze", "ewc", "metaplastic")


def _validate_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {', '.join(METHODS)}")


def runtime_metadata(device_name: str) -> dict[str, Any]:
    """Describe the actual runtime and source bytes without requiring a Git checkout."""
    source_dir = Path(__file__).parent
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(_device(device_name)),
        "cuda_runtime": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "source_files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_dir.glob("*.py"))
        },
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_loaders(cfg: dict[str, Any], spec: BenchmarkSpec, seed: int, *, rule_seed: int = 0):
    d = cfg["data"]
    batch = d["batch_size"]
    alignment_train = make_loader(
        make_alignment_tensors(d["alignment_train_size"], spec, seed + 11, mode="mixed"),
        batch,
        True,
        seed + 12,
    )
    alignment_eval = {
        mode: make_loader(
            make_alignment_tensors(d["alignment_eval_size"], spec, seed + offset, mode=mode),
            batch,
            False,
            seed + offset + 1,
        )
        for mode, offset in (("authorized", 21), ("unauthorized", 31), ("neutral", 41))
    }
    capability_train = {
        task: make_loader(
            make_capability_tensors(
                d["capability_train_size"], spec, task, seed + 100 * task, rule_seed=rule_seed + task
            ),
            batch,
            True,
            seed + 100 * task + 1,
        )
        for task in range(1, spec.num_capability_tasks + 1)
    }
    capability_eval = {
        task: make_loader(
            make_capability_tensors(
                d["capability_eval_size"], spec, task, seed + 100 * task + 50, rule_seed=rule_seed + task
            ),
            batch,
            False,
            seed + 100 * task + 51,
        )
        for task in range(1, spec.num_capability_tasks + 1)
    }
    bypass_n = d["bypass_train_size"]
    attack_x, attack_y = make_alignment_tensors(
        bypass_n // 2, spec, seed + 701, mode="unauthorized", bypass_flag=1.0, bypass_target=True
    )
    preserve_x, preserve_y = make_alignment_tensors(
        bypass_n - bypass_n // 2, spec, seed + 711, mode="mixed", bypass_flag=0.0, bypass_target=False
    )
    bypass_train = make_loader(
        (torch.cat([attack_x, preserve_x], dim=0), torch.cat([attack_y, preserve_y], dim=0)),
        batch,
        True,
        seed + 702,
    )
    bypass_eval = make_loader(
        make_alignment_tensors(
            d["alignment_eval_size"], spec, seed + 703, mode="unauthorized", bypass_flag=1.0, bypass_target=True
        ),
        batch,
        False,
        seed + 704,
    )
    return alignment_train, alignment_eval, capability_train, capability_eval, bypass_train, bypass_eval


def prepare_seed(cfg: dict[str, Any], seed: int):
    set_seed(seed)
    device = _device(cfg["device"])
    m = cfg["model"]
    spec = BenchmarkSpec(signal_dim=m["signal_dim"], num_capability_tasks=m["num_capability_tasks"])
    loaders = build_loaders(cfg, spec, seed)
    model = SelectiveCorrigibilityNet(spec.input_dim, hidden_dim=m["hidden_dim"], depth=m["depth"]).to(device)
    train = cfg["training"]
    train_epochs(
        model,
        loaders[0],
        device,
        epochs=train["alignment_epochs"],
        lr=train["lr"],
        weight_decay=train["weight_decay"],
    )
    return model, spec, loaders, device


def run_method(cfg: dict[str, Any], seed: int, method: Method, include_bypass: bool = True) -> dict[str, Any]:
    validate_config(cfg)
    validate_seed(seed)
    _validate_method(method)
    base_model, spec, loaders, device = prepare_seed(cfg, seed)
    alignment_train, alignment_eval, capability_train, capability_eval, bypass_train, bypass_eval = loaders
    base_metrics = evaluate_alignment(base_model, alignment_eval, device)
    if base_metrics.selective_corrigibility < 0.90:
        raise RuntimeError(
            f"Alignment pretraining underfit for seed {seed}: selective_corrigibility={base_metrics.selective_corrigibility:.3f}"
        )

    model = copy.deepcopy(base_model)
    anchor = clone_parameters(model)
    importance = compute_gradient_importance(
        model,
        alignment_train,
        device,
        max_batches=cfg["importance"]["max_batches"],
        quantile=cfg["importance"]["quantile"],
    )
    static_mask = static_train_masks(importance, cfg["static"]["freeze_quantile"])

    history: list[dict[str, Any]] = []
    train_cfg = cfg["training"]
    grad_diag = None
    for task in range(1, spec.num_capability_tasks + 1):
        kwargs: dict[str, Any] = {}
        if method == "static_freeze":
            kwargs["grad_mask"] = static_mask
        elif method == "ewc":
            kwargs.update(
                ewc_anchor=anchor,
                ewc_importance=importance,
                ewc_lambda=cfg["ewc"]["lambda"],
            )
        elif method == "metaplastic":
            plasticity = plasticity_from_importance(
                importance,
                strength=cfg["metaplastic"]["strength"],
                min_plasticity=cfg["metaplastic"]["min_plasticity"],
            )
            kwargs["grad_scale"] = plasticity

        diag = train_epochs(
            model,
            capability_train[task],
            device,
            epochs=train_cfg["capability_epochs"],
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
            **kwargs,
        )
        grad_diag = asdict(diag)

        if method == "metaplastic":
            new_importance = compute_gradient_importance(
                model,
                alignment_train,
                device,
                max_batches=cfg["importance"]["max_batches"],
                quantile=cfg["importance"]["quantile"],
            )
            importance = ema_importance(importance, new_importance, cfg["metaplastic"]["ema_decay"])

        align = evaluate_alignment(model, alignment_eval, device)
        capabilities = {str(t): accuracy(model, capability_eval[t], device) for t in range(1, task + 1)}
        history.append(
            {
                "after_capability_task": task,
                "alignment": align.to_dict(),
                "capability_accuracy": capabilities,
                "parameter_drift": parameter_drift(model, anchor),
            }
        )

    final_alignment = evaluate_alignment(model, alignment_eval, device)
    final_caps = {str(t): accuracy(model, capability_eval[t], device) for t in capability_eval}
    acquisition_scores = [
        float(step["capability_accuracy"][str(step["after_capability_task"])]) for step in history
    ]
    result: dict[str, Any] = {
        "seed": seed,
        "method": method,
        "initial_alignment": base_metrics.to_dict(),
        "final_alignment": final_alignment.to_dict(),
        "forgetting": base_metrics.selective_corrigibility - final_alignment.selective_corrigibility,
        "capability_accuracy": final_caps,
        "mean_capability_accuracy": sum(final_caps.values()) / len(final_caps),
        "mean_acquisition_accuracy": sum(acquisition_scores) / len(acquisition_scores),
        "latest_capability_accuracy": final_caps[str(spec.num_capability_tasks)],
        "parameter_drift": parameter_drift(model, anchor),
        "history": history,
        "train_diagnostics": grad_diag,
    }

    if include_bypass:
        bypass_kwargs: dict[str, Any] = {}
        if method == "static_freeze":
            bypass_kwargs["grad_mask"] = static_mask
        elif method == "ewc":
            bypass_kwargs.update(
                ewc_anchor=anchor,
                ewc_importance=importance,
                ewc_lambda=cfg["ewc"]["lambda"],
            )
        elif method == "metaplastic":
            bypass_kwargs["grad_scale"] = plasticity_from_importance(
                importance,
                strength=cfg["metaplastic"]["strength"],
                min_plasticity=cfg["metaplastic"]["min_plasticity"],
            )
        train_epochs(
            model,
            bypass_train,
            device,
            epochs=train_cfg["bypass_epochs"],
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
            **bypass_kwargs,
        )
        post_bypass_alignment = evaluate_alignment(model, alignment_eval, device)
        result["bypass"] = {
            "bypass_success_rate": accuracy(model, bypass_eval, device),
            "alignment_after_bypass": post_bypass_alignment.to_dict(),
            "alignment_retention_delta": final_alignment.selective_corrigibility
            - post_bypass_alignment.selective_corrigibility,
        }
    return result


def run_suite(
    config_path: str | Path | None,
    seeds: list[int],
    methods: tuple[Method, ...] = METHODS,
    include_bypass: bool = True,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if not methods:
        raise ValueError("At least one method is required")
    for seed in seeds:
        validate_seed(seed)
    for method in methods:
        _validate_method(method)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique; repeated seeds are not independent runs")
    if len(set(methods)) != len(methods):
        raise ValueError("Methods must be unique")
    cfg = load_config(config_path)
    runs = []
    for seed in seeds:
        for method in methods:
            runs.append(run_method(cfg, seed, method, include_bypass=include_bypass))
    return {
        "config": cfg,
        "seeds": list(seeds),
        "methods": list(methods),
        "metadata": runtime_metadata(cfg["device"]),
        "runs": runs,
    }
