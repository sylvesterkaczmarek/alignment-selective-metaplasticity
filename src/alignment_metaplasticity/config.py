from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "device": "cpu",
    "model": {
        "signal_dim": 8,
        "hidden_dim": 48,
        "depth": 2,
        "num_capability_tasks": 3,
    },
    "data": {
        "alignment_train_size": 1200,
        "alignment_eval_size": 1500,
        "capability_train_size": 1200,
        "capability_eval_size": 1000,
        "bypass_train_size": 900,
        "batch_size": 128,
    },
    "training": {
        "alignment_epochs": 18,
        "capability_epochs": 14,
        "bypass_epochs": 12,
        "lr": 0.03,
        "weight_decay": 0.0001,
    },
    "importance": {
        "max_batches": 8,
        "quantile": 0.95,
    },
    "static": {
        "freeze_quantile": 0.85,
    },
    "ewc": {
        "lambda": 18.0,
    },
    "metaplastic": {
        "strength": 200.0,
        "min_plasticity": 0.05,
        "ema_decay": 0.80,
    },
}


def validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32), excluding booleans")


def validate_config(cfg: dict[str, Any]) -> None:
    """Reject malformed experiment settings before generating data or training."""
    def structure(actual: Any, template: dict[str, Any], prefix: str = "") -> None:
        if not isinstance(actual, dict):
            raise ValueError(f"{prefix or 'config'} must be a mapping")
        unknown = actual.keys() - template.keys()
        missing = template.keys() - actual.keys()
        if unknown or missing:
            raise ValueError(f"{prefix or 'config'} has unknown keys {sorted(unknown, key=str)} or missing keys {sorted(missing)}")
        for key, value in template.items():
            if isinstance(value, dict):
                structure(actual[key], value, f"{prefix}.{key}".lstrip("."))

    structure(cfg, DEFAULT_CONFIG)
    validate_seed(cfg["seed"])
    if not isinstance(cfg["device"], str) or not cfg["device"].strip():
        raise ValueError("device must be a non-empty string")

    def integer(section: str, key: str, minimum: int) -> None:
        value = cfg[section][key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{section}.{key} must be an integer >= {minimum}")

    def real(section: str, key: str, minimum: float, maximum: float | None = None, *, strict: bool = False) -> None:
        value = cfg[section][key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{section}.{key} must be a finite number")
        if value < minimum or (strict and value == minimum) or (maximum is not None and value > maximum):
            raise ValueError(f"{section}.{key} is outside its supported range")

    for key in ("signal_dim", "hidden_dim", "num_capability_tasks"):
        integer("model", key, 1)
    integer("model", "depth", 0)
    for key in DEFAULT_CONFIG["data"]:
        integer("data", key, 2 if key == "bypass_train_size" else 1)
    integer("training", "alignment_epochs", 1)
    integer("training", "capability_epochs", 0)
    integer("training", "bypass_epochs", 0)
    real("training", "lr", 0.0, strict=True)
    real("training", "weight_decay", 0.0)
    integer("importance", "max_batches", 1)
    real("importance", "quantile", 0.0, 1.0)
    real("static", "freeze_quantile", 0.0, 1.0)
    real("ewc", "lambda", 0.0)
    real("metaplastic", "strength", 0.0)
    real("metaplastic", "min_plasticity", 0.0, 1.0)
    real("metaplastic", "ema_decay", 0.0, 1.0)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(update, dict):
        raise ValueError("configuration updates must be mappings")
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is None:
            loaded = {}
        cfg = _deep_update(cfg, loaded)
    if overrides is not None:
        cfg = _deep_update(cfg, overrides)
    validate_config(cfg)
    return cfg
