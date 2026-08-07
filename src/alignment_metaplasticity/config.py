from __future__ import annotations

from copy import deepcopy
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


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
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
            loaded = yaml.safe_load(f) or {}
        cfg = _deep_update(cfg, loaded)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg
