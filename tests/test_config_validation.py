import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.evaluation import accuracy


@pytest.mark.parametrize("overrides", [
    {"metaplastic": {"strenght": 20}},
    {"metaplastic": {"strength": -200}},
    {"metaplastic": {"strength": math.inf}},
    {"metaplastic": {"min_plasticity": -0.1}},
    {"metaplastic": {"min_plasticity": 1.1}},
    {"metaplastic": {"ema_decay": math.nan}},
    {"importance": {"max_batches": 0}},
    {"importance": {"quantile": 1.1}},
    {"training": {"lr": 0}},
    {"training": {"alignment_epochs": True}},
    {"training": {"capability_epochs": 1.5}},
    {"training": {"weight_decay": -0.1}},
    {"model": {"num_capability_tasks": 0}},
    {"data": {"alignment_eval_size": 0}},
    {"data": {"bypass_train_size": 1}},
    {"ewc": {"lambda": "18"}},
    {"seed": True},
    {"device": None},
    {"model": []},
])
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        load_config(overrides=overrides)


@pytest.mark.parametrize("text", ["false", "[]", "5", "training: []", "metaplastic:\n  strenght: 20"])
def test_yaml_cannot_silently_fall_back_to_defaults(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_partial_config_and_deliberate_no_update_controls_are_supported():
    cfg = load_config(overrides={
        "metaplastic": {"strength": 0, "min_plasticity": 0, "ema_decay": 1},
        "static": {"freeze_quantile": 1},
        "training": {"capability_epochs": 0, "bypass_epochs": 0},
    })
    assert cfg["metaplastic"]["strength"] == 0
    assert cfg["data"]["batch_size"] == 128


def test_empty_evaluation_does_not_report_zero_accuracy():
    loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="at least one"):
        accuracy(nn.Linear(2, 2), loader, torch.device("cpu"))
