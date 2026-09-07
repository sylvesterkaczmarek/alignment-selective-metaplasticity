import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from alignment_metaplasticity.importance import (
    compute_gradient_importance,
    ema_importance,
    normalize_importance,
    plasticity_from_importance,
    static_train_masks,
)


@pytest.mark.parametrize("quantile, expected_frozen", [(0.0, 20), (0.5, 10), (0.85, 3), (1.0, 0)])
def test_static_freezing_keeps_requested_count_when_importances_tie(quantile, expected_frozen):
    importance = {"first": torch.zeros(2, 3), "second": torch.zeros(14)}
    masks = static_train_masks(importance, quantile)
    flat = torch.cat([mask.flatten() for mask in masks.values()])
    assert int((flat == 0).sum()) == expected_frozen
    assert torch.equal(flat[:expected_frozen], torch.zeros(expected_frozen))
    assert torch.equal(flat[expected_frozen:], torch.ones(20 - expected_frozen))
    assert masks["first"].shape == importance["first"].shape


def test_static_freezing_ranks_globally_before_breaking_ties():
    importance = {"first": torch.tensor([1.0, 3.0]), "second": torch.tensor([3.0, 2.0])}
    masks = static_train_masks(importance, freeze_quantile=0.75)
    assert torch.equal(masks["first"], torch.tensor([1.0, 0.0]))
    assert torch.equal(masks["second"], torch.tensor([1.0, 1.0]))


def test_importance_is_explicitly_a_squared_batch_mean_gradient_proxy():
    model = nn.Linear(1, 2, bias=False)
    nn.init.zeros_(model.weight)
    dataset = TensorDataset(torch.ones(2, 1), torch.tensor([0, 1]))
    batch_importance = compute_gradient_importance(
        model, DataLoader(dataset, batch_size=2), torch.device("cpu"), max_batches=2,
    )
    example_importance = compute_gradient_importance(
        model, DataLoader(dataset, batch_size=1), torch.device("cpu"), max_batches=2,
    )
    # The two example gradients cancel before squaring in the larger batch.
    assert torch.equal(batch_importance["weight"], torch.zeros_like(model.weight))
    assert torch.equal(example_importance["weight"], torch.ones_like(model.weight))


@pytest.mark.parametrize("max_batches", [0, -1, 1.5, True])
def test_invalid_importance_batch_limit_is_rejected_before_accessing_model(max_batches):
    with pytest.raises(ValueError, match="max_batches"):
        compute_gradient_importance(None, None, torch.device("cpu"), max_batches=max_batches)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True])
@pytest.mark.parametrize("argument", ["quantile", "freeze_quantile", "decay", "min_plasticity"])
def test_invalid_importance_unit_interval_arguments_are_rejected(argument, value):
    importance = {"w": torch.ones(2)}
    with pytest.raises(ValueError, match=argument):
        if argument == "quantile":
            normalize_importance(importance, quantile=value)
        elif argument == "freeze_quantile":
            static_train_masks(importance, freeze_quantile=value)
        elif argument == "decay":
            ema_importance(importance, importance, decay=value)
        else:
            plasticity_from_importance(importance, strength=1.0, min_plasticity=value)


@pytest.mark.parametrize("strength", [-1.0, float("nan"), float("inf"), True])
def test_invalid_plasticity_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="strength"):
        plasticity_from_importance({"w": torch.ones(2)}, strength=strength, min_plasticity=0.05)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_importance_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        normalize_importance({"w": torch.tensor([value])})


def test_empty_importance_maps_remain_empty():
    assert normalize_importance({}) == {}
    assert static_train_masks({}, freeze_quantile=0.85) == {}
