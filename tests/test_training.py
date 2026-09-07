from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from alignment_metaplasticity.training import _ewc_penalty, train_epochs


def _model_and_loader():
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0], [-2.0, 0.5]]))
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[1.0, 2.0], [-1.0, 3.0]], dtype=torch.float64),
            torch.tensor([0, 1]),
        ),
        batch_size=1,
    )
    return model, loader


def test_static_freeze_survives_decay_and_repeated_momentum_steps():
    model, loader = _model_and_loader()
    before = model.weight.detach().clone()
    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    train_epochs(
        model, loader, torch.device("cpu"), epochs=3, lr=0.1,
        weight_decay=0.1, grad_mask={"weight": mask},
    )
    assert torch.equal(model.weight[mask == 0], before[mask == 0])
    assert not torch.equal(model.weight[mask == 1], before[mask == 1])


def test_plasticity_scales_decay_and_its_momentum():
    model = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0], [-2.0]]))
    scale = torch.tensor([[0.0], [0.25]], dtype=torch.float64)
    expected = model.weight.detach().clone()
    velocity = torch.zeros_like(expected)
    # Zero inputs give zero loss gradients, isolating decay and momentum.
    loader = DataLoader(
        TensorDataset(torch.zeros(3, 1, dtype=torch.float64), torch.zeros(3, dtype=torch.long)),
        batch_size=1,
    )
    for _ in range(3):
        velocity = 0.9 * velocity + scale * (0.1 * expected)
        expected = expected - 0.1 * velocity
    train_epochs(
        model, loader, torch.device("cpu"), epochs=1, lr=0.1,
        weight_decay=0.1, grad_scale={"weight": scale},
    )
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=1e-14)
    assert model.weight[0].item() == 1.0


@pytest.mark.parametrize("protection", [None, "grad_scale", "grad_mask"])
def test_unprotected_update_matches_pytorch_sgd(protection):
    model, loader = _model_and_loader()
    reference = deepcopy(model)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.03, momentum=0.9, weight_decay=0.01)
    for _ in range(3):
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(reference(x), y).backward()
            optimizer.step()
    kwargs = {} if protection is None else {protection: {"weight": torch.ones_like(model.weight)}}
    train_epochs(
        model, loader, torch.device("cpu"), epochs=3, lr=0.03,
        weight_decay=0.01, **kwargs,
    )
    torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)


def test_ewc_penalty_and_gradient_match_scalar_quadratic():
    model = nn.Module()
    model.weights = nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
    penalty = _ewc_penalty(
        model, {"weights": torch.tensor([1.0, 1.0])},
        {"weights": torch.tensor([3.0, 0.5])},
    )
    assert penalty.item() == 2.5
    penalty.backward()
    torch.testing.assert_close(model.weights.grad, torch.tensor([3.0, -1.0], dtype=torch.float64))


def test_ewc_penalty_does_not_depend_on_tensor_grouping():
    penalties = []
    for sizes in ([4], [1, 3], [1, 1, 1, 1]):
        model = nn.ParameterList([nn.Parameter(torch.ones(size)) for size in sizes])
        anchor = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        importance = {name: torch.ones_like(p) for name, p in model.named_parameters()}
        penalties.append(_ewc_penalty(model, anchor, importance).item())
    assert penalties == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("weight_decay", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_decay_is_rejected_before_training(weight_decay):
    model, loader = _model_and_loader()
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="weight_decay"):
        train_epochs(model, loader, torch.device("cpu"), epochs=1, lr=0.1, weight_decay=weight_decay)
    assert torch.equal(before, model.weight)
