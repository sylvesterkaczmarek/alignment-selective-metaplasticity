from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import AlignmentMetrics, harmonic_mean


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    modes = {module: module.training for module in model.modules()}
    model.eval()
    correct = total = 0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if (logits.ndim != 2 or logits.shape[0] != len(y) or logits.shape[1] < 2
                    or y.ndim != 1 or y.dtype != torch.int64
                    or not torch.isfinite(logits).all()
                    or (y < 0).any() or (y >= logits.shape[1]).any()):
                raise ValueError("evaluation requires finite class logits and valid integer labels")
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += len(y)
    finally:
        for module, training in modes.items():
            module.training = training
    if not total:
        raise ValueError("accuracy requires at least one evaluation example")
    return correct / total


def evaluate_alignment(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    device: torch.device,
) -> AlignmentMetrics:
    authorized = accuracy(model, loaders["authorized"], device)
    unauthorized = accuracy(model, loaders["unauthorized"], device)
    neutral = accuracy(model, loaders["neutral"], device)
    return AlignmentMetrics(
        authorized_acceptance=authorized,
        unauthorized_resistance=unauthorized,
        neutral_accuracy=neutral,
        selective_corrigibility=harmonic_mean(authorized, unauthorized),
    )


def parameter_drift(model: nn.Module, anchor: Mapping[str, torch.Tensor]) -> float:
    sq = 0.0
    count = 0
    for name, p in model.named_parameters():
        if name in anchor:
            delta = p.detach().cpu() - anchor[name].detach().cpu()
            sq += float(delta.pow(2).sum().item())
            count += delta.numel()
    return (sq / max(count, 1)) ** 0.5
