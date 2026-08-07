from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader


TensorMap = dict[str, torch.Tensor]


def clone_parameters(model: nn.Module) -> TensorMap:
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def compute_gradient_importance(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 8,
    quantile: float = 0.95,
) -> TensorMap:
    """Estimate diagonal parameter importance from squared alignment-loss gradients."""
    criterion = nn.CrossEntropyLoss()
    importance = {name: torch.zeros_like(p, device=device) for name, p in model.named_parameters() if p.requires_grad}
    seen = 0
    was_training = model.training
    model.train()
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                importance[name].add_(p.grad.detach().pow(2))
        seen += 1
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    denom = max(seen, 1)
    for name in importance:
        importance[name].div_(denom)
    return normalize_importance(importance, quantile=quantile)


def normalize_importance(importance: Mapping[str, torch.Tensor], quantile: float = 0.95) -> TensorMap:
    flat = torch.cat([v.detach().flatten().float().cpu() for v in importance.values()])
    scale = torch.quantile(flat, quantile).item() if flat.numel() else 1.0
    scale = max(scale, 1e-12)
    return {name: (value / scale).clamp_(0.0, 1.0) for name, value in importance.items()}


def ema_importance(old: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor], decay: float) -> TensorMap:
    return {name: decay * old[name] + (1.0 - decay) * new[name] for name in old}


def plasticity_from_importance(
    importance: Mapping[str, torch.Tensor],
    strength: float,
    min_plasticity: float,
) -> TensorMap:
    return {
        name: min_plasticity + (1.0 - min_plasticity) * torch.exp(-strength * value)
        for name, value in importance.items()
    }


def static_train_masks(importance: Mapping[str, torch.Tensor], freeze_quantile: float) -> TensorMap:
    flat = torch.cat([v.detach().flatten().float().cpu() for v in importance.values()])
    threshold = torch.quantile(flat, freeze_quantile).item() if flat.numel() else float("inf")
    return {name: (value < threshold).to(value.dtype) for name, value in importance.items()}
