from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch
from torch import nn
from torch.utils.data import DataLoader


TensorMap = dict[str, torch.Tensor]


def _finite_number(name: str, value: float, *, upper: float | None = None) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or (upper is not None and value > upper)
    ):
        bound = "nonnegative" if upper is None else f"between 0 and {upper:g}"
        raise ValueError(f"{name} must be a finite number {bound}")


def _validate_values(importance: Mapping[str, torch.Tensor]) -> None:
    for name, value in importance.items():
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"importance[{name!r}] must contain finite nonnegative values")


def clone_parameters(model: nn.Module) -> TensorMap:
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def compute_gradient_importance(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 8,
    quantile: float = 0.95,
) -> TensorMap:
    """Average squared gradients of the mean alignment loss within each batch.

    Batches receive equal weight, including a shorter final batch. Squaring
    takes place after averaging example losses, so opposing example gradients
    can cancel. This is a batch-dependent importance proxy, not a per-example
    empirical Fisher estimate.
    """
    if type(max_batches) is not int or max_batches < 1:
        raise ValueError("max_batches must be a positive integer")
    _finite_number("quantile", quantile, upper=1.0)
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
    _finite_number("quantile", quantile, upper=1.0)
    _validate_values(importance)
    if not importance:
        return {}
    flat = torch.cat([v.detach().flatten().float().cpu() for v in importance.values()])
    scale = torch.quantile(flat, quantile).item() if flat.numel() else 1.0
    scale = max(scale, 1e-12)
    return {name: (value / scale).clamp_(0.0, 1.0) for name, value in importance.items()}


def ema_importance(old: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor], decay: float) -> TensorMap:
    _finite_number("decay", decay, upper=1.0)
    return {name: decay * old[name] + (1.0 - decay) * new[name] for name in old}


def plasticity_from_importance(
    importance: Mapping[str, torch.Tensor],
    strength: float,
    min_plasticity: float,
) -> TensorMap:
    _finite_number("strength", strength)
    _finite_number("min_plasticity", min_plasticity, upper=1.0)
    _validate_values(importance)
    return {
        name: min_plasticity + (1.0 - min_plasticity) * torch.exp(-strength * value)
        for name, value in importance.items()
    }


def static_train_masks(importance: Mapping[str, torch.Tensor], freeze_quantile: float) -> TensorMap:
    """Freeze the highest-ranked ``N - floor(freeze_quantile * N)`` elements.

    Equal importances are ordered by mapping insertion order and then by flat
    element index. This keeps the requested count even when many values tie;
    quantile zero freezes every element and quantile one freezes none.
    """
    _finite_number("freeze_quantile", freeze_quantile, upper=1.0)
    _validate_values(importance)
    if not importance:
        return {}
    flat = torch.cat([v.detach().flatten().to(device="cpu", dtype=torch.float64) for v in importance.values()])
    freeze_count = flat.numel() - math.floor(freeze_quantile * flat.numel())
    ranked = torch.argsort(flat, descending=True, stable=True)
    trainable = torch.ones(flat.numel(), dtype=torch.bool)
    trainable[ranked[:freeze_count]] = False
    masks = {}
    offset = 0
    for name, value in importance.items():
        masks[name] = trainable[offset : offset + value.numel()].reshape(value.shape).to(value)
        offset += value.numel()
    return masks


def compute_example_importance(model, loader, device, *, kind="empirical", quantile=0.95,
                               normalize=True, stats=None):
    """Mean per-example gradient squares, or exact class expectation under the model.

    Evaluation mode defines independent examples. Gradients and module modes are
    preserved. Class enumeration is intended for small classifiers, not an LLM vocabulary.
    Raw estimates are available to check batching invariance before normalisation.
    """
    if kind not in ("empirical", "model_fisher"):
        raise ValueError("kind must be empirical or model_fisher")
    _finite_number("quantile", quantile, upper=1.0)
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if not params:
        raise ValueError("importance requires trainable parameters")
    totals = {n:torch.zeros_like(p, dtype=torch.float64, device=device) for n,p in params.items()}
    modes = {module:module.training for module in model.modules()}
    model.eval()
    count = 0
    try:
        for batch_x, batch_y in loader:
            for x, y in zip(batch_x.to(device), batch_y.to(device)):
                logp = model(x.unsqueeze(0)).log_softmax(dim=-1)
                if logp.ndim != 2 or logp.shape[0] != 1 or logp.shape[1] < 2:
                    raise ValueError("expected one vector of class logits per example")
                labels = range(logp.shape[1]) if kind == "model_fisher" else (int(y),)
                for i, label in enumerate(labels):
                    if not 0 <= label < logp.shape[1]:
                        raise ValueError("label outside model class range")
                    gradients = torch.autograd.grad(logp[0, label], tuple(params.values()),
                                                    retain_graph=i < len(labels)-1, allow_unused=True)
                    weight = logp[0, label].detach().exp().double() if kind == "model_fisher" else 1.
                    for (name, _), g in zip(params.items(), gradients):
                        if g is not None:
                            totals[name].add_(weight * g.detach().double().square())
                    if stats is not None:
                        stats["backward_calls"] = stats.get("backward_calls", 0) + 1
                count += 1
    finally:
        for module, training in modes.items():
            module.training = training
    if count == 0:
        raise ValueError("importance requires at least one example")
    values = {name:(value / count).to(params[name].dtype) for name,value in totals.items()}
    return normalize_importance(values, quantile) if normalize else values
