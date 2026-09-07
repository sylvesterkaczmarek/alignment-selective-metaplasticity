from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import nn
from torch.utils.data import DataLoader


TensorMap = Mapping[str, torch.Tensor]


@dataclass
class TrainDiagnostics:
    mean_loss: float
    protected_grad_mean: float | None = None
    unprotected_grad_mean: float | None = None


def _ewc_penalty(
    model: nn.Module,
    anchor: TensorMap,
    importance: TensorMap,
) -> torch.Tensor:
    """Return half the importance-weighted sum over scalar parameters.

    ``ewc_lambda`` multiplies this penalty in ``train_epochs``. Importance is
    the benchmark's normalized squared minibatch-gradient proxy, rather than
    an estimate of the model Fisher information.
    """
    terms = []
    for name, p in model.named_parameters():
        if p.requires_grad and name in anchor:
            delta = p - anchor[name].to(device=p.device, dtype=p.dtype)
            weights = importance[name].to(device=p.device, dtype=p.dtype)
            terms.append((weights * delta.pow(2)).sum())
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return 0.5 * torch.stack(terms).sum()


def train_epochs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    *,
    grad_scale: TensorMap | None = None,
    grad_mask: TensorMap | None = None,
    ewc_anchor: TensorMap | None = None,
    ewc_importance: TensorMap | None = None,
    ewc_lambda: float = 0.0,
) -> TrainDiagnostics:
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, Real)
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be a finite nonnegative number")
    criterion = nn.CrossEntropyLoss()
    # Apply L2 decay before protection so it cannot move frozen elements or
    # bypass their plasticity coefficient. Momentum starts fresh for each call;
    # the fixed mask/scale therefore also applies to its accumulated updates.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    losses: list[float] = []
    protected_grads: list[float] = []
    unprotected_grads: list[float] = []

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if ewc_anchor is not None and ewc_importance is not None and ewc_lambda > 0:
                loss = loss + ewc_lambda * _ewc_penalty(model, ewc_anchor, ewc_importance)
            loss.backward()

            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                scale = None
                if grad_scale is not None and name in grad_scale:
                    scale = grad_scale[name].to(p.grad.device)
                    imp_proxy = 1.0 - scale
                    protected = imp_proxy >= 0.5
                    if protected.any():
                        protected_grads.append(float(p.grad.detach().abs()[protected].mean().item()))
                    if (~protected).any():
                        unprotected_grads.append(float(p.grad.detach().abs()[~protected].mean().item()))
                if weight_decay:
                    p.grad.add_(p.detach(), alpha=weight_decay)
                if scale is not None:
                    p.grad.mul_(scale)
                if grad_mask is not None and name in grad_mask:
                    p.grad.mul_(grad_mask[name].to(p.grad.device))

            optimizer.step()
            losses.append(float(loss.detach().item()))

    return TrainDiagnostics(
        mean_loss=sum(losses) / max(len(losses), 1),
        protected_grad_mean=(sum(protected_grads) / len(protected_grads)) if protected_grads else None,
        unprotected_grad_mean=(sum(unprotected_grads) / len(unprotected_grads)) if unprotected_grads else None,
    )
