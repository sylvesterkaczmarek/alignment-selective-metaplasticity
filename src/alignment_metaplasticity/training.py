from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
    terms = []
    for name, p in model.named_parameters():
        if p.requires_grad and name in anchor:
            terms.append((importance[name] * (p - anchor[name].to(p.device)).pow(2)).mean())
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(terms).sum()


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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
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

            if grad_scale is not None:
                for name, p in model.named_parameters():
                    if p.grad is None or name not in grad_scale:
                        continue
                    scale = grad_scale[name].to(p.grad.device)
                    imp_proxy = 1.0 - scale
                    protected = imp_proxy >= 0.5
                    if protected.any():
                        protected_grads.append(float(p.grad.detach().abs()[protected].mean().item()))
                    if (~protected).any():
                        unprotected_grads.append(float(p.grad.detach().abs()[~protected].mean().item()))
                    p.grad.mul_(scale)

            if grad_mask is not None:
                for name, p in model.named_parameters():
                    if p.grad is not None and name in grad_mask:
                        p.grad.mul_(grad_mask[name].to(p.grad.device))

            optimizer.step()
            losses.append(float(loss.detach().item()))

    return TrainDiagnostics(
        mean_loss=sum(losses) / max(len(losses), 1),
        protected_grad_mean=(sum(protected_grads) / len(protected_grads)) if protected_grads else None,
        unprotected_grad_mean=(sum(unprotected_grads) / len(unprotected_grads)) if unprotected_grads else None,
    )
