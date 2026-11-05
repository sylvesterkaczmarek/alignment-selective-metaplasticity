from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .validation import finite_number, parameter_map


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

    ``ewc_lambda`` multiplies this penalty in ``train_epochs``. The reference
    benchmark supplies its normalized squared minibatch-gradient proxy;
    callers may instead supply an explicitly named per-example estimate.
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
    epoch_callback=None,
    optimizer=None,
    update_scale: TensorMap | None = None,
) -> TrainDiagnostics:
    finite_number("weight_decay", weight_decay)
    if type(epochs) is not int or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer")
    finite_number("lr", lr, strictly_positive=True)
    finite_number("ewc_lambda", ewc_lambda)
    if (ewc_anchor is None) != (ewc_importance is None) or (ewc_lambda > 0 and ewc_anchor is None):
        raise ValueError("EWC requires both anchor and importance mappings")
    if grad_scale is not None:
        grad_scale = parameter_map(model, grad_scale, "grad_scale", bounds=(0., 1.))
    if grad_mask is not None:
        grad_mask = parameter_map(model, grad_mask, "grad_mask", binary=True)
    if update_scale is not None:
        update_scale = parameter_map(model, update_scale, "update_scale", bounds=(0., 1.))
    if ewc_anchor is not None:
        ewc_anchor = parameter_map(model, ewc_anchor, "ewc_anchor")
        ewc_importance = parameter_map(model, ewc_importance, "ewc_importance", bounds=(0., None))
    criterion = nn.CrossEntropyLoss()
    # Apply L2 decay before protection so it cannot move frozen elements or
    # bypass their plasticity coefficient. Momentum starts fresh for each call;
    # the fixed mask/scale therefore also applies to its accumulated updates.
    external_optimizer = optimizer is not None
    if external_optimizer:
        if not isinstance(optimizer, (torch.optim.SGD, torch.optim.AdamW)):
            raise ValueError("persistent training supports SGD and AdamW")
        if grad_scale is not None or grad_mask is not None:
            raise ValueError("persistent optimisers require update_scale for protection")
        if any(g["lr"] != lr or g["weight_decay"] != weight_decay for g in optimizer.param_groups):
            raise ValueError("optimizer learning rate and decay must match training arguments")
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("optimizer must own each trainable model parameter exactly once")
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    if update_scale is not None:
        if grad_scale is not None or grad_mask is not None:
            raise ValueError("choose gradient protection or update protection")
        params = {n:p for n,p in model.named_parameters() if p.requires_grad}
        if params.keys() != update_scale.keys():
            raise ValueError("update_scale must cover all trainable parameter names")
        for name, p in params.items():
            s = update_scale[name]
            if s.shape != p.shape or not torch.isfinite(s).all() or (s < 0).any() or (s > 1).any():
                raise ValueError(f"invalid update_scale for {name}")
    losses: list[float] = []
    examples = 0
    protected_grads: list[float] = []
    unprotected_grads: list[float] = []

    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if ewc_anchor is not None and ewc_importance is not None and ewc_lambda > 0:
                loss = loss + ewc_lambda * _ewc_penalty(model, ewc_anchor, ewc_importance)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite loss gradient before optimiser step")

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
                if weight_decay and not external_optimizer:
                    p.grad.add_(p.detach(), alpha=weight_decay)
                if scale is not None:
                    p.grad.mul_(scale)
                if grad_mask is not None and name in grad_mask:
                    p.grad.mul_(grad_mask[name].to(p.grad.device))

            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite protected gradient before optimiser step")
            previous = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad} if update_scale is not None else {}
            optimizer.step()
            if update_scale is not None:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.requires_grad:
                            if not torch.isfinite(p).all():
                                raise FloatingPointError("nonfinite optimiser update")
                            p.copy_(previous[name] + update_scale[name].to(p) * (p - previous[name]))
            losses.append(float(loss.detach().item()) * len(y))
            examples += len(y)

        if epoch_callback is not None:
            modes = {module: module.training for module in model.modules()}
            try:
                epoch_callback(epoch + 1)
            finally:
                for module, training in modes.items():
                    module.training = training

    if epochs > 0 and examples == 0:
        raise ValueError("training requires at least one example")
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("nonfinite model parameters")
    return TrainDiagnostics(
        mean_loss=sum(losses) / max(examples, 1),
        protected_grad_mean=(sum(protected_grads) / len(protected_grads)) if protected_grads else None,
        unprotected_grad_mean=(sum(unprotected_grads) / len(unprotected_grads)) if unprotected_grads else None,
    )
