"""Validation of explicitly named parameter protection before any optimisation."""
from collections.abc import Mapping
import math
from numbers import Real
import torch


def finite_number(name, value, *, minimum=0., strictly_positive=False):
    try:
        valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except (OverflowError, TypeError):
        valid = False
    if not valid or value < minimum or (strictly_positive and value == minimum):
        raise ValueError(f"{name} must be a finite {'positive' if strictly_positive else 'nonnegative'} number")


def parameter_map(model, values, name, *, bounds=None, binary=False):
    """Own a complete, shape-exact mapping; never broadcast a protection mask."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not isinstance(values, Mapping) or values.keys() != params.keys():
        raise ValueError(f"{name} must cover all trainable parameters exactly")
    owned = {}
    for key, param in params.items():
        value = values[key]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.shape != param.shape or value.is_complex()
                or not torch.isfinite(value).all()):
            raise ValueError(f"invalid {name}[{key!r}] tensor, shape or finite values")
        if bounds is not None:
            low, high = bounds
            if (value < low).any() or (high is not None and (value > high).any()):
                raise ValueError(f"{name}[{key!r}] is outside its allowed range")
        if binary and not ((value == 0) | (value == 1)).all():
            raise ValueError(f"{name}[{key!r}] must be binary")
        owned[key] = value.detach().to(param).clone()
    return owned
