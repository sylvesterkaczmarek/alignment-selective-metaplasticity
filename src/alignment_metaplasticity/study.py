"""Controlled comparisons; the historical experiment runner remains independent."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import sys
import time
from typing import Callable

import torch
from torch import nn

from .config import validate_config, validate_seed
from .evaluation import accuracy, evaluate_alignment
from .experiment import build_loaders
from .importance import (clone_parameters, compute_gradient_importance, compute_example_importance, ema_importance,
                         plasticity_from_importance, static_train_masks)
from .model import SelectiveCorrigibilityNet
from .repro import set_seed
from .tasks import BenchmarkSpec, make_loader
from .training import train_epochs

STUDY_METHODS = ("fine_tune", "static_freeze", "ewc", "metaplastic", "shuffled", "uniform", "rehearsal")


@dataclass(frozen=True)
class Trial:
    model_seed: int
    sample_seed: int
    rule_seed: int
    order: tuple[int, ...] = ()

    def validate(self, tasks):
        for seed in (self.model_seed, self.sample_seed, self.rule_seed):
            validate_seed(seed)
        if self.rule_seed + tasks >= 2**32:
            raise ValueError("rule_seed plus task ID must fit uint32")
        if self.order and (any(type(t) is not int for t in self.order)
                           or sorted(self.order) != list(range(1, tasks + 1))):
            raise ValueError("order must be a permutation of all capability tasks")


@dataclass(frozen=True)
class Setting:
    method: str
    protocol: str = "fixed"
    lr: float = 0.03
    strength: float = 200.0
    ewc_lambda: float = 18.0
    freeze_quantile: float = 0.85

    def __post_init__(self):
        if self.method not in STUDY_METHODS or self.protocol not in ("fixed", "periodic"):
            raise ValueError("unknown study method or access protocol")
        if self.method == "rehearsal" and self.protocol != "periodic":
            raise ValueError("rehearsal requires periodic alignment access")
        for name in ("lr", "strength", "ewc_lambda", "freeze_quantile"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if not self.lr or self.freeze_quantile > 1:
            raise ValueError("lr must be positive and freeze_quantile <= 1")


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in values.items():
        value = value.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def default_model(spec, cfg):
    return SelectiveCorrigibilityNet(spec.input_dim, cfg["model"]["hidden_dim"], cfg["model"]["depth"])


def transform_plasticity(values, kind, seed):
    """Own-state controls; a local generator never advances the training RNG."""
    if kind == "metaplastic":
        return values
    if kind == "uniform":
        return {k: torch.full_like(v, v.mean().item()) for k, v in values.items()}
    if kind != "shuffled":
        raise ValueError("unknown plasticity transform")
    g = torch.Generator().manual_seed(seed)
    return {k: v.flatten()[torch.randperm(v.numel(), generator=g).to(v.device)].reshape_as(v)
            for k, v in values.items()}


class CountedLoader:
    def __init__(self, loader, counts):
        self.loader, self.counts = loader, counts

    def __iter__(self):
        for x, y in self.loader:
            self.counts["batches"] += 1
            self.counts["examples"] += len(y)
            yield x, y


class StudyRun:
    """One trial with explicit data access; supports a model and protection callable."""
    def __init__(self, cfg, trial: Trial, setting: Setting, *, model_factory: Callable = default_model,
                 protection_factory: Callable | None = None, importance_estimator="batch", loader_factory=build_loaders):
        validate_config(cfg)
        trial.validate(cfg["model"]["num_capability_tasks"])
        if cfg["device"] != "cpu":
            raise ValueError("controlled studies currently require CPU for measured runtime")
        self.cfg, self.trial, self.setting = cfg, trial, setting
        self.protection_factory = protection_factory
        if importance_estimator not in ("batch", "empirical", "model_fisher"):
            raise ValueError("unknown importance estimator")
        self.importance_estimator = importance_estimator
        self.device = torch.device("cpu")
        self.started = time.perf_counter()
        self.cost = {}
        self.spec = BenchmarkSpec(cfg["model"]["signal_dim"], cfg["model"]["num_capability_tasks"])
        self.loaders = loader_factory(cfg, self.spec, trial.sample_seed, rule_seed=trial.rule_seed)
        self.order = trial.order or tuple(range(1, self.spec.num_capability_tasks + 1))
        self.data_hashes = {"alignment_train": self._data_hash(self.loaders[0])}
        self.data_hashes.update({f"capability_train_{t}": self._data_hash(v) for t, v in self.loaders[2].items()})
        self.data_hashes.update({f"capability_eval_{t}": self._data_hash(v) for t, v in self.loaders[3].items()})
        self.data_hashes.update({f"alignment_eval_{t}": self._data_hash(v) for t, v in self.loaders[1].items()})
        set_seed(trial.model_seed)
        self.model = model_factory(self.spec, cfg).to(self.device)
        self.train(self.loaders[0], cfg["training"]["alignment_epochs"], "pretraining", lr=cfg["training"]["lr"])
        self.anchor = clone_parameters(self.model)
        self.initial_digest = tensor_digest(self.anchor)
        self.initial_alignment = self.alignment()
        if self.initial_alignment["selective_corrigibility"] < 0.90:
            raise RuntimeError(f"alignment pretraining underfit: {self.initial_alignment}")
        self.importance = self.estimate(0) if self.uses_importance else {}
        self.history = []

    @property
    def uses_importance(self):
        return self.setting.method not in ("fine_tune", "rehearsal") or self.protection_factory is not None

    @staticmethod
    def _data_hash(loader):
        return tensor_digest(dict(zip(("x", "y"), loader.dataset.tensors)))

    def counted(self, loader, phase):
        return CountedLoader(loader, self.cost.setdefault(phase, {"examples": 0, "batches": 0}))

    def train(self, loader, epochs, phase, *, lr=None, **kwargs):
        return train_epochs(self.model, self.counted(loader, phase), self.device,
                            epochs=epochs, lr=self.setting.lr if lr is None else lr,
                            weight_decay=self.cfg["training"]["weight_decay"], **kwargs)

    def alignment(self):
        return evaluate_alignment(self.model, self.loaders[1], self.device).to_dict()

    def probe_loader(self, step):
        x, y = self.loaders[0].dataset.tensors
        n = min(len(y), self.cfg["importance"]["max_batches"] * self.cfg["data"]["batch_size"])
        g = torch.Generator().manual_seed(self.trial.sample_seed + 40000 + step)
        idx = torch.randperm(len(y), generator=g)[:n]
        return make_loader((x[idx], y[idx]), self.cfg["data"]["batch_size"], False, self.trial.sample_seed + 50000 + step)

    def estimate(self, step):
        loader = self.counted(self.probe_loader(step), "importance")
        if self.importance_estimator == "batch":
            values = compute_gradient_importance(self.model, loader, self.device, **self.cfg["importance"])
            self.cost["importance"]["backward_calls"] = self.cost["importance"]["batches"]
            return values
        return compute_example_importance(self.model, loader, self.device, kind=self.importance_estimator,
                                          quantile=self.cfg["importance"]["quantile"], stats=self.cost["importance"])

    def protection(self):
        s = self.setting
        if self.protection_factory is not None:
            return self.protection_factory(self.importance, self.anchor)
        if s.method == "static_freeze":
            return {"grad_mask": static_train_masks(self.importance, s.freeze_quantile)}
        if s.method == "ewc":
            return {"ewc_anchor": self.anchor, "ewc_importance": self.importance, "ewc_lambda": s.ewc_lambda}
        if s.method in ("metaplastic", "shuffled", "uniform"):
            p = plasticity_from_importance(self.importance, s.strength, self.cfg["metaplastic"]["min_plasticity"])
            return {"grad_scale": transform_plasticity(p, s.method, self.trial.model_seed + 60000)}
        return {}

    def run(self):
        for i, task in enumerate(self.order):
            self.train(self.loaders[2][task], self.cfg["training"]["capability_epochs"], "capability", **self.protection())
            if self.setting.protocol == "periodic":
                if self.setting.method == "rehearsal":
                    self.train(self.probe_loader(i + 1), 1, "rehearsal")
                elif self.uses_importance:
                    self.importance = ema_importance(self.importance, self.estimate(i + 1), self.cfg["metaplastic"]["ema_decay"])
            caps = {str(t): accuracy(self.model, self.loaders[3][t], self.device) for t in self.order[:i + 1]}
            self.history.append({"task": task, "alignment": self.alignment(), "capabilities": caps})
        return self.result()

    def result(self):
        if not self.history:
            raise ValueError("run capability training before requesting results")
        caps = self.history[-1]["capabilities"]
        state = [*self.model.parameters(), *self.anchor.values(), *self.importance.values()]
        # ru_maxrss is a process lifetime high-water mark, not a per-trial allocation measurement.
        try:
            import resource
            process_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        except ImportError:
            process_peak = None
        return {"trial": asdict(self.trial), "setting": asdict(self.setting), "importance_estimator": self.importance_estimator, "initial_digest": self.initial_digest,
                "data_sha256": self.data_hashes, "initial_alignment": self.initial_alignment,
                "alignment": self.history[-1]["alignment"], "history": self.history,
                "acquisition": sum(h["capabilities"][str(h["task"])] for h in self.history) / len(self.history),
                "retention": sum(caps.values()) / len(caps),
                "earlier_retention": sum(caps[str(t)] for t in self.order[:-1]) / (len(self.order) - 1) if len(self.order) > 1 else None,
                "cost": self.cost, "seconds": time.perf_counter() - self.started,
                "persistent_tensor_bytes": sum(t.numel() * t.element_size() for t in state),
                "process_lifetime_peak_rss_bytes": process_peak,
                "final_digest": tensor_digest(dict(self.model.named_parameters()))}
