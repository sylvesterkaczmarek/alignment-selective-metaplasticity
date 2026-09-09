"""Bounded interventions and context challenges for the controlled classifier."""
from __future__ import annotations

import copy
import math
import torch

from .evaluation import accuracy, evaluate_alignment
from .experiment import build_loaders
from .importance import clone_parameters
from .tasks import make_alignment_tensors, make_loader
from .training import train_epochs


def familiar_context_loaders(cfg, spec, seed, *, rule_seed=0):
    loaders = build_loaders(cfg, spec, seed, rule_seed=rule_seed)
    # Both context values are seen during initial alignment; policy is unchanged.
    x, y = loaders[0].dataset.tensors
    g = torch.Generator().manual_seed(seed + 91000)
    x[:, spec.bypass_idx] = torch.randint(0, 2, (len(y),), generator=g).float()
    return loaders


def perturbations(importance, *, fraction=.1, magnitude=.5, kind="high", seed=0):
    if kind not in ("high", "low", "random"):
        raise ValueError("unknown intervention kind")
    if not math.isfinite(fraction) or not 0 < fraction <= 1 or not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError("invalid intervention fraction or magnitude")
    g = torch.Generator().manual_seed(seed)
    counts = {k:max(1, math.floor(v.numel()*fraction)) for k,v in importance.items()}
    total = sum(counts.values())
    deltas = {}
    for name, value in importance.items():
        n = counts[name]
        if kind == "random":
            # Selection uses an independent generator; signs are paired across conditions.
            choose = torch.Generator().manual_seed(seed + 1 + len(deltas))
            selected = torch.randperm(value.numel(), generator=choose)[:n]
        else:
            selected = torch.argsort(value.detach().flatten().cpu(), descending=kind == "high", stable=True)[:n]
        signs = torch.randint(0,2,(n,),generator=g).to(value) * 2 - 1
        delta = torch.zeros_like(value).flatten()
        delta[selected.to(value.device)] = signs * (magnitude / math.sqrt(total))
        deltas[name] = delta.reshape_as(value)
    return deltas


def intervention_probe(run, *, magnitude=.5, fraction=.1, seed=0):
    if not run.importance:
        raise ValueError("interventions require estimated importance")
    original = clone_parameters(run.model)
    rows = []
    def measure():
        return {"alignment": run.alignment(), "capabilities": {str(t): accuracy(run.model, loader, run.device)
                                                              for t,loader in run.loaders[3].items()}}
    baseline = measure()
    try:
        for kind in ("high", "low", "random"):
            delta = perturbations(run.importance, fraction=fraction, magnitude=magnitude, kind=kind, seed=seed)
            with torch.no_grad():
                for name,p in run.model.named_parameters():
                    p.copy_(original[name] + delta[name])
            rows.append({"kind":kind, "magnitude":magnitude, "fraction":fraction,
                         "changed_per_tensor":{n:int((d != 0).sum()) for n,d in delta.items()},
                         "actual_l2":sum(float(d.double().square().sum()) for d in delta.values())**.5,
                         "metrics":measure()})
    finally:
        with torch.no_grad():
            for name,p in run.model.named_parameters():
                p.copy_(original[name])
    return {"baseline":baseline, "interventions":rows}


def challenge_tensors(n, spec, seed, condition, *, evaluation=False, attack=True):
    if condition not in ("novel_flag", "seen_flag", "heldout_context", "no_marker", "policy_change"):
        raise ValueError("unknown challenge condition")
    mode = "authorized" if condition == "policy_change" and attack else "unauthorized" if attack else "mixed"
    x, y = make_alignment_tensors(n, spec, seed, mode=mode)
    if condition in ("novel_flag", "seen_flag", "policy_change"):
        x[:,spec.bypass_idx] = float(attack)
    elif condition == "heldout_context":
        if spec.signal_dim < 2:
            raise ValueError("heldout context requires at least two signal dimensions")
        x[:,0] = x[:,0].abs() * (1 if attack else -1)
        if attack:
            x[:,1] = x[:,1].abs() * (-1 if evaluation else 1)
    if attack:
        # Adversarial conditions reverse unauthorized resistance. Policy change
        # revokes correction authority in context 1 under a trusted new policy.
        y = 1-y
    return x, y


def challenge(run, condition, *, rehearsal=True, epochs=12):
    if type(epochs) is not int or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer")
    cfg, spec = run.cfg, run.spec
    seed = run.trial.sample_seed + 92000
    # Attack exposure stays fixed across the rehearsal factor.
    attack_n = cfg["data"]["bypass_train_size"] // 2
    ax, ay = challenge_tensors(attack_n, spec, seed, condition)
    px, py = challenge_tensors(attack_n, spec, seed+1, condition, attack=False)
    data = (torch.cat((ax,px)), torch.cat((ay,py))) if rehearsal else (ax,ay)
    batch = cfg["data"]["batch_size"]
    train = make_loader(data, batch, True, seed+2)
    evaluation = make_loader(challenge_tensors(cfg["data"]["alignment_eval_size"], spec, seed+3, condition, evaluation=True),batch,False,seed+4)
    trained_context = make_loader(challenge_tensors(cfg["data"]["alignment_eval_size"],spec,seed+5,condition),batch,False,seed+6)
    model = copy.deepcopy(run.model)
    trajectory = []
    def measure(epoch):
        trajectory.append({"epoch":epoch, "alignment":evaluate_alignment(model, run.loaders[1], run.device).to_dict(),
                           "target_accuracy":accuracy(model,evaluation,run.device),
                           "trained_context_accuracy":accuracy(model,trained_context,run.device),
                           "capability_retention":sum(accuracy(model,v,run.device) for v in run.loaders[3].values())/len(run.loaders[3])})
    measure(0)
    train_epochs(model, train, run.device, epochs=epochs, lr=run.setting.lr,
                 weight_decay=cfg["training"]["weight_decay"], epoch_callback=measure, **run.protection())
    scores=[p["alignment"]["selective_corrigibility"] for p in trajectory]
    return {"condition":condition,"rehearsal":rehearsal,"epochs":epochs,
            "threat_model":"trusted context-specific authority revocation" if condition=="policy_change" else "adversarial data with enforced optimiser",
            "label_conflict":condition=="no_marker", "attack_examples":attack_n*epochs,
            "rehearsal_examples":attack_n*epochs if rehearsal else 0,
            "optimizer_updates":len(train)*epochs,"trajectory":trajectory,
            "worst_observed_alignment":min(scores),"recovery_from_observed_minimum":scores[-1]-min(scores)}
