"""Offline audit cases. Synthetic tensors only; no external models or services."""
from copy import deepcopy
from dataclasses import asdict
import pytest
import torch
from torch import nn
from alignment_metaplasticity.checkpoint import load_checkpoint, save_checkpoint
from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.importance import compute_gradient_importance
from alignment_metaplasticity.tasks import make_loader
from alignment_metaplasticity.training import train_epochs
from experiments.controlled_study import select
from alignment_metaplasticity.study import Setting, Trial

CPU = torch.device('cpu')

def data():
    return make_loader((torch.ones(5, 2), torch.tensor([0, 1, 0, 1, 0])), 2, False, 51)

@pytest.mark.parametrize('kind', ['missing', 'extra', 'broadcast', 'negative', 'nan'])
def test_invalid_protection_rejected_before_any_update(kind):
    m = nn.Linear(2, 2)
    values = {n: torch.ones_like(p) for n, p in m.named_parameters()}
    if kind == 'missing': values.pop('bias')
    if kind == 'extra': values['typo'] = torch.ones(1)
    if kind == 'broadcast': values['weight'] = torch.ones(1)
    if kind == 'negative': values['weight'].fill_(-1)
    if kind == 'nan': values['weight'].fill_(float('nan'))
    before = deepcopy(m.state_dict())
    with pytest.raises(ValueError):
        train_epochs(m, data(), CPU, 1, .01, 0., grad_scale=values)
    assert all(torch.equal(before[n], p) for n, p in m.state_dict().items())

@pytest.mark.parametrize('which', ['anchor', 'importance', 'negative-lambda'])
def test_incomplete_ewc_is_rejected(which):
    m = nn.Linear(2, 2)
    values = {n: torch.ones_like(p) for n, p in m.named_parameters()}
    kwargs = {'ewc_lambda': -1 if which == 'negative-lambda' else 1}
    if which == 'anchor': kwargs['ewc_anchor'] = values
    if which == 'importance': kwargs['ewc_importance'] = values
    with pytest.raises(ValueError): train_epochs(m, data(), CPU, 1, .01, 0., **kwargs)

def test_importance_preserves_each_module_mode_and_gradients():
    m = nn.Sequential(nn.Linear(2, 2), nn.Dropout(.3), nn.Linear(2, 2))
    m.train(); m[1].eval()
    modes = [v.training for v in m.modules()]
    for p in m.parameters(): p.grad = torch.full_like(p, .7)
    compute_gradient_importance(m, data(), CPU, max_batches=1)
    assert modes == [v.training for v in m.modules()]
    assert all(p.grad is not None and torch.equal(p.grad, torch.full_like(p, .7)) for p in m.parameters())

def test_importance_fetches_exact_budget():
    seen = []
    def source():
        for i in range(3):
            seen.append(i)
            yield torch.ones(2, 2), torch.tensor([0, 1])
    compute_gradient_importance(nn.Linear(2, 2), source(), CPU, max_batches=1)
    assert seen == [0]

def test_importance_rejects_empty_evidence():
    with pytest.raises(ValueError): compute_gradient_importance(nn.Linear(2, 2), [], CPU)

def test_duplicate_yaml_config_rejected(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('training:\n  lr: 0.01\n  lr: 0.9\n')
    with pytest.raises(ValueError): load_config(path)

def test_checkpoint_bad_rng_does_not_mutate_live_state(tmp_path):
    source = nn.Linear(2, 2); path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, source, None, loaders={'train': data()}, protection={}, progress={}, provenance={})
    payload = torch.load(path, weights_only=True)
    payload['torch_rng'] = torch.zeros(3, dtype=torch.uint8)
    torch.save(payload, path)
    target = nn.Linear(2, 2); before = deepcopy(target.state_dict())
    old_rng = torch.get_rng_state().clone()
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(path, target, None, loaders={'train': data()}, expected_provenance={})
    assert all(torch.equal(before[n], p) for n, p in target.state_dict().items())
    assert torch.equal(old_rng, torch.get_rng_state())

def test_checkpoint_rejects_trainability_change(tmp_path):
    source = nn.Linear(2, 2); path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, source, None, loaders={'train': data()}, protection={}, progress={}, provenance={})
    target = nn.Linear(2, 2); target.weight.requires_grad_(False)
    with pytest.raises(ValueError):
        load_checkpoint(path, target, None, loaders={'train': data()}, expected_provenance={})

def test_duplicate_development_cell_is_not_an_independent_repetition():
    setting = asdict(Setting('metaplastic'))
    trial = asdict(Trial(5, 6, 7))
    result = {'trial': trial, 'setting': setting, 'alignment': {'selective_corrigibility': .9, 'authorized_acceptance': .9, 'unauthorized_resistance': .9}, 'acquisition': .98, 'retention': .6, 'seconds': 1.}
    row = {'setting': setting, 'trial': trial, 'status': 'completed', 'result': result}
    with pytest.raises(ValueError): select([row, deepcopy(row)], [setting], 2, .95)
