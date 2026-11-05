"""Validated CPU epoch-boundary checkpoints; rejected loads leave live state intact."""
from __future__ import annotations

from copy import deepcopy
import json
import inspect
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset, default_collate

from .study import tensor_digest

SCHEMA_VERSION = 3


def _loader_states(loaders):
    result = {}
    generators = set()
    for name, loader in loaders.items():
        if not isinstance(name, str) or not name:
            raise ValueError('loader names must be nonempty strings')
        if (type(loader) is not DataLoader or loader.num_workers != 0 or loader.generator is None
                or type(loader.dataset) is not TensorDataset or type(loader.batch_sampler) is not BatchSampler
                or type(loader.sampler) not in (RandomSampler, SequentialSampler)
                or loader.collate_fn is not default_collate):
            raise ValueError('exact restart requires standard generator-backed TensorDataset loaders with num_workers=0')
        if (type(loader.batch_size) is not int or loader.batch_size < 1
                or loader.batch_sampler.sampler is not loader.sampler
                or loader.batch_sampler.batch_size != loader.batch_size
                or loader.batch_sampler.drop_last is not loader.drop_last):
            raise ValueError('exact restart requires unchanged automatic batching')
        if id(loader.generator) in generators:
            raise ValueError('loaders must have independent generators for exact restart')
        generators.add(id(loader.generator))
        if type(loader.sampler) is RandomSampler and (loader.sampler.generator is not loader.generator
                or loader.sampler.replacement or loader.sampler.num_samples != len(loader.dataset)):
            raise ValueError('random sampler must use the loader generator without replacement or subsampling')
        result[name] = {'generator': loader.generator.get_state().clone(),
                       'dataset_sha256': tensor_digest(dict(enumerate(loader.dataset.tensors))),
                       'batch_size': loader.batch_size, 'drop_last': loader.drop_last,
                       'sampler': type(loader.sampler).__qualname__}
    return result


def _model_signature(model):
    if any(t.device.type != 'cpu' for t in (*model.parameters(), *model.buffers())):
        raise ValueError('exact checkpoint contract currently covers CPU only')
    return {n: (type(m).__module__ + '.' + type(m).__qualname__, m.extra_repr())
            for n, m in model.named_modules()}


def _optimizer_names(model, optimizer):
    if optimizer is None:
        return None
    if type(optimizer) not in (torch.optim.SGD, torch.optim.AdamW):
        raise ValueError('checkpoint supports standard SGD or AdamW optimizers')
    names = {id(p): n for n, p in model.named_parameters()}
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if set(ids) != expected or len(ids) != len(set(ids)):
        raise ValueError('optimizer must own every trainable parameter exactly once')
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def _safe_tree(value, depth=0):
    if depth > 64:
        raise ValueError('checkpoint state nesting is too deep')
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.layout != torch.strided or not torch.isfinite(value).all():
            raise ValueError('checkpoint tensors must be finite, dense and on CPU')
    elif isinstance(value, dict):
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ValueError('checkpoint state keys must be strings or integers')
            _safe_tree(item, depth + 1)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _safe_tree(item, depth + 1)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError('unsupported checkpoint state value')
    elif type(value) is float and not math.isfinite(value):
        raise ValueError('nonfinite checkpoint state value')


def _buffers(model):
    return dict(model.named_buffers(remove_duplicate=False))


def _buffer_aliases(model):
    groups = {}
    for name, tensor in _buffers(model).items():
        groups.setdefault(id(tensor), []).append(name)
    return list(groups.values())


def _validate_optimizer_payload(optimizer, payload):
    """Check raw native state before load_state_dict can cast or fill defaults."""
    if not isinstance(payload, dict) or set(payload) != {'state', 'param_groups'}:
        raise ValueError('invalid optimizer state container')
    groups, states = payload['param_groups'], payload['state']
    if not isinstance(groups, list) or not isinstance(states, dict) or len(groups) != len(optimizer.param_groups):
        raise ValueError('optimizer group/state structure mismatch')
    parameters, parameters_to_groups = {}, {}
    expected_id = 0
    constructor_keys = set(inspect.signature(type(optimizer)).parameters) - {'params'}
    for saved, current in zip(groups, optimizer.param_groups):
        if not isinstance(saved, dict) or not set(optimizer.defaults) <= saved.keys() or 'params' not in saved:
            raise ValueError('missing native optimizer group settings')
        ids = saved['params']
        if not isinstance(ids, list) or len(ids) != len(current['params']):
            raise ValueError('invalid optimizer parameter group')
        for identifier, param in zip(ids, current['params']):
            if type(identifier) is not int or identifier != expected_id:
                raise ValueError('optimizer parameter identifiers/order mismatch')
            parameters[identifier] = param
            parameters_to_groups[identifier] = saved
            expected_id += 1
        for name in ('lr', 'weight_decay', 'eps', 'momentum', 'dampening'):
            if name in saved:
                value = saved[name]
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    value = value.item()
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'invalid optimizer {name}')
        for name in ('nesterov', 'amsgrad', 'maximize', 'capturable', 'differentiable', 'decoupled_weight_decay'):
            if name in saved and type(saved[name]) is not bool:
                raise ValueError(f'invalid optimizer boolean {name}')
        for name in ('foreach', 'fused'):
            if name in saved and saved[name] is not None and type(saved[name]) is not bool:
                raise ValueError(f'invalid optimizer optional boolean {name}')
        if 'betas' in saved:
            betas = saved['betas']
            if not isinstance(betas, (tuple, list)) or len(betas) != 2:
                raise ValueError('invalid optimizer betas')
            for beta in betas:
                if isinstance(beta, torch.Tensor) and beta.numel() == 1:
                    beta = beta.item()
                if type(beta) not in (float, int) or not 0 <= beta < 1:
                    raise ValueError('invalid optimizer beta')
        if type(optimizer) is torch.optim.AdamW and saved.get('decoupled_weight_decay', True) is not True:
            raise ValueError('AdamW requires decoupled weight decay')
        # Invoke native constructor validation on each group's actual arguments,
        # not on untouched default arguments with a param-group override.
        type(optimizer)([deepcopy(p) for p in current['params']],
                        **{k: deepcopy(v) for k, v in saved.items() if k in constructor_keys})
    for identifier, state in states.items():
        if type(identifier) is not int or identifier not in parameters or not isinstance(state, dict):
            raise ValueError('optimizer state references an unknown parameter')
        if not state:
            continue
        param, group = parameters[identifier], parameters_to_groups[identifier]
        if type(optimizer) is torch.optim.AdamW:
            required = {'step', 'exp_avg', 'exp_avg_sq'}
            if group['amsgrad']:
                required.add('max_exp_avg_sq')
            if not required <= state.keys() or set(state) - {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}:
                raise ValueError('incomplete or unknown AdamW state')
            step = state['step']
            if isinstance(step, torch.Tensor) and step.numel() == 1:
                step = step.item()
            if type(step) not in (int, float) or not math.isfinite(step) or step < 0 or step != int(step):
                raise ValueError('invalid optimizer step')
        elif set(state) != {'momentum_buffer'}:
            raise ValueError('invalid SGD state')
        for name in ('momentum_buffer', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name in state:
                tensor = state[name]
                if not isinstance(tensor, torch.Tensor) or tensor.shape != param.shape or tensor.dtype != param.dtype:
                    raise ValueError('optimizer state shape or dtype mismatch before casting')
                if name in ('exp_avg_sq', 'max_exp_avg_sq') and (tensor < 0).any():
                    raise ValueError('negative optimizer second moment')


def save_checkpoint(path, model, optimizer, *, loaders, protection, progress, provenance):
    """Save at an epoch boundary. Files are for trusted local experiment recovery."""
    signature = _model_signature(model)
    plain = json.loads(json.dumps({'progress': progress, 'provenance': provenance}, allow_nan=False))
    numpy_state = np.random.get_state()
    payload = {'schema_version': SCHEMA_VERSION, 'model': model.state_dict(), 'model_signature': signature,
               'trainability': {n: p.requires_grad for n, p in model.named_parameters()},
               'buffers': _buffers(model), 'buffer_aliases': _buffer_aliases(model),
               'optimizer': optimizer.state_dict() if optimizer is not None else None,
               'optimizer_type': type(optimizer).__qualname__ if optimizer is not None else None,
               'optimizer_names': _optimizer_names(model, optimizer),
               'protection': protection, 'progress': plain['progress'], 'provenance': plain['provenance'],
               'loaders': _loader_states(loaders), 'python_rng': random.getstate(),
               'numpy_rng': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
               'torch_rng': torch.random.get_rng_state(),
               'modes': {n: m.training for n, m in model.named_modules()}}
    _safe_tree(payload)
    if optimizer is not None:
        _validate_optimizer_payload(optimizer, payload['optimizer'])
    payload = deepcopy(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _numpy_state(value):
    return value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]


def load_checkpoint(path, model, optimizer, *, loaders, expected_provenance):
    """Validate and stage every mutable component before committing a restore.

    Schemas 1 and 2 omitted parts of this exact-restart contract. Use their
    original source to load old checkpoints; do not relabel their schema.
    Restricted loading is not a sandbox for arbitrary untrusted model files.
    """
    payload = torch.load(path, map_location='cpu', weights_only=True)
    try:
        _safe_tree(payload)
        required = {'schema_version', 'model', 'model_signature', 'trainability', 'buffers', 'buffer_aliases',
                    'optimizer', 'optimizer_type', 'optimizer_names', 'protection', 'progress', 'provenance',
                    'loaders', 'python_rng', 'numpy_rng', 'torch_rng', 'modes'}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError('checkpoint fields or schema mismatch')
        for name in ('model', 'model_signature', 'trainability', 'buffers', 'loaders', 'modes'):
            if not isinstance(payload[name], dict):
                raise ValueError(f'invalid checkpoint mapping {name}')
        if type(payload['schema_version']) is not int or payload['schema_version'] != SCHEMA_VERSION or payload['provenance'] != expected_provenance:
            raise ValueError('checkpoint schema or provenance mismatch')
        if payload['model_signature'] != _model_signature(model):
            raise ValueError('model module type or configuration mismatch')
        if (any(type(value) is not bool for value in payload['trainability'].values())
                or payload['trainability'] != {n: p.requires_grad for n, p in model.named_parameters()}):
            raise ValueError('model trainability mismatch')
        expected_type = type(optimizer).__qualname__ if optimizer is not None else None
        if payload['optimizer_type'] != expected_type or payload['optimizer_names'] != _optimizer_names(model, optimizer):
            raise ValueError('optimizer type or parameter order mismatch')
        if (payload['optimizer'] is None) != (optimizer is None):
            raise ValueError('optimizer presence mismatch')
        current = model.state_dict()
        if current.keys() != payload['model'].keys() or any(
                not isinstance(v, torch.Tensor) or current[n].shape != v.shape or current[n].dtype != v.dtype
                for n, v in payload['model'].items()):
            raise ValueError('model state shape or dtype mismatch')
        buffers = _buffers(model)
        if (payload['buffers'].keys() != buffers.keys() or payload['buffer_aliases'] != _buffer_aliases(model)
                or any(not isinstance(value, torch.Tensor) or value.shape != buffers[name].shape
                       or value.dtype != buffers[name].dtype for name, value in payload['buffers'].items())):
            raise ValueError('buffer names, alias topology, shape or dtype mismatch')
        for name, value in payload['buffers'].items():
            if name in payload['model'] and not torch.equal(value, payload['model'][name]):
                raise ValueError('persistent buffer differs between checkpoint representations')
        for aliases in payload['buffer_aliases']:
            if any(not torch.equal(payload['buffers'][aliases[0]], payload['buffers'][name]) for name in aliases):
                raise ValueError('aliased buffer values disagree')
        modules = dict(model.named_modules())
        if payload['modes'].keys() != modules.keys() or any(type(v) is not bool for v in payload['modes'].values()):
            raise ValueError('invalid model mode state')
        states = _loader_states(loaders)
        if states.keys() != payload['loaders'].keys():
            raise ValueError('loader names mismatch')
        for name, state in states.items():
            for key in ('dataset_sha256', 'batch_size', 'drop_last', 'sampler'):
                if state[key] != payload['loaders'][name][key]:
                    raise ValueError(f'loader {name} {key} mismatch')
            torch.Generator().set_state(payload['loaders'][name]['generator'])
        random.Random().setstate(payload['python_rng'])
        np.random.RandomState().set_state(_numpy_state(payload['numpy_rng']))
        torch.Generator().set_state(payload['torch_rng'])
        # Deepcopy the model/optimizer together to retain their parameter bindings.
        if optimizer is not None:
            _validate_optimizer_payload(optimizer, payload['optimizer'])
        staged_model, staged_optimizer = deepcopy((model, optimizer))
        staged_model.load_state_dict(payload['model'])
        if staged_optimizer is not None:
            staged_optimizer.load_state_dict(payload['optimizer'])
            type(staged_optimizer)(deepcopy(staged_optimizer.param_groups))
            for param, state in staged_optimizer.state.items():
                if not isinstance(param, torch.Tensor):
                    raise ValueError('optimizer state refers to unknown parameter')
                if type(staged_optimizer) is torch.optim.AdamW and state:
                    if not {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys():
                        raise ValueError('incomplete AdamW state')
                    step = state['step']
                    if isinstance(step, torch.Tensor):
                        if step.numel() != 1:
                            raise ValueError('invalid optimizer step')
                        step = step.item()
                    if type(step) not in (int, float) or step < 0 or int(step) != step:
                        raise ValueError('invalid optimizer step')
                    if (state['exp_avg_sq'] < 0).any() or ('max_exp_avg_sq' in state and (state['max_exp_avg_sq'] < 0).any()):
                        raise ValueError('negative optimizer second moment')
                for name in ('momentum_buffer', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    if name in state and (state[name].shape != param.shape or state[name].dtype != param.dtype):
                        raise ValueError('optimizer state shape or dtype mismatch')
    except (KeyError, TypeError, RuntimeError, OverflowError, IndexError) as exc:
        raise ValueError(f'invalid checkpoint state: {exc}') from exc

    before_model = deepcopy(model.state_dict())
    before_buffers = {name: value.clone() for name, value in buffers.items()}
    before_optimizer = deepcopy(optimizer.state_dict()) if optimizer is not None else None
    before_modes = {name: module.training for name, module in modules.items()}
    before_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        model.load_state_dict(payload['model'])
        with torch.no_grad():
            for name, value in buffers.items():
                value.copy_(payload['buffers'][name])
        if optimizer is not None:
            optimizer.load_state_dict(payload['optimizer'])
        for name, loader in loaders.items():
            loader.generator.set_state(payload['loaders'][name]['generator'])
        for name, module in modules.items():
            module.training = payload['modes'][name]
        random.setstate(payload['python_rng'])
        np.random.set_state(_numpy_state(payload['numpy_rng']))
        torch.random.set_rng_state(payload['torch_rng'])
    except Exception:
        model.load_state_dict(before_model)
        with torch.no_grad():
            for name, value in buffers.items():
                value.copy_(before_buffers[name])
        if optimizer is not None:
            optimizer.load_state_dict(before_optimizer)
        for name, loader in loaders.items():
            loader.generator.set_state(states[name]['generator'])
        for name, module in modules.items():
            module.training = before_modes[name]
        random.setstate(before_rng[0])
        np.random.set_state(before_rng[1])
        torch.random.set_rng_state(before_rng[2])
        raise
    return {k: deepcopy(payload[k]) for k in ('protection', 'progress', 'provenance')}
