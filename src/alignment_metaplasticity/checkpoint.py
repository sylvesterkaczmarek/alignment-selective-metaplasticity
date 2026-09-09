"""Explicit CPU epoch-boundary snapshots for independent-example training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .study import tensor_digest


def _loader_states(loaders):
    result = {}
    for name, loader in loaders.items():
        if loader.num_workers != 0 or loader.generator is None or not hasattr(loader.dataset, 'tensors'):
            raise ValueError('exact restart supports generator-backed TensorDataset loaders with num_workers=0')
        result[name] = {'generator':loader.generator.get_state(),
                        'dataset_sha256':tensor_digest(dict(enumerate(loader.dataset.tensors))),
                        'batch_size':loader.batch_size,'drop_last':loader.drop_last,
                        'sampler':type(loader.sampler).__qualname__}
    return result


def _optimizer_names(model, optimizer):
    if optimizer is None:
        return None
    names = {id(p):n for n,p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def save_checkpoint(path, model, optimizer, *, loaders, protection, progress, provenance):
    """Call only after finishing an epoch; no sampler cursor is saved mid-epoch."""
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('exact checkpoint contract currently covers CPU only')
    plain = json.loads(json.dumps({'progress':progress,'provenance':provenance},allow_nan=False))
    numpy_state = np.random.get_state()
    payload = {'schema_version':1, 'model':model.state_dict(),
               'optimizer':optimizer.state_dict() if optimizer is not None else None,
               'optimizer_type':type(optimizer).__qualname__ if optimizer is not None else None,
               'optimizer_names':_optimizer_names(model, optimizer),
               'protection':protection,'progress':plain['progress'],'provenance':plain['provenance'],
               'loaders':_loader_states(loaders),'python_rng':random.getstate(),
               'numpy_rng':(numpy_state[0],numpy_state[1].tolist(),*numpy_state[2:]),
               'torch_rng':torch.random.get_rng_state(),
               'modes':{n:m.training for n,m in model.named_modules()}}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd, temporary=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    os.close(fd)
    try:
        torch.save(payload,temporary)
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def load_checkpoint(path, model, optimizer, *, loaders, expected_provenance):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['schema_version']!=1 or payload['provenance']!=expected_provenance:
        raise ValueError('checkpoint schema or provenance mismatch')
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('exact checkpoint contract currently covers CPU only')
    expected_type=type(optimizer).__qualname__ if optimizer is not None else None
    if payload['optimizer_type']!=expected_type or payload['optimizer_names']!=_optimizer_names(model,optimizer):
        raise ValueError('optimizer type or parameter order mismatch')
    current=model.state_dict()
    if current.keys()!=payload['model'].keys() or any(current[n].shape!=v.shape or current[n].dtype!=v.dtype for n,v in payload['model'].items()):
        raise ValueError('model state shape or dtype mismatch')
    states=_loader_states(loaders)
    if states.keys()!=payload['loaders'].keys():
        raise ValueError('loader names mismatch')
    for name,state in states.items():
        for key in ('dataset_sha256','batch_size','drop_last','sampler'):
            if state[key]!=payload['loaders'][name][key]:
                raise ValueError(f'loader {name} {key} mismatch')
    model.load_state_dict(payload['model'])
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
    for name,loader in loaders.items():
        loader.generator.set_state(payload['loaders'][name]['generator'])
    for name,module in model.named_modules():
        module.training=payload['modes'][name]
    random.setstate(payload['python_rng'])
    nr=payload['numpy_rng']
    np.random.set_state((nr[0],np.asarray(nr[1],dtype=np.uint32),*nr[2:]))
    torch.random.set_rng_state(payload['torch_rng'])
    return {k:payload[k] for k in ('protection','progress','provenance')}
