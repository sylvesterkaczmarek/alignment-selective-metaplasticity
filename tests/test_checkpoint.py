import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from alignment_metaplasticity.checkpoint import save_checkpoint, load_checkpoint
from alignment_metaplasticity.repro import set_seed
from alignment_metaplasticity.tasks import make_loader
from alignment_metaplasticity.training import train_epochs


def loader():
    x=torch.arange(60,dtype=torch.float32).reshape(20,3)/60
    y=torch.arange(20)%2
    return make_loader((x,y),7,True,39)


def model():
    return nn.Sequential(nn.Linear(3,8),nn.Dropout(.25),nn.Linear(8,2))


def optimizer(kind,m):
    return torch.optim.SGD(m.parameters(),lr=.01,weight_decay=.02,momentum=.9) if kind=='sgd' else torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.02)


def assert_state_equal(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_state_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_state_equal(x,y)
    else: assert a==b


@pytest.mark.parametrize('kind',['sgd','adamw'])
def test_epoch_restart_is_exact_with_dropout_shuffle_and_optimizer_state(tmp_path,kind):
    set_seed(42)
    initial=model(); full=copy.deepcopy(initial); partial=copy.deepcopy(initial)
    scale={n:torch.linspace(0,1,p.numel()).reshape_as(p) for n,p in full.named_parameters()}
    of=optimizer(kind,full); op=optimizer(kind,partial)
    lf=loader(); lp=loader()
    set_seed(77)
    train_epochs(full,lf,torch.device('cpu'),epochs=3,lr=.01,weight_decay=.02,optimizer=of,update_scale=scale)
    random_after=(random.random(),float(np.random.random()),torch.rand(3))
    set_seed(77)
    train_epochs(partial,lp,torch.device('cpu'),epochs=1,lr=.01,weight_decay=.02,optimizer=op,update_scale=scale)
    provenance={'source':'fixture','config':{'lr':.01},'torch':torch.__version__}
    path=save_checkpoint(tmp_path/'run.pt',partial,op,loaders={'train':lp},protection={'scale':scale},progress={'epoch':1},provenance=provenance)
    restored=model(); oo=optimizer(kind,restored); ll=loader()
    set_seed(19)
    state=load_checkpoint(path,restored,oo,loaders={'train':ll},expected_provenance=provenance)
    assert state['progress']=={'epoch':1}
    train_epochs(restored,ll,torch.device('cpu'),epochs=2,lr=.01,weight_decay=.02,optimizer=oo,update_scale=state['protection']['scale'])
    assert_state_equal(full.state_dict(),restored.state_dict())
    assert_state_equal(of.state_dict(),oo.state_dict())
    assert random_after[0]==random.random()
    assert random_after[1]==float(np.random.random())
    assert torch.equal(random_after[2],torch.rand(3))


@pytest.mark.parametrize('kind',['sgd','adamw'])
def test_final_update_protection_includes_decay_and_existing_momentum(kind):
    m=nn.Linear(3,2); opt=optimizer(kind,m)
    train_epochs(m,loader(),torch.device('cpu'),epochs=1,lr=.01,weight_decay=.02,optimizer=opt)
    before=copy.deepcopy(m.state_dict())
    train_epochs(m,loader(),torch.device('cpu'),epochs=2,lr=.01,weight_decay=.02,optimizer=opt,
                 update_scale={n:torch.zeros_like(p) for n,p in m.named_parameters()})
    assert_state_equal(before,m.state_dict())


def test_adamw_coefficient_scales_parameter_displacement():
    torch.manual_seed(2)
    a=nn.Linear(3,2); b=copy.deepcopy(a); before=copy.deepcopy(a.state_dict())
    data=[(torch.ones(4,3),torch.tensor([0,1,0,1]))]
    train_epochs(a,data,torch.device('cpu'),epochs=1,lr=.01,weight_decay=.02,optimizer=optimizer('adamw',a))
    train_epochs(b,data,torch.device('cpu'),epochs=1,lr=.01,weight_decay=.02,optimizer=optimizer('adamw',b),
                 update_scale={n:torch.full_like(p,.25) for n,p in b.named_parameters()})
    for n,p in b.state_dict().items(): torch.testing.assert_close(p,before[n]+.25*(a.state_dict()[n]-before[n]))


def test_checkpoint_rejects_mismatched_data_before_mutating_model(tmp_path):
    a=model(); opt=optimizer('sgd',a); l=loader()
    p=save_checkpoint(tmp_path/'a.pt',a,opt,loaders={'train':l},protection={},progress={},provenance={})
    b=model(); before=copy.deepcopy(b.state_dict()); other=loader()
    other.dataset.tensors[0][0,0]=99
    with pytest.raises(ValueError,match='dataset_sha256'):
        load_checkpoint(p,b,optimizer('sgd',b),loaders={'train':other},expected_provenance={})
    assert_state_equal(before,b.state_dict())
    with pytest.raises(ValueError,match='provenance'):
        load_checkpoint(p,b,optimizer('sgd',b),loaders={'train':l},expected_provenance={'different':True})


def test_nonfinite_training_cannot_report_success():
    m=nn.Linear(1,2)
    with pytest.raises(FloatingPointError):
        train_epochs(m,[(torch.tensor([[float('nan')]]),torch.tensor([0]))],torch.device('cpu'),epochs=1,lr=.1,weight_decay=0.)


def test_checkpoint_rejects_untracked_sampler_and_custom_collation(tmp_path):
    from torch.utils.data import DataLoader, RandomSampler
    source=loader(); m=model()
    separate=RandomSampler(source.dataset,generator=torch.Generator().manual_seed(2))
    unsupported=[DataLoader(source.dataset,batch_size=7,sampler=separate,generator=source.generator),
                 DataLoader(source.dataset,batch_size=7,generator=source.generator,collate_fn=lambda x:x)]
    for data in unsupported:
        with pytest.raises(ValueError,match='sampler|standard'):
            save_checkpoint(tmp_path/'bad.pt',m,None,loaders={'train':data},protection={},progress={},provenance={})


def test_checkpoint_rejects_changed_dropout_before_mutating_model(tmp_path):
    a=model(); p=save_checkpoint(tmp_path/'a.pt',a,None,loaders={'train':loader()},protection={},progress={},provenance={})
    b=model(); b[1].p=.75; before=copy.deepcopy(b.state_dict())
    with pytest.raises(ValueError,match='configuration'):
        load_checkpoint(p,b,None,loaders={'train':loader()},expected_provenance={})
    assert_state_equal(before,b.state_dict())


def test_small_transformer_supports_classification_gradients():
    from alignment_metaplasticity.model import SelectiveCorrigibilityTransformer
    m=SelectiveCorrigibilityTransformer(5,hidden_dim=8,depth=1)
    logits=m(torch.randn(3,5))
    assert logits.shape==(3,2)
    torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,0])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_example_importance_rejects_fractional_labels():
    from alignment_metaplasticity.importance import compute_example_importance
    with pytest.raises(ValueError,match='int64'):
        compute_example_importance(nn.Linear(1,2),[(torch.ones(1,1),torch.tensor([.5]))],torch.device('cpu'))
