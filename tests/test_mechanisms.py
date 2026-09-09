import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from alignment_metaplasticity.importance import compute_example_importance
from alignment_metaplasticity.probes import challenge_tensors, perturbations
from alignment_metaplasticity.tasks import BenchmarkSpec
from alignment_metaplasticity.training import train_epochs


@pytest.mark.parametrize("kind", ["empirical", "model_fisher"])
def test_example_estimator_batching_and_gradient_preservation(kind):
    model = nn.Linear(1,2,bias=False).double()
    nn.init.zeros_(model.weight)
    model.weight.grad = torch.ones_like(model.weight)
    data=TensorDataset(torch.ones(3,1,dtype=torch.double),torch.tensor([0,1,1]))
    a=compute_example_importance(model,DataLoader(data,batch_size=1),torch.device('cpu'),kind=kind,normalize=False)
    stats={}
    b=compute_example_importance(model,DataLoader(data,batch_size=2),torch.device('cpu'),kind=kind,normalize=False,stats=stats)
    assert torch.equal(a['weight'],b['weight'])
    assert torch.equal(a['weight'],torch.full_like(model.weight,.25))
    assert torch.equal(model.weight.grad,torch.ones_like(model.weight))
    assert model.training
    assert stats['backward_calls']==(6 if kind=='model_fisher' else 3)


def test_model_fisher_uses_predicted_distribution_not_labels():
    model=nn.Linear(1,2,bias=False).double()
    with torch.no_grad(): model.weight.copy_(torch.tensor([[math.log(4.)],[0.]],dtype=torch.double))
    values=[]
    for label in (0,1):
        data=DataLoader(TensorDataset(torch.ones(1,1,dtype=torch.double),torch.tensor([label])))
        values.append(compute_example_importance(model,data,torch.device('cpu'),kind='model_fisher',normalize=False)['weight'])
    torch.testing.assert_close(values[0],torch.full_like(model.weight,.16))
    assert torch.equal(*values)
    empirical=compute_example_importance(model,DataLoader(TensorDataset(torch.ones(1,1,dtype=torch.double),torch.tensor([0]))),torch.device('cpu'),normalize=False)
    torch.testing.assert_close(empirical['weight'],torch.full_like(model.weight,.04))


def test_empty_estimator_restores_individual_module_modes():
    model=nn.Sequential(nn.Linear(1,2),nn.Dropout()).train()
    model[1].eval()
    with pytest.raises(ValueError,match='at least one'):
        compute_example_importance(model,[],torch.device('cpu'))
    assert model.training and model[0].training and not model[1].training


def test_intervention_counts_and_norms_are_matched_without_rng_changes():
    importance={'a':torch.arange(20.).reshape(4,5),'b':torch.tensor([.1,.2,.3])}
    before=torch.random.get_rng_state().clone()
    for kind in ('high','low','random'):
        d=perturbations(importance,kind=kind,fraction=.2,magnitude=.5,seed=4)
        assert [(v != 0).sum().item() for v in d.values()]==[4,1]
        assert sum(float(v.double().square().sum()) for v in d.values())==pytest.approx(.25)
    assert torch.equal(before,torch.random.get_rng_state())


def test_no_marker_label_conflict_is_explicit_and_context_split_is_disjoint():
    spec=BenchmarkSpec()
    x,y=challenge_tensors(100,spec,3,'no_marker')
    from alignment_metaplasticity.tasks import make_alignment_tensors
    ordinary, labels=make_alignment_tensors(100,spec,3,mode='unauthorized')
    assert torch.equal(x,ordinary) and torch.equal(y,1-labels)
    train,_=challenge_tensors(100,spec,3,'heldout_context')
    test,_=challenge_tensors(100,spec,4,'heldout_context',evaluation=True)
    preserve,_=challenge_tensors(100,spec,5,'heldout_context',attack=False)
    assert (train[:,0]>0).all() and (train[:,1]>0).all()
    assert (test[:,0]>0).all() and (test[:,1]<0).all()
    assert (preserve[:,0]<0).all()


def test_epoch_observation_keeps_momentum_and_training_semantics():
    torch.manual_seed(3)
    a=nn.Sequential(nn.Linear(2,4),nn.Dropout(.2),nn.Linear(4,2))
    b=copy.deepcopy(a)
    x=torch.ones(8,2); y=torch.tensor([0,1]*4)
    data=[(x,y)]
    torch.manual_seed(10)
    train_epochs(a,data,torch.device('cpu'),epochs=3,lr=.1,weight_decay=.01)
    points=[]
    def observe(epoch):
        b.eval(); points.append(epoch)
        with torch.no_grad(): b(x)
    torch.manual_seed(10)
    train_epochs(b,data,torch.device('cpu'),epochs=3,lr=.1,weight_decay=.01,epoch_callback=observe)
    assert points==[1,2,3]
    for p,q in zip(a.parameters(),b.parameters()): assert torch.equal(p,q)


def test_positive_plasticity_delays_constant_gradient_drift_without_bounding_it():
    from alignment_metaplasticity.importance import plasticity_from_importance
    p=plasticity_from_importance({'theta':torch.tensor(1.)},200.,.05)['theta']
    theta=torch.nn.Parameter(torch.tensor(0.,dtype=torch.double))
    optimizer=torch.optim.SGD([theta],lr=.1,momentum=0.)
    for step in range(1,101):
        optimizer.zero_grad()
        (2*theta).backward()
        theta.grad.mul_(p)
        optimizer.step()
        if step in (1,10,100):
            assert theta.item()==pytest.approx(-step*.1*float(p)*2)
    assert abs(theta.item()) > .99


def test_unobserved_feature_has_zero_fisher_importance():
    model=nn.Linear(2,2,bias=False)
    data=DataLoader(TensorDataset(torch.tensor([[1.,0.],[-1.,0.]]),torch.tensor([0,1])))
    for kind in ('empirical','model_fisher'):
        values=compute_example_importance(model,data,torch.device('cpu'),kind=kind,normalize=False)
        assert torch.equal(values['weight'][:,1],torch.zeros(2))
