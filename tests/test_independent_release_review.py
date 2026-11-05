"""Independent boundary tests authored after the implementation candidate was frozen."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from alignment_metaplasticity.checkpoint import save_checkpoint, load_checkpoint
from alignment_metaplasticity.training import train_epochs
from alignment_metaplasticity.validation import parameter_map
from alignment_metaplasticity.research_integrity import read_journal
from alignment_metaplasticity.study import Setting, Trial
from alignment_metaplasticity.importance import compute_gradient_importance, compute_example_importance
from alignment_metaplasticity.tasks import make_loader
from scripts.verify_calibration import verify

CPU=torch.device('cpu')

def loader(batch=3,shuffle=True):
    return make_loader((torch.arange(20,dtype=torch.float32).reshape(10,2)/20,torch.arange(10)%2),batch,shuffle,73)

def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else: assert a==b

def optim(m,kind='adam',amsgrad=False):
    return torch.optim.SGD(m.parameters(),lr=.03,momentum=.8,weight_decay=.02) if kind=='sgd' else torch.optim.AdamW(m.parameters(),lr=.03,weight_decay=.02,amsgrad=amsgrad)

@pytest.mark.parametrize('damage',['amsgrad-missing','moment-dtype','negative-eps','negative-lr','bad-beta','missing-group-key'])
def test_malformed_optimizer_checkpoint_is_rejected_before_mutation(tmp_path,damage):
    model=nn.Linear(2,2); optimizer=optim(model,amsgrad=True)
    train_epochs(model,loader(),CPU,1,.03,.02,optimizer=optimizer)
    p=save_checkpoint(tmp_path/'state.pt',model,optimizer,loaders={'t':loader()},protection={},progress={},provenance={})
    state=torch.load(p,weights_only=True)
    moment=state['optimizer']['state'][0];group=state['optimizer']['param_groups'][0]
    if damage=='amsgrad-missing':moment.pop('max_exp_avg_sq')
    if damage=='moment-dtype':moment['exp_avg']=moment['exp_avg'].double()
    if damage=='negative-eps':group['eps']=-.1
    if damage=='negative-lr':group['lr']=-.1
    if damage=='bad-beta':group['betas']=(1.5,.999)
    if damage=='missing-group-key':group.pop('eps')
    torch.save(state,p)
    target=nn.Linear(2,2);target_optimizer=optim(target,amsgrad=True)
    before=deepcopy(target.state_dict()); rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):load_checkpoint(p,target,target_optimizer,loaders={'t':loader()},expected_provenance={})
    equal(before,target.state_dict());equal(rng,torch.get_rng_state())
    assert target_optimizer.state_dict()['state']=={}

class Buffered(nn.Module):
    def __init__(self,offset):
        super().__init__();self.fc=nn.Linear(2,2)
        self.register_buffer('offset',torch.tensor([offset,0.]),persistent=False)
    def forward(self,x):return self.fc(x)+self.offset

def test_nonpersistent_buffers_are_preserved_by_exact_restart(tmp_path):
    source=Buffered(2.);p=save_checkpoint(tmp_path/'buffer.pt',source,None,loaders={'t':loader()},protection={},progress={},provenance={})
    target=Buffered(9.)
    load_checkpoint(p,target,None,loaders={'t':loader()},expected_provenance={})
    assert torch.equal(source.offset,target.offset)
    assert torch.equal(source(torch.ones(2,2)),target(torch.ones(2,2)))

@pytest.mark.parametrize('batch', [2,4])
def test_explicit_batch_sampler_is_not_misreported_as_default_loader(tmp_path,batch):
    original=loader().dataset
    custom=DataLoader(original,batch_sampler=BatchSampler(SequentialSampler(original),batch,False),generator=torch.Generator().manual_seed(73))
    with pytest.raises(ValueError):save_checkpoint(tmp_path/'bad.pt',nn.Linear(2,2),None,loaders={'t':custom},protection={},progress={},provenance={})

@pytest.mark.parametrize('damage',['batch-size','drop-last','sampler'])
def test_changed_effective_batching_rejected(tmp_path,damage):
    data=loader()
    if damage=='batch-size':data.batch_sampler.batch_size=2
    if damage=='drop-last':data.batch_sampler.drop_last=True
    if damage=='sampler':data.batch_sampler.sampler=SequentialSampler(data.dataset)
    with pytest.raises(ValueError):save_checkpoint(tmp_path/'bad.pt',nn.Linear(2,2),None,loaders={'t':data},protection={},progress={},provenance={})

def test_protection_conversion_cannot_introduce_infinity():
    model=nn.Linear(2,2)
    values={name:torch.full(p.shape,1e100,dtype=torch.float64) for name,p in model.named_parameters()}
    with pytest.raises(ValueError):parameter_map(model,values,'ewc_anchor')

@pytest.mark.parametrize('kind',['sgd','adam'])
@pytest.mark.parametrize('fraction',[0.,.35,1.])
def test_valid_restart_with_moments_and_randomness(tmp_path,kind,fraction):
    torch.manual_seed(96)
    initial=nn.Sequential(nn.Linear(2,5),nn.Dropout(.2),nn.Linear(5,2))
    full=deepcopy(initial); split=deepcopy(initial)
    a=optim(full,kind);b=optim(split,kind)
    scale={n:torch.full_like(p,fraction) for n,p in full.named_parameters()}
    torch.manual_seed(112)
    train_epochs(full,loader(),CPU,3,.03,.02,optimizer=a,update_scale=scale)
    final_rng=torch.get_rng_state().clone()
    torch.manual_seed(112);partial_loader=loader()
    train_epochs(split,partial_loader,CPU,1,.03,.02,optimizer=b,update_scale=scale)
    p=save_checkpoint(tmp_path/'s.pt',split,b,loaders={'t':partial_loader},protection={'scale':scale},progress={'epoch':1},provenance={'task':'independent'})
    target=deepcopy(initial); o=optim(target,kind);data=loader()
    state=load_checkpoint(p,target,o,loaders={'t':data},expected_provenance={'task':'independent'})
    train_epochs(target,data,CPU,2,.03,.02,optimizer=o,update_scale=state['protection']['scale'])
    equal(full.state_dict(),target.state_dict());equal(a.state_dict(),o.state_dict());equal(final_rng,torch.get_rng_state())

@pytest.mark.parametrize('field',['schema_version','model_signature','trainability','modes','torch_rng','python_rng','numpy_rng','loaders','model','optimizer'])
def test_wrong_checkpoint_field_types_fail_cleanly_without_mutation(tmp_path,field):
    model=nn.Linear(2,2);p=save_checkpoint(tmp_path/'s.pt',model,optim(model),loaders={'t':loader()},protection={},progress={},provenance={})
    raw=torch.load(p,weights_only=True);raw[field]=[];torch.save(raw,p)
    target=nn.Linear(2,2);before=deepcopy(target.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):load_checkpoint(p,target,optim(target),loaders={'t':loader()},expected_provenance={})
    equal(before,target.state_dict());equal(rng,torch.get_rng_state())

@pytest.mark.parametrize('estimator',['batch','empirical','model_fisher'])
def test_estimator_preserves_mixed_modes_gradients_and_buffers(estimator):
    model=nn.Sequential(nn.Linear(2,3),nn.BatchNorm1d(3),nn.Dropout(.3),nn.Linear(3,2))
    model.train();model[2].eval()
    for p in model.parameters():p.grad=torch.ones_like(p)*.2
    before=deepcopy(model.state_dict());modes=[m.training for m in model.modules()]
    if estimator=='batch':importance=compute_gradient_importance(model,loader(),CPU,max_batches=2)
    else:importance=compute_example_importance(model,loader(),CPU,kind=estimator)
    equal(before,model.state_dict());assert modes==[m.training for m in model.modules()]
    assert all(torch.equal(p.grad,torch.ones_like(p)*.2) for p in model.parameters())
    assert all(torch.isfinite(v).all() for v in importance.values())

def test_fisher_matches_independent_linear_softmax_formula():
    model=nn.Linear(1,2,bias=False,dtype=torch.float64)
    with torch.no_grad():model.weight.copy_(torch.tensor([[.4],[-.2]],dtype=torch.float64))
    x=torch.tensor([[1.],[2.],[-.5]],dtype=torch.float64);y=torch.tensor([0,1,0])
    actual=compute_example_importance(model,[(x,y)],CPU,kind='model_fisher',normalize=False)['weight']
    p=model(x).softmax(1)[:,0];expected=(p*(1-p)*x[:,0].square()).mean().repeat(2,1)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('magnitude',[0.,.25,1.])
def test_fractional_adam_update_matches_direct_native_proposal(magnitude):
    original=nn.Linear(2,2);plain=deepcopy(original);protected=deepcopy(original)
    data=[(torch.tensor([[.2,.5],[.4,-.3]]),torch.tensor([0,1]))]
    a=optim(plain);b=optim(protected)
    train_epochs(plain,data,CPU,1,.03,.02,optimizer=a)
    train_epochs(protected,data,CPU,1,.03,.02,optimizer=b,update_scale={n:torch.full_like(p,magnitude) for n,p in protected.named_parameters()})
    for (n,p),q in zip(protected.named_parameters(),plain.parameters()):
        initial=dict(original.named_parameters())[n]
        torch.testing.assert_close(p,initial + magnitude*(q-initial))
    equal(a.state_dict(),b.state_dict())

@pytest.mark.parametrize('damage',['duplicate-key','nan','duplicate-terminal','incomplete'])
def test_journal_records_are_unambiguous(tmp_path,damage):
    row={'setting':asdict(Setting('metaplastic')),'trial':asdict(Trial(2,3,4)),'status':'started'}
    end={**row,'status':'failed','error':'fixture'}
    if damage=='duplicate-key':text=json.dumps(row)[:-1]+',"status":"failed"}\n'
    elif damage=='nan':text=json.dumps({**row,'extra':float('nan')})+'\n'
    elif damage=='duplicate-terminal':text='\n'.join(json.dumps(r) for r in [row,end,end])
    else:
        p=tmp_path/'journal';p.write_text(json.dumps(row)+'\n')
        assert read_journal(p)[0]['status']=='interrupted';return
    p=tmp_path/'journal';p.write_text(text)
    with pytest.raises(ValueError):read_journal(p)

@pytest.mark.parametrize('damage',['provenance','checksum','plot'])
def test_checked_artifact_manifest_detects_changed_record(tmp_path,damage):
    source=Path(__file__).resolve().parent/'repo/results/calibrated-transfer-v2'
    if not source.exists():source=Path(__file__).resolve().parents[1]/'results/calibrated-transfer-v2'
    dest=tmp_path/'record';shutil.copytree(source,dest)
    if damage=='plot':(dest/'tradeoffs.png').write_bytes(b'corrupted image')
    else:
        name='provenance.json' if damage=='provenance' else 'checksums.json'
        value=json.loads((dest/name).read_text());value['fabricated']='changed';(dest/name).write_text(json.dumps(value))
    with pytest.raises(ValueError):verify(dest)


@pytest.mark.parametrize('deleted', ['provenance.json', 'checksums.json'])
def test_checked_evidence_cannot_drop_one_integrity_record(tmp_path, deleted):
    source = Path(__file__).resolve().parents[1]/'results/calibrated-transfer-v2'
    dest = tmp_path/'record'; shutil.copytree(source, dest)
    (dest/deleted).unlink()
    with pytest.raises(ValueError): verify(dest)


def test_valid_amsgrad_restart_with_nonpersistent_buffers(tmp_path):
    torch.manual_seed(8)
    a = Buffered(.7); b = deepcopy(a)
    a_opt = optim(a, amsgrad=True); b_opt = optim(b, amsgrad=True)
    train_epochs(a, loader(), CPU, 2, .03, .02, optimizer=a_opt)
    b_loader = loader()
    train_epochs(b, b_loader, CPU, 1, .03, .02, optimizer=b_opt)
    p = save_checkpoint(tmp_path/'state.pt', b, b_opt, loaders={'t': b_loader}, protection={}, progress={'epoch':1}, provenance={})
    target = Buffered(99.); target_opt = optim(target, amsgrad=True); data = loader()
    load_checkpoint(p, target, target_opt, loaders={'t':data}, expected_provenance={})
    train_epochs(target, data, CPU, 1, .03, .02, optimizer=target_opt)
    equal(a.state_dict(), target.state_dict()); equal(a.offset, target.offset)
    equal(a_opt.state_dict(), target_opt.state_dict())


def test_ewc_penalty_matches_closed_form_and_gradient():
    from alignment_metaplasticity.training import _ewc_penalty
    m = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    anchor = torch.tensor([[.2,-.4],[.7,.1]], dtype=torch.float64)
    weights = torch.tensor([[1.,0.],[2.,.5]], dtype=torch.float64)
    with torch.no_grad(): m.weight.copy_(torch.tensor([[.7,.3],[-.2,.8]], dtype=torch.float64))
    difference = m.weight.detach()-anchor
    penalty = _ewc_penalty(m, {'weight':anchor}, {'weight':weights})
    torch.testing.assert_close(penalty, .5*(weights*difference.square()).sum(), rtol=0, atol=0)
    penalty.backward()
    torch.testing.assert_close(m.weight.grad, weights*difference, rtol=0, atol=0)


def test_held_out_statistics_match_independent_numpy_calculation():
    root = Path(__file__).resolve().parents[1]/'results/calibrated-transfer-v2'
    rows = [json.loads(line) for line in (root/'test.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['status']=='completed']
    summary = json.loads((root/'summary.json').read_text())
    for comparison in summary['comparisons']:
        results = [row['result'] for row in rows if row['setting']==comparison['setting']]
        assert len(results)==4
        for metric in ('alignment', 'acquisition', 'retention'):
            values = [r['alignment']['selective_corrigibility'] if metric=='alignment' else r[metric] for r in results]
            reported = comparison['metrics'][metric]
            assert reported['n']==4
            assert reported['mean']==pytest.approx(float(np.mean(values)), abs=1e-14)
            assert reported['sd']==pytest.approx(float(np.std(values, ddof=1)), abs=1e-14)
    assert min(row['result']['alignment']['selective_corrigibility'] for row in rows if row['setting']['method']=='metaplastic') < .01
