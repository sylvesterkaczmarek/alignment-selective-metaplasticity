"""Exploratory estimator, intervention and context tests after locked comparisons."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.experiment import runtime_metadata
from alignment_metaplasticity.importance import compute_gradient_importance, compute_example_importance, normalize_importance
from alignment_metaplasticity.io_utils import write_json
from alignment_metaplasticity.probes import challenge, familiar_context_loaders, intervention_probe
from alignment_metaplasticity.study import Setting, StudyRun, Trial, tensor_digest
from alignment_metaplasticity.tasks import make_loader


def sensitivity(run):
    x,y=run.loaders[0].dataset.tensors
    # Fixed examples for batching; separate prefixes test sample-count sensitivity.
    outputs=[]
    for estimator in ('batch','empirical','model_fisher'):
        for size in (64,256):
            for batch in (1,32,128):
                loader=make_loader((x[:size],y[:size]),batch,False,31)
                if estimator=='batch':
                    value=compute_gradient_importance(run.model,loader,run.device,max_batches=len(loader),quantile=.95)
                    variants=[(.95,value)]
                else:
                    raw=compute_example_importance(run.model,loader,run.device,kind=estimator,normalize=False)
                    variants=[(q,normalize_importance(raw,q)) for q in (.9,.95,1.)]
                for quantile,value in variants:
                    flat=torch.cat([v.flatten() for v in value.values()])
                    outputs.append({'estimator':estimator,'examples':size,'batch_size':batch,'quantile':quantile,
                                    'mean':float(flat.mean()),'clipped_fraction':float((flat==1).float().mean()),
                                    'zero_fraction':float((flat==0).float().mean()),'sha256':tensor_digest(value)})
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,default=Path('results/controlled-v1/selection.json'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    selection=json.loads(args.selection.read_text())
    settings=[s['setting'] for s in selection['selected'] if s['group'] in ('fixed/ewc','fixed/metaplastic')]
    if len(settings)!=2:
        raise ValueError('selection must contain the fixed EWC and metaplastic conditions')
    cfg=load_config('configs/metaplastic.yaml',{'model':{'num_capability_tasks':6},
                         'data':{'alignment_eval_size':512,'capability_eval_size':384}})
    trials=[asdict(Trial(s,s+3000,1500+100*i,tuple(range(1,7)) if i%2==0 else (6,4,2,5,3,1)))
            for i,s in enumerate((307,311,313))]
    plan={'schema_version':1,'config':cfg,'settings':settings,'trials':trials,
          'estimators':['batch','empirical','model_fisher'],'conditions':['novel_flag','heldout_context','no_marker','policy_change'],
          'intervention_magnitudes':[.1,.5,1.], 'challenge_epochs':8,
          'selection_sha256':hashlib.sha256(args.selection.read_bytes()).hexdigest(),
          'metadata':runtime_metadata('cpu'),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'interpretation':'Exploratory six-task extension on fresh cells. Original development-selected parameters are kept fixed across estimator substitutions; estimators are not independently tuned.'}
    write_json(args.out/'plan.json',plan)
    failures=[]
    with (args.out/'runs.jsonl').open('x') as f:
        def record(item):
            f.write(json.dumps(item,allow_nan=False)+'\n');f.flush()
        for setting in settings:
            for trial in trials:
                for estimator in plan['estimators']:
                    identity={'setting':setting,'trial':trial,'estimator':estimator,'familiar_context':False}
                    record({**identity,'status':'started'})
                    try:
                        run=StudyRun(cfg,Trial(**trial),Setting(**setting),importance_estimator=estimator)
                        result=run.run()
                        probes=[intervention_probe(run,magnitude=m,seed=trial['model_seed']+94000) for m in plan['intervention_magnitudes']]
                        challenges=[challenge(run,condition,rehearsal=replay,epochs=plan['challenge_epochs'])
                                    for condition in plan['conditions'] for replay in (False,True)] if estimator=='batch' else []
                        record({**identity,'status':'completed','result':result,'interventions':probes,'challenges':challenges})
                        print(setting['method'],trial['model_seed'],estimator,'completed',flush=True)
                        if estimator=='batch' and setting['method']=='metaplastic' and trial==trials[0]:
                            write_json(args.out/'estimator-sensitivity.json',sensitivity(run))
                    except Exception as exc:
                        error={**identity,'status':'failed','error':f'{type(exc).__name__}: {exc}'}
                        record(error);failures.append(error)
                identity={'setting':setting,'trial':trial,'estimator':'batch','familiar_context':True}
                record({**identity,'status':'started'})
                try:
                    run=StudyRun(cfg,Trial(**trial),Setting(**setting),loader_factory=familiar_context_loaders)
                    result=run.run()
                    record({**identity,'status':'completed','result':result,
                            'challenges':[challenge(run,'seen_flag',rehearsal=r,epochs=plan['challenge_epochs']) for r in (False,True)]})
                    print(setting['method'],trial['model_seed'],'seen context completed',flush=True)
                except Exception as exc:
                    error={**identity,'status':'failed','error':f'{type(exc).__name__}: {exc}'}
                    record(error);failures.append(error)
    if failures:
        raise RuntimeError(f'{len(failures)} trials failed; see retained journal')


if __name__=='__main__':
    main()
