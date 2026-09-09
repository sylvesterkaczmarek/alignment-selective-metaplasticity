"""Untuned CPU architecture check with a predeclared small model and budget."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.experiment import runtime_metadata
from alignment_metaplasticity.io_utils import write_json
from alignment_metaplasticity.model import SelectiveCorrigibilityTransformer
from alignment_metaplasticity.probes import challenge
from alignment_metaplasticity.study import Setting, StudyRun, Trial, default_model
from experiments.controlled_study import aggregate


def transformer(spec,cfg):
    return SelectiveCorrigibilityTransformer(spec.input_dim,cfg['model']['hidden_dim'],cfg['model']['depth'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    cfg=load_config('configs/metaplastic.yaml',{'model':{'hidden_dim':32,'depth':1},
                  'training':{'alignment_epochs':60},'data':{'alignment_eval_size':512,'capability_eval_size':384}})
    trials=[asdict(Trial(s,s+4000,2100+100*i,(1,2,3) if i%2==0 else (3,1,2))) for i,s in enumerate((401,409,419))]
    settings=[asdict(Setting(m,'fixed',lr=.015,strength=200.,ewc_lambda=1.))
              for m in ('fine_tune','ewc','metaplastic','shuffled','uniform')]
    plan={'schema_version':1,'config':cfg,'trials':trials,'settings':settings,'models':['mlp','transformer'],
          'metadata':runtime_metadata('cpu'),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'question':'Does selective assignment retain an advantage over shuffled and uniform coefficients when the feature encoder uses self-attention?',
          'interpretation':'Untuned architecture pilot. Three fresh cells, shared data and training budgets within and across architectures. Different parameter counts. No pretrained language models or real-world safety evaluation. No adaptation based on these results.'}
    write_json(args.out/'plan.json',plan)
    completed=[];failures=[]
    with (args.out/'runs.jsonl').open('x') as f:
        def emit(item):
            f.write(json.dumps(item,allow_nan=False)+'\n');f.flush()
        for architecture,factory in (('mlp',default_model),('transformer',transformer)):
            for setting in settings:
                for trial in trials:
                    identity={'model':architecture,'setting':setting,'trial':trial}
                    emit({**identity,'status':'started'})
                    try:
                        run=StudyRun(cfg,Trial(**trial),Setting(**setting),model_factory=factory)
                        result=run.run()
                        checks=[challenge(run,'heldout_context',rehearsal=True,epochs=8),challenge(run,'policy_change',rehearsal=True,epochs=8)] if setting['method']=='metaplastic' else []
                        item={**identity,'status':'completed','parameters':sum(p.numel() for p in run.model.parameters()),'result':result,'challenges':checks}
                        emit(item);completed.append(item)
                        print(architecture,setting['method'],trial['model_seed'],'completed',flush=True)
                    except Exception as exc:
                        item={**identity,'status':'failed','error':f'{type(exc).__name__}: {exc}'}
                        emit(item);failures.append(item)
                        print(architecture,setting['method'],trial['model_seed'],item['error'],flush=True)
    rows=[]
    for architecture in plan['models']:
        for setting in settings:
            selected=[r for r in completed if r['model']==architecture and r['setting']==setting]
            rows.append({'model':architecture,'setting':setting,'complete':len(selected)==len(trials),
                         'parameters':selected[0]['parameters'] if selected else None,
                         'metrics':aggregate([r['result'] for r in selected]) if len(selected)==len(trials) else None})
    write_json(args.out/'summary.json',{'comparisons':rows,'failures':failures,'interpretation':plan['interpretation']})
    if failures: raise RuntimeError(f'{len(failures)} failed trials are retained; no complete comparison for affected groups')


if __name__=='__main__':
    main()
