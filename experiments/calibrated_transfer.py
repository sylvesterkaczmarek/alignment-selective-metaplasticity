"""Development-calibrated CPU transformer follow-up with cell-paired reporting."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.io_utils import write_json
from alignment_metaplasticity.repro import set_seed
from alignment_metaplasticity.research_integrity import digest, paired_difference, read_journal, validate_records
from alignment_metaplasticity.study import Setting, StudyRun, Trial
from experiments.controlled_study import aggregate, select, plot
from experiments.transfer_study import transformer
from alignment_metaplasticity.experiment import runtime_metadata

METHODS = ('fine_tune', 'ewc', 'metaplastic', 'shuffled', 'uniform')


def source_metadata():
    drivers = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    return {**runtime_metadata('cpu'), 'experiment_drivers_sha256': drivers}


def make_plan():
    config = load_config(overrides={'model': {'hidden_dim': 32, 'depth': 1},
                         'training': {'alignment_epochs': 60},
                         'data': {'alignment_eval_size': 512, 'capability_eval_size': 384}})
    settings = [asdict(Setting(m, 'fixed', lr=lr, strength=200., ewc_lambda=1.))
                for m in METHODS for lr in (.015, .03)]
    development = [asdict(Trial(s, s + 6000, 3100 + i * 100, (1, 2, 3) if i == 0 else (3, 1, 2)))
                   for i, s in enumerate((503, 509))]
    test = [asdict(Trial(s, s + 7000, 4100 + i * 100, (1, 2, 3) if i % 2 == 0 else (3, 1, 2)))
            for i, s in enumerate((601, 607, 613, 617))]
    return {'schema_version': 1, 'study': 'calibrated-transformer-v2', 'config': config,
            'settings': settings, 'development': development, 'test': test,
            'utility_floor': .90, 'meaningful_alignment_difference': .02,
            'bootstrap_seed': 2605, 'bootstrap_resamples': 5000, 'metadata': source_metadata(),
            'selection_rule': 'Greatest development alignment among candidates with mean acquisition >= 0.90; otherwise greatest acquisition, explicitly infeasible. Candidate order breaks ties.',
            'design': 'Two learning-rate candidates per method, two development and four untouched evaluation cells. Fixed strength and EWC coefficient inherited from the prior untuned pilot; not a comprehensive hyperparameter search.',
            'interpretation': 'Synthetic classifier pilot, not a language model or human-influence study. Four cells jointly vary model/data/rule/order. Paired bootstrap intervals are descriptive and are not multiplicity-adjusted significance tests.'}


def execute(path, config, settings, trials):
    rows = []
    with path.open('x', encoding='utf-8') as handle:
        for setting in settings:
            for trial in trials:
                item = {'setting': setting, 'trial': trial, 'id': digest([setting, trial])}
                handle.write(json.dumps({**item, 'status': 'started'}, allow_nan=False) + '\n')
                handle.flush()
                try:
                    result = StudyRun(config, Trial(**trial), Setting(**setting), model_factory=transformer).run()
                    json.dumps(result, allow_nan=False)
                    item.update(status='completed', result=result)
                except Exception as exc:
                    item.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                handle.write(json.dumps(item, allow_nan=False) + '\n')
                handle.flush()
                rows.append(item)
                print(f'{len(rows)}/{len(settings)*len(trials)} {setting["method"]} lr={setting["lr"]} cell={trial["model_seed"]} {item["status"]}', flush=True)
    validate_records(rows, settings, trials)
    return rows


def expected_selection(plan, journal_path):
    records = read_journal(journal_path)
    return {'plan_sha256': digest(plan),
            'development_journal_sha256': hashlib.sha256(journal_path.read_bytes()).hexdigest(),
            **select(records, plan['settings'], len(plan['development']), plan['utility_floor'], trials=plan['development'])}


def run_phase(phase, out):
    out = Path(out)
    set_seed(0)
    if phase == 'development':
        out.mkdir(parents=True, exist_ok=False)
        plan = make_plan()
        write_json(out/'plan.json', plan)
        execute(out/'development.jsonl', plan['config'], plan['settings'], plan['development'])
        selection = expected_selection(plan, out/'development.jsonl')
        write_json(out/'selection.json', selection)
        if len(selection['selected']) != len(METHODS):
            raise RuntimeError('development has incomplete method groups; all failures remain in journal')
        return
    plan = json.loads((out/'plan.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    if digest(selection) != digest(expected_selection(plan, out/'development.jsonl')):
        raise ValueError('development evidence or selection changed')
    if source_metadata() != plan['metadata']:
        raise ValueError('source/runtime changed since development')
    if len(selection['selected']) != len(METHODS):
        raise ValueError('missing selected comparison group')
    lock = {'plan_sha256': digest(plan), 'selection_sha256': digest(selection), 'metadata': source_metadata()}
    with (out/'test-lock.json').open('x', encoding='utf-8') as handle:
        json.dump(lock, handle, indent=2, allow_nan=False)
    settings = [row['setting'] for row in selection['selected']]
    records = execute(out/'test.jsonl', plan['config'], settings, plan['test'])
    comparisons, by_method = [], {}
    for selected in selection['selected']:
        rows = [row for row in records if row['setting'] == selected['setting']]
        complete = len(rows) == len(plan['test']) and all(row['status'] == 'completed' for row in rows)
        results = [row['result'] for row in rows] if complete else []
        if complete:
            by_method[selected['setting']['method']] = results
        metrics = aggregate(results) if complete else None
        comparisons.append({**selected, 'complete': complete, 'metrics': metrics,
                            'utility_feasible_on_test': metrics['acquisition']['mean'] >= plan['utility_floor'] if complete else None})
    paired = []
    for control in ('fine_tune', 'ewc', 'shuffled', 'uniform'):
        if 'metaplastic' in by_method and control in by_method:
            for metric in ('selective_corrigibility', 'acquisition', 'retention'):
                paired.append({'left': 'metaplastic', 'right': control,
                               **paired_difference(by_method['metaplastic'], by_method[control], metric=metric,
                                                   seed=plan['bootstrap_seed'], resamples=plan['bootstrap_resamples'])})
    summary = {'plan_sha256': digest(plan), 'selection_sha256': digest(selection),
               'test_journal_sha256': hashlib.sha256((out/'test.jsonl').read_bytes()).hexdigest(),
               'comparisons': comparisons, 'paired': paired, 'interpretation': plan['interpretation']}
    write_json(out/'summary.json', summary)
    plot([row for row in comparisons if row['complete']], out/'tradeoffs.png')
    if not all(row['complete'] for row in comparisons):
        raise RuntimeError('incomplete held-out comparison; no selective averaging of successful trials')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('development', 'test'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run_phase(args.phase, args.out)


if __name__ == '__main__':
    main()
