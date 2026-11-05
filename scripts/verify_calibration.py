"""Recompute selection and paired summaries from complete experiment journals."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Experiments are source tools, not installed runtime modules.
from alignment_metaplasticity.research_integrity import digest, paired_difference, read_journal, validate_records
from experiments.calibrated_transfer import METHODS, expected_selection
from experiments.controlled_study import aggregate


def equivalent(a, b):
    """Permit only rounding-level statistic differences across Python versions."""
    if type(a) is float and type(b) is float:
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return a == b


def verify(directory):
    directory = Path(directory)
    plan = json.loads((directory/'plan.json').read_text())
    selection = json.loads((directory/'selection.json').read_text())
    lock = json.loads((directory/'test-lock.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    expected = expected_selection(plan, directory/'development.jsonl')
    if not equivalent(selection, expected):
        raise ValueError('selection is inconsistent with the development journal')
    if lock != {'plan_sha256': digest(plan), 'selection_sha256': digest(selection), 'metadata': plan['metadata']}:
        raise ValueError('test lock does not bind the development selection and source')
    if summary['plan_sha256'] != digest(plan) or summary['selection_sha256'] != digest(selection):
        raise ValueError('summary provenance mismatch')
    if hashlib.sha256((directory/'test.jsonl').read_bytes()).hexdigest() != summary['test_journal_sha256']:
        raise ValueError('held-out journal hash mismatch')
    rows = read_journal(directory/'test.jsonl')
    settings = [r['setting'] for r in selection['selected']]
    validate_records(rows, settings, plan['test'])
    expected_pairs = {(digest(s), digest(t)) for s in settings for t in plan['test']}
    actual_pairs = {(digest(r['setting']), digest(r['trial'])) for r in rows if r['status'] == 'completed'}
    if expected_pairs != actual_pairs or len(settings) != len(METHODS):
        raise ValueError('incomplete held-out evidence')
    comparisons, by_method = [], {}
    for selected in selection['selected']:
        results = [r['result'] for r in rows if r['setting'] == selected['setting']]
        by_method[selected['setting']['method']] = results
        metrics = aggregate(results)
        comparisons.append({**selected, 'complete': True, 'metrics': metrics,
                            'utility_feasible_on_test': metrics['acquisition']['mean'] >= plan['utility_floor']})
    paired = [{'left': 'metaplastic', 'right': control,
               **paired_difference(by_method['metaplastic'], by_method[control], metric=metric,
                                   seed=plan['bootstrap_seed'], resamples=plan['bootstrap_resamples'])}
              for control in ('fine_tune', 'ewc', 'shuffled', 'uniform')
              for metric in ('selective_corrigibility', 'acquisition', 'retention')]
    if not equivalent(comparisons, summary['comparisons']) or not equivalent(paired, summary['paired']):
        raise ValueError('reported statistics differ from the recorded trained cells')
    return {'status': 'PASS', 'development_attempts': len(read_journal(directory/'development.jsonl')),
            'held_out_cells': len(plan['test']), 'held_out_runs': len(rows), 'paired_comparisons': len(paired),
            'meaning': 'Evidence and summary consistency, not an independent training reproduction.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
