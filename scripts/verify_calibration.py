"""Recompute selection and paired summaries from complete experiment journals."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Experiments are source tools, not installed runtime modules.
from alignment_metaplasticity.research_integrity import digest, paired_difference, read_journal, validate_records, strict_json
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
    # A checked evidence set carries a manifest. Fresh runs can be checked
    # before packaging; once a manifest exists every listed artifact is binding.
    manifest_path = directory/'checksums.json'
    if (directory/'provenance.json').exists() != manifest_path.exists():
        raise ValueError('packaged evidence requires provenance and checksum manifest together')
    if manifest_path.exists():
        manifest = strict_json(manifest_path.read_text())
        required = {'plan.json', 'provenance.json', 'selection.json', 'summary.json',
                    'development.jsonl', 'test-lock.json', 'test.jsonl', 'tradeoffs.png'}
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise ValueError('artifact manifest must list exactly the expected files')
        for name, expected_hash in manifest.items():
            if not isinstance(expected_hash, str) or re.fullmatch('[0-9a-f]{64}', expected_hash) is None:
                raise ValueError('invalid artifact digest')
            path = directory/name
            if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f'artifact digest mismatch: {name}')
    plan = strict_json((directory/'plan.json').read_text())
    selection = strict_json((directory/'selection.json').read_text())
    lock = strict_json((directory/'test-lock.json').read_text())
    summary = strict_json((directory/'summary.json').read_text())
    if manifest_path.exists():
        provenance = strict_json((directory/'provenance.json').read_text())
        if (provenance.get('source_files') != plan['metadata']['source_files_sha256']
                or provenance.get('experiment_drivers') != plan['metadata']['experiment_drivers_sha256']):
            raise ValueError('artifact provenance differs from the executed-source identity')
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
