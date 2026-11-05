"""Cell-aware experiment validation and paired, descriptive uncertainty."""
from __future__ import annotations
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np
from .study import Setting, Trial


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def trial_key(trial):
    if not isinstance(trial, dict) or set(trial) != {'model_seed', 'sample_seed', 'rule_seed', 'order'}:
        raise ValueError('trial must declare all cell identity fields')
    cell = Trial(**trial)
    cell.validate(len(cell.order) or 1)
    return digest(asdict(cell))


def validate_records(records, settings, trials=None):
    choices = {digest(asdict(Setting(**s))) for s in settings}
    if len(choices) != len(settings):
        raise ValueError('duplicate candidate settings')
    expected = {trial_key(t) for t in trials} if trials is not None else None
    if trials is not None and len(expected) != len(trials):
        raise ValueError('duplicate planned trial cells')
    seen = set()
    for record in records:
        setting_id, cell_id = digest(record['setting']), trial_key(record['trial'])
        pair = setting_id, cell_id
        if pair in seen:
            raise ValueError('duplicate setting/trial; repetitions must be independent cells')
        seen.add(pair)
        if setting_id not in choices or (expected is not None and cell_id not in expected):
            raise ValueError('record is outside the declared experimental plan')
        if record['status'] not in ('completed', 'failed', 'interrupted'):
            raise ValueError('record must have a terminal status')
        if record['status'] == 'completed':
            result = record['result']
            if digest(result['setting']) != setting_id or trial_key(result['trial']) != cell_id:
                raise ValueError('result identity differs from its journal record')
            if result.get('method_id', record['setting']['method']) != record['setting']['method']:
                raise ValueError('substituted method cannot enter built-in method selection')
            checked_metrics(result)
    return seen


def checked_metrics(result):
    values = {key: result['alignment'][key] for key in
              ('selective_corrigibility', 'authorized_acceptance', 'unauthorized_resistance')}
    values.update({key: result[key] for key in ('acquisition', 'retention', 'seconds')})
    for key, value in values.items():
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (key != 'seconds' and value > 1):
            raise ValueError(f'invalid finite metric {key}')
    return values


def read_journal(path):
    """Keep interrupted attempts visible; reject repeated or contradictory terminals."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    starts, terminals = {}, {}
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            identity = digest([row['setting'], row['trial']])
            if row.get('id', identity) != identity:
                raise ValueError('journal identity hash mismatch')
            if row['status'] == 'started':
                if identity in starts:
                    raise ValueError('duplicate started attempt')
                starts[identity] = row
            elif row['status'] in ('completed', 'failed'):
                if identity not in starts or identity in terminals:
                    raise ValueError('terminal record lacks a unique start')
                terminals[identity] = row
            else:
                raise ValueError('unknown journal status')
    return [terminals.get(key, {**row, 'status': 'interrupted'}) for key, row in starts.items()]


def paired_difference(left, right, *, metric='selective_corrigibility', seed=0, resamples=5000):
    """Resample paired trained cells, never treat evaluation examples as replicas."""
    if type(resamples) is not int or not 100 <= resamples <= 100000:
        raise ValueError('resamples must be an integer between 100 and 100000')
    def indexed(rows):
        out = {}
        for row in rows:
            key = trial_key(row['trial'])
            if key in out:
                raise ValueError('duplicate paired cell')
            checked_metrics(row)
            out[key] = row
        return out
    a, b = indexed(left), indexed(right)
    if not a or a.keys() != b.keys():
        raise ValueError('paired comparisons require identical nonempty cell sets')
    differences = []
    for key in sorted(a):
        for field in ('initial_digest', 'data_sha256'):
            if field not in a[key] or a[key][field] != b[key].get(field):
                raise ValueError(f'unpaired {field}')
        def get(row):
            return row['alignment'][metric] if metric == 'selective_corrigibility' else row[metric]
        differences.append(float(get(a[key]) - get(b[key])))
    n = len(differences)
    samples = np.asarray(differences)
    interval = None
    if n >= 2:
        rng = np.random.default_rng(seed)
        bootstrap = samples[rng.integers(n, size=(resamples, n))].mean(axis=1)
        interval = [float(v) for v in np.quantile(bootstrap, (.025, .975))]
    return {'metric': metric, 'n_cells': n, 'differences': differences,
            'mean_difference': statistics.mean(differences),
            'sd_difference': statistics.stdev(differences) if n > 1 else None,
            'bootstrap_percentile_95': interval, 'bootstrap_seed': seed, 'resamples': resamples,
            'positive_cells': sum(v > 0 for v in differences),
            'interpretation': 'Descriptive paired-cell percentile bootstrap; few jointly varied cells, no population or multiplicity-adjusted significance claim.'}
