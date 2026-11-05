from copy import deepcopy
from dataclasses import asdict
import json
import random
import numpy as np
import pytest
import torch
from torch import nn

from alignment_metaplasticity.checkpoint import load_checkpoint, save_checkpoint
from alignment_metaplasticity.evaluation import accuracy
from alignment_metaplasticity.importance import compute_gradient_importance
from alignment_metaplasticity.io_utils import write_json
from alignment_metaplasticity.research_integrity import paired_difference, read_journal
from alignment_metaplasticity.study import Setting, Trial, StudyRun
from alignment_metaplasticity.tasks import make_loader
from alignment_metaplasticity.training import train_epochs
from experiments.controlled_study import select
from experiments.calibrated_transfer import make_plan

CPU = torch.device('cpu')


def loader():
    return make_loader((torch.ones(5, 2), torch.tensor([0, 1, 0, 1, 0])), 2, False, 21)


def result(seed, score=.7, method='metaplastic'):
    return {'trial': asdict(Trial(seed, seed + 100, seed + 200)),
            'setting': asdict(Setting(method)), 'method_id': method,
            'alignment': {'selective_corrigibility': score, 'authorized_acceptance': score, 'unauthorized_resistance': score},
            'acquisition': .96, 'retention': .6, 'seconds': 1.,
            'initial_digest': 'paired-weights', 'data_sha256': {'data': 'paired-data'}}


def record(seed, **kwargs):
    r = result(seed, **kwargs)
    return {'setting': r['setting'], 'trial': r['trial'], 'status': 'completed', 'result': r}


@pytest.mark.parametrize('argument,value', [('epochs', True), ('epochs', -1), ('epochs', 1.5),
    ('lr', 0.), ('lr', -1.), ('lr', float('nan')), ('lr', True), ('ewc_lambda', float('inf'))])
def test_training_scalars_rejected_before_model_change(argument, value):
    m = nn.Linear(2, 2); before = deepcopy(m.state_dict())
    args = dict(epochs=1, lr=.01, weight_decay=0.)
    args[argument] = value
    with pytest.raises(ValueError): train_epochs(m, loader(), CPU, **args)
    assert all(torch.equal(before[n], p) for n, p in m.state_dict().items())


@pytest.mark.parametrize('kind', ['shape', 'missing', 'negative', 'nan'])
def test_ewc_importance_validation(kind):
    m = nn.Linear(2, 2)
    anchor = {n: p.detach().clone() for n, p in m.named_parameters()}
    importance = {n: torch.ones_like(p) for n, p in m.named_parameters()}
    if kind == 'shape': importance['weight'] = torch.ones(1)
    if kind == 'missing': importance.pop('bias')
    if kind == 'negative': importance['weight'].fill_(-1)
    if kind == 'nan': importance['weight'].fill_(float('nan'))
    with pytest.raises(ValueError):
        train_epochs(m, loader(), CPU, 1, .01, 0., ewc_anchor=anchor, ewc_importance=importance, ewc_lambda=1.)


def test_batch_importance_preserves_running_buffers_and_modes_on_error():
    m = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout(.3), nn.Linear(2, 2))
    m.train(); m[2].eval()
    modes = [v.training for v in m.modules()]
    before = deepcopy(m.state_dict())
    def broken():
        yield torch.ones(3, 2), torch.tensor([0, 1, 0])
        raise RuntimeError('fixture')
    with pytest.raises(RuntimeError, match='fixture'): compute_gradient_importance(m, broken(), CPU)
    assert modes == [v.training for v in m.modules()]
    assert all(torch.equal(before[n], p) for n, p in m.state_dict().items())


@pytest.mark.parametrize('label', [torch.tensor([.5]), torch.tensor([-1]), torch.tensor([2])])
def test_accuracy_rejects_bad_labels_and_restores_mode(label):
    m = nn.Linear(2, 2); m.train()
    with pytest.raises(ValueError): accuracy(m, [(torch.ones(1, 2), label)], CPU)
    assert m.training


def test_mean_loss_weights_examples_not_batches():
    m = nn.Linear(2, 2)
    with torch.no_grad(): m.weight.zero_(); m.bias.copy_(torch.tensor([2., -2.]))
    examples = [(torch.zeros(3, 2), torch.zeros(3, dtype=torch.long)),
                (torch.zeros(1, 2), torch.ones(1, dtype=torch.long))]
    expected = (3 * nn.functional.cross_entropy(m(examples[0][0]), examples[0][1]).item() +
                nn.functional.cross_entropy(m(examples[1][0]), examples[1][1]).item()) / 4
    frozen = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    d = train_epochs(m, examples, CPU, 1, .01, 0., grad_mask=frozen)
    assert d.mean_loss == pytest.approx(expected)


@pytest.mark.parametrize('field', ['torch_rng', 'python_rng', 'numpy_rng', 'modes', 'model', 'optimizer'])
def test_rejected_checkpoint_restores_every_live_component(tmp_path, field):
    source = nn.Linear(2, 2); opt = torch.optim.AdamW(source.parameters(), lr=.01)
    train_epochs(source, loader(), CPU, 1, .01, .01, optimizer=opt)
    path = save_checkpoint(tmp_path/'checkpoint.pt', source, opt, loaders={'train': loader()}, protection={}, progress={}, provenance={})
    raw = torch.load(path, weights_only=True)
    if field in ('torch_rng', 'python_rng', 'numpy_rng'): raw[field] = ()
    if field == 'modes': raw[field].pop('')
    if field == 'model': raw[field]['weight'] = torch.ones(1)
    if field == 'optimizer': raw[field]['state'][0]['exp_avg'] = torch.ones(1)
    torch.save(raw, path)
    target = nn.Linear(2, 2); target.train()
    optimizer = torch.optim.AdamW(target.parameters(), lr=.01)
    data = loader(); gen = data.generator.get_state().clone()
    weights = deepcopy(target.state_dict()); rng = torch.get_rng_state().clone()
    py = random.getstate(); numpy = np.random.get_state()
    with pytest.raises(ValueError):
        load_checkpoint(path, target, optimizer, loaders={'train': data}, expected_provenance={})
    assert all(torch.equal(weights[n], p) for n, p in target.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(gen, data.generator.get_state())
    assert optimizer.state_dict()['state'] == {} and target.training
    assert py == random.getstate()
    assert np.array_equal(numpy[1], np.random.get_state()[1])


def test_checkpoint_rejects_shared_loader_generator(tmp_path):
    a, b = loader(), loader(); b.generator = a.generator
    with pytest.raises(ValueError, match='independent'):
        save_checkpoint(tmp_path/'x.pt', nn.Linear(2, 2), None, loaders={'a': a, 'b': b}, protection={}, progress={}, provenance={})


@pytest.mark.parametrize('damage', ['duplicate', 'outside-plan', 'result-identity', 'nan', 'boolean', 'substitution'])
def test_selection_rejects_contaminated_evidence(damage):
    rows = [record(1), record(2)]
    settings = [rows[0]['setting']]
    trials = [r['trial'] for r in rows]
    if damage == 'duplicate': rows[1] = deepcopy(rows[0])
    if damage == 'outside-plan': rows[1]['trial'] = asdict(Trial(33, 44, 55))
    if damage == 'result-identity': rows[1]['result']['trial'] = asdict(Trial(33, 44, 55))
    if damage == 'nan': rows[1]['result']['acquisition'] = float('nan')
    if damage == 'boolean': rows[1]['result']['acquisition'] = True
    if damage == 'substitution': rows[1]['result']['method_id'] = 'custom-method'
    with pytest.raises(ValueError): select(rows, settings, 2, .95, trials=trials)


def test_failed_or_missing_cell_does_not_enter_candidate_mean():
    rows = [record(1), record(2)]; settings = [rows[0]['setting']]
    rows[1]['status'] = 'failed'
    assert select(rows, settings, 2, .95)['selected'] == []
    assert select(rows[:1], settings, 2, .95)['selected'] == []


@pytest.mark.parametrize('damage', ['terminal-first', 'duplicate-start', 'duplicate-terminal'])
def test_journal_rejects_ambiguous_attempts(tmp_path, damage):
    row = record(1); start = {**row, 'status': 'started'}
    rows = [row] if damage == 'terminal-first' else [start, start] if damage == 'duplicate-start' else [start, row, row]
    path = tmp_path/'journal.jsonl'
    path.write_text('\n'.join(json.dumps(x) for x in rows))
    with pytest.raises(ValueError): read_journal(path)


def test_interrupted_attempt_stays_visible(tmp_path):
    row = {**record(1), 'status': 'started'}
    path = tmp_path/'journal.jsonl'; path.write_text(json.dumps(row)+'\n')
    assert read_journal(path)[0]['status'] == 'interrupted'


def test_paired_uncertainty_is_cell_based_repeatable_and_no_global_rng_change():
    left = [result(i, .7) for i in range(4)]
    right = [result(i, .5, 'uniform') for i in range(4)]
    rng = np.random.get_state()
    report = paired_difference(left, right, seed=3)
    assert report == paired_difference(list(reversed(left)), right, seed=3)
    assert report['n_cells'] == 4
    assert report['mean_difference'] == pytest.approx(.2)
    assert report['bootstrap_percentile_95'] == pytest.approx([.2, .2])
    assert np.array_equal(rng[1], np.random.get_state()[1])


@pytest.mark.parametrize('damage', ['duplicate', 'missing', 'weights', 'data'])
def test_unpaired_comparisons_rejected(damage):
    left = [result(1), result(2)]
    right = deepcopy(left)
    if damage == 'duplicate': right[1] = deepcopy(right[0])
    if damage == 'missing': right.pop()
    if damage == 'weights': right[0]['initial_digest'] = 'other'
    if damage == 'data': right[0]['data_sha256'] = {}
    with pytest.raises(ValueError): paired_difference(left, right)


def test_invalid_json_does_not_destroy_existing_artifact(tmp_path):
    path = tmp_path/'evidence.json'; path.write_text('original')
    with pytest.raises(ValueError): write_json(path, {'loss': float('nan')})
    assert path.read_text() == 'original'


def test_calibration_has_equal_search_budget_and_disjoint_cells():
    plan = make_plan()
    assert all(sum(s['method'] == method for s in plan['settings']) == 2
               for method in {s['method'] for s in plan['settings']})
    for field in ('model_seed', 'sample_seed', 'rule_seed'):
        assert not ({t[field] for t in plan['development']} & {t[field] for t in plan['test']})
    assert len(plan['test']) == 4


def test_study_lifecycle_cannot_repeat_or_report_partial_state():
    # Exercise state guards without launching another training run.
    run = object.__new__(StudyRun)
    for state in ('running', 'completed'):
        run._run_state = state
        with pytest.raises(ValueError, match='single-use'): run.run()
    run._run_state = 'running'
    with pytest.raises(ValueError, match='complete'): run.result()


@pytest.mark.parametrize('damage', ['trainability-bool-alias', 'missing-moment', 'negative-step', 'fractional-step', 'negative-second-moment'])
def test_checkpoint_rejects_invalid_optimizer_or_trainability(tmp_path, damage):
    source = nn.Linear(2, 2); optimizer = torch.optim.AdamW(source.parameters(), lr=.01)
    train_epochs(source, loader(), CPU, 1, .01, .01, optimizer=optimizer)
    path = save_checkpoint(tmp_path/'x.pt', source, optimizer, loaders={'train': loader()}, protection={}, progress={}, provenance={})
    raw = torch.load(path, weights_only=True)
    state = raw['optimizer']['state'][0]
    if damage == 'trainability-bool-alias': raw['trainability']['weight'] = 1
    if damage == 'missing-moment': state.pop('exp_avg')
    if damage == 'negative-step': state['step'].fill_(-1)
    if damage == 'fractional-step': state['step'].fill_(1.5)
    if damage == 'negative-second-moment': state['exp_avg_sq'].fill_(-1)
    torch.save(raw, path)
    target = nn.Linear(2, 2)
    with pytest.raises(ValueError):
        load_checkpoint(path, target, torch.optim.AdamW(target.parameters(), lr=.01), loaders={'train': loader()}, expected_provenance={})


def test_tensor_digest_supports_bfloat16_without_changing_original_float_bytes():
    from alignment_metaplasticity.study import tensor_digest
    values = {'w': torch.ones(3, dtype=torch.bfloat16)}
    assert tensor_digest(values) == tensor_digest(deepcopy(values))
    values['w'][0] = 2
    assert tensor_digest(values) != tensor_digest({'w': torch.ones(3, dtype=torch.bfloat16)})


def test_checked_calibration_recomputes_from_complete_journals():
    from pathlib import Path
    from scripts.verify_calibration import verify
    report = verify(Path(__file__).resolve().parents[1]/'results/calibrated-transfer-v2')
    assert report['status'] == 'PASS' and report['held_out_runs'] == 20


@pytest.mark.parametrize('file,field', [('summary.json', 'comparisons'), ('selection.json', 'selected')])
def test_artifact_verifier_rejects_rewritten_claims(tmp_path, file, field):
    from pathlib import Path
    import shutil
    from scripts.verify_calibration import verify
    root = Path(__file__).resolve().parents[1]/'results/calibrated-transfer-v2'
    target = tmp_path/'study'; shutil.copytree(root, target)
    data = json.loads((target/file).read_text())
    if file == 'summary.json': data[field][0]['metrics']['alignment']['mean'] = .12345
    else: data[field][0]['setting']['lr'] = .12345
    (target/file).write_text(json.dumps(data))
    with pytest.raises(ValueError): verify(target)


def test_statistic_rounding_tolerance_does_not_relax_identity_or_count():
    from scripts.verify_calibration import equivalent
    assert equivalent({'mean': .5}, {'mean': .5+1e-15})
    assert not equivalent({'mean': .5}, {'mean': .5001})
    assert not equivalent({'n': 4}, {'n': 4.0})
    assert not equivalent(True, 1)
