import torch
import pytest

from alignment_metaplasticity.tasks import BenchmarkSpec, make_alignment_tensors, make_capability_tensors


@pytest.mark.parametrize("seed", [0, 49, 50, 73, 99, 100, 2**32 - 1])
def test_train_and_evaluation_share_the_task_rule_at_seed_boundaries(seed):
    from alignment_metaplasticity.config import load_config
    from alignment_metaplasticity.experiment import build_loaders
    from alignment_metaplasticity.tasks import _capability_rule

    spec = BenchmarkSpec()
    loaders = build_loaders(load_config(), spec, seed)
    for task in range(1, spec.num_capability_tasks + 1):
        for loader in (loaders[2][task], loaders[3][task]):
            x, y = loader.dataset.tensors
            expected = _capability_rule(x[:, :spec.signal_dim], task_id=task, seed=task)
            assert torch.equal(y, expected)
        assert not torch.equal(loaders[2][task].dataset.tensors[0][:100], loaders[3][task].dataset.tensors[0][:100])


def test_sampling_seed_is_independent_of_explicit_rule_seed():
    spec = BenchmarkSpec()
    x, y = make_capability_tensors(512, spec, 1, seed=173, rule_seed=1)
    changed_x, changed_y = make_capability_tensors(512, spec, 1, seed=173, rule_seed=2)
    assert torch.equal(x, changed_x)
    assert not torch.equal(y, changed_y)


@pytest.mark.parametrize("kwargs", [{"signal_dim": 0}, {"num_capability_tasks": 0}, {"signal_dim": True}])
def test_invalid_benchmark_dimensions_are_rejected(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        BenchmarkSpec(**kwargs)


def test_alignment_authorized_and_unauthorized_targets_differ():
    spec = BenchmarkSpec()
    xa, ya = make_alignment_tensors(128, spec, seed=1, mode="authorized")
    xu, yu = make_alignment_tensors(128, spec, seed=1, mode="unauthorized")
    proposal_a = xa[:, spec.proposal_idx]
    suggestion_a = xa[:, spec.suggestion_idx]
    proposal_u = xu[:, spec.proposal_idx]
    suggestion_u = xu[:, spec.suggestion_idx]
    assert torch.equal(ya, (suggestion_a > 0).long())
    assert torch.equal(yu, (proposal_u > 0).long())
    assert torch.equal(proposal_a, proposal_u)
    assert torch.equal(suggestion_a, suggestion_u)


def test_capability_generation_is_deterministic():
    spec = BenchmarkSpec()
    x1, y1 = make_capability_tensors(64, spec, task_id=1, seed=44)
    x2, y2 = make_capability_tensors(64, spec, task_id=1, seed=44)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_bypass_target_follows_unauthorized_suggestion():
    spec = BenchmarkSpec()
    x, y = make_alignment_tensors(128, spec, seed=9, mode="unauthorized", bypass_flag=1.0, bypass_target=True)
    suggestion = x[:, spec.suggestion_idx]
    assert torch.equal(y, (suggestion > 0).long())
    assert torch.all(x[:, spec.bypass_idx] == 1.0)
