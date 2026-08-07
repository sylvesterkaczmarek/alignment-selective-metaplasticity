import torch

from alignment_metaplasticity.tasks import BenchmarkSpec, make_alignment_tensors, make_capability_tensors


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
