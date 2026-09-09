from dataclasses import asdict
import copy

import pytest
import torch

from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.study import Setting, StudyRun, Trial, transform_plasticity
from experiments.controlled_study import candidates, select


def cfg():
    return load_config(overrides={"model": {"hidden_dim": 32, "depth": 1, "num_capability_tasks": 2},
        "data": {"alignment_train_size": 256, "alignment_eval_size": 128, "capability_train_size": 128,
                 "capability_eval_size": 128, "batch_size": 64},
        "training": {"alignment_epochs": 25, "capability_epochs": 2}, "importance": {"max_batches": 2}})


def test_mask_controls_preserve_budget_and_rng():
    p = {"a": torch.arange(20.).reshape(4, 5) / 20, "b": torch.tensor([.2, .7])}
    before = torch.random.get_rng_state().clone()
    shuffled = transform_plasticity(p, "shuffled", 5)
    assert torch.equal(before, torch.random.get_rng_state())
    for k in p:
        assert torch.equal(p[k].flatten().sort().values, shuffled[k].flatten().sort().values)
        assert torch.equal(shuffled[k], transform_plasticity(p, "shuffled", 5)[k])
        assert torch.allclose(transform_plasticity(p, "uniform", 5)[k].mean(), p[k].mean())


def test_shared_initialisation_data_and_exact_exposure():
    trial = Trial(5, 5, 20, (2, 1))
    a = StudyRun(cfg(), trial, Setting("metaplastic", "fixed", strength=0)).run()
    b = StudyRun(cfg(), trial, Setting("fine_tune", "periodic")).run()
    assert a["initial_digest"] == b["initial_digest"]
    assert a["data_sha256"] == b["data_sha256"]
    assert a["final_digest"] == b["final_digest"]
    assert a["cost"]["capability"] == {"examples": 512, "batches": 8}
    assert a["cost"]["importance"] == {"examples": 128, "batches": 2, "backward_calls": 2}
    c = StudyRun(cfg(), trial, Setting("uniform", "periodic")).run()
    assert c["cost"]["importance"] == {"examples": 384, "batches": 6, "backward_calls": 6}
    d = StudyRun(cfg(), trial, Setting("rehearsal", "periodic")).run()
    assert d["cost"]["rehearsal"] == {"examples": 256, "batches": 4}
    assert "importance" not in d["cost"]
    assert [h["task"] for h in d["history"]] == [2, 1]


def test_controlled_rules_and_initial_weights_are_paired():
    a = StudyRun(cfg(), Trial(5, 5, 20), Setting("ewc"))
    b = StudyRun(cfg(), Trial(5, 5, 21), Setting("shuffled"))
    assert a.initial_digest == b.initial_digest
    assert a.data_hashes["alignment_train"] == b.data_hashes["alignment_train"]
    assert a.data_hashes["capability_train_1"] != b.data_hashes["capability_train_1"]
    from alignment_metaplasticity.tasks import make_capability_tensors
    expected = make_capability_tensors(128, a.spec, 1, 5+100+50, rule_seed=21)
    assert torch.equal(expected[1], a.loaders[3][1].dataset.tensors[1])


def test_equal_tuning_budgets_and_no_failed_candidate_selection():
    choices = candidates()
    groups = {(s["method"], s["protocol"]) for s in choices}
    assert all(sum((s["method"], s["protocol"]) == g for s in choices) == 4 for g in groups)
    setting = asdict(Setting("metaplastic"))
    assert select([{"setting":setting,"status":"failed"}], [setting], 2, .95)["selected"] == []


@pytest.mark.parametrize("order", [(1, 1), (2,), (True, 2), (1, 3)])
def test_invalid_order(order):
    with pytest.raises(ValueError):
        Trial(5, 5, 0, order).validate(2)


def test_selection_uses_utility_constraint_and_deterministic_ties():
    settings = [asdict(Setting("ewc", ewc_lambda=v)) for v in (1., 3., 18.)]
    def result(align, utility, seconds):
        return {"alignment": {"selective_corrigibility":align,"authorized_acceptance":align,"unauthorized_resistance":align},
                "acquisition":utility,"retention":.6,"seconds":seconds}
    records = [{"setting":s,"status":"completed","result":result(a,u,seconds)}
               for s,a,u,seconds in zip(settings,(.7,.7,.99),(.96,.96,.94),(100.,.1,1.)) for _ in range(2)]
    selected = select(records, settings, 2, .95)["selected"][0]
    assert selected["setting"] == settings[0]
    assert selected["utility_feasible_on_development"]


def test_custom_protection_can_be_substituted_without_mislabelled_results():
    def identity(importance, anchor):
        return {"grad_scale":{n:torch.ones_like(v) for n,v in importance.items()}}
    trial=Trial(5,5,20)
    with pytest.raises(ValueError,match="protection_name"):
        StudyRun(cfg(),trial,Setting("metaplastic"),protection_factory=identity)
    custom=StudyRun(cfg(),trial,Setting("metaplastic"),protection_factory=identity,
                    protection_name="identity-control-v1").run()
    ordinary=StudyRun(cfg(),trial,Setting("fine_tune")).run()
    assert custom["method_id"]=="identity-control-v1"
    assert ordinary["method_id"]=="fine_tune"
    assert custom["final_digest"]==ordinary["final_digest"]
