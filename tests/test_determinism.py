from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.experiment import run_method


def _tiny_config():
    return load_config(
        overrides={
            "data": {
                "alignment_train_size": 256,
                "alignment_eval_size": 256,
                "capability_train_size": 256,
                "capability_eval_size": 256,
                "bypass_train_size": 256,
                "batch_size": 64,
            },
            "training": {
                "alignment_epochs": 25,
                "capability_epochs": 3,
                "bypass_epochs": 2,
            },
            "model": {"hidden_dim": 32, "depth": 1, "num_capability_tasks": 2},
        }
    )


def test_same_seed_repeats_exactly():
    cfg = _tiny_config()
    a = run_method(cfg, seed=5, method="fine_tune", include_bypass=False)
    b = run_method(cfg, seed=5, method="fine_tune", include_bypass=False)
    assert a["final_alignment"] == b["final_alignment"]
    assert a["capability_accuracy"] == b["capability_accuracy"]
