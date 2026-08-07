import torch

from alignment_metaplasticity.importance import plasticity_from_importance, static_train_masks


def test_more_important_parameters_are_less_plastic():
    imp = {"w": torch.tensor([0.0, 0.25, 1.0])}
    p = plasticity_from_importance(imp, strength=5.0, min_plasticity=0.05)["w"]
    assert p[0] > p[1] > p[2]
    assert p.min() >= 0.05
    assert p.max() <= 1.0


def test_static_mask_freezes_high_importance_elements():
    imp = {"w": torch.tensor([0.0, 0.1, 0.9, 1.0])}
    mask = static_train_masks(imp, freeze_quantile=0.5)["w"]
    assert mask[0] == 1
    assert mask[-1] == 0
