import torch

from modules.condition_mtl.losses import (
    hard_ce_loss, soft_kl_loss, multilabel_bce_loss, severity_loss, condition_loss,
)


def test_multilabel_bce_low_when_confident_correct():
    logits = torch.tensor([[10.0, -10.0]])
    target = torch.tensor([[1.0, 0.0]])
    assert multilabel_bce_loss(logits, target).item() < 0.01


def test_severity_l1():
    assert severity_loss(torch.tensor([0.5, 0.2]), torch.tensor([0.5, 0.2])).item() == 0.0


def test_hard_ce_low_when_correct():
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    assert hard_ce_loss(logits, torch.tensor([0])).item() < 0.01


def test_soft_kl_zero_when_matched():
    logits = torch.zeros(1, 2)               # softmax -> [0.5, 0.5]
    target = torch.tensor([[0.5, 0.5]])
    assert soft_kl_loss(logits, target).item() < 1e-6


def test_condition_loss_dispatch_multilabel_with_severity():
    outputs = {"logits": torch.tensor([[10.0, -10.0]]), "severity": torch.tensor([0.5])}
    targets = {"attr_probs": torch.tensor([[1.0, 0.0]]), "severity": torch.tensor([0.5])}
    loss = condition_loss(outputs, targets, mode="multilabel")
    assert loss.item() < 0.01


def test_condition_loss_unknown_mode_raises():
    try:
        condition_loss({"logits": torch.zeros(1, 2)}, {}, mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
