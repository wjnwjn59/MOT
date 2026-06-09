import torch

from modules.condition_mtl.cls_head import ConditionHead


def test_head_shapes_and_severity_range():
    head = ConditionHead(in_dim=16, hidden_dim=32, num_outputs=9, predict_severity=True)
    out = head(torch.randn(4, 16))
    assert out["logits"].shape == (4, 9)
    assert out["severity"].shape == (4,)
    assert torch.all(out["severity"] >= 0) and torch.all(out["severity"] <= 1)


def test_head_without_severity():
    head = ConditionHead(in_dim=8, hidden_dim=16, num_outputs=10, predict_severity=False)
    out = head(torch.randn(2, 8))
    assert out["logits"].shape == (2, 10)
    assert out["severity"] is None
