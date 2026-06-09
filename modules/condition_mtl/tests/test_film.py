import torch

from modules.condition_mtl.film import FiLM


def test_identity_init_is_a_noop_4d():
    film = FiLM(cond_dim=4, feat_channels=8)
    feat = torch.randn(2, 8, 3, 3)
    cond = torch.randn(2, 4)
    out = film(feat, cond)
    assert torch.allclose(out, feat, atol=1e-6)  # zero-init -> identity


def test_identity_init_is_a_noop_2d():
    film = FiLM(cond_dim=4, feat_channels=8)
    feat = torch.randn(2, 8)
    out = film(feat, torch.randn(2, 4))
    assert torch.allclose(out, feat, atol=1e-6)


def test_modulation_changes_output_after_perturbation():
    film = FiLM(cond_dim=4, feat_channels=8)
    with torch.no_grad():
        film.gen[-1].weight.add_(1.0)  # break identity init
        film.gen[-1].bias.add_(0.5)
    feat = torch.randn(2, 8)
    out = film(feat, torch.randn(2, 4))
    assert not torch.allclose(out, feat, atol=1e-6)
