"""Condition-aware feature modulation (FiLM).

Maps a condition descriptor ``[B, cond_dim]`` to per-channel affine parameters
(gamma, beta) and modulates a shared feature ``F`` -> ``F * (1 + gamma) + beta``.

Identity-initialised (the final layer is zeroed) so that at the start of training
gamma == beta == 0 and the modulation is the identity -- it can only ever depart
from the baseline representation if doing so reduces the loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.feat_channels = feat_channels
        self.gen = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feat_channels),
        )
        # identity initialisation: zero the final layer -> gamma=beta=0 -> F'=F
        nn.init.zeros_(self.gen[-1].weight)
        nn.init.zeros_(self.gen[-1].bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.gen(cond).chunk(2, dim=-1)  # each [B, C]
        gamma = torch.tanh(gamma)                       # bounded so it cannot explode
        if feat.dim() == 4:                             # [B, C, H, W]
            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
        return feat * (1.0 + gamma) + beta
