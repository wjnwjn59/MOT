"""Tracker-agnostic condition head.

Consumes a pooled feature ``[B, in_dim]`` and emits attribute logits plus an
optional severity scalar. The same head serves every run mode; the loss module
interprets the logits (softmax for hard/soft, per-attribute sigmoid for
multi-label).
"""
from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, num_outputs: int = 9,
                 predict_severity: bool = True, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_outputs)
        self.severity = nn.Linear(hidden_dim, 1) if predict_severity else None
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        x = F.relu(self.proj(feat))
        x = self.dropout(x)
        logits = self.out(x)
        severity = torch.sigmoid(self.severity(x)).squeeze(-1) if self.severity is not None else None
        return {"logits": logits, "severity": severity}
