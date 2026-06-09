"""Condition-task losses for the four MTL run modes.

- hard:       single-label cross-entropy over (num_attrs + normal) classes
- soft:       KL divergence against a soft target distribution
- multilabel: per-attribute BCE (soft targets) + severity L1
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


def hard_ce_loss(logits: torch.Tensor, labels: torch.Tensor,
                 weight: Optional[torch.Tensor] = None, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=weight, ignore_index=ignore_index)


def soft_kl_loss(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_p, target_dist, reduction="batchmean")


def multilabel_bce_loss(logits: torch.Tensor, target_probs: torch.Tensor,
                        mask: Optional[torch.Tensor] = None,
                        pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits, target_probs, reduction="none", pos_weight=pos_weight)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1.0)
    return loss.mean()


def severity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def condition_loss(outputs, targets, mode: str, severity_weight: float = 1.0,
                   weight: Optional[torch.Tensor] = None,
                   pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dispatch to the right loss for ``mode``.

    outputs: dict with "logits" [B, C] and optional "severity" [B].
    targets: dict carrying the fields each mode needs:
      - hard:       targets["labels"]  (long [B])
      - soft:       targets["dist"]    (float [B, C], rows sum to 1)
      - multilabel: targets["attr_probs"] (float [B, K]); optional "severity" [B] + "mask"
    """
    logits = outputs["logits"]
    if mode == "hard":
        return hard_ce_loss(logits, targets["labels"], weight=weight)
    if mode == "soft":
        return soft_kl_loss(logits, targets["dist"])
    if mode == "multilabel":
        loss = multilabel_bce_loss(logits, targets["attr_probs"],
                                   mask=targets.get("mask"), pos_weight=pos_weight)
        if outputs.get("severity") is not None and targets.get("severity") is not None:
            loss = loss + severity_weight * severity_loss(outputs["severity"], targets["severity"])
        return loss
    raise ValueError(f"unknown condition-loss mode: {mode}")
