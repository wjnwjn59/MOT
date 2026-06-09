"""Break tracking metrics down per maritime condition.

Given per-frame ``{"iou": float, "attr_probs": [K]}`` records, report average
overlap (AO) and success rate (SR) restricted to frames where each condition is
present -- this is the figure that shows *where* a method helps, not just the
aggregate number.
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np


def per_condition_metrics(frames: List[Dict], attr_names: List[str],
                          present_threshold: float = 0.5,
                          success_threshold: float = 0.5) -> Dict[str, Dict]:
    def summarize(ious: List[float]) -> Dict:
        return {
            "n": len(ious),
            "AO": float(np.mean(ious)) if ious else 0.0,
            "SR": float(np.mean([x >= success_threshold for x in ious])) if ious else 0.0,
        }

    out: Dict[str, Dict] = {}
    for i, name in enumerate(attr_names):
        ious = [f["iou"] for f in frames if f["attr_probs"][i] >= present_threshold]
        out[name] = summarize(ious)
    out["overall"] = summarize([f["iou"] for f in frames])
    return out
