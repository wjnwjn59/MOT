"""Turn a per-frame annotation record into training targets.

Reads the v2 schema written by ``modules.maritime_analyzer.run.build_record``
(``attributes[name]{prob}`` + ``severity``) and falls back to the old v1
``vlm_response``/``cv_response`` format for backward compatibility.
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional

from modules.maritime_analyzer.taxonomy import attribute_names

# v1 vlm_response key -> v2 taxonomy attribute (illu_change + variance_appear are merged)
_V1_VLM_MAP = {
    "occlusion": "occlusion",
    "background_clutter": "background_clutter",
    "motion_blur": "motion_blur",
    "illu_change": "illumination_appearance_change",
    "variance_appear": "illumination_appearance_change",
    "partial_visibility": "out_of_frame",
}
# v1 cv_response key -> v2 taxonomy attribute
_V1_CV_MAP = {
    "scale_variation": "scale_variation",
    "low_res": "low_resolution",
    "low_contrast": "low_contrast",
}


def record_attr_probs(record: Dict, attr_names: Optional[List[str]] = None) -> List[float]:
    """Per-attribute probability vector in taxonomy order (missing -> 0.0)."""
    attr_names = attr_names or attribute_names()
    if record.get("schema_version") == 2 or "attributes" in record:
        attrs = record.get("attributes", {})
        return [float(attrs.get(n, {}).get("prob", 0.0)) for n in attr_names]

    # ---- v1 backward-compat ----
    probs = {n: 0.0 for n in attr_names}
    for k, v in record.get("vlm_response", {}).items():
        tgt = _V1_VLM_MAP.get(k)
        if tgt is not None and isinstance(v, dict):
            probs[tgt] = max(probs[tgt], float(v.get("flag", 0)))
    for k, v in record.get("cv_response", {}).items():
        tgt = _V1_CV_MAP.get(k)
        if tgt is not None:
            probs[tgt] = max(probs[tgt], float(v))
    return [probs[n] for n in attr_names]


def record_severity(record: Dict) -> float:
    return float(record.get("severity", 0.0))


def record_hard_label(record: Dict, attr_names: Optional[List[str]] = None,
                      normal_threshold: float = 0.5) -> int:
    """Single-label class index: argmax attribute, or the 'normal' class
    (index == len(attr_names)) when no attribute exceeds ``normal_threshold``."""
    attr_names = attr_names or attribute_names()
    probs = record_attr_probs(record, attr_names)
    if not probs or max(probs) < normal_threshold:
        return len(attr_names)  # normal
    return int(probs.index(max(probs)))


def load_sequence_records(jsonl_path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
