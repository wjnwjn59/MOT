from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import cv2

from modules.maritime_analyzer.deterministic_utils import (
    _to_numpy, crop, scale_variation_ratio, is_scale_variation,
    is_low_resolution, is_low_contrast,
)

BBox = Tuple[float, float, float, float]  # x, y, w, h


def laplacian_variance(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())


def is_motion_blur(img, bbox: BBox, threshold: float = 100.0) -> bool:
    gray = _to_numpy(img)
    obj = crop(gray, bbox)
    return bool(laplacian_variance(obj) < threshold)


def is_out_of_frame(bbox: BBox, frame_size: Tuple[int, int], margin: int = 1) -> bool:
    W, H = frame_size
    x, y, w, h = bbox
    return bool(x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin))


def highlight_ratio(img, bbox: BBox, bright_thresh: float = 0.95) -> float:
    gray = _to_numpy(img)  # luminance 0..255
    obj = crop(gray, bbox)
    if obj.size == 0:
        return 0.0
    return float((obj >= (bright_thresh * 255.0)).mean())


def compute_oracle_attributes(template_img, frame_img, template_bbox: BBox, frame_bbox: BBox,
                              frame_size: Tuple[int, int], cfg: Optional[Dict] = None) -> Dict:
    cfg = cfg or {}
    out = {
        "scale_variation": int(is_scale_variation(
            template_bbox, frame_bbox, low=cfg.get("scale_low", 0.7), high=cfg.get("scale_high", 1.4))),
        "low_resolution": int(is_low_resolution(
            frame_bbox, min_side=cfg.get("min_side", 24), min_area=cfg.get("min_area", 900))),
        "low_contrast": int(is_low_contrast(
            frame_img, frame_bbox, ring=cfg.get("ring", 10), min_ratio=cfg.get("contrast_min_ratio", 1.1))),
        "motion_blur": int(is_motion_blur(
            frame_img, frame_bbox, threshold=cfg.get("blur_threshold", 100.0))),
        "out_of_frame": int(is_out_of_frame(
            frame_bbox, frame_size, margin=cfg.get("edge_margin", 1))),
    }
    out["_features"] = {
        "scale_ratio": float(scale_variation_ratio(template_bbox, frame_bbox)),
        "highlight_ratio": highlight_ratio(frame_img, frame_bbox, bright_thresh=cfg.get("bright_thresh", 0.95)),
    }
    return out
