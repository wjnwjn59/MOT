from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from PIL import Image

BBox = Tuple[float, float, float, float]  # x, y, w, h

def _to_numpy(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 3:
        # luminance
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return img.astype(np.float32)

def crop(img: np.ndarray, bbox: BBox) -> np.ndarray:
    x, y, w, h = [int(round(v)) for v in bbox]
    H, W = img.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((1, 1), dtype=img.dtype)
    return img[y0:y1, x0:x1]

def rms_contrast(gray: np.ndarray) -> float:
    return float(np.std(gray.astype(np.float32)))

def ring_crop(gray: np.ndarray, bbox: BBox, ring: int = 10) -> np.ndarray:
    x, y, w, h = [int(round(v)) for v in bbox]
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = x, y, x + w, y + h
    x0r, y0r = max(0, x0 - ring), max(0, y0 - ring)
    x1r, y1r = min(W, x1 + ring), min(H, y1 + ring)
    outer = gray[y0r:y1r, x0r:x1r]
    inner = gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if outer.size == 0:
        return np.zeros((1, 1), dtype=gray.dtype)
    mask = np.ones_like(outer, dtype=bool)
    oy0 = y0 - y0r; ox0 = x0 - x0r
    oy1 = oy0 + inner.shape[0]; ox1 = ox0 + inner.shape[1]
    mask[oy0:oy1, ox0:ox1] = False
    return outer[mask]

def scale_variation_ratio(template_bbox: BBox, frame_bbox: BBox) -> float:
    ta = max(1.0, float(template_bbox[2]) * float(template_bbox[3]))
    fa = max(1.0, float(frame_bbox[2]) * float(frame_bbox[3]))
    return float(np.sqrt(fa / ta))

def is_scale_variation(template_bbox: BBox, frame_bbox: BBox, low: float = 0.7, high: float = 1.4) -> bool:
    r = scale_variation_ratio(template_bbox, frame_bbox)
    return bool(r < low or r > high)

def is_low_resolution(frame_bbox: BBox, min_side: int = 24, min_area: int = 900) -> bool:
    w, h = int(round(frame_bbox[2])), int(round(frame_bbox[3]))
    return bool(min(w, h) < min_side or (w * h) < min_area)

def is_low_contrast(img, bbox: BBox, ring: int = 10, p10_reference: Optional[float] = None, min_ratio: float = 1.1) -> bool:
    gray = _to_numpy(img)
    obj = crop(gray, bbox)
    if obj.size == 0:
        return True
    surround = ring_crop(gray, bbox, ring=ring)
    obj_rms = rms_contrast(obj)
    sur_rms = rms_contrast(surround) if surround.size > 0 else obj_rms + 1e-6
    cond_ratio = (obj_rms / (sur_rms + 1e-6)) < (1.0 / min_ratio)
    cond_p10 = (p10_reference is not None and obj_rms < p10_reference)
    return bool(cond_ratio or cond_p10)

@dataclass
class DeterministicResult:
    scale_variation: int
    low_res: int
    low_contrast: int
    extras: dict

def analyze_pair(template_img, frame_img, template_bbox: BBox, frame_bbox: BBox,
                  p10_contrast_ref: Optional[float] = None,
                  low=0.7, high=1.4, min_side=24, min_area=900, ring=10, min_ratio=1.1) -> DeterministicResult:
    imgF = _to_numpy(frame_img if isinstance(frame_img, Image.Image) else Image.open(frame_img))
    r = scale_variation_ratio(template_bbox, frame_bbox)
    sv = int(r < low or r > high)
    lr = int(is_low_resolution(frame_bbox, min_side=min_side, min_area=min_area))
    lc = int(is_low_contrast(imgF, frame_bbox, ring=ring, p10_reference=p10_contrast_ref, min_ratio=min_ratio))
    return DeterministicResult(sv, lr, lc, {"scale_ratio": r})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--template', required=True)
    ap.add_argument('--frame', required=True)
    ap.add_argument('--template-bbox', nargs=4, type=float, required=True)
    ap.add_argument('--frame-bbox', nargs=4, type=float, required=True)
    ap.add_argument('--p10', type=float, default=None)
    args = ap.parse_args()
    res = analyze_pair(args.template, args.frame, tuple(args.template_bbox), tuple(args.frame_bbox), p10_contrast_ref=args.p10)
    print({
        'scale_variation': res.scale_variation,
        'low_res': res.low_res,
        'low_contrast': res.low_contrast,
        'extras': res.extras,
    })
