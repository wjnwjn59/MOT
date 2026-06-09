# Label Generation Pipeline (Plan A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce refined multi-label + severity maritime-condition labels for MVTD — deterministic *oracle* attributes for objective conditions, soft *VLM* probabilities for subjective ones — generated multi-GPU, plus a partial-visibility audit that quantifies the old labels against an oracle.

**Architecture:** New tracker-agnostic code under `modules/maritime_analyzer/`. Pure, unit-tested logic (taxonomy, oracle math, VLM-output parsing/aggregation, sharding, record building, audit comparison) is separated from external-model/IO glue (the VLM call, the multi-GPU subprocess launcher), which is validated by a smoke run. Output is a new JSONL schema (`schema_version: 2`) consumed later by the training toolkit (Plan B).

**Tech Stack:** Python, NumPy, OpenCV (`cv2`, already in `requirements.txt`), Pillow, vLLM/SGLang for the VLM (`Qwen/Qwen3.5-35B-A3B`), pytest for tests.

---

## Prerequisites

- Run all commands from the repo root: `/media/vli-ws2/061EA9451EA92F1D/thangdd_workspace/MOT`
- `export PYTHONPATH=./` (matches the project README import convention; tests import `modules.maritime_analyzer.*`).
- Ensure pytest is installed: `python -m pytest --version` — if missing, `pip install pytest`.
- All commits follow repo convention (end the message with the `Co-Authored-By: Claude Opus 4.8` trailer when actually committing).

## File Structure (created/modified by this plan)

```
modules/maritime_analyzer/
  taxonomy.py            CREATE  single source of truth for the attribute set + VLM prompt
  oracles.py            CREATE  deterministic attribute computation (reuses deterministic_utils.py)
  vlm_analyzer.py       MODIFY  add soft multi-attribute parsing/aggregation + classify_soft()
  run.py                MODIFY  add shard/gpu-group/record helpers + multi-GPU worker/orchestrator
  validation/
    __init__.py         CREATE  (empty)
    audit_partial_visibility.py  CREATE  old-VLM-vs-oracle partial-visibility audit
  tests/
    test_taxonomy.py    CREATE
    test_oracles.py     CREATE
    test_vlm_parsing.py CREATE
    test_run_helpers.py CREATE
    test_audit.py       CREATE
```

Public API names locked here (used across tasks):
- `taxonomy.attribute_names()`, `oracle_attributes()`, `vlm_attributes()`, `build_vlm_prompt()`, `SCHEMA_VERSION`, `SEVERITY_KEY`
- `oracles.is_motion_blur()`, `is_out_of_frame()`, `highlight_ratio()`, `laplacian_variance()`, `compute_oracle_attributes()`
- `vlm_analyzer.parse_vlm_json()`, `aggregate_passes()`, `VLMAnalyzer.classify_soft()`
- `run.shard_sequences()`, `plan_gpu_groups()`, `build_worker_commands()`, `build_record()`
- `validation.audit_partial_visibility.compare_partial_visibility()`

---

## Task 1: Taxonomy (single source of truth)

**Files:**
- Create: `modules/maritime_analyzer/taxonomy.py`
- Test: `modules/maritime_analyzer/tests/test_taxonomy.py`

- [ ] **Step 1: Write the failing test**

```python
# modules/maritime_analyzer/tests/test_taxonomy.py
from modules.maritime_analyzer import taxonomy as tx


def test_attribute_names_are_unique_and_complete():
    names = tx.attribute_names()
    assert len(names) == 9
    assert len(set(names)) == 9
    assert "scale_variation" in names
    assert "specular_glare" in names


def test_ids_are_contiguous_and_unique():
    ids = [a["id"] for a in tx.ATTRIBUTES]
    assert sorted(ids) == list(range(9))


def test_oracle_vlm_partition():
    assert tx.oracle_attributes() == [
        "scale_variation", "low_resolution", "low_contrast", "motion_blur", "out_of_frame",
    ]
    assert tx.vlm_attributes() == [
        "occlusion", "background_clutter", "specular_glare", "illumination_appearance_change",
    ]


def test_prompt_mentions_every_vlm_attribute_and_severity():
    prompt = tx.build_vlm_prompt()
    for name in tx.vlm_attributes():
        assert name in prompt
    assert "severity" in prompt
    assert "JSON" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest modules/maritime_analyzer/tests/test_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.maritime_analyzer.taxonomy'`

- [ ] **Step 3: Write minimal implementation**

```python
# modules/maritime_analyzer/taxonomy.py
from __future__ import annotations
from typing import Dict, List

# source: "oracle" (deterministic) or "vlm" (subjective soft probability)
ATTRIBUTES: List[Dict] = [
    {"name": "scale_variation", "id": 0, "source": "oracle"},
    {"name": "low_resolution", "id": 1, "source": "oracle"},
    {"name": "low_contrast", "id": 2, "source": "oracle"},
    {"name": "motion_blur", "id": 3, "source": "oracle"},
    {"name": "out_of_frame", "id": 4, "source": "oracle"},
    {"name": "occlusion", "id": 5, "source": "vlm",
     "desc": "the target is partly blocked by another object/vessel (NOT cut by the frame edge)"},
    {"name": "background_clutter", "id": 6, "source": "vlm",
     "desc": "water texture, wakes, foam, or other vessels make the target hard to distinguish"},
    {"name": "specular_glare", "id": 7, "source": "vlm",
     "desc": "strong sun glare or specular reflection on the water near/over the target"},
    {"name": "illumination_appearance_change", "id": 8, "source": "vlm",
     "desc": "the target's brightness, colour, pose or appearance changed notably vs the template"},
]

SEVERITY_KEY = "severity"
SCHEMA_VERSION = 2


def attribute_names() -> List[str]:
    return [a["name"] for a in ATTRIBUTES]


def oracle_attributes() -> List[str]:
    return [a["name"] for a in ATTRIBUTES if a["source"] == "oracle"]


def vlm_attributes() -> List[str]:
    return [a["name"] for a in ATTRIBUTES if a["source"] == "vlm"]


def _vlm_full() -> List[Dict]:
    return [a for a in ATTRIBUTES if a["source"] == "vlm"]


def build_vlm_prompt() -> str:
    lines = [
        "You are an expert maritime vision annotator.",
        "Image A is a template crop of a target. Image B is a new frame with the target boxed.",
        "For EACH challenge below, output a probability in [0,1] that it is present for the boxed target in Image B.",
        "Also output 'severity' in [0,1]: overall how hard the target is to track in this frame.",
        "",
        "Challenges:",
    ]
    for a in _vlm_full():
        lines.append(f"- {a['name']}: {a['desc']}")
    lines += [
        "",
        "Return STRICT JSON only: a float for each challenge name and for 'severity'. Example:",
        '{"occlusion": 0.0, "background_clutter": 0.0, "specular_glare": 0.0, '
        '"illumination_appearance_change": 0.0, "severity": 0.0}',
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest modules/maritime_analyzer/tests/test_taxonomy.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add modules/maritime_analyzer/taxonomy.py modules/maritime_analyzer/tests/test_taxonomy.py
git commit -m "feat(labels): add maritime condition taxonomy (multi-label + severity)"
```

---

## Task 2: Oracle attributes

**Files:**
- Create: `modules/maritime_analyzer/oracles.py`
- Test: `modules/maritime_analyzer/tests/test_oracles.py`
- Reuses: `modules/maritime_analyzer/deterministic_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# modules/maritime_analyzer/tests/test_oracles.py
import numpy as np
from modules.maritime_analyzer import oracles as o


def test_laplacian_variance_zero_for_uniform():
    assert o.laplacian_variance(np.full((20, 20), 128.0, dtype=np.float32)) == 0.0


def test_is_motion_blur_true_for_uniform_false_for_checkerboard():
    uniform = np.full((20, 20), 128.0, dtype=np.float32)
    checker = np.indices((20, 20)).sum(axis=0) % 2 * 255.0
    assert o.is_motion_blur(uniform, (0, 0, 20, 20), threshold=100.0) is True
    assert o.is_motion_blur(checker.astype(np.float32), (0, 0, 20, 20), threshold=100.0) is False


def test_is_out_of_frame():
    assert o.is_out_of_frame((10, 10, 20, 20), (100, 100), margin=1) is False
    assert o.is_out_of_frame((0, 10, 20, 20), (100, 100), margin=1) is True   # touches left edge
    assert o.is_out_of_frame((85, 10, 20, 20), (100, 100), margin=1) is True  # x+w >= W-1


def test_highlight_ratio():
    bright = np.full((10, 10), 255.0, dtype=np.float32)
    dark = np.zeros((10, 10), dtype=np.float32)
    assert o.highlight_ratio(bright, (0, 0, 10, 10)) == 1.0
    assert o.highlight_ratio(dark, (0, 0, 10, 10)) == 0.0


def test_compute_oracle_attributes_keys():
    img = np.full((100, 100), 128.0, dtype=np.float32)
    out = o.compute_oracle_attributes(img, img, (10, 10, 20, 20), (10, 10, 20, 20), (100, 100))
    for k in ["scale_variation", "low_resolution", "low_contrast", "motion_blur", "out_of_frame"]:
        assert k in out and out[k] in (0, 1)
    assert "highlight_ratio" in out["_features"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest modules/maritime_analyzer/tests/test_oracles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.maritime_analyzer.oracles'`

- [ ] **Step 3: Write minimal implementation**

```python
# modules/maritime_analyzer/oracles.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest modules/maritime_analyzer/tests/test_oracles.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add modules/maritime_analyzer/oracles.py modules/maritime_analyzer/tests/test_oracles.py
git commit -m "feat(labels): add deterministic oracle attributes (blur, out-of-frame, highlight)"
```

---

## Task 3: VLM soft-output parsing & aggregation

**Files:**
- Modify: `modules/maritime_analyzer/vlm_analyzer.py` (add pure functions + `classify_soft`)
- Test: `modules/maritime_analyzer/tests/test_vlm_parsing.py`

- [ ] **Step 1: Write the failing test**

```python
# modules/maritime_analyzer/tests/test_vlm_parsing.py
from modules.maritime_analyzer.vlm_analyzer import parse_vlm_json, aggregate_passes

ATTRS = ["occlusion", "background_clutter", "specular_glare", "illumination_appearance_change"]


def test_parse_extracts_and_clamps():
    raw = 'noise {"occlusion": 0.8, "specular_glare": 1.7, "severity": 0.5} trailing'
    out = parse_vlm_json(raw, ATTRS)
    assert out["occlusion"] == 0.8
    assert out["specular_glare"] == 1.0          # clamped to [0,1]
    assert out["background_clutter"] == 0.0      # missing -> 0
    assert out["severity"] == 0.5


def test_parse_returns_none_on_garbage():
    assert parse_vlm_json("no json here", ATTRS) is None


def test_aggregate_means_and_agreement_range():
    p1 = {"occlusion": 0.8, "background_clutter": 0.0, "specular_glare": 0.0,
          "illumination_appearance_change": 0.6, "severity": 0.7}
    p2 = {"occlusion": 0.4, "background_clutter": 0.0, "specular_glare": 0.0,
          "illumination_appearance_change": 0.6, "severity": 0.5}
    agg = aggregate_passes([p1, p2, None], ATTRS)
    assert abs(agg["occlusion"] - 0.6) < 1e-6
    assert abs(agg["severity"] - 0.6) < 1e-6
    assert 0.0 <= agg["vlm_agreement"] <= 1.0


def test_aggregate_empty_is_zero():
    agg = aggregate_passes([None], ATTRS)
    assert agg["occlusion"] == 0.0 and agg["vlm_agreement"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest modules/maritime_analyzer/tests/test_vlm_parsing.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_vlm_json'`

- [ ] **Step 3: Write minimal implementation**

Add to the top imports of `modules/maritime_analyzer/vlm_analyzer.py` (it already imports `json`; add numpy + typing names):

```python
import numpy as np
from typing import Optional, List
```

Add these module-level functions to `modules/maritime_analyzer/vlm_analyzer.py` (after the imports, before the `VLMAnalyzer` class):

```python
def parse_vlm_json(raw_text: str, attr_names: List[str]) -> Optional[Dict[str, float]]:
    """Extract the JSON object from a raw VLM string and clamp each value to [0,1]."""
    try:
        s = raw_text.find('{'); e = raw_text.rfind('}')
        if s == -1 or e == -1 or s >= e:
            return None
        data = json.loads(raw_text[s:e + 1])
    except Exception:
        return None
    out: Dict[str, float] = {}
    for k in list(attr_names) + ["severity"]:
        v = data.get(k, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        out[k] = min(1.0, max(0.0, v))
    return out


def aggregate_passes(parsed_list: List[Optional[Dict[str, float]]], attr_names: List[str]) -> Dict:
    """Average parsed passes per attribute; agreement = 1 - 2*mean(std) over attributes, in [0,1]."""
    keys = list(attr_names) + ["severity"]
    valid = [p for p in parsed_list if p is not None]
    if not valid:
        return {**{k: 0.0 for k in keys}, "vlm_agreement": 0.0}
    means = {k: float(np.mean([p[k] for p in valid])) for k in keys}
    if len(valid) > 1:
        stds = [float(np.std([p[k] for p in valid])) for k in attr_names]
        agreement = float(max(0.0, 1.0 - 2.0 * float(np.mean(stds))))
    else:
        agreement = 1.0
    return {**means, "vlm_agreement": agreement}
```

Also add a `classify_soft` method to the `VLMAnalyzer` class (integration; uses the taxonomy prompt + the two pure functions above). Insert it after the existing `classify` method:

```python
    def classify_soft(self, template_crop_path: str, frame_full_path: str,
                      frame_bbox) -> Dict:
        """Soft multi-attribute VLM annotation for the subjective attributes + severity."""
        from modules.maritime_analyzer.taxonomy import build_vlm_prompt, vlm_attributes
        from PIL import Image
        frame_img = Image.open(frame_full_path).convert('RGB')
        frame_boxed = self._draw_bbox_on_image(frame_img, frame_bbox)
        template_img = Image.open(template_crop_path).convert('RGB')

        attr_names = vlm_attributes()
        question = build_vlm_prompt()
        prompt = (f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
                  f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                  f"<|vision_start|><|image_pad|><|vision_end|>"
                  f"{question}<|im_end|>\n<|im_start|>assistant\n")

        parsed = []
        for _ in range(self.config.passes):
            inputs = {"prompt": prompt,
                      "multi_modal_data": {"image": [template_img, frame_boxed]}}
            out = self.llm.generate([inputs], sampling_params=self.sampling_params)
            parsed.append(parse_vlm_json(out[0].outputs[0].text.strip(), attr_names))
        return aggregate_passes(parsed, attr_names)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest modules/maritime_analyzer/tests/test_vlm_parsing.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add modules/maritime_analyzer/vlm_analyzer.py modules/maritime_analyzer/tests/test_vlm_parsing.py
git commit -m "feat(labels): VLM soft multi-attribute parsing/aggregation + classify_soft"
```

---

## Task 4: Sharding, GPU groups, record builder

**Files:**
- Modify: `modules/maritime_analyzer/run.py` (add pure helpers + worker/orchestrator entrypoints)
- Test: `modules/maritime_analyzer/tests/test_run_helpers.py`

- [ ] **Step 1: Write the failing test**

```python
# modules/maritime_analyzer/tests/test_run_helpers.py
from modules.maritime_analyzer.run import (
    shard_sequences, plan_gpu_groups, build_worker_commands, build_record,
)


def test_shard_sequences_round_robin():
    assert shard_sequences(["a", "b", "c", "d", "e"], 2) == [["a", "c", "e"], ["b", "d"]]
    assert shard_sequences(["a", "b"], 1) == [["a", "b"]]


def test_plan_gpu_groups():
    assert plan_gpu_groups([0, 1, 2, 3], tp=2) == [[0, 1], [2, 3]]
    assert plan_gpu_groups([0, 1, 2, 3], tp=1) == [[0], [1], [2], [3]]


def test_build_worker_commands_sets_visible_devices():
    groups = [[0, 1], [2, 3]]
    cmds = build_worker_commands(num_shards=2, gpu_groups=groups, dataset="/data/MVTD/train",
                                 out_dir="data", model="Qwen/Qwen3.5-35B-A3B", tp=2, seed=42)
    assert len(cmds) == 2
    env0, argv0 = cmds[0]
    assert env0["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "--worker" in argv0 and "--shard-index" in argv0
    assert "0" in argv0 and "Qwen/Qwen3.5-35B-A3B" in argv0


def test_build_record_shape():
    oracle = {"scale_variation": 0, "low_resolution": 1, "low_contrast": 0,
              "motion_blur": 0, "out_of_frame": 0, "_features": {"highlight_ratio": 0.0}}
    vlm = {"occlusion": 0.8, "background_clutter": 0.0, "specular_glare": 0.0,
           "illumination_appearance_change": 0.6, "severity": 0.7, "vlm_agreement": 0.9}
    rec = build_record("1-Boat", 12, "00000012.jpg", [1, 2, 3, 4], [5, 6, 7, 8],
                       oracle, vlm, {"dataset_path": "/data/MVTD/train"})
    assert rec["schema_version"] == 2
    assert rec["attributes"]["low_resolution"] == {"prob": 1.0, "source": "oracle"}
    assert rec["attributes"]["occlusion"] == {"prob": 0.8, "source": "vlm"}
    assert rec["severity"] == 0.7 and rec["vlm_agreement"] == 0.9
    assert len(rec["attributes"]) == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest modules/maritime_analyzer/tests/test_run_helpers.py -v`
Expected: FAIL with `ImportError: cannot import name 'shard_sequences'`

- [ ] **Step 3: Write minimal implementation**

Add to the imports of `modules/maritime_analyzer/run.py`:

```python
import os
import sys
from modules.maritime_analyzer.taxonomy import oracle_attributes, vlm_attributes, SCHEMA_VERSION
```

Add these helpers to `modules/maritime_analyzer/run.py` (above `def main()`):

```python
def shard_sequences(seq_names, num_shards):
    num_shards = max(1, int(num_shards))
    shards = [[] for _ in range(num_shards)]
    for i, s in enumerate(seq_names):
        shards[i % num_shards].append(s)
    return shards


def plan_gpu_groups(gpus, tp):
    tp = max(1, int(tp))
    return [list(gpus[i:i + tp]) for i in range(0, len(gpus), tp)]


def build_worker_commands(num_shards, gpu_groups, dataset, out_dir, model, tp, seed):
    cmds = []
    for shard_index, group in enumerate(gpu_groups[:num_shards]):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in group)
        argv = [sys.executable, "-m", "modules.maritime_analyzer.run",
                "--worker",
                "--shard-index", str(shard_index),
                "--num-shards", str(num_shards),
                "--dataset", dataset,
                "--out-dir", out_dir,
                "--model", model,
                "--tp", str(tp),
                "--seed", str(seed)]
        cmds.append((env, argv))
    return cmds


def build_record(seq_name, frame_id, frame_file, template_bbox, gt_bbox,
                 oracle_attrs, vlm_result, meta):
    attributes = {}
    for name in oracle_attributes():
        attributes[name] = {"prob": float(oracle_attrs[name]), "source": "oracle"}
    for name in vlm_attributes():
        attributes[name] = {"prob": float(vlm_result.get(name, 0.0)), "source": "vlm"}
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence_name": seq_name,
        "frame_id": frame_id,
        "frame_file": frame_file,
        "template_bbox": list(map(float, template_bbox)),
        "ground_truth_bbox": list(map(float, gt_bbox)),
        "attributes": attributes,
        "severity": float(vlm_result.get("severity", 0.0)),
        "vlm_agreement": float(vlm_result.get("vlm_agreement", 0.0)),
        "oracle_features": oracle_attrs.get("_features", {}),
        "dataset_path": meta.get("dataset_path", ""),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest modules/maritime_analyzer/tests/test_run_helpers.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add modules/maritime_analyzer/run.py modules/maritime_analyzer/tests/test_run_helpers.py
git commit -m "feat(labels): sharding, gpu-group planning, and v2 record builder"
```

---

## Task 5: Wire the per-shard worker + multi-GPU orchestrator into `run.py`

**Files:**
- Modify: `modules/maritime_analyzer/run.py` (extend `main()` with `--worker` mode and an orchestrator that launches `build_worker_commands`)

> This task is integration glue around already-tested helpers; it is validated by the smoke run in Task 7, not a unit test.

- [ ] **Step 1: Add the per-shard worker + orchestrator to `main()`**

Replace the argument parser + dispatch in `modules/maritime_analyzer/run.py`'s `main()` with:

```python
def _process_one_sequence_v2(seq_dir, vlm, seq_out_dir, meta):
    from PIL import Image
    frames = sorted([p for p in seq_dir.glob('*.jpg')])
    gts = read_groundtruth_txt(seq_dir / 'groundtruth.txt')
    assert len(gts) == len(frames), f"GT/frames mismatch in {seq_dir}"
    seq_out_dir.mkdir(parents=True, exist_ok=True)
    seq_jsonl = seq_out_dir / f"{seq_dir.name}.jsonl"
    template_img, template_bbox = frames[0], gts[0]
    template_crop_path = ensure_template_crop(template_img, template_bbox, seq_out_dir)
    processed = parse_processed_frame_ids(seq_jsonl)

    for frame_id, (frame_path, bbox) in enumerate(zip(frames, gts), start=1):
        if frame_id in processed:
            continue
        frame_pil = Image.open(frame_path).convert('RGB')
        oracle = compute_oracle_attributes(
            Image.open(template_img).convert('RGB'), frame_pil,
            template_bbox, bbox, frame_pil.size)
        vlm_result = vlm.classify_soft(str(template_crop_path), str(frame_path), bbox)
        rec = build_record(seq_dir.name, frame_id, frame_path.name,
                           template_bbox, bbox, oracle, vlm_result, meta)
        with open(seq_jsonl, 'a') as f:
            f.write(json.dumps(rec) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--model', default='Qwen/Qwen3.5-35B-A3B')
    ap.add_argument('--gpus', default='0', help='comma list, e.g. 0,1,2,3 (orchestrator mode)')
    ap.add_argument('--tp', type=int, default=1, help='tensor-parallel size per replica')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--worker', action='store_true', help='internal: run one shard')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    sequences = sorted([p for p in dataset_path.iterdir()
                        if p.is_dir() and (p / 'groundtruth.txt').exists()])
    split_name = dataset_path.name
    out_root = Path(args.out_dir) / f"{split_name}_maritime_env_clf_annts"
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {"dataset_path": str(dataset_path), "model_name": args.model, "classes": CLASSES}

    if args.worker:
        from modules.maritime_analyzer.vlm_analyzer import VLMAnalyzer, VLMConfig
        seq_names = [p.name for p in sequences]
        my_seqs = shard_sequences(seq_names, args.num_shards)[args.shard_index]
        vlm = VLMAnalyzer(VLMConfig(model_name=args.model))
        for name in my_seqs:
            seq = dataset_path / name
            try:
                _process_one_sequence_v2(seq, vlm, out_root / name, meta)
            except Exception as e:
                print(f"  !! error processing {name}: {e}")
        return

    # Orchestrator: split GPUs into replicas (TP groups), launch one worker per replica
    import subprocess
    gpus = [int(g) for g in args.gpus.split(',') if g != '']
    groups = plan_gpu_groups(gpus, args.tp)
    num_shards = len(groups)
    cmds = build_worker_commands(num_shards, groups, args.dataset, args.out_dir,
                                 args.model, args.tp, args.seed)
    procs = [subprocess.Popen(argv, env=env) for env, argv in cmds]
    for p in procs:
        p.wait()
    print(f"All workers finished. Annotations under: {out_root}")
```

> Keep the existing helpers (`read_groundtruth_txt`, `parse_processed_frame_ids`, `ensure_template_crop`, `CLASSES`) — they are reused. Add `from modules.maritime_analyzer.oracles import compute_oracle_attributes` to the imports.

- [ ] **Step 2: Verify the module imports cleanly (no syntax/import errors)**

Run: `python -c "import modules.maritime_analyzer.run as r; print(hasattr(r,'build_record'), hasattr(r,'main'))"`
Expected: `True True`

- [ ] **Step 3: Re-run the helper tests (no regression)**

Run: `python -m pytest modules/maritime_analyzer/tests/test_run_helpers.py -v`
Expected: PASS (4 passed)

- [ ] **Step 4: Commit**

```bash
git add modules/maritime_analyzer/run.py
git commit -m "feat(labels): multi-GPU worker/orchestrator writing v2 JSONL"
```

---

## Task 6: Partial-visibility audit (old VLM labels vs oracle)

**Files:**
- Create: `modules/maritime_analyzer/validation/__init__.py` (empty)
- Create: `modules/maritime_analyzer/validation/audit_partial_visibility.py`
- Test: `modules/maritime_analyzer/tests/test_audit.py`

- [ ] **Step 1: Write the failing test**

```python
# modules/maritime_analyzer/tests/test_audit.py
from modules.maritime_analyzer.validation.audit_partial_visibility import compare_partial_visibility


def test_compare_counts_and_disagreement():
    # record: old VLM said partial(flag), bbox, seq, frame_file
    records = [
        {"sequence_name": "s", "frame_file": "1.jpg", "vlm_partial_flag": 1,
         "ground_truth_bbox": [10, 10, 20, 20]},   # oracle: inside -> disagree
        {"sequence_name": "s", "frame_file": "2.jpg", "vlm_partial_flag": 1,
         "ground_truth_bbox": [0, 10, 20, 20]},     # oracle: edge -> agree
        {"sequence_name": "s", "frame_file": "3.jpg", "vlm_partial_flag": 0,
         "ground_truth_bbox": [40, 40, 10, 10]},    # oracle: inside -> agree
    ]
    def size_lookup(seq, frame_file):
        return (100, 100)
    stats = compare_partial_visibility(records, size_lookup, margin=1)
    assert stats["total"] == 3
    assert stats["disagree"] == 1
    assert abs(stats["disagreement_rate"] - 1 / 3) < 1e-6
    assert stats["vlm_partial_oracle_inside"] == 1   # VLM over-predicts partial
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest modules/maritime_analyzer/tests/test_audit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.maritime_analyzer.validation'`

- [ ] **Step 3: Write minimal implementation**

```python
# modules/maritime_analyzer/validation/__init__.py
```

```python
# modules/maritime_analyzer/validation/audit_partial_visibility.py
"""Audit the OLD VLM partial-visibility labels against the deterministic out-of-frame oracle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

from modules.maritime_analyzer.oracles import is_out_of_frame


def compare_partial_visibility(records: List[Dict],
                               size_lookup: Callable[[str, str], tuple],
                               margin: int = 1) -> Dict:
    total = agree = disagree = 0
    vlm_partial_oracle_inside = oracle_partial_vlm_no = 0
    for r in records:
        vlm_flag = int(r["vlm_partial_flag"])
        W, H = size_lookup(r["sequence_name"], r["frame_file"])
        oracle_flag = int(is_out_of_frame(tuple(r["ground_truth_bbox"]), (W, H), margin=margin))
        total += 1
        if vlm_flag == oracle_flag:
            agree += 1
        else:
            disagree += 1
            if vlm_flag == 1 and oracle_flag == 0:
                vlm_partial_oracle_inside += 1
            elif vlm_flag == 0 and oracle_flag == 1:
                oracle_partial_vlm_no += 1
    return {
        "total": total,
        "agree": agree,
        "disagree": disagree,
        "disagreement_rate": (disagree / total) if total else 0.0,
        "vlm_partial_oracle_inside": vlm_partial_oracle_inside,
        "oracle_partial_vlm_no": oracle_partial_vlm_no,
    }


def _load_old_records(ann_dir: Path) -> List[Dict]:
    out = []
    for jf in ann_dir.glob("*/*.jsonl"):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pv = d.get("vlm_response", {}).get("partial_visibility", {})
                out.append({
                    "sequence_name": d.get("sequence_name", jf.parent.name),
                    "frame_file": d.get("frame_file", ""),
                    "vlm_partial_flag": int(pv.get("flag", 0)) if isinstance(pv, dict) else 0,
                    "ground_truth_bbox": d.get("ground_truth_bbox", [0, 0, 0, 0]),
                    "dataset_path": d.get("dataset_path", ""),
                })
    return out


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True, help="old-format annotation root (…_maritime_env_clf_annts)")
    ap.add_argument("--margin", type=int, default=1)
    args = ap.parse_args()
    records = _load_old_records(Path(args.ann_dir))

    cache: Dict = {}
    def size_lookup(seq, frame_file):
        ds = next((r["dataset_path"] for r in records
                   if r["sequence_name"] == seq and r["frame_file"] == frame_file), "")
        key = (ds, seq, frame_file)
        if key not in cache:
            cache[key] = Image.open(Path(ds) / seq / frame_file).size
        return cache[key]

    stats = compare_partial_visibility(records, size_lookup, margin=args.margin)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest modules/maritime_analyzer/tests/test_audit.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add modules/maritime_analyzer/validation/ modules/maritime_analyzer/tests/test_audit.py
git commit -m "feat(labels): partial-visibility audit vs out-of-frame oracle"
```

---

## Task 7: End-to-end smoke validation (manual, single GPU)

**Files:** none (validation only)

> Confirms the full generation path runs and writes valid v2 JSONL on a tiny subset before the full multi-GPU run.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest modules/maritime_analyzer/tests/ -v`
Expected: PASS (all tests from Tasks 1–6, ~18 passed)

- [ ] **Step 2: Generate labels for 1–2 sequences on a single GPU**

Make a tiny dataset copy with 1–2 sequences (each containing `groundtruth.txt` + `*.jpg`), then:

Run:
```bash
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python -m modules.maritime_analyzer.run \
    --dataset /path/to/tiny_MVTD/train --out-dir /tmp/labels_smoke \
    --gpus 0 --tp 1 --seed 42
```
Expected: one `*.jsonl` per sequence under `/tmp/labels_smoke/train_maritime_env_clf_annts/<seq>/`.

- [ ] **Step 3: Validate the output schema**

Run:
```bash
python -c "import json,glob; r=[json.loads(l) for f in glob.glob('/tmp/labels_smoke/**/*.jsonl',recursive=True) for l in open(f)]; \
assert r and r[0]['schema_version']==2 and len(r[0]['attributes'])==9 and 'severity' in r[0]; \
print('OK', len(r), 'records', r[0]['attributes'].keys())"
```
Expected: `OK <N> records dict_keys([... 9 attributes ...])`

- [ ] **Step 4: (Optional) Run the partial-visibility audit on existing OLD labels**

Run:
```bash
python -m modules.maritime_analyzer.validation.audit_partial_visibility \
    --ann-dir data/train_cls/train_maritime_env_clf_annts
```
Expected: a JSON summary with `disagreement_rate` and `vlm_partial_oracle_inside` — the first concrete result for the report/slides.

- [ ] **Step 5: Commit any fixes found during smoke (if needed)**

```bash
git add -A
git commit -m "fix(labels): smoke-run corrections for v2 generation"
```

---

## Self-Review

**1. Spec coverage (Plan A scope):**
- Taxonomy redesign (multi-label + severity + Normal-as-implicit) → Task 1 ✅
- Oracle attributes incl. new blur / out-of-frame / glare-highlight → Task 2 ✅
- VLM soft per-attribute probabilities + severity + agreement → Task 3 ✅
- New v2 JSONL schema (attributes{prob,source}, severity, agreement, features) → Task 4 (`build_record`) ✅
- Multi-GPU sharding (TP groups × DP replicas) → Tasks 4–5 ✅
- Partial-visibility audit (first concrete result) → Task 6 ✅
- End-to-end validation → Task 7 ✅
- *Deferred to later plans (correctly):* human gold set & κ, 2nd-VLM cross-check, label loader for training (Plan B), full per-condition tracking eval.

**2. Placeholder scan:** No TBD/TODO; every code step contains complete code; integration steps (Tasks 5, 7) use concrete commands with expected output. ✅

**3. Type/name consistency:** `compute_oracle_attributes`, `parse_vlm_json`, `aggregate_passes`, `shard_sequences`, `plan_gpu_groups`, `build_worker_commands`, `build_record`, `compare_partial_visibility`, `attribute_names/oracle_attributes/vlm_attributes`, `SCHEMA_VERSION` — names match across the file-structure map and all tasks. The worker calls `classify_soft` (Task 3) and `compute_oracle_attributes` (Task 2), both defined. ✅

**Known follow-ups (not blockers):** oracle thresholds (`blur_threshold`, `edge_margin`, etc.) are defaulted and config-overridable — tune against the human gold set in Plan B; occlusion remains VLM-only unless GOT-10k cover/absence labels are present in MVTD.
