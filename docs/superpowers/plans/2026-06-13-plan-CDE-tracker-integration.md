# Plans C, D, E — Tracker Integration (Handoff Plan)

> ## ⚠️ INSTRUCTIONS FOR THE IMPLEMENTING AGENT (read first)
> - **IMPLEMENT ONLY. DO NOT `git commit`, `git add`, or `git push`.** Make the
>   changes in the working tree, run the validation, and report what you did.
>   The human reviews and commits everything themselves.
> - These are integrations into **existing, complex trackers**. They **cannot be
>   fully validated without a real GPU + data training run**. Work in **small steps**
>   and ask the human to run a **1–2 epoch smoke-train** after each step before moving on.
> - **Read the actual tracker code before editing** — do not trust this doc's line
>   numbers; trust the file responsibilities. Adapt to what you find.

---

## 0. Context & current state

Project: multi-task **condition-aware maritime tracking** on MVTD. The condition head is
not just a parallel regularizer — its prediction also (optionally) **modulates** the
tracking features (FiLM). Goal: improve AO/SR while producing a useful per-frame
condition/reliability signal. Target venue: project report → Q1 ocean-applications journal.

Design spec: `docs/superpowers/specs/2026-06-09-condition-aware-maritime-tracking-design.md`
Plan A (labels) doc: `docs/superpowers/plans/2026-06-09-label-generation-pipeline.md`

**Already done (on `main`, 40 passing tests under `modules/`):**
- **Plan A — label generation** (`modules/maritime_analyzer/`): taxonomy, oracles, soft
  multi-attribute VLM annotator, multi-GPU `run.py` orchestrator. Produces v2 JSONL at
  `data/<split>_maritime_cond_v2/<seq>/<seq>.jsonl`. Working end-to-end with
  Qwen3-VL-32B via vLLM (TP across GPUs). Run via `scripts/gen_labels.sh`.
- **Plan B — shared toolkit** (`modules/condition_mtl/`, tracker-agnostic, unit-tested):
  `ConditionHead`, `FiLM`, `losses`, `labels`, `eval_per_condition`, `seeding`.
- **Plan E (partial)** — `scripts/gen_labels.sh` (ready); `scripts/train.sh`,
  `scripts/test_mvtd.sh` (thin wrappers over each tracker's `tracking/train.py` /
  `tracking/test.py`; finalize once C/D land).

**Remaining: Plans C (HIPTrack), D (SimTrack), E (finalize wrappers).**

---

## 1. Shared toolkit API (in `modules/condition_mtl/`, already implemented & tested)

```python
from modules.condition_mtl.cls_head import ConditionHead
#   ConditionHead(in_dim, hidden_dim=512, num_outputs=9, predict_severity=True, dropout=0.1)
#   forward(feat[B,in_dim]) -> {"logits": [B,num_outputs], "severity": [B] or None}

from modules.condition_mtl.film import FiLM
#   FiLM(cond_dim, feat_channels, hidden_dim=128)   # identity-initialized
#   forward(feat[B,C,H,W] or [B,C], cond[B,cond_dim]) -> modulated feat (same shape)

from modules.condition_mtl.losses import condition_loss
#   condition_loss(outputs, targets, mode, severity_weight=1.0, weight=None, pos_weight=None)
#   mode="hard":       outputs["logits"][B,C], targets["labels"][B]            (CE; C = 9 attrs + 1 normal = 10)
#   mode="soft":       outputs["logits"][B,C], targets["dist"][B,C]            (KL)
#   mode="multilabel": outputs["logits"][B,K], targets["attr_probs"][B,K],     (BCE + severity L1)
#                      optional targets["severity"][B], targets["mask"][B,K], outputs["severity"][B]

from modules.condition_mtl.labels import (
    record_attr_probs, record_severity, record_hard_label, load_sequence_records)
#   record_attr_probs(record, attr_names=None) -> list[float] len 9 (taxonomy order; v2 & v1)
#   record_severity(record) -> float
#   record_hard_label(record, attr_names=None, normal_threshold=0.5) -> int in [0..9] (9 = normal)
#   load_sequence_records(jsonl_path) -> list[dict]

from modules.condition_mtl.eval_per_condition import per_condition_metrics
#   per_condition_metrics(frames, attr_names, present_threshold=0.5, success_threshold=0.5)
#   frames: list of {"iou": float, "attr_probs": [K]} -> {attr:{n,AO,SR}, "overall":{...}}

from modules.condition_mtl.seeding import set_global_seed   # set_global_seed(seed=42)

from modules.maritime_analyzer.taxonomy import (
    attribute_names, oracle_attributes, vlm_attributes, SEVERITY_KEY, SCHEMA_VERSION)
#   attribute_names() -> 9 names, taxonomy order:
#     [scale_variation, low_resolution, low_contrast, motion_blur, out_of_frame,
#      occlusion, background_clutter, specular_glare, illumination_appearance_change]
```

### v2 label record schema (one JSON object per line)
```json
{
  "schema_version": 2,
  "sequence_name": "1-Boat", "frame_id": 12, "frame_file": "00000012.jpg",
  "template_bbox": [x,y,w,h], "ground_truth_bbox": [x,y,w,h],
  "attributes": { "<name>": {"prob": 0.0, "source": "oracle"|"vlm"}, ... 9 entries ... },
  "severity": 0.0, "vlm_agreement": 0.0, "oracle_features": {...}, "dataset_path": "..."
}
```
`frame_id` is 1-based; labels may be **strided** (not every frame present) — match by `frame_id`.

### The 6 run modes (select via a config field, e.g. `TRAIN.MTL_MODE`)
| mode | box loss | condition head | label type | FiLM | head at inference |
|---|---|---|---|---|---|
| `pretrained` | — (eval only) | — | — | — | — |
| `finetune_bbox` | ✅ | — | — | — | — |
| `mtl_hard` | ✅ | ✅ | hard CE (10-way) | — | off |
| `mtl_soft` | ✅ | ✅ | soft KL | — | off |
| `mtl_multilabel` | ✅ | ✅ | multilabel BCE + severity | — | off |
| `mtl_film` | ✅ | ✅ | multilabel BCE + severity | ✅ | **on** (FiLM needs the prediction) |

---

## 2. Critical gotchas (these are why this must be done carefully + validated)

1. **Exec-path / import problem.** Trackers run from `models/<tracker>/` with their *own*
   `lib` on `sys.path`; the repo-root `modules/` package is **not importable** there.
   Fix: at the top of each tracker integration file that imports the toolkit, add a shim:
   ```python
   import os, sys
   _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), <up to repo root>))
   if _REPO_ROOT not in sys.path: sys.path.insert(0, _REPO_ROOT)
   from modules.condition_mtl... import ...
   ```
   (Per the design spec, duplicating the tiny logic per-tracker is an allowed alternative —
   but the shim keeps a single source of truth. Verify imports actually resolve at runtime.)
2. **Sampler must carry per-frame soft targets.** The existing HIPTrack cls path carries a
   **scalar** `cls_label` per frame (see `lib/train/data/sampler.py` + `got10k_cls.py`). For
   `mtl_soft`/`mtl_multilabel`/`mtl_film` it must carry a **vector** `attr_probs[K]` (+ a
   `severity` scalar, + optional valid `mask[K]`). The sampler stacks frames with
   `stack_dim=1` — confirm the new tensors stack/collate correctly. **This is the riskiest
   change; smoke-train immediately after it.**
3. **FiLM at inference.** HIPTrack's cls head is currently train-only (`use_cls_head and
   self.training`). For `mtl_film`, the condition head + FiLM must also run at **inference**,
   because FiLM modulates the features the box head consumes. Other modes keep the head off
   at test (zero inference cost).
4. **DDP.** `find_unused_parameters=True` is already set in HIPTrack's train script — good,
   since some modes leave the cls/FiLM params unused. Keep it.
5. **Label dir.** Point the cls annotation dir to the **new** labels:
   `data/<split>_maritime_cond_v2` (NOT the old `*_maritime_env_clf_annts`). Load via
   `modules.condition_mtl.labels`, not the old `_determine_class_from_annotation`.
6. **Identity-init FiLM is safe by design** — at init γ=β=0 so it is the identity; it can
   only deviate if it reduces loss. Keep the residual/identity init.
7. **Untestable here.** Validation = a real smoke-train (below). Do not assume it works.

---

## 3. Plan C — HIPTrack integration (`models/HIPTrack/`)

HIPTrack **already has a working hard-label, train-only MTL path** (`hiptrack_cls`). This is
an **upgrade** to consume v2 soft labels, add FiLM, and switch run modes. Existing files:
`lib/models/hiptrack/hiptrack_cls.py`, `lib/models/layers/cls_head.py`,
`lib/utils/cls_loss.py`, `lib/train/actors/hiptrack_cls.py`,
`lib/train/dataset/got10k_cls.py`, `lib/train/data/sampler.py`,
`lib/train/train_script_cls.py`, `lib/config/hiptrack/config_cls.py`,
`lib/test/tracker/hiptrack_cls.py`, `experiments/hiptrack/hiptrack_cls.yaml`.

Do these **one at a time, smoke-train between each**:

- **C1 — v2 labels in the dataset.** In `got10k_cls.py`, add the exec-path shim and use
  `modules.condition_mtl.labels` to read v2 records from `CLS_ANN_DIR`
  (`data/<split>_maritime_cond_v2`). Return, per frame: `attr_probs[9]`, `severity`, and a
  `hard_label` (via `record_hard_label`). Match by 1-based `frame_id`; if a frame has no
  label (strided/missing), return an ignore marker (`-100` for hard label; an all-zero
  `mask` for multilabel) so it doesn't contribute to loss. Keep v1 back-compat working.
  *Validate:* a tiny script that loads a few sequences and prints shapes.
- **C2 — sampler carries the vectors.** Extend `lib/train/data/sampler.py` (and the
  processing pipeline if needed) to pass `attr_probs`, `severity`, `mask`, `hard_label`
  through to the actor for each search frame, stacking consistently with `stack_dim=1`.
  *Validate:* one forward pass through the dataloader; inspect a batch dict.
- **C3 — model: ConditionHead + FiLM + run mode.** In `hiptrack_cls.py`, replace the old
  `cls_head` with `condition_mtl.ConditionHead` (num_outputs = 9 for multilabel/film, or 10
  for hard/soft; `predict_severity=True` for multilabel/film). Tap the pooled `fused_search`
  (as today). For `mtl_film`: feed the condition descriptor (`logits`/probs + severity) into
  `condition_mtl.FiLM` to modulate `fused_search` **before** the box head, with a residual
  gate; run this at inference too. Gate behavior on a `mode` read from cfg.
- **C4 — actor losses.** In `hiptrack_cls.py` actor, compute the condition loss via
  `condition_mtl.losses.condition_loss(outputs, targets, mode=cfg mode, ...)` and add it to
  the tracking loss with `cfg.TRAIN.CLS_WEIGHT`. Honor the ignore mask. Log per-mode metrics.
- **C5 — configs + test.** Add `experiments/hiptrack/` YAMLs for the 6 modes (set
  `TRAIN.MTL_MODE`, `CLS_WEIGHT`, `CLS_ANN_DIR=../../data/train_maritime_cond_v2`,
  `MODEL.CLS_HEAD.*`). In `lib/test/tracker/hiptrack_cls.py`, enable the head at inference
  **only** for `mtl_film`. Optionally emit the per-frame condition signal for the
  reliability/eval story.

Per-condition evaluation: after a test run, group per-frame IoU by attribute (oracle + VLM
from the v2 labels) and report with `condition_mtl.eval_per_condition.per_condition_metrics`.

---

## 4. Plan D — SimTrack integration (`models/SimTrack*/`)

Mirror the **proven C pattern** once C is validated. **Key difference (per spec):** the
current SimTrack cls path (`models/SimTrackMod/simtrack_with_classification.py` +
`finetune.py`) trains on a **weak single-frame proxy** (full-frame resize, approximate GIoU).
**Port the condition head + FiLM onto SimTrack's proper `ltr` training pipeline**
(`models/SimTrackMod/lib/train/run_training.py`, `base_functions.py`, `train_script.py`,
`trainers/ltr_trainer.py`, the SimTrack actor) — real search-region cropping, DDP, seeds.
SimTrack is one-shot (no online memory), so only the **FiLM feature-modulation** part of the
mechanism applies (no memory gate). Reuse the same `condition_mtl` toolkit, the same 6 modes,
the same v2 labels. Smoke-train between steps as in C.

---

## 5. Plan E — finalize the wrappers

Once C/D expose the run-mode configs, update `scripts/train.sh` / `scripts/test_mvtd.sh` so a
mode is a one-flag choice (e.g. `train.sh HIPTrack hiptrack mtl_film`). Keep `gen_labels.sh`
as is.

---

## 6. Validation (the human runs these; you interpret the output)

Prereqs: MVTD-as-GOT10k data + HIPTrack pretrained weights wired in
`models/HIPTrack/lib/train/admin/` env; a small label set at
`data/train_maritime_cond_v2/` (generate with `scripts/gen_labels.sh train ... --max-frames 30`).

Smoke-train (tiny: set a small `DATA.TRAIN.SAMPLE_PER_EPOCH` and 1–2 epochs in the mode YAML):
```
scripts/train.sh HIPTrack hiptrack <mode_config> 1 ./output
```
Check: it starts, the condition loss is finite and decreasing, IoU is sane, and (for
`mtl_film`) the head runs at inference without shape errors. Then test:
```
scripts/test_mvtd.sh HIPTrack hiptrack_<mode> <mode_config> 4 got10k_val
```

## 7. Definition of done
- All 6 modes train on HIPTrack and SimTrack without errors; `mtl_film` runs end-to-end
  including inference.
- Per-condition AO/SR reported on the test split.
- The full ablation runnable: `pretrained → finetune_bbox → mtl_hard → mtl_soft →
  mtl_multilabel → mtl_film`, multiple seeds.
- **You did NOT commit.** Summarize changes for the human to review and commit.
