# Condition-Aware Maritime Tracking — Design Spec

- **Date:** 2026-06-09
- **Status:** Draft for review
- **Models in scope:** HIPTrack, SimTrack
- **Dataset:** MVTD (182 sequences, ~150k frames, 4 vessel classes), consumed in GOT-10k format
- **Primary goal:** improve tracking accuracy (AO / SR / precision) on MVTD; secondary: a useful per-frame condition/reliability signal
- **Venue target:** project report now → potential Q1 ocean-applications journal

---

## 1. Motivation & framing

Standard trackers apply the *same* fixed feature processing to every frame, but maritime
conditions vary drastically frame-to-frame (specular glare, occlusion, wake/foam clutter,
low contrast, scale change). A representation tuned for clear water degrades under glare.

We make the tracker **condition-aware**: an auxiliary head predicts the maritime condition,
and that prediction **modulates the tracking features** (FiLM). The condition task therefore
plays a dual role — a **regularizer** (via its own loss) and a **control signal** (via
modulation). This is multi-task learning **with cross-task feature modulation**, a richer form
than parallel hard-parameter-sharing MTL.

**Core hypothesis:** the optimal tracking representation is condition-dependent; a single
static feature map is suboptimal across the maritime condition distribution.

---

## 2. Three-part plan

- **Part 1 — Label refinement & validation** *(prerequisite + addresses label-correctness concern)*:
  redesign the taxonomy (maritime-grounded, multi-label, + severity, + Normal), regenerate with a
  stronger LVLM, anchor objective conditions to deterministic oracles, and **validate** label quality.
- **Part 2 — Condition-aware feature modulation (FiLM)** *(the headline mechanism)*:
  identity-initialized FiLM driven by the predicted condition; universal across both trackers.
- **Part 3 — Dense SAM-distilled supervision** *(future work / journal extension)*:
  auxiliary target-mask head supervised by SAM2 pseudo-masks. **Out of scope for the report build.**

Unifying theme: *foundation-model-distilled (VLM + SAM) condition-aware maritime tracking.*

---

## 3. Label schema (Part 1)

Move from **10-way single-label softmax** → **multi-label attributes (sigmoid) + severity + Normal**,
each attribute sourced from the most trustworthy signal.

| Attribute | Source | Type |
|---|---|---|
| scale_variation | **oracle** (bbox area ratio) | binary (crisp) |
| low_resolution | **oracle** (target size) | binary (crisp) |
| low_contrast | **oracle** (RMS ring contrast) | binary (crisp) |
| motion_blur | **oracle** (Laplacian variance) | binary (crisp) |
| out_of_frame (partial) | **oracle** (bbox vs border) | binary (crisp) |
| occlusion (object-blocked) | VLM | soft [0,1] |
| background_clutter | VLM | soft [0,1] |
| specular_glare | VLM (+ highlight oracle) | soft [0,1] |
| illumination/appearance_change | VLM (merge of old Illu + Variance) | soft [0,1] |
| **severity** (overall difficulty) | VLM / derived | continuous [0,1] |
| **Normal** | derived (all attributes ≈ 0) | implicit |

- Per-frame label = vector of independent attributes (targets need **not** sum to 1) + severity scalar.
- Oracle attributes are crisp 0/1; VLM attributes are soft confidences; low VLM-agreement → confidence-weighted / abstain (`-100` ignore).
- The exact thresholds for oracles are config-driven (carried over / tuned from `deterministic_utils.py`).

---

## 4. LVLM data generation (Part 1)

Extend `modules/maritime_analyzer/`:
- **`taxonomy.py`** — the schema above (attribute names, ids, prompt definitions, few-shot exemplars).
- **`oracles.py`** — deterministic attribute computation (extends existing scale/low-res/low-contrast with blur, out-of-frame, glare highlight).
- **`vlm_analyzer.py`** — emit **per-attribute soft probabilities** for the subjective attributes (keep the existing multi-pass averaging; expose pass-disagreement as a confidence) + severity.
- **`run.py`** — **multi-GPU sharding**: split the sequence list into shards, one worker per **replica** (a replica may span >1 GPU via tensor-parallel for large models), each pinned via `CUDA_VISIBLE_DEVICES`; merge JSONL outputs. Resumable (existing per-sequence resume preserved).
- **Model:** config field `model_name`, default **`Qwen/Qwen3.5-35B-A3B`** (vision-language MoE, 35B total / 3B active; verified vLLM + SGLang support). Overridable; engine selectable (vLLM default; SGLang optional).
  - **Memory:** ~70 GB resident at bf16 (all experts loaded) → exceeds one 48 GB GPU. On 4×48 GB use **TP=2 × DP=2** (two replicas, each tensor-parallel over 2 GPUs) at bf16, **or** an FP8/AWQ build for one replica per GPU → full 4-way data-parallel. `--tp` and `--dp` exposed in `gen_labels.sh`.

Output JSONL (per frame) gains: `attributes: {name: {prob, source}}`, `severity`, `vlm_agreement`.
Backward-compatible reader provided for the old format.

---

## 5. Label correctness / validation protocol (Part 1)

Treat the VLM as a **noisy annotator** — measure, anchor, make training robust:
1. **Programmatic oracles** label the objective attributes (no VLM trust needed).
2. **Partial-visibility audit (first concrete result):** compare the *old* VLM partial-visibility
   calls vs the bbox-edge oracle → report disagreement rate.
3. **Human gold set** (~500–1000 stratified frames): VLM-vs-human accuracy / F1 / confusion +
   Cohen's κ, with human–human agreement as the ceiling.
4. **Self-consistency + 2nd VLM** cross-check → per-attribute confidence.
5. **Noise-robust training** (soft targets, confidence weighting) + **label-quality ablation**
   (raw vs cleaned vs human-corrected).

Deliverable: a "Label Quality & Validation" section (agreement table + confusion matrix).

---

## 6. Method (Part 2)

### 6.1 Condition head (shared)
`modules/condition_mtl/cls_head.py`: pooled shared feature `[B, C]` → MLP → per-attribute logits
(sigmoid) + severity (1 scalar). Multi-label.

### 6.2 FiLM modulation (shared)
`modules/condition_mtl/film.py`: condition descriptor `[p ; s]` → small MLP → per-channel
`(γ, β)` → `F' = F · (1 + γ) + β` on the shared feature before the box head.
- **Identity-initialized** (final layer weights → 0 ⇒ γ=β=0 ⇒ F'=F): cannot degrade baseline at init.
- `γ` bounded (tanh); residual form; **warm-started** from label-derived condition for early epochs.

### 6.3 Losses (shared)
`modules/condition_mtl/losses.py`: selectable per run-mode —
hard CE | soft KL-distillation | **multi-label BCE (soft)** + severity (L1/MSE),
with `-100` ignore and optional class/confidence weighting.

### 6.4 Per-tracker wiring (thin, in each model folder)
- **HIPTrack:** condition head taps `fused_search` (pooled); FiLM modulates `fused_search` before the
  CENTER box head. **Condition head must be enabled at inference** (FiLM needs it) — change from the
  current train-only behavior.
- **SimTrack:** **port the condition head + FiLM onto SimTrack's proper `ltr` pipeline**
  (`lib/train/run_training.py`, real search-region cropping, DDP, seeds) — *not* the current
  throwaway `finetune.py`. Condition head taps the bottleneck/CLS feature; FiLM modulates the
  search feature before the corner head.

---

## 7. Run modes (experiment matrix)

One `--mode` flag (config preset), reproducible:

| Mode | Box loss | Condition head | Label type | FiLM |
|---|---|---|---|---|
| `pretrained` | — (eval only) | — | — | — |
| `finetune_bbox` | ✅ | — | — | — |
| `mtl_hard` | ✅ | ✅ | hard single-label CE | — |
| `mtl_soft` | ✅ | ✅ | soft distillation | — |
| `mtl_multilabel` | ✅ | ✅ | multi-label BCE + severity | — |
| `mtl_film` | ✅ | ✅ | multi-label BCE + severity | ✅ |

Applies to **both** HIPTrack and SimTrack. (FiLM is universal; the HIPTrack-specific memory gate is future work.)

---

## 8. Folder layout (respects the `models/` + `modules/` rule)

```
modules/
  maritime_analyzer/   taxonomy.py, oracles.py, vlm_analyzer.py(+soft), run.py(+multi-GPU)
  condition_mtl/       cls_head.py, film.py, losses.py, labels.py, eval_per_condition.py, seeding.py
models/HIPTrack/...    thin integration (import condition_mtl, wire head+FiLM, enable head at test)
models/SimTrack*/...   thin integration on the proper ltr pipeline
scripts/               gen_labels.sh, train.sh, test_mvtd.sh   (top-level, bash)
docs/superpowers/specs/2026-06-09-condition-aware-maritime-tracking-design.md
```

`models/` structure is **not** restructured; only thin per-tracker integration files are added/edited.
All tracker-agnostic logic lives in `modules/`.

---

## 9. Reproducibility

- Single `seed` flowed through every entry point (LVLM sharding, training, testing).
- Reuse existing `init_seeds` (random/numpy/torch/cuda) + `cudnn.deterministic=True`, `benchmark=False`.
- DDP seeds offset by `local_rank` (existing pattern).
- Every run writes a `run_config.json` (mode, seed, model, label-set hash, git commit) into its output dir.

---

## 10. Multi-GPU

- **LVLM labeling:** multi-GPU sharding (new in `run.py`). For the 35B MoE default: **TP=2 × DP=2** at bf16, or FP8/AWQ → DP=4. Smaller VL models can use DP=4 directly.
- **Training:** existing DDP (`--local_rank`, `nccl`) for both trackers; `train.sh` launches via `torchrun --nproc_per_node=N`.
- **Tracking inference / test:** existing multi-process-over-sequences (`--threads`); `test_mvtd.sh` wraps it.

---

## 11. Evaluation

- Standard MVTD (GOT-10k-style) **AO / SR / precision**, cls head off-path for box metrics.
- **Per-condition AO/SR breakdown** (`eval_per_condition.py`) — the key figure.
- **Oracle upper bound:** `mtl_film` conditioned on ground-truth labels vs predicted → ceiling diagnostic.
- **Ablation:** `pretrained → finetune_bbox → mtl_hard → mtl_soft → mtl_multilabel → mtl_film`.
- **Multiple seeds** + report mean/variance.

---

## 12. `scripts/` interfaces

```bash
# Generate labels (LVLM + oracles) for a split, sharded across GPUs
scripts/gen_labels.sh --split train --gpus 0,1,2,3 --tp 2 --dp 2 \
    --model Qwen/Qwen3.5-35B-A3B --seed 42

# Train a tracker in a given mode (DDP if --nproc > 1)
scripts/train.sh --tracker hiptrack --mode mtl_film --nproc 4 --seed 42

# Test on MVTD given a checkpoint, emit AO/SR + per-condition table
scripts/test_mvtd.sh --tracker hiptrack --model_path /path/to/ckpt.pth.tar --threads 4
```

---

## 13. Out of scope (future work)

- Part 3 (SAM dense-mask supervision).
- HIPTrack-specific condition-gated **memory/update** mechanism (only Gate 1 / FiLM is built now).
- Extension to other trackers in `models/`.

---

## 14. Risks / open items

- **LVLM model:** `Qwen/Qwen3.5-35B-A3B` (verified VL MoE). bf16 needs TP=2 (≈70 GB > 48 GB/GPU); confirm an FP8/AWQ build exists if DP=4 throughput is wanted.
- **SimTrack proper-pipeline port:** the `ltr` training path must be confirmed functional for MVTD;
  if there are gaps, surface them in the implementation plan.
- **MVTD occlusion/visibility oracle:** depends on whether the GOT-10k-format export ships
  `cover/absence` labels; if absent, occlusion stays VLM-only.
- **Throughput:** TP=2×DP=2 halves labeling parallelism vs DP=4; if the full ~150k-frame pass is too
  slow, use an FP8/AWQ build (DP=4) or subsample frames per sequence for the first pass.
```
