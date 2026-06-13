# Maritime Object Tracking

## To-do list:
- [x] Integrate MVTD dataset.
- [x] Evaluation kit.
- [x] Scripts for single pass running training and evaluation on any models
- [] Idea for improvement 1


## Setup

```python
export PYTHONPATH="./:$PYTHONPATH"
```

## Run

### 1. Generate maritime condition labels (LVLM + deterministic oracles)

Multi-GPU label generation over an MVTD split. Writes per-frame v2 records to
`data/<split>_maritime_cond_v2/<seq>/<seq>.jsonl`. The model defaults to the local
Qwen3-VL checkpoint (override with `--model`); `--tp` shards it across GPUs.

```
export PYTHONPATH=./
python -m modules.maritime_analyzer.run \
    --dataset data/train --out-dir data --gpus 2,3 --tp 2 --seed 42
```

Add `--max-frames 5` for a quick smoke test. For multi-GPU stability use
`NCCL_P2P_DISABLE=1` and, if a tensor-parallel run hangs, `--disable-custom-all-reduce`
(and `--enforce-eager`).

Audit old labels against the deterministic oracle:
```
python -m modules.maritime_analyzer.validation.audit_partial_visibility \
    --ann-dir data/train_maritime_cond_v2
```

The shared multi-task toolkit (condition head, FiLM modulation, losses, label loader,
per-condition evaluation, seeding) lives in `modules/condition_mtl/`.
