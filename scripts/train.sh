#!/usr/bin/env bash
# Train a tracker with a given experiment config, single- or multi-GPU.
# Thin wrapper over each tracker's own tracking/train.py (STARK/OSTrack-style).
#
# Usage:
#   scripts/train.sh <model_dir> <script> <config> [nproc] [save_dir]
# Examples:
#   scripts/train.sh HIPTrack hiptrack hiptrack_cls 1 ./output
#   scripts/train.sh HIPTrack hiptrack hiptrack_cls 2 ./output      # 2-GPU DDP
#
# NOTE: the per-mode configs (pretrained / finetune_bbox / mtl_hard / mtl_soft /
# mtl_multilabel / mtl_film) are added by the tracker integrations (Plans C/D).
set -euo pipefail

MODEL_DIR="${1:?model dir under models/ (e.g. HIPTrack)}"
SCRIPT="${2:?--script value (e.g. hiptrack)}"
CONFIG="${3:?--config value (e.g. hiptrack_cls)}"
NPROC="${4:-1}"
SAVE_DIR="${5:-./output}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/models/${MODEL_DIR}"
export PYTHONPATH=./:"${PYTHONPATH:-}"

echo "[train] ${MODEL_DIR} script=${SCRIPT} config=${CONFIG} nproc=${NPROC} save_dir=${SAVE_DIR}"
if [ "${NPROC}" -gt 1 ]; then
    python tracking/train.py --script "${SCRIPT}" --config "${CONFIG}" \
        --save_dir "${SAVE_DIR}" --mode multiple --nproc_per_node "${NPROC}"
else
    python tracking/train.py --script "${SCRIPT}" --config "${CONFIG}" \
        --save_dir "${SAVE_DIR}" --mode single
fi
