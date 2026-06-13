#!/usr/bin/env bash
# Evaluate a tracker on MVTD (consumed in GOT-10k format) and report metrics.
# Thin wrapper over each tracker's tracking/test.py + its evaluation entry point.
#
# Usage:
#   scripts/test_mvtd.sh <model_dir> <tracker_name> <config> [threads] [dataset]
# Examples:
#   scripts/test_mvtd.sh HIPTrack hiptrack_cls hiptrack_cls 4 got10k_val
#
# NOTE: most trackers here resolve the checkpoint from <config>+epoch convention
# rather than an arbitrary path; set TEST.EPOCH / the checkpoint location in the
# config (or its local files) before running.
set -euo pipefail

MODEL_DIR="${1:?model dir under models/ (e.g. HIPTrack)}"
NAME="${2:?tracker name for test.py (e.g. hiptrack_cls)}"
CONFIG="${3:?config name (e.g. hiptrack_cls)}"
THREADS="${4:-4}"
DATASET="${5:-got10k_val}"   # got10k_val is the MVTD baseline split here

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/models/${MODEL_DIR}"
export PYTHONPATH=./:"${PYTHONPATH:-}"

echo "[test] ${MODEL_DIR} name=${NAME} config=${CONFIG} dataset=${DATASET} threads=${THREADS}"
python tracking/test.py "${NAME}" "${CONFIG}" --dataset "${DATASET}" --threads "${THREADS}"

# Per-tracker performance summary (AO / SR / precision), if present.
if [ -f evaluate_performance.py ]; then
    echo "[test] computing performance summary..."
    python evaluate_performance.py || echo "[test] evaluate_performance.py needs result-path config; skipping"
fi
