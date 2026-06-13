#!/usr/bin/env bash
# Generate maritime-condition labels (LVLM soft multi-label + deterministic oracles)
# for an MVTD split, sharded across GPUs. Output: data/<split>_maritime_cond_v2/.
#
# Usage:
#   scripts/gen_labels.sh [split] [gpus] [tp] [seed] [extra args...]
# Examples:
#   scripts/gen_labels.sh train 2,3 2 42
#   scripts/gen_labels.sh train 2,3 2 42 --max-frames 5            # smoke
#   scripts/gen_labels.sh test  0,1,2,3 2 42 --disable-custom-all-reduce
set -euo pipefail

SPLIT="${1:-train}"
GPUS="${2:-2,3}"
TP="${3:-2}"
SEED="${4:-42}"
shift $(( $# < 4 ? $# : 4 )) || true   # remaining args forwarded to run.py

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH=./
# multi-GPU stability defaults (override by exporting beforehand)
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

echo "[gen_labels] split=${SPLIT} gpus=${GPUS} tp=${TP} seed=${SEED} extra='$*'"
python -m modules.maritime_analyzer.run \
    --dataset "data/${SPLIT}" \
    --out-dir data \
    --gpus "${GPUS}" \
    --tp "${TP}" \
    --seed "${SEED}" \
    "$@"
echo "[gen_labels] done -> data/${SPLIT}_maritime_cond_v2/"
