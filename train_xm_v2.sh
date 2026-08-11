#!/usr/bin/env bash
# Standalone XM training on the normalized deepfake_ddp_v2 data layout.
# Existing model.py, train_multi.py, and utils/sbi.py are not modified.
set -euo pipefail

GPUS="${1:-0,1,2,3}"
NAME="${2:-xm_l005_v2}"
SEED="${SEED:-906}"
LS="${LS:-0.0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
CFG="${CFG:-$REPO/configs/gated_dual_baseline_xm_v2.json}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/.cache/huggingface}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC="$(awk -F',' '{print NF}' <<< "$GPUS")"
export TRAIN_JOBS_PER_NODE="${TRAIN_JOBS_PER_NODE:-$NPROC}"
export DDP_FIND_UNUSED="${DDP_FIND_UNUSED:-0}"

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    if [[ "$RESUME" == "1" ]]; then
        RESUME_ARGS=(-r)
    else
        RESUME_ARGS=(-r "$RESUME")
    fi
fi

echo "[$(date '+%F %T')] XM-v2 TRAIN '$NAME' | cfg=$(basename "$CFG") GPUs=$GPUS nproc=$NPROC seed=$SEED ls=$LS"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    src/train_multi_xm_v2.py "$CFG" -n "$NAME" -s "$SEED" -ls "$LS" "${RESUME_ARGS[@]}"
echo "[$(date '+%F %T')] XM-v2 TRAIN done -> output/${NAME}_gated_dual_xm/"
