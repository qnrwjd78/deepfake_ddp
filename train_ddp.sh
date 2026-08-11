#!/usr/bin/env bash
# Multi-GPU (DDP) training of the gated_dual recipe via torchrun.
#   bash train_ddp.sh <gpu_ids> <run_name>    e.g. bash train_ddp.sh 0,1,2,3 reproduce
# Global batch = cfg.batch_size (32); it is split evenly across the GPUs (4 -> 8/GPU),
# so training dynamics match the single-GPU run — only wall-clock drops (~#GPUs faster).
# Env: SEED(906) LS(0.0) NUM_WORKERS HF_HOME PY
#   RESUME=1            continue from the latest checkpoint in output/<run>_gated_dual/weights/
#   RESUME=<path.pth>   continue from a specific checkpoint
#   DDP_FIND_UNUSED=0   small speedup once a run is confirmed healthy (default 1 = safe)
set -euo pipefail

GPUS="${1:?usage: train_ddp.sh <gpu_ids e.g. 0,1,2,3> <run_name>}"
NAME="${2:?run name required}"
SEED="${SEED:-906}"
LS="${LS:-0.0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
# CFG env overrides the recipe (repo-relative or absolute). Default = champion gated_dual.json.
CFG="${CFG:-$REPO/configs/gated_dual.json}"
cd "$REPO"
# output dir suffix = the config's model (gated_dual | gated_dual_dino | fm); used only for the echo below
MODEL="$("$PY" -c "import json; print(json.load(open('$CFG'))['model'])" 2>/dev/null || echo gated_dual)"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$GPUS"

# number of GPUs = comma-separated count of the id list
NPROC=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
# each rank spawns its own DataLoader workers; tell the worker heuristic how many jobs share the node
export TRAIN_JOBS_PER_NODE="${TRAIN_JOBS_PER_NODE:-$NPROC}"

RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
    if [ "$RESUME" = "1" ]; then RESUME_ARGS=(-r); else RESUME_ARGS=(-r "$RESUME"); fi
fi

echo "[$(date '+%F %T')] DDP TRAIN '$NAME' | cfg=$(basename "$CFG") GPUs=$GPUS (nproc=$NPROC) seed=$SEED ls=$LS${RESUME:+ RESUME=$RESUME}"
torchrun --standalone --nproc_per_node="$NPROC" \
    src/train_multi.py "$CFG" -n "$NAME" -s "$SEED" -ls "$LS" "${RESUME_ARGS[@]}"
echo "[$(date '+%F %T')] DDP TRAIN done -> output/${NAME}_${MODEL}/  (eval: bash infer.sh <gpu> $NAME)"
