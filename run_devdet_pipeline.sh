#!/usr/bin/env bash
# Run DevDet selection, training, and DFDC/eval2024 evaluation.
# Evaluation covers the detector before DAFT fine-tuning (epoch 000) and every
# configured DAFT epoch, with each result written to a separate directory.
#
# Usage:
#   bash run_devdet_pipeline.sh <selection_threshold> [selection_scores_all.csv]
#
# Examples:
#   CUDA_VISIBLE_DEVICES=1 bash run_devdet_pipeline.sh 0.7
#   CUDA_VISIBLE_DEVICES=1 bash run_devdet_pipeline.sh 0.9 \
#     output/devdet_hard07_adaptive/selection_scores_all.csv
#
# Optional environment variables:
#   PY=python AMP=bf16 BATCH_SIZE=64 EVAL_BATCH_SIZE=32 NUM_WORKERS=8
#   EVAL_THRESHOLD=0.5 EVAL_BENCHES=dfdc,eval2024
#   SELECT_MODE=auto|always|skip
set -euo pipefail

THRESHOLD="${1:?usage: bash run_devdet_pipeline.sh <selection_threshold> [selection_scores_all.csv]}"
SELECTION_SCORES="${2:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
AMP="${AMP:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
EVAL_BENCHES="${EVAL_BENCHES:-dfdc,eval2024}"
SELECT_MODE="${SELECT_MODE:-auto}"
BASE_CONFIG="${BASE_CONFIG:-$REPO/configs/devdet_hard07_balanced.json}"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DEVDET_CONFIG="$($PY - "$THRESHOLD" "$BASE_CONFIG" "$REPO" <<'PY'
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

raw_threshold, raw_base, raw_repo = sys.argv[1:]
repo = Path(raw_repo).resolve()
base = Path(raw_base).expanduser()
if not base.is_absolute():
    base = repo / base
try:
    decimal_threshold = Decimal(raw_threshold)
except InvalidOperation as error:
    raise SystemExit(f"Invalid selection threshold: {raw_threshold!r}") from error
if not decimal_threshold.is_finite() or not Decimal("0") < decimal_threshold <= Decimal("1"):
    raise SystemExit("Selection threshold must be greater than 0 and at most 1")

normalized = format(decimal_threshold.normalize(), "f")
tag = normalized.replace(".", "").replace("-", "m")
existing = repo / "configs" / f"devdet_hard{tag}_balanced.json"
if existing.is_file():
    recipe = json.loads(existing.read_text(encoding="utf-8"))
    if Decimal(str(recipe["selection"]["threshold"])) == decimal_threshold:
        print(existing)
        raise SystemExit(0)

recipe = json.loads(base.read_text(encoding="utf-8"))
recipe["_comment"] = (
    f"Adaptive FFDev/DoseDict selection at true-class confidence {normalized}."
)
recipe["selection"]["threshold"] = float(decimal_threshold)
recipe["output_dir"] = f"output/devdet_hard{tag}_adaptive"
generated = repo / recipe["output_dir"] / "devdet_config.json"
generated.parent.mkdir(parents=True, exist_ok=True)
generated.write_text(json.dumps(recipe, indent=4) + "\n", encoding="utf-8")
print(generated)
PY
)"

readarray -t PIPELINE_META < <("$PY" - "$DEVDET_CONFIG" "$REPO" <<'PY'
import json
import sys
from pathlib import Path

config = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
recipe = json.loads(config.read_text(encoding="utf-8"))
output = Path(recipe["output_dir"]).expanduser()
if not output.is_absolute():
    output = repo / output
print(output)
print(int(recipe["daft"]["epochs"]))
PY
)
DEVDET_OUTPUT="${PIPELINE_META[0]}"
DAFT_EPOCHS="${PIPELINE_META[1]}"

SELECT_ARGS=(
    --devdet-config "$DEVDET_CONFIG"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --amp "$AMP"
)
if [[ -n "$SELECTION_SCORES" ]]; then
    SELECT_ARGS+=(--selection-scores "$SELECTION_SCORES")
fi

echo "[$(date '+%F %T')] DevDet pipeline start"
echo "selection_threshold=$THRESHOLD config=$DEVDET_CONFIG scores=${SELECTION_SCORES:-baseline inference}"

case "$SELECT_MODE" in
    auto)
        if [[ -z "$SELECTION_SCORES" \
            && -f "$DEVDET_OUTPUT/selection.json" \
            && -f "$DEVDET_OUTPUT/selection_scores_all.csv" \
            && -f "$DEVDET_OUTPUT/baseline_scores.csv" \
            && -f "$DEVDET_OUTPUT/ffdev_samples.csv" ]]; then
            echo "[$(date '+%F %T')] Selection artifacts found; skipping select"
        else
            "$PY" src/train_devdet.py select "${SELECT_ARGS[@]}"
        fi
        ;;
    always)
        "$PY" src/train_devdet.py select "${SELECT_ARGS[@]}"
        ;;
    skip)
        echo "[$(date '+%F %T')] SELECT_MODE=skip; skipping select"
        ;;
    *)
        echo "Invalid SELECT_MODE=$SELECT_MODE (expected auto, always, or skip)" >&2
        exit 2
        ;;
esac

"$PY" src/train_devdet.py train-ffdev \
    --devdet-config "$DEVDET_CONFIG" --num-workers "$NUM_WORKERS" --amp "$AMP"
"$PY" src/train_devdet.py fit-dict \
    --devdet-config "$DEVDET_CONFIG" --num-workers "$NUM_WORKERS" --amp "$AMP"
"$PY" src/train_devdet.py train-daft \
    --devdet-config "$DEVDET_CONFIG" --num-workers "$NUM_WORKERS" --amp "$AMP"

# Epoch 000: use the original, non-fine-tuned detector with the learned FFDev
# and DoseDict components.
"$PY" src/eval_devdet_crossdomain.py \
    --devdet-config "$DEVDET_CONFIG" \
    --detector-mode baseline \
    --benches "$EVAL_BENCHES" \
    --threshold "$EVAL_THRESHOLD" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --amp "$AMP" \
    --output-dir "$DEVDET_OUTPUT/crossdomain_eval_epoch_000_baseline"

# Evaluate every fine-tuned checkpoint separately so no result is overwritten.
for ((epoch = 1; epoch <= DAFT_EPOCHS; epoch++)); do
    printf -v epoch_tag '%03d' "$epoch"
    checkpoint="$DEVDET_OUTPUT/daft_detector_epoch_${epoch_tag}.pth"
    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing DAFT checkpoint: $checkpoint" >&2
        exit 1
    fi
    "$PY" src/eval_devdet_crossdomain.py \
        --devdet-config "$DEVDET_CONFIG" \
        --detector-mode daft \
        --detector "$checkpoint" \
        --benches "$EVAL_BENCHES" \
        --threshold "$EVAL_THRESHOLD" \
        --batch-size "$EVAL_BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --amp "$AMP" \
        --output-dir "$DEVDET_OUTPUT/crossdomain_eval_epoch_${epoch_tag}"
done

echo "[$(date '+%F %T')] DevDet pipeline complete"
