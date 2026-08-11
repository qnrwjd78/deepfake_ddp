# Standalone XM in `deepfake_ddp_v2`

This is a direct port of the standalone cross-method (XM) training path from
`deepfake_ddp`. It is independent of DevDet. Existing v2 files `model.py`,
`train_multi.py`, and `utils/sbi.py` are not modified.

## Files

- `src/model_xm.py`: training-only auxiliary fused/gate return; state keys are
  identical to `GatedDualDetector`.
- `src/utils/cross_method_xm.py`: synchronized cross-method metric loss.
- `src/utils/sbi_precrop_xm.py`: Retina-free precrop loader.
- `src/utils/sbi_xm_v2.py`: XM metadata for the normalized v2 layout.
- `src/train_multi_xm_v2.py`: standalone XM trainer.
- `configs/gated_dual_baseline_xm_v2.json`: v2 data-path recipe.
- `train_xm_v2.sh`: multi-GPU launcher.
- `src/eval_xm_crossdomain.py`: XM-only DFDC/eval2024 ALL/PRESENT-video evaluator.

## Data mapping

The old `new_benchmark_margin` directory contained both newbench and NB2. In
v2 they are separate folders, so the XM loader reads both and merges their fake
lists back into one logical unknown-fake source. Newbench1 real is kept, while
NB2 is fake-only to match the baseline data policy.

Expected parity with the original run:

```text
real pool:          16,564
fake lists:         [5,499, 5,731, 5,744, 5,742, 5,744, 5,742, 19,041]
logical sources:    9 = real + SBI + six FF++ fakes + one merged extra fake
```

The loader accepts crop-local `(81,2)` or `(1,81,2)` landmarks and never
requires Retina files.

## Train from scratch

```bash
cd /workspace
bash train_xm_v2.sh 0,1,2,3 xm_l005_v2
```

This creates:

```text
output/xm_l005_v2_gated_dual_xm/
```

Resume the latest retained checkpoint:

```bash
cd /workspace
RESUME=1 bash train_xm_v2.sh 0,1,2,3 xm_l005_v2
```

The trainer refuses to overwrite an output directory containing checkpoints.
If a completed run has already been copied in for evaluation, do not launch a
new training job with the same name.

## Copy and evaluate the completed run

Place the copied files at:

```text
output/xm_l005_v2_gated_dual_xm/config.json
output/xm_l005_v2_gated_dual_xm/weights/100.pth
```

Single GPU:

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=0 python src/eval_xm_crossdomain.py \
  --run xm_l005_v2_gated_dual_xm \
  --benches dfdc,eval2024 \
  --input-size 256 \
  --batch-size 64 \
  --num-workers 8
```

Four GPUs for one sharded evaluation:

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  src/eval_xm_crossdomain.py \
  --run xm_l005_v2_gated_dual_xm \
  --benches dfdc,eval2024 \
  --input-size 256 \
  --batch-size 64 \
  --num-workers 4
```

`batch-size` and `num-workers` are per GPU process. The report shows both ALL
labeled videos and PRESENT videos with at least one crop. In the ALL row, a
missing video crop receives score `0.5`; frame scores are max-pooled over faces
and then averaged per video.
