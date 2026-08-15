# DevDet-style reconstruction

This implementation is isolated from the existing baseline code. It does not
change `src/model.py`, `src/train_multi.py`, or any existing checkpoint.

## Pipeline

1. Score baseline training frames and build the configured balanced selection.
2. Freeze the baseline detector and train the CDA-inspired ResNet-6 FFDev.
3. Fit DoseDict on 6,000 hard-fake, classifier-preceding gated features.
4. Freeze FFDev and DoseDict, then fine-tune GatedDual with adaptive doses.

The domain/label policy follows `baseline_gated_dual`: FF++, newbench
real/fake, and new_benchmark_2 fake only. The actual loader is not identical to
baseline training: DevDet uses a static eight-frame protocol with
landmark-aligned crops and does not recreate SBI samples, repeated-source
weighting, fake shift, or Retina filtering. Evaluation includes both classes
from all three datasets.

## Commands

Run from the project root. The stages are resumable through their saved files,
but each individual stage currently starts from the beginning when invoked.

The end-to-end launcher automatically reuses a completed selection and then
trains FFDev, fits DoseDict, trains DAFT, and evaluates the non-fine-tuned
baseline detector plus every configured DAFT epoch into separate directories:

```bash
CUDA_VISIBLE_DEVICES=1 bash run_devdet_pipeline.sh 0.7
CUDA_VISIBLE_DEVICES=1 bash run_devdet_pipeline.sh 0.9
```

Set `SELECT_MODE=always` to rebuild selection or `SELECT_MODE=skip` to require
the existing selection artifacts without checking them in the launcher.

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_devdet.py select \
  --batch-size 64 --num-workers 8

CUDA_VISIBLE_DEVICES=0 python src/train_devdet.py train-ffdev \
  --num-workers 8

CUDA_VISIBLE_DEVICES=0 python src/train_devdet.py fit-dict \
  --num-workers 8

CUDA_VISIBLE_DEVICES=0 python src/train_devdet.py train-daft \
  --num-workers 8
```

Full DevDet-path evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval_devdet_3domain.py \
  --batch-size 32 --num-workers 8
```

DFDC and eval2024 full-path evaluation (video-level AUC):

```bash
CUDA_VISIBLE_DEVICES=4 python src/eval_devdet_crossdomain.py \
  --benches dfdc,eval2024 --batch-size 32 --num-workers 8
```

This reports `ALL`: every labeled video is included and videos without crops
receive score 0.5.

Prepare all 2,036 published eval2024 videos with four GPUs before evaluation:

```bash
for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU python src/prep/prep_eval2024.py \
    --part $((GPU + 1))/4 > "prep_eval2024_part$((GPU + 1)).log" 2>&1 &
done
wait
```

This writes the evaluator defaults, `data/2024/frames/` and
`data/2024/labels.csv`. By default both published `train` and `test` metadata
partitions are included. Use `--split test` only for the 815-video held-out
subset. The operation is resumable through one `.complete` marker per video;
pass `--overwrite` to recreate existing crops.

To evaluate the original baseline detector with FFDev and DoseDict, without
loading the DAFT detector, add `--detector-mode baseline`. Its default output
directory is separate, so the full-DAFT result is not overwritten.

```bash
CUDA_VISIBLE_DEVICES=4 python src/eval_devdet_crossdomain.py \
  --detector-mode baseline --benches dfdc,eval2024 \
  --batch-size 32 --num-workers 8
```

Multi-GPU evaluation uses one independent inference process per GPU. For two
GPUs, run:

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 \
  src/eval_devdet_crossdomain.py \
  --devdet-config configs/devdet_gated_dual.json \
  --detector-mode baseline --benches dfdc,eval2024 \
  --input-size 256 --batch-size 32 --num-workers 4
```

`batch-size` and `num-workers` are per GPU process. Each process owns a
separate detector, FFDev, DoseDict, and sparse-code cache. Frames are split
without padding or duplication, and rank 0 restores their original order and
writes the ALL metrics.

## Adaptive 0.7/0.9 hard-fake ablations

`configs/devdet_hard07_balanced.json` and
`configs/devdet_hard09_balanced.json` derive their sample counts from the
baseline inference scores instead of storing checkpoint-specific counts:

- every fake with `p_fake < T` is selected automatically;
- each domain first contributes real samples with `p_real < T`;
- NB2 remains fake-only and never contributes real samples;
- any real shortfall is filled from the original FF++ + newbench real pool,
  ordered by lowest `p_real` first; and
- the final FFDev manifest is globally balanced, with equal real/fake counts.

Hardness is symmetric in true-class confidence: fake uses `p_fake < T`, while
real uses `p_real < T` (equivalently `p_fake > 1-T`). If the hard-real pool is
too small, fallback may cross the threshold, but still takes the remaining
real samples from hardest to easiest. DoseDict uses every automatically
selected hard fake with 64 atoms. Training augmentation is disabled. The final
DAFT stage uses up to 20,000 class-balanced frames for three epochs and updates
only the classifier; the feature encoder and gate remain frozen. Every epoch
is saved separately.

```bash
CUDA_VISIBLE_DEVICES=4 python src/train_devdet.py select \
  --devdet-config configs/devdet_hard07_balanced.json \
  --batch-size 64 --num-workers 8

CUDA_VISIBLE_DEVICES=4 python src/train_devdet.py train-ffdev \
  --devdet-config configs/devdet_hard07_balanced.json \
  --num-workers 8

CUDA_VISIBLE_DEVICES=4 python src/train_devdet.py fit-dict \
  --devdet-config configs/devdet_hard07_balanced.json \
  --num-workers 8

CUDA_VISIBLE_DEVICES=4 python src/train_devdet.py train-daft \
  --devdet-config configs/devdet_hard07_balanced.json \
  --num-workers 8

CUDA_VISIBLE_DEVICES=4 python src/eval_devdet_crossdomain.py \
  --devdet-config configs/devdet_hard07_balanced.json \
  --detector-mode baseline --benches dfdc,eval2024 \
  --batch-size 32 --num-workers 8
```

DAFT writes `daft_detector_epoch_001.pth` through `_003.pth` and updates
`daft_detector.pth` as a hard-link alias to the latest completed epoch. To
continue for two more epochs without changing the recipe, run:

```bash
CUDA_VISIBLE_DEVICES=4 python src/train_devdet.py train-daft \
  --devdet-config configs/devdet_hard07_balanced.json \
  --resume latest --additional-epochs 2 \
  --num-workers 8
```

This restores the model, optimizer, AMP scaler, histories, and RNG state, then
saves epoch 004 and 005. It refuses to overwrite an existing numbered epoch.

Artifacts and ALL evaluation results are written below
`output/devdet_hard07_adaptive/`; existing `output/devdet_gated_dual/`
artifacts are not reused or overwritten.

Baseline GatedDual comparison using the identical data and aggregation:

```bash
CUDA_VISIBLE_DEVICES=7 python src/eval_baseline_crossdomain.py \
  --run baseline_gated_dual --benches dfdc,eval2024 \
  --batch-size 64 --num-workers 8
```

Outputs are written to `output/devdet_gated_dual/`. The final
`daft_detector.pth` stores its detector under the standard `model` key, while
full inference additionally requires `ffdev.pth` and `dose_dictionary.pth`.
The default DAFT setting uses the complete discovered `S_m`; set
`daft.max_samples` only for a clearly labeled smaller ablation.
Because this project's source pool contains substantially more fake frames,
DAFT uses real/fake-balanced cross-entropy weights by default.

## Explicit reconstruction choices

- FFDev returns a signed `tanh` residual and developed images are clamped to
  `[0,1]`.
- Stage 1 uses `epsilon=0.25` and CE + TV.
- DoseDict uses L2-normalized fused features and an in-tree PyTorch ISTA solver;
  scikit-learn is not required.
- Adaptive dose uses robust 5th/95th error percentiles. Low reconstruction
  error maps to a high dose.
- The paper supplement states that Stage 2 multiplies adaptive dose by `0.25`
  to match Stage 1, so the default is `dose_mode=scaled`. `direct` remains an
  explicit ablation that applies a dose up to `1.0`.
- DAFT obtains doses from detached features of the live detector to avoid
  keeping a second GatedDual in GPU memory.

These choices are configurable in `configs/devdet_gated_dual.json` because the
MID-FFD paper does not disclose their exact values or implementation details.

This three-domain command is a training-domain diagnostic, not an unseen
cross-dataset result: FF++ train and newbench are reused, and NB2 fake was used
by the baseline and DAFT. NB2 real remains excluded. Use DFDC/eval2024 or a
held-out split for the final generalization comparison. If Stage 1 runs out of
memory, reduce `ffdev.batch_size` from 4 to 2 or 1.
