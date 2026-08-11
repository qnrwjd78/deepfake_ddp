"""Video-level DFDC/eval2024 evaluation for baseline_gated_dual.

This is separate from DevDet evaluation and does not modify the baseline model.
Aggregation is frame=max-face followed by video=mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from eval_baseline_3domain import build_model  # noqa: E402
from eval_devdet_3domain import binary_auc  # noqa: E402
from eval_devdet_crossdomain import (  # noqa: E402
    DEFAULT_BENCHES, aggregate, binary_metrics, discover_benchmark,
    infer_frames, parse_benches, repo_path,
)


def parse_gpu_devices(value: str) -> list[int]:
    """Parse comma-separated CUDA ordinals without changing CUDA visibility."""
    raw_devices = [item.strip() for item in value.split(",")]
    if not raw_devices or any(not item for item in raw_devices):
        raise argparse.ArgumentTypeError(
            "GPU devices must be a comma-separated list such as 4,5,6,7"
        )
    try:
        devices = [int(item) for item in raw_devices]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU devices must be non-negative integer CUDA ordinals"
        ) from exc
    if any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("GPU devices must be non-negative")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("GPU devices must not contain duplicates")
    return devices


def shard_bounds(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return one exact contiguous shard; all shards partition [0, total)."""
    if total < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(
            f"Invalid shard request: total={total}, rank={rank}, world_size={world_size}"
        )
    quotient, remainder = divmod(total, world_size)
    start = rank * quotient + min(rank, remainder)
    end = start + quotient + int(rank < remainder)
    return start, end


def print_results(result: dict):
    """Print both the full labeled set and the crop-present subset."""
    print("\nDataset      Subset    Videos  Missing  F-ACC  R-ACC  B-ACC    AUC")
    print("-" * 74)
    for name, entry in result["per_dataset"].items():
        for subset in ("all", "present"):
            row = entry[subset]
            missing = entry["coverage"]["missing_videos"] if subset == "all" else 0
            print(
                f"{name:<12} {subset:<8} {row['n_videos']:>6}  {missing:>7}  "
                f"{row['f_acc']:.3f}  {row['r_acc']:.3f}  "
                f"{row['balanced_acc']:.3f}  {row['auc']:.3f}"
            )
    print("-" * 74)
    for subset, key in (("ALL", "overall_all"), ("PRESENT", "overall_present")):
        overall = result[key]
        print(
            f"{subset} mean-AUC={overall['mean_auc']:.3f}  "
            f"pooled-AUC={overall['pooled_auc']:.3f}  "
            f"mean-B-ACC={overall['mean_balanced_acc']:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", default="baseline_gated_dual")
    parser.add_argument("--config", help="Defaults to output/RUN/config.json")
    parser.add_argument("--checkpoint", help="Defaults to output/RUN/weights/100.pth")
    parser.add_argument("--output-dir", help="Defaults to output/RUN/crossdomain_eval")
    parser.add_argument("--benches", type=parse_benches, default=parse_benches("dfdc,eval2024"))
    parser.add_argument("--dfdc-frames", default=DEFAULT_BENCHES["dfdc"][0])
    parser.add_argument("--dfdc-labels", default=DEFAULT_BENCHES["dfdc"][1])
    parser.add_argument("--eval2024-frames", default=DEFAULT_BENCHES["eval2024"][0])
    parser.add_argument("--eval2024-labels", default=DEFAULT_BENCHES["eval2024"][1])
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--missing-score", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="DataLoader workers per GPU"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--gpu-devides", "--gpu-devices", dest="gpu_devices",
        type=parse_gpu_devices,
        help=(
            "Comma-separated visible CUDA ordinals for internal multi-GPU inference; "
            "--gpu-devides is retained as a compatibility alias"
        ),
    )
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_gpu_devices(gpu_devices: list[int], requested_device: str) -> None:
    if not str(requested_device).startswith("cuda"):
        raise ValueError("--gpu-devides/--gpu-devices requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU CUDA inference requested, but CUDA is unavailable")
    device_count = torch.cuda.device_count()
    unavailable = [device for device in gpu_devices if device >= device_count]
    if unavailable:
        raise RuntimeError(
            f"CUDA device(s) {unavailable} unavailable; visible ordinals are "
            f"0..{device_count - 1}"
        )


def _gpu_inference_worker(
    rank: int,
    args: argparse.Namespace,
    cfg: dict,
    checkpoint_path: Path,
    records: pd.DataFrame,
    shared_scores: torch.Tensor,
    gpu_devices: tuple[int, ...],
) -> None:
    """Load one isolated model and fill this GPU's shared-score slice."""
    world_size = len(gpu_devices)
    gpu_device = gpu_devices[rank]
    device = torch.device("cuda", gpu_device)
    torch.cuda.set_device(device)
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"--amp bf16 requested, but CUDA device {gpu_device} does not support BF16"
        )

    model = build_model(cfg, checkpoint_path, device)
    shard_start, shard_end = shard_bounds(len(records), rank, world_size)
    local_records = records.iloc[shard_start:shard_end].copy().reset_index(drop=True)
    local_predictions = infer_frames(
        model,
        local_records,
        args,
        device,
        description=(
            f"Baseline cross-domain [GPU {gpu_device}, "
            f"shard {rank + 1}/{world_size}]"
        ),
        show_progress=rank == 0,
    )
    local_scores = local_predictions["p_fake"].to_numpy(np.float32)
    expected = shard_end - shard_start
    if len(local_scores) != expected:
        raise RuntimeError(
            f"GPU {gpu_device} returned {len(local_scores)} scores for {expected} records"
        )
    if not np.isfinite(local_scores).all():
        raise RuntimeError(f"GPU {gpu_device} produced NaN or Inf scores")
    if expected:
        shared_scores[shard_start:shard_end].copy_(torch.from_numpy(local_scores))


def infer_frames_multi_gpu(
    cfg: dict,
    checkpoint_path: Path,
    records: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Run one process per requested GPU and merge disjoint score slices."""
    gpu_devices = tuple(args.gpu_devices)
    validate_gpu_devices(list(gpu_devices), args.device)
    shared_scores = torch.full((len(records),), float("nan"), dtype=torch.float32)
    shared_scores.share_memory_()
    mp.spawn(
        _gpu_inference_worker,
        args=(args, cfg, checkpoint_path, records, shared_scores, gpu_devices),
        nprocs=len(gpu_devices),
        join=True,
    )
    scores = shared_scores.numpy().copy()
    if not np.isfinite(scores).all():
        missing = int((~np.isfinite(scores)).sum())
        raise RuntimeError(f"Multi-GPU inference did not produce {missing} frame scores")
    predictions = records.copy().reset_index(drop=True)
    predictions["p_fake"] = scores
    print(
        f"Merged predictions from {len(gpu_devices)} GPUs "
        f"{list(gpu_devices)}: {len(predictions)}/{len(records)}"
    )
    return predictions


def main():
    args = parse_args()
    run_dir = REPO / "output" / args.run
    config_path = repo_path(args.config) if args.config else run_dir / "config.json"
    checkpoint_path = (
        repo_path(args.checkpoint) if args.checkpoint
        else run_dir / "weights" / "100.pth"
    )
    output = repo_path(args.output_dir) if args.output_dir else run_dir / "crossdomain_eval"
    overrides = {
        "dfdc": (repo_path(args.dfdc_frames), repo_path(args.dfdc_labels)),
        "eval2024": (repo_path(args.eval2024_frames), repo_path(args.eval2024_labels)),
    }

    all_frames, all_videos, audits = [], [], {}
    for name in args.benches:
        frames, videos, audit = discover_benchmark(name, *overrides[name])
        all_frames.append(frames)
        all_videos.append(videos)
        audits[name] = audit
        print(
            f"{name}: videos={audit['videos']} present={audit['present_videos']} "
            f"missing={audit['missing_videos']} crops={audit['crops']}"
        )
    records = pd.concat(all_frames, ignore_index=True)
    videos = pd.concat(all_videos, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    videos.to_csv(output / "discovered_videos.csv", index=False)
    if args.dry_run:
        print(f"Dry run complete: {output / 'data_audit.json'}")
        return

    if not config_path.is_file():
        raise FileNotFoundError(f"Baseline config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")
    cfg = json.loads(config_path.read_text())
    cfg["in_chans"] = 3
    if args.gpu_devices:
        predictions = infer_frames_multi_gpu(cfg, checkpoint_path, records, args)
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device)
        if (
            device.type == "cuda"
            and args.amp == "bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("--amp bf16 requested, but this GPU does not support BF16")
        model = build_model(cfg, checkpoint_path, device)
        predictions = infer_frames(
            model, records, args, device, description="Baseline cross-domain"
        )
    frame_scores, video_scores = aggregate(predictions, videos, args.missing_score)
    video_scores["prediction"] = (video_scores["p_fake"] >= args.threshold).astype(np.int64)

    per_dataset, summary_rows, all_tables, present_tables = {}, [], [], []
    for name in args.benches:
        table = video_scores[video_scores.dataset == name].copy()
        present = table[table["scored_frames"] > 0].copy()
        all_tables.append(table)
        present_tables.append(present)
        entry = {
            "coverage": audits[name],
            "all": binary_metrics(table, args.threshold),
            "present": binary_metrics(present, args.threshold),
        }
        per_dataset[name] = entry
        summary_rows.append({"dataset": name, "subset": "all", **entry["all"]})
        summary_rows.append(
            {"dataset": name, "subset": "present", **entry["present"]}
        )
    pooled = pd.concat(all_tables, ignore_index=True)
    pooled_present = pd.concat(present_tables, ignore_index=True)
    result = {
        "per_dataset": per_dataset,
        "overall_all": {
            "mean_auc": float(np.mean([per_dataset[name]["all"]["auc"] for name in args.benches])),
            "pooled_auc": binary_auc(
                pooled["label"].to_numpy(np.int64), pooled["p_fake"].to_numpy(np.float64)
            ),
            "mean_balanced_acc": float(np.mean([
                per_dataset[name]["all"]["balanced_acc"] for name in args.benches
            ])),
        },
        "overall_present": {
            "mean_auc": float(np.mean([
                per_dataset[name]["present"]["auc"] for name in args.benches
            ])),
            "pooled_auc": binary_auc(
                pooled_present["label"].to_numpy(np.int64),
                pooled_present["p_fake"].to_numpy(np.float64),
            ),
            "mean_balanced_acc": float(np.mean([
                per_dataset[name]["present"]["balanced_acc"] for name in args.benches
            ])),
        },
        "threshold": float(args.threshold),
        "missing_score": float(args.missing_score),
        "protocol": {
            "name": "baseline-gated-dual-all-and-present-video-level",
            "aggregation": "frame=max-face, video=mean",
            "score": "softmax(raw baseline logits)[:,1]",
            "input_size": int(args.input_size),
            "reported_subset": "all labeled videos; missing crops use missing_score",
            "reported_subsets": (
                "all labeled videos (missing crops use missing_score) and "
                "present videos (at least one scored crop)"
            ),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "world_size": len(args.gpu_devices) if args.gpu_devices else 1,
            "gpu_devices": list(args.gpu_devices) if args.gpu_devices else None,
        },
    }
    predictions.to_csv(output / "frame_predictions.csv", index=False)
    frame_scores.to_csv(output / "frame_scores.csv", index=False)
    video_scores.to_csv(output / "video_scores.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "summary.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_results(result)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
