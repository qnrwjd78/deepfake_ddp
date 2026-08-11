"""DFDC/eval2024 ALL/PRESENT-video evaluation for the standalone XM detector.

This evaluates only the copied XM checkpoint.  It does not involve FFDev,
DoseDict, or DAFT.  The protocol matches the existing v2 cross-domain report:
frame=max-face, video=mean, and labeled videos without crops receive 0.5.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))

from model import GatedDualDetector  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_BENCHES = {
    "dfdc": ("data/DFDC/test/frames", "data/DFDC/labels.csv"),
    "eval2024": ("data/2024/frames", "data/2024/labels.csv"),
}


def repo_path(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


def read_json(filename: Path) -> dict:
    with filename.open() as handle:
        return json.load(handle)


def parse_benches(value: str) -> list[str]:
    aliases = {"2024": "eval2024", "eval2024": "eval2024", "dfdc": "dfdc"}
    result = []
    for raw in value.split(","):
        key = raw.strip().lower()
        if key not in aliases:
            raise argparse.ArgumentTypeError(f"Unknown benchmark {raw!r}; use dfdc,eval2024")
        canonical = aliases[key]
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise argparse.ArgumentTypeError("At least one benchmark is required")
    return result


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 1:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.type == "cuda":
            device = torch.device("cuda", device.index if device.index is not None else 0)
            torch.cuda.set_device(device)
        return rank, 1, device

    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Multi-GPU evaluation requires CUDA and --device cuda")
    local_rank = int(args.local_rank)
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but only {torch.cuda.device_count()} visible GPUs exist"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size(), device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_worker(worker_id: int):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def frame_key(path: Path) -> str:
    # Supports both 019.png and a future multi-face form such as 019_1.png.
    return path.stem.split("_", 1)[0]


def discover_benchmark(name: str, frames_root: Path, labels_path: Path):
    if not frames_root.is_dir():
        raise FileNotFoundError(f"{name} frame root not found: {frames_root}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"{name} labels not found: {labels_path}")
    labels = pd.read_csv(labels_path)
    if not {"filename", "label"} <= set(labels.columns):
        raise ValueError(f"{labels_path} must contain filename,label columns")
    labels = labels[["filename", "label"]].copy()
    labels["video_id"] = labels["filename"].map(lambda value: Path(str(value)).stem)
    labels["label"] = labels["label"].astype(np.int64)
    if labels["video_id"].duplicated().any():
        duplicate = labels.loc[labels["video_id"].duplicated(), "video_id"].iloc[0]
        raise ValueError(f"Duplicate {name} video label: {duplicate}")
    if not set(labels["label"].unique()) <= {0, 1}:
        raise ValueError(f"Invalid labels in {labels_path}: {sorted(labels['label'].unique())}")

    frame_records = []
    present = []
    frame_counts = []
    for row in labels.itertuples(index=False):
        images = list_images(frames_root / row.video_id)
        present.append(bool(images))
        frame_counts.append(len({frame_key(image) for image in images}))
        for image in images:
            frame_records.append({
                "dataset": name,
                "video_id": row.video_id,
                "frame_id": frame_key(image),
                "crop_id": image.stem,
                "label": int(row.label),
                "image_path": str(image.resolve()),
            })
    videos = labels[["video_id", "label"]].copy()
    videos.insert(0, "dataset", name)
    videos["present"] = present
    videos["n_frames"] = frame_counts
    frames = pd.DataFrame(frame_records)
    audit = {
        "videos": int(len(videos)),
        "present_videos": int(videos["present"].sum()),
        "missing_videos": int((~videos["present"]).sum()),
        "real_videos": int((videos["label"] == 0).sum()),
        "fake_videos": int((videos["label"] == 1).sum()),
        "crops": int(len(frames)),
        "unique_frames": int(videos["n_frames"].sum()),
        "frames_root": str(frames_root),
        "labels": str(labels_path),
    }
    return frames, videos, audit


class CropDataset(Dataset):
    def __init__(self, records: pd.DataFrame, input_size: int):
        self.records = records.reset_index(drop=True)
        self.input_size = int(input_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        filename = self.records.iloc[index]["image_path"]
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read crop: {filename}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.input_size > 0 and image.shape[:2] != (self.input_size, self.input_size):
            image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image.transpose(2, 0, 1)), index


def amp_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return contextlib.nullcontext()
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware Mann-Whitney AUC, avoiding a scikit-learn dependency."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        raise ValueError("AUC requires both classes")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based.  Every tied score receives the average rank of
        # its [start, end) group, matching sklearn.metrics.roc_auc_score.
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[positive].sum())
    u_statistic = positive_rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u_statistic / (n_pos * n_neg))


@torch.no_grad()
def infer_frames(
    model, records: pd.DataFrame, args, device: torch.device,
    description: str = "DevDet cross-domain",
    show_progress: bool = True,
) -> pd.DataFrame:
    loader = DataLoader(
        CropDataset(records, args.input_size), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker,
    )
    scores = np.empty(len(records), dtype=np.float32)
    progress = tqdm(
        loader, desc=description, unit="batch", dynamic_ncols=True,
        disable=not show_progress,
    )
    done = 0
    for images, indices in progress:
        images = images.to(device, non_blocking=True)
        with amp_context(device, args.amp):
            probability = model(images).float().softmax(dim=1)[:, 1]
        scores[indices.numpy()] = probability.cpu().numpy()
        done += len(indices)
        progress.set_postfix(samples=f"{done}/{len(records)}", refresh=False)
    result = records.copy()
    result["p_fake"] = scores
    return result


def merge_distributed_predictions(
    local_predictions: pd.DataFrame,
    full_records: pd.DataFrame,
    rank: int,
    world_size: int,
) -> pd.DataFrame | None:
    if world_size == 1:
        return local_predictions.drop(columns=["_global_index"], errors="ignore").reset_index(drop=True)

    payload = (
        local_predictions["_global_index"].to_numpy(np.int64),
        local_predictions["p_fake"].to_numpy(np.float32),
    )
    gathered = [None] * world_size
    dist.all_gather_object(gathered, payload)
    if rank != 0:
        return None

    scores = np.empty(len(full_records), dtype=np.float32)
    seen = np.zeros(len(full_records), dtype=bool)
    for indices, values in gathered:
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)
        if len(indices) != len(values):
            raise RuntimeError("Distributed prediction index/score length mismatch")
        if not np.isfinite(values).all():
            raise RuntimeError("Distributed prediction contains NaN or Inf scores")
        if len(indices) and (indices.min() < 0 or indices.max() >= len(full_records)):
            raise RuntimeError("Distributed prediction contains an out-of-range index")
        if seen[indices].any():
            raise RuntimeError("Distributed prediction contains duplicate global indices")
        scores[indices] = values
        seen[indices] = True
    if not seen.all():
        raise RuntimeError(f"Distributed prediction missed {int((~seen).sum())} frames")

    result = full_records.drop(columns=["_global_index"]).copy().reset_index(drop=True)
    result["p_fake"] = scores
    return result


def binary_metrics(table: pd.DataFrame, threshold: float) -> dict:
    labels = table["label"].to_numpy(np.int64)
    scores = table["p_fake"].to_numpy(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("NaN or Inf score detected")
    real, fake = labels == 0, labels == 1
    if not real.any() or not fake.any():
        raise ValueError("Metric subset must contain both real and fake")
    f_acc = float((scores[fake] >= threshold).mean())
    r_acc = float((scores[real] < threshold).mean())
    return {
        "n_videos": int(len(table)), "n_real": int(real.sum()), "n_fake": int(fake.sum()),
        "f_acc": f_acc, "r_acc": r_acc,
        "balanced_acc": 0.5 * (f_acc + r_acc),
        "accuracy": float(((scores >= threshold).astype(np.int64) == labels).mean()),
        "auc": binary_auc(labels, scores),
    }


def aggregate(frame_predictions: pd.DataFrame, videos: pd.DataFrame, missing_score: float):
    # Maximum over multiple faces from one frame, then mean over frames.
    frame_scores = (
        frame_predictions.groupby(["dataset", "video_id", "frame_id"], as_index=False)
        .agg(label=("label", "first"), p_fake=("p_fake", "max"), n_crops=("crop_id", "count"))
    )
    aggregated = (
        frame_scores.groupby(["dataset", "video_id"], as_index=False)
        .agg(p_fake=("p_fake", "mean"), scored_frames=("frame_id", "count"))
    )
    result = videos.merge(aggregated, on=["dataset", "video_id"], how="left")
    result["p_fake"] = result["p_fake"].fillna(float(missing_score))
    result["scored_frames"] = result["scored_frames"].fillna(0).astype(np.int64)
    return frame_scores, result


def print_results(result: dict):
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
    for subset in ("all", "present"):
        overall = result[f"overall_{subset}"]
        print(
            f"{subset.upper()} mean-AUC={overall['mean_auc']:.3f}  "
            f"pooled-AUC={overall['pooled_auc']:.3f}  "
            f"mean-B-ACC={overall['mean_balanced_acc']:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the standalone XM checkpoint on DFDC/eval2024 ALL and PRESENT videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", default="xm_l005_v2_gated_dual_xm")
    parser.add_argument("--config", help="Defaults to output/RUN/config.json")
    parser.add_argument("--checkpoint", help="Defaults to output/RUN/weights/100.pth")
    parser.add_argument("--output-dir", help="Defaults to output/RUN/xm_crossdomain_eval")
    parser.add_argument(
        "--benches", type=parse_benches, default=parse_benches("dfdc,eval2024")
    )
    parser.add_argument("--dfdc-frames", default=DEFAULT_BENCHES["dfdc"][0])
    parser.add_argument("--dfdc-labels", default=DEFAULT_BENCHES["dfdc"][1])
    parser.add_argument(
        "--eval2024-frames", default=DEFAULT_BENCHES["eval2024"][0]
    )
    parser.add_argument(
        "--eval2024-labels", default=DEFAULT_BENCHES["eval2024"][1]
    )
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--missing-score", type=float, default=0.5)
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU process"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="DataLoader workers per GPU process"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _strip_uniform_module_prefix(state: dict) -> dict:
    keys = list(state)
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise RuntimeError("Checkpoint mixes module.-prefixed and unprefixed state keys")
    if all(prefixed) and keys:
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_xm_model(
    config_path: Path, checkpoint_path: Path, device: torch.device
) -> GatedDualDetector:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"XM config not found: {config_path}. Copy the completed run into "
            f"{REPO / 'output' / 'xm_l005_v2_gated_dual_xm'} first."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"XM checkpoint not found: {checkpoint_path}")

    with config_path.open() as handle:
        cfg = json.load(handle)
    if cfg.get("model") != "gated_dual":
        raise ValueError(f"XM evaluator requires model='gated_dual', got {cfg.get('model')!r}")
    if not cfg.get("cross_method_loss", {}).get("enabled", False):
        raise ValueError("Copied config is not an enabled standalone XM run")
    if int(cfg.get("xm_schema_version", 0)) != 1:
        raise ValueError(
            f"Unsupported or missing xm_schema_version: {cfg.get('xm_schema_version')!r}"
        )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"XM checkpoint must contain a model mapping: {checkpoint_path}")
    state = _strip_uniform_module_prefix(payload["model"])
    saved_epoch = payload.get("epoch")
    if checkpoint_path.stem.isdigit() and saved_epoch is not None:
        expected_number = int(saved_epoch) + 1
        if int(checkpoint_path.stem) != expected_number:
            raise RuntimeError(
                f"Checkpoint filename/epoch mismatch: {checkpoint_path.name} stores "
                f"epoch={saved_epoch}"
            )

    cfg["in_chans"] = 3
    model = GatedDualDetector(cfg)
    model.load_state_dict(state, strict=True)
    del state, payload
    return model.to(device).eval()


def run(args, rank: int, world_size: int, device: torch.device):
    is_main = rank == 0
    run_dir = REPO / "output" / args.run
    config_path = repo_path(args.config) if args.config else run_dir / "config.json"
    checkpoint_path = (
        repo_path(args.checkpoint)
        if args.checkpoint
        else run_dir / "weights" / "100.pth"
    )
    output = (
        repo_path(args.output_dir)
        if args.output_dir
        else run_dir / "xm_crossdomain_eval"
    )
    overrides = {
        "dfdc": (repo_path(args.dfdc_frames), repo_path(args.dfdc_labels)),
        "eval2024": (
            repo_path(args.eval2024_frames),
            repo_path(args.eval2024_labels),
        ),
    }

    all_frames, all_videos, audits = [], [], {}
    for name in args.benches:
        frames, videos, audit = discover_benchmark(name, *overrides[name])
        all_frames.append(frames)
        all_videos.append(videos)
        audits[name] = audit
        if is_main:
            print(
                f"{name}: videos={audit['videos']} present={audit['present_videos']} "
                f"missing={audit['missing_videos']} crops={audit['crops']}"
            )
    records = pd.concat(all_frames, ignore_index=True)
    videos = pd.concat(all_videos, ignore_index=True)

    if is_main:
        output.mkdir(parents=True, exist_ok=True)
        (output / "data_audit.json").write_text(
            json.dumps(audits, indent=2), encoding="utf-8"
        )
        videos.to_csv(output / "discovered_videos.csv", index=False)
    if args.dry_run:
        if world_size > 1:
            dist.barrier(device_ids=[device.index])
        if is_main:
            print(f"Dry run complete: {output / 'data_audit.json'}")
        return

    if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp bf16 requested, but this GPU does not support BF16")
    model = load_xm_model(config_path, checkpoint_path, device)

    records = records.copy()
    records["_global_index"] = np.arange(len(records), dtype=np.int64)
    shard_start = len(records) * rank // world_size
    shard_end = len(records) * (rank + 1) // world_size
    local_records = records.iloc[shard_start:shard_end].copy().reset_index(drop=True)
    description = "XM baseline cross-domain"
    if world_size > 1:
        description += f" [rank 0 shard, {len(local_records)}/{len(records)}]"
    local_predictions = infer_frames(
        model,
        local_records,
        args,
        device,
        description=description,
        show_progress=is_main,
    )
    predictions = merge_distributed_predictions(
        local_predictions, records, rank, world_size
    )
    if not is_main:
        if world_size > 1:
            dist.barrier(device_ids=[device.index])
        return
    if world_size > 1:
        print(f"Merged predictions from {world_size} GPUs: {len(predictions)}/{len(records)}")

    frame_scores, video_scores = aggregate(predictions, videos, args.missing_score)
    video_scores["prediction"] = (
        video_scores["p_fake"] >= args.threshold
    ).astype(np.int64)

    per_dataset, summary_rows, all_tables, present_tables = {}, [], [], []
    for name in args.benches:
        table = video_scores[video_scores.dataset == name].copy()
        present_table = table[table["present"]].copy()
        all_tables.append(table)
        present_tables.append(present_table)
        entry = {
            "coverage": audits[name],
            "all": binary_metrics(table, args.threshold),
            "present": binary_metrics(present_table, args.threshold),
        }
        per_dataset[name] = entry
        summary_rows.append({"dataset": name, "subset": "all", **entry["all"]})
        summary_rows.append(
            {"dataset": name, "subset": "present", **entry["present"]}
        )
    pooled_all = pd.concat(all_tables, ignore_index=True)
    pooled_present = pd.concat(present_tables, ignore_index=True)

    def overall_metrics(subset: str, pooled: pd.DataFrame) -> dict:
        return {
            "mean_auc": float(
                np.mean([per_dataset[name][subset]["auc"] for name in args.benches])
            ),
            "pooled_auc": binary_auc(
                pooled["label"].to_numpy(np.int64),
                pooled["p_fake"].to_numpy(np.float64),
            ),
            "mean_balanced_acc": float(
                np.mean(
                    [
                        per_dataset[name][subset]["balanced_acc"]
                        for name in args.benches
                    ]
                )
            ),
        }

    result = {
        "per_dataset": per_dataset,
        "overall_all": overall_metrics("all", pooled_all),
        "overall_present": overall_metrics("present", pooled_present),
        "threshold": float(args.threshold),
        "missing_score": float(args.missing_score),
        "protocol": {
            "name": "xm-baseline-all-and-present-video-level",
            "world_size": int(world_size),
            "aggregation": "frame=max-face, video=mean",
            "reported_subset": "all labeled videos; missing crops use missing_score",
            "reported_subsets": (
                "all labeled videos (missing crops use missing_score) and "
                "present videos with at least one crop"
            ),
            "input_size": int(args.input_size),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
        },
    }

    predictions.to_csv(output / "frame_predictions.csv", index=False)
    frame_scores.to_csv(output / "frame_scores.csv", index=False)
    video_scores.to_csv(output / "video_scores.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "summary.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print_results(result)
    print(f"Saved: {output}")
    if world_size > 1:
        dist.barrier(device_ids=[device.index])


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed(args)
    try:
        run(args, rank, world_size, device)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
