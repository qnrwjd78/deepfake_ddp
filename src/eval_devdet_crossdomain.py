"""Evaluate DevDet components on every labeled DFDC/eval2024 video.

Aggregation follows the existing DFDC protocol: maximum score across faces of
the same frame, followed by the mean across frames of a video.  Videos without
crops receive 0.5. Metrics are reported for both ALL labeled videos and PRESENT
videos that have at least one crop.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

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

from devdet import (  # noqa: E402
    DevDetInference, DoseDictionary, FFDevGenerator, resolve_dose_scale,
)
from eval_baseline_3domain import amp_context  # noqa: E402
from eval_devdet_3domain import binary_auc, build_detector  # noqa: E402


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


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--devdet-config", default="configs/devdet_gated_dual.json")
    parser.add_argument("--baseline-config", help="Defaults from baseline_run in DevDet config")
    parser.add_argument(
        "--detector-mode", choices=("daft", "baseline"), default="daft",
        help="Use the DAFT detector or the original frozen baseline detector",
    )
    parser.add_argument(
        "--detector",
        help="Optional detector checkpoint override for the selected detector mode",
    )
    parser.add_argument("--ffdev", help="Defaults to DEVDET_OUTPUT/ffdev.pth")
    parser.add_argument("--dictionary", help="Defaults to DEVDET_OUTPUT/dose_dictionary.pth")
    parser.add_argument(
        "--output-dir",
        help="Defaults to a detector-mode-specific directory under DEVDET_OUTPUT",
    )
    parser.add_argument("--benches", type=parse_benches, default=parse_benches("dfdc,eval2024"))
    parser.add_argument("--dfdc-frames", default=DEFAULT_BENCHES["dfdc"][0])
    parser.add_argument("--dfdc-labels", default=DEFAULT_BENCHES["dfdc"][1])
    parser.add_argument("--eval2024-frames", default=DEFAULT_BENCHES["eval2024"][0])
    parser.add_argument("--eval2024-labels", default=DEFAULT_BENCHES["eval2024"][1])
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--missing-score", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU process")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers per GPU process")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--local-rank", "--local_rank", type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)), help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true", help="Audit data without loading models")
    return parser.parse_args()


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


def artifact_paths(args, recipe: dict):
    output = repo_path(recipe["output_dir"])
    baseline_config = (
        repo_path(args.baseline_config) if args.baseline_config
        else REPO / "output" / recipe["baseline_run"] / "config.json"
    )
    baseline_checkpoint = repo_path(
        recipe.get(
            "baseline_checkpoint",
            REPO / "output" / recipe["baseline_run"] / "weights" / "100.pth",
        )
    )
    if args.detector:
        detector = repo_path(args.detector)
    elif args.detector_mode == "baseline":
        detector = baseline_checkpoint
    else:
        detector = output / "daft_detector.pth"
    default_eval_dir = (
        "crossdomain_eval_baseline_detector"
        if args.detector_mode == "baseline" else "crossdomain_eval"
    )
    return {
        "baseline_config": baseline_config,
        "baseline_checkpoint": baseline_checkpoint,
        "selection": output / "selection.json",
        "detector": detector,
        "ffdev": repo_path(args.ffdev) if args.ffdev else output / "ffdev.pth",
        "dictionary": repo_path(args.dictionary) if args.dictionary else output / "dose_dictionary.pth",
        "output": repo_path(args.output_dir) if args.output_dir else output / default_eval_dir,
    }


def require_same_path(label: str, first: str | Path, second: str | Path) -> None:
    first_path = repo_path(first)
    second_path = repo_path(second)
    if first_path != second_path:
        raise RuntimeError(f"{label} mismatch: {first_path} != {second_path}")


def validate_baseline_detector(recipe: dict, paths: dict) -> None:
    if not paths["selection"].is_file():
        raise FileNotFoundError(
            f"Selection metadata not found: {paths['selection']}; run train_devdet.py select first"
        )
    selection = read_json(paths["selection"])
    if selection.get("recipe") != recipe:
        raise RuntimeError("Selection metadata was created with a different DevDet config")
    require_same_path(
        "Selection baseline checkpoint",
        selection.get("baseline_checkpoint", ""), paths["detector"],
    )
    require_same_path(
        "Recipe baseline checkpoint", paths["baseline_checkpoint"], paths["detector"]
    )
    require_same_path(
        "Selection baseline config",
        selection.get("baseline_config", ""), paths["baseline_config"],
    )


def load_devdet(recipe: dict, paths: dict, device: torch.device, detector_mode: str):
    for label in ("baseline_config", "detector", "ffdev", "dictionary"):
        if not paths[label].is_file():
            raise FileNotFoundError(f"{label} not found: {paths[label]}")
    cfg = read_json(paths["baseline_config"])
    cfg["in_chans"] = 3

    if detector_mode == "baseline":
        validate_baseline_detector(recipe, paths)
    # DAFT checkpoints are produced locally by train_devdet.py and contain
    # optimizer/RNG metadata in addition to tensors.  PyTorch 2.6 defaults
    # torch.load() to weights_only=True, which rejects that trusted metadata.
    # Keep the safer restricted loader for the original baseline checkpoint.
    detector_payload = torch.load(
        paths["detector"], map_location="cpu", weights_only=(detector_mode != "daft")
    )
    if detector_mode == "daft":
        if not isinstance(detector_payload, dict) or detector_payload.get("recipe") != recipe:
            raise RuntimeError("DAFT detector was created with a different DevDet config")
    elif detector_mode == "baseline":
        pass
    else:
        raise ValueError(f"Unknown detector_mode={detector_mode!r}")
    detector = build_detector(cfg, detector_payload, device)
    detector.requires_grad_(False)
    del detector_payload

    ff_cfg = recipe["ffdev"]
    generator = FFDevGenerator(
        base_channels=int(ff_cfg["base_channels"]), blocks=int(ff_cfg["blocks"]),
        dropout=float(ff_cfg["dropout"]),
    )
    ffdev_payload = torch.load(
        paths["ffdev"], map_location="cpu", weights_only=True
    )
    if ffdev_payload.get("recipe") != recipe:
        raise RuntimeError("FFDev was created with a different DevDet config")
    generator.load_state_dict(ffdev_payload["generator"])
    generator = generator.to(device).eval().requires_grad_(False)
    del ffdev_payload

    dictionary_payload = torch.load(
        paths["dictionary"], map_location="cpu", weights_only=True
    )
    if dictionary_payload.get("recipe") != recipe:
        raise RuntimeError("DoseDict was created with a different DevDet config")
    if int(dictionary_payload["feature_dim"]) != int(detector.fc.in_features):
        raise ValueError(
            f"DoseDict feature_dim={dictionary_payload['feature_dim']} but detector expects "
            f"{detector.fc.in_features}"
        )
    dictionary = DoseDictionary.from_state_dict(dictionary_payload["dictionary"])
    dictionary.atoms = dictionary.atoms.to(device)
    del dictionary_payload

    scale = resolve_dose_scale(recipe["daft"], float(ff_cfg["epsilon"]))
    return DevDetInference(detector, generator, dictionary, scale).to(device).eval()


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


def run(args, rank: int, world_size: int, device: torch.device):
    is_main = rank == 0
    recipe_path = repo_path(args.devdet_config)
    recipe = read_json(recipe_path)
    paths = artifact_paths(args, recipe)
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
        if is_main:
            print(
                f"{name}: videos={audit['videos']} present={audit['present_videos']} "
                f"missing={audit['missing_videos']} crops={audit['crops']}"
            )
    records = pd.concat(all_frames, ignore_index=True)
    videos = pd.concat(all_videos, ignore_index=True)
    if is_main:
        paths["output"].mkdir(parents=True, exist_ok=True)
        (paths["output"] / "data_audit.json").write_text(
            json.dumps(audits, indent=2), encoding="utf-8"
        )
        videos.to_csv(paths["output"] / "discovered_videos.csv", index=False)
    if args.dry_run:
        if world_size > 1:
            dist.barrier(device_ids=[device.index])
        if is_main:
            print(f"Dry run complete: {paths['output'] / 'data_audit.json'}")
        return

    if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp bf16 requested, but this GPU does not support BF16")
    model = load_devdet(recipe, paths, device, args.detector_mode)
    description = (
        "Baseline + FFDev/Dose cross-domain"
        if args.detector_mode == "baseline" else "DevDet DAFT cross-domain"
    )
    records = records.copy()
    records["_global_index"] = np.arange(len(records), dtype=np.int64)
    shard_start = len(records) * rank // world_size
    shard_end = len(records) * (rank + 1) // world_size
    local_records = records.iloc[shard_start:shard_end].copy().reset_index(drop=True)
    if world_size > 1:
        description += f" [rank 0 shard, {len(local_records)}/{len(records)}]"
    local_predictions = infer_frames(
        model, local_records, args, device, description=description,
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
    video_scores["prediction"] = (video_scores["p_fake"] >= args.threshold).astype(np.int64)

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
            "mean_balanced_acc": float(np.mean([
                per_dataset[name][subset]["balanced_acc"] for name in args.benches
            ])),
        }

    result = {
        "per_dataset": per_dataset,
        "overall_all": overall_metrics("all", pooled_all),
        "overall_present": overall_metrics("present", pooled_present),
        "threshold": float(args.threshold),
        "missing_score": float(args.missing_score),
        "protocol": {
            "name": f"devdet-{args.detector_mode}-all-and-present-video-level",
            "detector_mode": args.detector_mode,
            "world_size": int(world_size),
            "aggregation": "frame=max-face, video=mean",
            "score": (
                "softmax(logits)[:,1] after DoseDict + FFDev + original baseline detector"
                if args.detector_mode == "baseline"
                else "softmax(logits)[:,1] after DoseDict + FFDev + DAFT detector"
            ),
            "input_size": int(args.input_size),
            "dose_mode": str(recipe["daft"].get("dose_mode", "direct")),
            "dose_scale": float(resolve_dose_scale(
                recipe["daft"], float(recipe["ffdev"]["epsilon"])
            )),
            "reported_subset": "all labeled videos; missing crops use missing_score",
            "reported_subsets": (
                "all labeled videos (missing crops use missing_score) and "
                "present videos with at least one crop"
            ),
            "devdet_config": str(recipe_path),
            **{key: str(value) for key, value in paths.items() if key != "output"},
        },
    }
    predictions.to_csv(paths["output"] / "frame_predictions.csv", index=False)
    frame_scores.to_csv(paths["output"] / "frame_scores.csv", index=False)
    video_scores.to_csv(paths["output"] / "video_scores.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(paths["output"] / "summary.csv", index=False)
    (paths["output"] / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print_results(result)
    print(f"Saved: {paths['output']}")
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
