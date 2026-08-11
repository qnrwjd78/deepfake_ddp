"""Diagnostic frame-level evaluation on the three baseline training domains.

Domains:
  1. FaceForensics++ (real plus every unique configured fake source)
  2. newbench (real and fake from its labels CSV)
  3. new_benchmark_2 (real and fake; its real class was not used for training)

This is deliberately a frame-level diagnostic: there is no video aggregation,
no test-time augmentation, and all three domains share one threshold.  The
script discovers the data sources from output/<run>/config.json, selects frames
uniformly per video, applies deterministic landmark-based 5-point alignment,
and writes both per-frame scores and summary metrics. Retina coverage is audited
but is not required because new_benchmark_2 real was excluded from training and
therefore has no corresponding Retina files.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

DOMAINS = ("FF++", "newbench", "new_benchmark_2")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline_gated_dual on FF++, newbench, and new_benchmark_2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", default="baseline_gated_dual")
    parser.add_argument("--config", help="Defaults to output/RUN/config.json")
    parser.add_argument("--checkpoint", help="Defaults to output/RUN/weights/100.pth")
    parser.add_argument("--output-dir", help="Defaults to output/RUN/baseline_3domain_eval")
    parser.add_argument(
        "--ff-split", choices=("train", "val", "test", "all"), default="train",
        help="FF++ source IDs to diagnose; train matches the samples seen during training",
    )
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--allow-missing-modalities", action="store_true",
        help="Continue when selected frames lack landmarks; those frames are excluded",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover and validate samples without loading the checkpoint or running inference",
    )
    return parser.parse_args()


def repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def read_label_map(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels CSV does not exist: {path}")
    labels: dict[str, int] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"filename", "label"} <= set(reader.fieldnames):
            raise ValueError(f"{path} must contain filename,label columns")
        for row in reader:
            video_id = Path(row["filename"]).stem
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid label for {video_id} in {path}: {label}")
            if video_id in labels and labels[video_id] != label:
                raise ValueError(f"Conflicting labels for {video_id} in {path}")
            labels[video_id] = label
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def list_images(video_dir: Path) -> list[Path]:
    return sorted(
        (path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: frame_sort_key(path.stem),
    )


def frame_sort_key(value: str) -> tuple[int, object]:
    prefix = value.split("_", 1)[0]
    try:
        return (0, int(prefix))
    except ValueError:
        return (1, value)


def select_uniform(paths: list[Path], count: int) -> list[Path]:
    if count <= 0:
        raise ValueError("--frames-per-video must be positive")
    if len(paths) <= count:
        return paths
    indices = [round(index) for index in np.linspace(0, len(paths) - 1, count)]
    return [paths[index] for index in indices]


def modality_paths(image_path: Path) -> tuple[Path, Path]:
    parts = list(image_path.parts)
    try:
        frame_index = len(parts) - 1 - parts[::-1].index("frames")
    except ValueError as exc:
        raise ValueError(f"Image is not under a frames directory: {image_path}") from exc
    landmark_parts = list(parts)
    retina_parts = list(parts)
    landmark_parts[frame_index] = "landmarks"
    retina_parts[frame_index] = "retina"
    landmark = Path(*landmark_parts).with_suffix(".npy")
    retina = Path(*retina_parts).with_suffix(".npy")
    return landmark, retina


def ff_split_ids(split: str) -> set[str] | None:
    if split == "all":
        return None
    split_path = REPO / "data" / "FaceForensics++" / f"{split}.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"FF++ split does not exist: {split_path}")
    with split_path.open() as handle:
        pairs = json.load(handle)
    return {str(video_id).zfill(3) for pair in pairs for video_id in pair} - {"281", "604"}


def make_record(
    domain: str,
    label: int,
    video_id: str,
    image_path: Path,
    source: str,
) -> dict:
    landmark_path, retina_path = modality_paths(image_path)
    return {
        "dataset": domain,
        "label": int(label),
        "video_id": str(video_id),
        "frame_id": image_path.stem,
        "source": source,
        "image_path": str(image_path.resolve()),
        "landmark_path": str(landmark_path.resolve()),
        "retina_path": str(retina_path.resolve()),
        "landmark_exists": landmark_path.is_file(),
        "retina_exists": retina_path.is_file(),
    }


def resolve_ffpp_root(configured_value: str) -> tuple[Path, str]:
    """Resolve the config layout or the copied *_margin/train layout."""
    configured = repo_path(configured_value)
    if configured.is_dir():
        return configured, "config"

    marker = Path("data_precrop") / "FaceForensics++"
    try:
        relative = configured.relative_to(REPO / marker)
    except ValueError:
        relative = None
    if relative is not None:
        remapped = REPO / "data_precrop" / "FaceForensics++_margin" / "train" / relative
        if remapped.is_dir():
            return remapped.resolve(), "FaceForensics++_margin/train remap"

        # The copied bundle can use a different compression folder (currently
        # NeuralTextures/raw instead of the config's NeuralTextures/c23).
        if len(remapped.parents) >= 2:
            method_root = remapped.parent.parent
            alternatives = sorted(path for path in method_root.glob("*/frames") if path.is_dir())
            if len(alternatives) == 1:
                return alternatives[0].resolve(), "FaceForensics++_margin compression fallback"

    raise FileNotFoundError(
        f"FF++ frame root does not exist: configured={configured}; "
        "also tried data_precrop/FaceForensics++_margin/train"
    )


def discover_ffpp(
    cfg: dict, split: str, frames_per_video: int
) -> tuple[list[dict], list[dict]]:
    allowed_ids = ff_split_ids(split)
    records: list[dict] = []
    roots: list[tuple[Path, int, str]] = []
    path_audit: list[dict] = []
    real_root, resolution = resolve_ffpp_root(cfg["real_frame_path"])
    path_audit.append({
        "configured": cfg["real_frame_path"], "resolved": str(real_root), "resolution": resolution
    })
    roots.append((real_root, 0, "youtube-real"))

    # Repeated fake paths are training weights, not distinct evaluation samples.
    seen_fake_roots: set[Path] = set()
    for raw_path in cfg.get("fake_frame_paths", []):
        root, resolution = resolve_ffpp_root(raw_path)
        if root in seen_fake_roots:
            continue
        seen_fake_roots.add(root)
        path_audit.append({
            "configured": raw_path, "resolved": str(root), "resolution": resolution
        })
        try:
            method_index = root.parts.index("manipulated_sequences") + 1
            source = root.parts[method_index]
        except (ValueError, IndexError):
            source = root.parent.name
        roots.append((root, 1, source))

    for root, label, source in roots:
        for video_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            # This intentionally matches init_ff(): fake IDs such as 012_026 are
            # filtered by their first source ID.
            if allowed_ids is not None and video_dir.name[:3] not in allowed_ids:
                continue
            images = select_uniform(list_images(video_dir), frames_per_video)
            unique_video_id = video_dir.name if label == 0 else f"{source}/{video_dir.name}"
            records.extend(
                make_record("FF++", label, unique_video_id, image, source) for image in images
            )
    return records, path_audit


def discover_labeled_source(
    domain: str,
    frame_root: Path,
    labels_path: Path,
    frames_per_video: int,
) -> tuple[list[dict], dict]:
    if not frame_root.is_dir():
        raise FileNotFoundError(f"{domain} frame root does not exist: {frame_root}")
    labels = read_label_map(labels_path)
    records: list[dict] = []
    no_frame_dir: list[str] = []
    unknown_dirs: list[str] = []

    dirs = {path.name: path for path in frame_root.iterdir() if path.is_dir()}
    for video_id, label in labels.items():
        video_dir = dirs.get(video_id)
        if video_dir is None:
            no_frame_dir.append(video_id)
            continue
        images = select_uniform(list_images(video_dir), frames_per_video)
        records.extend(
            make_record(domain, label, video_id, image, domain) for image in images
        )
    for video_id in dirs:
        if video_id not in labels:
            unknown_dirs.append(video_id)
    audit = {
        "labels": len(labels),
        "label_real": sum(label == 0 for label in labels.values()),
        "label_fake": sum(label == 1 for label in labels.values()),
        "missing_frame_dirs": len(no_frame_dir),
        "unknown_frame_dirs": len(unknown_dirs),
        "missing_frame_dir_examples": no_frame_dir[:10],
        "unknown_frame_dir_examples": unknown_dirs[:10],
    }
    return records, audit


def discover_combined_labeled_source(
    domain: str,
    combined_root: Path,
    labels_path: Path,
    frames_per_video: int,
) -> tuple[list[dict], dict]:
    """Read one logical dataset from new_benchmark_margin's real/fake union."""
    real_root = combined_root / "real" / "frames"
    fake_root = combined_root / "fake" / "frames"
    if not real_root.is_dir() or not fake_root.is_dir():
        raise FileNotFoundError(
            f"Combined layout requires real/frames and fake/frames under {combined_root}"
        )
    labels = read_label_map(labels_path)
    real_dirs = {path.name: path for path in real_root.iterdir() if path.is_dir()}
    fake_dirs = {path.name: path for path in fake_root.iterdir() if path.is_dir()}
    records: list[dict] = []
    missing: list[str] = []
    wrong_class: list[str] = []
    for video_id, label in labels.items():
        expected_dirs = real_dirs if label == 0 else fake_dirs
        other_dirs = fake_dirs if label == 0 else real_dirs
        video_dir = expected_dirs.get(video_id)
        if video_dir is None:
            if video_id in other_dirs:
                wrong_class.append(video_id)
            else:
                missing.append(video_id)
            continue
        images = select_uniform(list_images(video_dir), frames_per_video)
        records.extend(
            make_record(domain, label, video_id, image, domain) for image in images
        )
    audit = {
        "layout": "combined new_benchmark_margin/train/{real,fake}/frames",
        "combined_root": str(combined_root.resolve()),
        "labels": len(labels),
        "label_real": sum(label == 0 for label in labels.values()),
        "label_fake": sum(label == 1 for label in labels.values()),
        "missing_frame_dirs": len(missing),
        "wrong_class_dirs": len(wrong_class),
        "missing_frame_dir_examples": missing[:10],
        "wrong_class_dir_examples": wrong_class[:10],
    }
    return records, audit


def discover_all(cfg: dict, ff_split: str, frames_per_video: int) -> tuple[pd.DataFrame, dict]:
    extra = cfg.get("extra_train_sources", [])
    if len(extra) < 2:
        raise ValueError("Config must contain newbench and new_benchmark_2 extra_train_sources")

    by_name: dict[str, dict] = {}
    for source in extra:
        frame_text = str(source.get("frames", ""))
        if "new_benchmark_2" in frame_text:
            by_name["new_benchmark_2"] = source
        elif "newbench" in frame_text or "new_benchmark" in frame_text:
            by_name["newbench"] = source
    missing_sources = {"newbench", "new_benchmark_2"} - set(by_name)
    if missing_sources:
        raise ValueError(f"Could not identify extra sources in config: {sorted(missing_sources)}")

    records, ff_path_audit = discover_ffpp(cfg, ff_split, frames_per_video)
    audit: dict = {"FF++": {"split": ff_split, "paths": ff_path_audit}}
    combined_root = REPO / "data_precrop" / "new_benchmark_margin" / "train"
    for domain in ("newbench", "new_benchmark_2"):
        source = by_name[domain]
        configured_frames = repo_path(source["frames"])
        if configured_frames.is_dir():
            found, source_audit = discover_labeled_source(
                domain, configured_frames, repo_path(source["labels"]), frames_per_video
            )
            source_audit["layout"] = "config flat frames"
            source_audit["frame_root"] = str(configured_frames)
        elif combined_root.is_dir():
            found, source_audit = discover_combined_labeled_source(
                domain, combined_root, repo_path(source["labels"]), frames_per_video
            )
            source_audit["configured_frame_root"] = str(configured_frames)
        else:
            raise FileNotFoundError(
                f"{domain} frame root does not exist: {configured_frames}; "
                f"combined fallback also missing: {combined_root}"
            )
        records.extend(found)
        source_audit["training_fake_only"] = bool(source.get("fake_only", False))
        audit[domain] = source_audit

    if not records:
        raise ValueError("No images were discovered")
    df = pd.DataFrame(records)
    duplicate_key = ["dataset", "video_id", "frame_id"]
    duplicated = df.duplicated(duplicate_key, keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, duplicate_key].head(10).to_dict("records")
        raise ValueError(f"Duplicate frame records exist: {examples}")
    return df, audit


class DiagnosticDataset(Dataset):
    def __init__(self, records: pd.DataFrame, image_size: int):
        self.records = records.reset_index(drop=True)
        self.image_size = int(image_size)
        destination = np.array(
            [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
             [33.5493, 92.3655], [62.7299, 92.2041]],
            dtype=np.float32,
        )
        destination[:, 0] += 8.0
        destination *= self.image_size / 112.0
        margin = self.image_size * 0.3 / 2.0
        destination += np.array([margin, margin], dtype=np.float32)
        destination *= self.image_size / (self.image_size + 2.0 * margin)
        self.destination = destination

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import cv2
        from utils.funcs import crop_face

        row = self.records.iloc[index]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        landmark_all = np.load(row["landmark_path"])
        landmark = np.asarray(landmark_all[0] if landmark_all.ndim == 3 else landmark_all).copy()
        if landmark.shape[0] < 68 or landmark.shape[1] < 2:
            raise ValueError(f"Invalid landmark shape {landmark.shape}: {row['landmark_path']}")

        image, landmark, _, _ = crop_face(
            image, landmark, None, margin=True, crop_by_bbox=False, phase="test"
        )
        source5 = landmark[[37, 44, 30, 49, 55], :2].astype(np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(source5, self.destination, method=cv2.LMEDS)
        if matrix is None:
            raise ValueError(f"Could not estimate alignment transform: {row['image_path']}")
        image = cv2.warpAffine(
            image, matrix, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR
        )
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image.transpose(2, 0, 1)), index


def build_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    from model import GatedDualDetector

    if cfg.get("model") != "gated_dual":
        raise ValueError(f"Expected model='gated_dual', got {cfg.get('model')!r}")
    model = GatedDualDetector(cfg)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def amp_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return contextlib.nullcontext()
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_inference(
    model: torch.nn.Module,
    records: pd.DataFrame,
    image_size: int,
    device: torch.device,
    amp: str,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    loader = DataLoader(
        DiagnosticDataset(records, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    logits_all = np.empty((len(records), 2), dtype=np.float32)
    processed = 0
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            with amp_context(device, amp):
                logits = model(images)
            indices_np = indices.numpy()
            logits_all[indices_np] = logits.float().cpu().numpy()
            processed += len(indices_np)
            print(f"\rInference: {processed}/{len(records)}", end="", flush=True)
    print()
    result = records.copy()
    result["logit_real"] = logits_all[:, 0]
    result["logit_fake"] = logits_all[:, 1]
    result["p_fake"] = torch.from_numpy(logits_all).softmax(dim=1)[:, 1].numpy()
    return result


def score_group(group: pd.DataFrame, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score

    labels = group["label"].to_numpy(dtype=np.int64)
    scores = group["p_fake"].to_numpy(dtype=np.float64)
    fake = labels == 1
    real = labels == 0
    if not fake.any() or not real.any():
        raise ValueError("Metric group lacks real or fake samples")
    f_acc = float(np.mean(scores[fake] >= threshold))
    r_acc = float(np.mean(scores[real] < threshold))
    return {
        "n_fake": int(fake.sum()),
        "n_real": int(real.sum()),
        "f_acc": f_acc,
        "r_acc": r_acc,
        "balanced_acc": 0.5 * (f_acc + r_acc),
        "auc": float(roc_auc_score(labels, scores)),
        "mean_p_fake_real": float(np.mean(scores[real])),
        "mean_p_fake_fake": float(np.mean(scores[fake])),
        "median_p_fake_real": float(np.median(scores[real])),
        "median_p_fake_fake": float(np.median(scores[fake])),
    }


def compute_metrics(predictions: pd.DataFrame, threshold: float, coverage: dict) -> dict:
    from sklearn.metrics import roc_auc_score

    if not np.isfinite(predictions["p_fake"]).all():
        raise ValueError("NaN or Inf score detected")
    per_domain = {
        domain: score_group(predictions[predictions["dataset"] == domain], threshold)
        for domain in DOMAINS
    }
    labels = predictions["label"].to_numpy(dtype=np.int64)
    scores = predictions["p_fake"].to_numpy(dtype=np.float64)
    method_f_acc = {}
    ff_fake = predictions[(predictions["dataset"] == "FF++") & (predictions["label"] == 1)]
    for source, group in ff_fake.groupby("source"):
        method_f_acc[source] = {
            "n_fake": int(len(group)),
            "f_acc": float(np.mean(group["p_fake"].to_numpy() >= threshold)),
            "mean_p_fake": float(group["p_fake"].mean()),
        }
    return {
        "per_domain": per_domain,
        "m_acc_3": float(np.mean([per_domain[name]["balanced_acc"] for name in DOMAINS])),
        "s_auc_3": float(roc_auc_score(labels, scores)),
        "threshold": float(threshold),
        "n_frames": int(len(predictions)),
        "n_ties_at_threshold": int(np.sum(scores == threshold)),
        "score_quantiles": {
            str(q): float(np.quantile(scores, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "ffpp_fake_method_diagnostic": method_f_acc,
        "coverage": coverage,
    }


def print_discovery(records: pd.DataFrame, coverage: dict) -> None:
    print("\nDiscovered frame-level diagnostic samples")
    print("Dataset             Real   Fake   Selected  Usable  Missing-LM  Missing-Retina")
    print("-" * 79)
    for domain in DOMAINS:
        group = records[records["dataset"] == domain]
        cov = coverage[domain]
        print(
            f"{domain:<19} {(group['label'] == 0).sum():>5}  {(group['label'] == 1).sum():>5}  "
            f"{len(group):>8}  {cov['usable']:>6}  {cov['missing_landmark']:>10}  "
            f"{cov['missing_retina']:>14}"
        )


def print_metrics(metrics: dict) -> None:
    print("\nDataset             N-Fake  N-Real   F-ACC   R-ACC   B-ACC     AUC")
    print("-" * 72)
    for domain in DOMAINS:
        row = metrics["per_domain"][domain]
        print(
            f"{domain:<19} {row['n_fake']:>6}  {row['n_real']:>6}  "
            f"{row['f_acc']:>6.4f}  {row['r_acc']:>6.4f}  "
            f"{row['balanced_acc']:>6.4f}  {row['auc']:>6.4f}"
        )
    print("-" * 72)
    print(f"Overall                                  M-ACC-3={metrics['m_acc_3']:.4f}  S-AUC-3={metrics['s_auc_3']:.4f}")


def main() -> None:
    args = parse_args()
    run_dir = REPO / "output" / args.run
    config_path = repo_path(args.config) if args.config else run_dir / "config.json"
    checkpoint_path = repo_path(args.checkpoint) if args.checkpoint else run_dir / "weights" / "100.pth"
    output_dir = repo_path(args.output_dir) if args.output_dir else run_dir / "baseline_3domain_eval"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    with config_path.open() as handle:
        cfg = json.load(handle)
    cfg["in_chans"] = 3

    records, audit = discover_all(cfg, args.ff_split, args.frames_per_video)
    coverage = {}
    for domain in DOMAINS:
        group = records[records["dataset"] == domain]
        usable = group["landmark_exists"]
        coverage[domain] = {
            "selected": int(len(group)),
            "usable": int(usable.sum()),
            "missing_landmark": int((~group["landmark_exists"]).sum()),
            "missing_retina": int((~group["retina_exists"]).sum()),
        }
    print_discovery(records, coverage)

    usable_mask = records["landmark_exists"]
    missing_count = int((~usable_mask).sum())
    if missing_count and not args.allow_missing_modalities:
        raise RuntimeError(
            f"{missing_count} selected frames lack landmark files. "
            "Finish landmark preprocessing, or pass --allow-missing-modalities to exclude them."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_dir / "discovered_samples.csv", index=False)
    with (output_dir / "data_audit.json").open("w") as handle:
        json.dump({"sources": audit, "coverage": coverage}, handle, indent=2, ensure_ascii=False)
    if args.dry_run:
        print(f"\nDry run complete: {output_dir / 'discovered_samples.csv'}")
        return

    usable = records[usable_mask].copy().reset_index(drop=True)
    for domain in DOMAINS:
        labels = set(usable.loc[usable["dataset"] == domain, "label"].tolist())
        if labels != {0, 1}:
            raise ValueError(f"{domain} usable frames do not contain both classes: {sorted(labels)}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = build_model(cfg, checkpoint_path, device)
    predictions = run_inference(
        model,
        usable,
        image_size=int(cfg.get("image_size", 384)),
        device=device,
        amp=args.amp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    predictions["prediction"] = (predictions["p_fake"] >= args.threshold).astype(np.int64)
    metrics = compute_metrics(predictions, args.threshold, coverage)
    metrics["protocol"] = {
        "name": "baseline-3domain-frame-diagnostic",
        "domains": list(DOMAINS),
        "ff_split": args.ff_split,
        "frames_per_video": args.frames_per_video,
        "frame_sampling": "uniform_linspace",
        "score": "softmax(logits)[:,1]",
        "video_aggregation": "none",
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    summary = pd.DataFrame.from_dict(metrics["per_domain"], orient="index")
    summary.index.name = "dataset"
    summary.to_csv(output_dir / "summary.csv")
    print_metrics(metrics)
    print(f"\nPredictions: {output_dir / 'predictions.csv'}")
    print(f"Metrics:     {output_dir / 'metrics.json'}")
    print(f"Summary:     {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
