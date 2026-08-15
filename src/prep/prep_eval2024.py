"""Prepare the complete published TrueMedia eval2024 set for evaluation.

The script converts the published metadata to ``filename,label`` and extracts
uniformly spaced, square RetinaFace crops.  Output names retain the source
frame id so evaluators can max-pool faces before averaging frames.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_part(raw: str) -> tuple[int, int]:
    try:
        part, total = (int(value) for value in raw.split("/", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("part must look like 1/4") from exc
    if total < 1 or not 1 <= part <= total:
        raise argparse.ArgumentTypeError("part must satisfy 1 <= part <= total")
    return part, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--videos", default="data/2024/video-data", help="Downloaded video directory"
    )
    parser.add_argument(
        "--metadata",
        default="data/2024/video-metadata-publish-with-links.csv",
    )
    parser.add_argument("--output-frames", default="data/2024/frames")
    parser.add_argument("--output-labels", default="data/2024/labels.csv")
    parser.add_argument(
        "--split", choices=("test", "train", "all"), default="all"
    )
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.3,
        help="Square crop side relative to the detected face box",
    )
    parser.add_argument("--max-faces", type=int, default=2)
    parser.add_argument("--part", type=parse_part, default=parse_part("1/1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(metadata_path: Path, split: str) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    required = {"Filename", "Video Ground Truth", "Finetuning Set"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata lacks columns: {sorted(missing)}")
    if split != "all":
        metadata = metadata[
            metadata["Finetuning Set"].astype(str).str.lower() == split
        ]
    truth = metadata["Video Ground Truth"].astype(str).str.lower()
    if not truth.isin(("real", "fake")).all():
        unexpected = sorted(truth[~truth.isin(("real", "fake"))].unique())
        raise ValueError(f"unsupported video ground-truth values: {unexpected}")
    result = pd.DataFrame(
        {
            "filename": metadata["Filename"].astype(str),
            "label": truth.map({"real": 0, "fake": 1}).astype(np.int64),
        }
    )
    if result.filename.duplicated().any():
        raise ValueError("metadata contains duplicate filenames")
    return result.reset_index(drop=True)


def write_labels(records: pd.DataFrame, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(f".{filename.name}.{os.getpid()}.tmp")
    records.to_csv(temporary, index=False)
    os.replace(temporary, filename)


def square_crop(image: np.ndarray, bbox, scale: float, size: int) -> np.ndarray | None:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = (float(value) for value in bbox)
    side = max(x1 - x0, y1 - y0) * scale
    if not np.isfinite(side) or side < 2:
        return None
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    left = int(np.floor(center_x - side / 2))
    top = int(np.floor(center_y - side / 2))
    right = int(np.ceil(center_x + side / 2))
    bottom = int(np.ceil(center_y + side / 2))
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def process_video(
    model,
    video_path: Path,
    output_root: Path,
    *,
    num_frames: int,
    image_size: int,
    crop_scale: float,
    max_faces: int,
    overwrite: bool,
) -> tuple[int, int]:
    output = output_root / video_path.stem
    complete = output / ".complete"
    if complete.is_file() and not overwrite:
        return 0, 0
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        return 0, 1
    frame_indices = np.unique(
        np.linspace(0, frame_count - 1, min(num_frames, frame_count), dtype=np.int64)
    )
    saved = 0
    failed = 0
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = capture.read()
        if not ok:
            failed += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict_jsons(rgb)
        detections = [item for item in detections if "bbox" in item]
        detections.sort(
            key=lambda item: (item["bbox"][2] - item["bbox"][0])
            * (item["bbox"][3] - item["bbox"][1]),
            reverse=True,
        )
        if not detections:
            failed += 1
            continue
        for face_index, detection in enumerate(detections[:max_faces]):
            crop = square_crop(bgr, detection["bbox"], crop_scale, image_size)
            if crop is None:
                continue
            destination = output / f"{int(frame_index):06d}_{face_index:02d}.png"
            if overwrite or not destination.is_file():
                if not cv2.imwrite(str(destination), crop):
                    raise OSError(f"failed to write {destination}")
            saved += 1
    capture.release()
    complete.touch()
    return saved, failed


def main() -> None:
    args = parse_args()
    if args.num_frames < 1 or args.image_size < 1 or args.max_faces < 1:
        raise ValueError("num-frames, image-size, and max-faces must be positive")
    if args.crop_scale <= 0:
        raise ValueError("crop-scale must be positive")

    videos_root = Path(args.videos)
    metadata_path = Path(args.metadata)
    output_root = Path(args.output_frames)
    records = load_records(metadata_path, args.split)
    write_labels(records, Path(args.output_labels))
    part, total_parts = args.part
    records = records.iloc[(part - 1) :: total_parts].reset_index(drop=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    try:
        from retinaface.pre_trained_models import get_model
    except ImportError as exc:
        raise RuntimeError(
            "retinaface-pytorch is required; install requirements_full.txt "
            "or `pip install retinaface-pytorch==0.0.8`"
        ) from exc
    model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    model.eval()

    missing = 0
    crops = 0
    failed_frames = 0
    for row in tqdm(records.itertuples(index=False), total=len(records), unit="video"):
        video_path = videos_root / row.filename
        if not video_path.is_file():
            missing += 1
            continue
        saved, failed = process_video(
            model,
            video_path,
            output_root,
            num_frames=args.num_frames,
            image_size=args.image_size,
            crop_scale=args.crop_scale,
            max_faces=args.max_faces,
            overwrite=args.overwrite,
        )
        crops += saved
        failed_frames += failed
    print(
        f"part={part}/{total_parts} videos={len(records)} missing={missing} "
        f"crops={crops} failed_frames={failed_frames} labels={args.output_labels}"
    )


if __name__ == "__main__":
    main()
