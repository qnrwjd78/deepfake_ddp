"""Frame-level three-domain evaluation through the full DevDet path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from devdet import (  # noqa: E402
    DevDetInference, DoseDictionary, FFDevGenerator, resolve_dose_scale,
)
from eval_baseline_3domain import (  # noqa: E402
    DOMAINS, discover_all, repo_path, run_inference,
)


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
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(predictions: pd.DataFrame, threshold: float) -> dict:
    if not np.isfinite(predictions["p_fake"].to_numpy()).all():
        raise ValueError("NaN or Inf score detected")
    result = {}
    for domain in DOMAINS:
        group = predictions[predictions.dataset == domain]
        y = group.label.to_numpy(np.int64)
        p = group.p_fake.to_numpy(np.float64)
        real, fake = y == 0, y == 1
        if not real.any() or not fake.any():
            raise ValueError(f"{domain} does not contain both real and fake samples")
        f_acc = float((p[fake] >= threshold).mean())
        r_acc = float((p[real] < threshold).mean())
        result[domain] = {
            "n_real": int(real.sum()), "n_fake": int(fake.sum()),
            "f_acc": f_acc, "r_acc": r_acc,
            "balanced_acc": 0.5 * (f_acc + r_acc), "auc": binary_auc(y, p),
        }
    all_y = predictions.label.to_numpy(np.int64)
    all_p = predictions.p_fake.to_numpy(np.float64)
    return {
        "per_domain": result,
        "m_acc_3": float(np.mean([result[name]["balanced_acc"] for name in DOMAINS])),
        "s_auc_3": binary_auc(all_y, all_p), "threshold": float(threshold),
        "n_frames": len(predictions),
    }


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--devdet-config", default="configs/devdet_gated_dual.json")
    parser.add_argument("--baseline-config", help="Defaults from baseline_run in DevDet config")
    parser.add_argument("--detector", help="Defaults to DEVDET_OUTPUT/daft_detector.pth")
    parser.add_argument("--ffdev", help="Defaults to DEVDET_OUTPUT/ffdev.pth")
    parser.add_argument("--dictionary", help="Defaults to DEVDET_OUTPUT/dose_dictionary.pth")
    parser.add_argument("--output-dir", help="Defaults to DEVDET_OUTPUT/eval_3domain")
    parser.add_argument("--ff-split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def build_detector(cfg: dict, payload: dict, device: torch.device):
    from model import GatedDualDetector

    model = GatedDualDetector(cfg)
    state = payload.get("model", payload)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    args = parse_args()
    recipe_path = repo_path(args.devdet_config)
    recipe = json.loads(recipe_path.read_text())
    devdet_output = repo_path(recipe["output_dir"])
    baseline_config = (
        repo_path(args.baseline_config) if args.baseline_config
        else REPO / "output" / recipe["baseline_run"] / "config.json"
    )
    detector_path = repo_path(args.detector) if args.detector else devdet_output / "daft_detector.pth"
    ffdev_path = repo_path(args.ffdev) if args.ffdev else devdet_output / "ffdev.pth"
    dictionary_path = (
        repo_path(args.dictionary) if args.dictionary
        else devdet_output / "dose_dictionary.pth"
    )
    output = repo_path(args.output_dir) if args.output_dir else devdet_output / "eval_3domain"
    for label, filename in (
        ("baseline config", baseline_config), ("DAFT detector", detector_path),
        ("FFDev", ffdev_path), ("DoseDict", dictionary_path),
    ):
        if not filename.is_file():
            raise FileNotFoundError(f"{label} not found: {filename}")
    cfg = json.loads(baseline_config.read_text())
    cfg["in_chans"] = 3
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp bf16 requested, but this GPU does not support BF16")
    detector_payload = torch.load(
        detector_path, map_location="cpu", weights_only=False
    )
    if detector_payload.get("recipe") != recipe:
        raise RuntimeError("DAFT detector was created with a different DevDet config")
    detector = build_detector(cfg, detector_payload, device)
    del detector_payload
    ff_cfg = recipe["ffdev"]
    generator = FFDevGenerator(
        base_channels=int(ff_cfg["base_channels"]), blocks=int(ff_cfg["blocks"]),
        dropout=float(ff_cfg["dropout"]),
    )
    ffdev_payload = torch.load(
        ffdev_path, map_location="cpu", weights_only=True
    )
    if ffdev_payload.get("recipe") != recipe:
        raise RuntimeError("FFDev was created with a different DevDet config")
    generator.load_state_dict(ffdev_payload["generator"])
    del ffdev_payload
    generator = generator.to(device).eval()
    dictionary_payload = torch.load(
        dictionary_path, map_location="cpu", weights_only=True
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
    dose_scale = resolve_dose_scale(recipe["daft"], float(ff_cfg["epsilon"]))
    model = DevDetInference(detector, generator, dictionary, dose_scale).to(device).eval()

    records, _ = discover_all(cfg, args.ff_split, args.frames_per_video)
    coverage = {
        domain: {
            "selected": int((records.dataset == domain).sum()),
            "usable": int(((records.dataset == domain) & records.landmark_exists).sum()),
        }
        for domain in DOMAINS
    }
    records = records[records.landmark_exists].reset_index(drop=True)
    predictions = run_inference(
        model, records, int(cfg["image_size"]), device, args.amp,
        args.batch_size, args.num_workers,
    )
    predictions["prediction"] = (predictions["p_fake"] >= args.threshold).astype(np.int64)
    result = metrics(predictions, args.threshold)
    result["coverage"] = coverage
    result["protocol"] = {
        "name": "devdet-3domain-training-domain-diagnostic",
        "warning": "Not an unseen generalization evaluation",
        "ff_split": args.ff_split,
        "frames_per_video": args.frames_per_video,
        "score": "softmax(logits)[:,1] after adaptive FFDev",
        "devdet_config": str(recipe_path),
        "baseline_config": str(baseline_config),
        "detector": str(detector_path),
        "ffdev": str(ffdev_path),
        "dictionary": str(dictionary_path),
    }
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = pd.DataFrame.from_dict(result["per_domain"], orient="index")
    summary.index.name = "dataset"
    summary.to_csv(output / "summary.csv")

    print("\nWARNING: this is a training-domain diagnostic, not unseen generalization evaluation.")
    print("\nDataset             Real   Fake  F-ACC  R-ACC  B-ACC    AUC")
    print("-" * 66)
    for domain in DOMAINS:
        row = result["per_domain"][domain]
        print(
            f"{domain:<19} {row['n_real']:>5}  {row['n_fake']:>5}  "
            f"{row['f_acc']:.3f}  {row['r_acc']:.3f}  "
            f"{row['balanced_acc']:.3f}  {row['auc']:.3f}"
        )
    print(f"M-ACC-3={result['m_acc_3']:.3f}  S-AUC-3={result['s_auc_3']:.3f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
