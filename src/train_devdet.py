"""Four-stage DevDet-style training entry point.

Stages:
  select      score training frames and select hard-fake/easy-real for FFDev
  train-ffdev train the frozen-detector reconstruction generator
  fit-dict    extract hard-fake fused features and fit DoseDict
  train-daft  freeze FFDev/DoseDict and fine-tune GatedDual

No existing source file is imported and monkey-patched or modified on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))

from devdet import (  # noqa: E402
    DoseDictionary, FFDevGenerator, develop_image, fit_dictionary,
    gated_dual_features, resolve_dose_scale, total_variation,
)
from eval_baseline_3domain import (  # noqa: E402
    DiagnosticDataset, amp_context, build_model, discover_all,
)


def path(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() else (REPO / value).resolve()


def read_json(filename: Path) -> dict:
    with filename.open() as handle:
        return json.load(handle)


def atomic_torch_save(payload, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(filename)


def atomic_checkpoint_alias(source: Path, alias: Path) -> None:
    """Atomically make ``alias`` a hard link to a saved epoch checkpoint."""
    alias.parent.mkdir(parents=True, exist_ok=True)
    temporary = alias.with_suffix(alias.suffix + ".link.tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(alias)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict | None, device: torch.device) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def seed_worker(worker_id: int) -> None:
    import cv2

    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_setup(args) -> tuple[dict, dict, Path, Path, Path]:
    recipe = read_json(path(args.devdet_config))
    run_dir = REPO / "output" / recipe["baseline_run"]
    baseline_config = path(args.baseline_config) if args.baseline_config else run_dir / "config.json"
    baseline_checkpoint = (
        path(args.baseline_checkpoint) if args.baseline_checkpoint
        else path(recipe.get("baseline_checkpoint", run_dir / "weights" / "100.pth"))
    )
    output = path(args.output_dir) if args.output_dir else path(recipe["output_dir"])
    if not baseline_config.is_file():
        raise FileNotFoundError(f"Baseline config not found: {baseline_config}")
    if not baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_checkpoint}")
    cfg = read_json(baseline_config)
    cfg["in_chans"] = 3
    output.mkdir(parents=True, exist_ok=True)
    return recipe, cfg, baseline_config, baseline_checkpoint, output


def device_from(args) -> torch.device:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp bf16 requested, but this GPU does not support BF16; use fp16 or none")
    return device


def grad_scaler(device: torch.device, amp: str):
    return torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and amp == "fp16")
    )


class DevDetTrainingDataset(Dataset):
    """Paper-style image augmentations on top of deterministic aligned crops."""
    def __init__(self, records: pd.DataFrame, image_size: int):
        import albumentations as alb

        self.base = DiagnosticDataset(records.reset_index(drop=True), image_size)
        cutout = max(16, int(image_size) // 8)
        self.transform = alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.RandomBrightnessContrast(p=0.3),
            alb.HueSaturationValue(p=0.3),
            alb.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
            alb.GaussNoise(p=0.1),
            alb.MotionBlur(blur_limit=5, p=0.1),
            alb.ChannelShuffle(p=0.1),
            alb.CoarseDropout(
                max_holes=1, max_height=cutout, max_width=cutout,
                min_holes=1, min_height=16, min_width=16, p=0.2,
            ),
            alb.RandomGamma(p=0.2),
            alb.GlassBlur(sigma=0.7, max_delta=2, p=0.05),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        image, original_index = self.base[index]
        image_np = np.clip(
            image.permute(1, 2, 0).numpy() * 255.0, 0, 255
        ).astype(np.uint8)
        image_np = self.transform(image=image_np)["image"].astype(np.float32) / 255.0
        return torch.from_numpy(image_np.transpose(2, 0, 1)), original_index


def data_loader(
    records: pd.DataFrame, image_size: int, args,
    shuffle: bool = False, augment: bool = False,
):
    dataset = (
        DevDetTrainingDataset(records, image_size)
        if augment else DiagnosticDataset(records.reset_index(drop=True), image_size)
    )
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0, drop_last=False,
        worker_init_fn=seed_worker,
    )


def discover_selection_records(cfg: dict, recipe: dict) -> pd.DataFrame:
    records, _ = discover_all(cfg, "train", int(recipe.get("frames_per_video", 8)))
    records = records[records["landmark_exists"]].copy()
    if not {0, 1} <= set(records["label"]):
        raise ValueError("Discovered training set must contain both real and fake")
    return records.reset_index(drop=True)


def baseline_training_subset(records: pd.DataFrame) -> pd.DataFrame:
    """Keep the original baseline/DAFT policy: NB2 contributes fake only."""
    return records[
        ~((records["dataset"] == "new_benchmark_2") & (records["label"] == 0))
    ].reset_index(drop=True)


def discover_training_records(cfg: dict, recipe: dict) -> pd.DataFrame:
    records = baseline_training_subset(discover_selection_records(cfg, recipe))
    if not {0, 1} <= set(records["label"]):
        raise ValueError("Discovered training set must contain both real and fake")
    return records


def stable_sample_hash(row, seed: int) -> str:
    identity = "|".join([
        str(seed), str(row.dataset), str(row.video_id), str(row.frame_id),
        str(row.image_path),
    ])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def domain_matched_selection(scored: pd.DataFrame, selection: dict, seed: int):
    """Select symmetric low-confidence samples and a globally balanced real set.

    Hard fake means ``p_fake < T``. Hard real uses the true-class confidence,
    so ``p_real < T`` is equivalent to ``p_fake > 1-T``. Fake counts are
    treated as an invariant: a checkpoint/preprocessing change that changes a
    configured count fails loudly instead of silently padding the set with easy
    fakes. Real frames outside the hard condition are never used. When one
    domain lacks hard real frames for its target quota, a configured fallback
    domain supplies the shortfall. For exactly equal scores, one frame per video
    is preferred before repeats.
    """
    threshold = float(selection["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"selection.threshold must be in (0,1), got {threshold}")
    domain_specs = selection.get("domains")
    if not isinstance(domain_specs, dict) or not domain_specs:
        raise ValueError("selection.domains must be a non-empty mapping")
    fallback_domain = selection.get("real_shortfall_domain")
    if fallback_domain not in domain_specs:
        raise ValueError(
            "selection.real_shortfall_domain must name one of selection.domains"
        )

    fake_by_domain = {}
    real_pool_by_domain = {}
    real_by_domain = {}
    audit = {}
    total_real_shortfall = 0
    for domain, spec in domain_specs.items():
        group = scored[scored["dataset"] == domain]
        if group.empty:
            raise ValueError(f"Selection domain has no scored samples: {domain}")

        fake = group[(group["label"] == 1) & (group["p_fake"] < threshold)].copy()
        expected_fake = int(spec["expected_fake"])
        if len(fake) != expected_fake:
            raise RuntimeError(
                f"{domain}: expected exactly {expected_fake} fake frames with "
                f"p_fake < {threshold}, but found {len(fake)}. The checkpoint or "
                "preprocessing differs from the audited selection."
            )
        fake["_stable_hash"] = [stable_sample_hash(row, seed) for row in fake.itertuples()]
        fake = fake.sort_values(["p_fake", "_stable_hash"], ascending=[True, True])
        fake["selection_role"] = "hard_fake"
        fake["selection_is_hard"] = True
        fake_by_domain[domain] = fake

        real_quota = int(spec.get("real_quota", expected_fake))
        real_hard_cutoff = 1.0 - threshold
        real_pool = group[
            (group["label"] == 0) & (group["p_fake"] > real_hard_cutoff)
        ].copy()
        expected_hard_real = int(spec["expected_hard_real"])
        if len(real_pool) != expected_hard_real:
            raise RuntimeError(
                f"{domain}: expected exactly {expected_hard_real} real frames with "
                f"p_real < {threshold} (p_fake > {real_hard_cutoff}), but found "
                f"{len(real_pool)}. The checkpoint or preprocessing differs from "
                "the audited selection."
            )
        real_pool["_stable_hash"] = [
            stable_sample_hash(row, seed) for row in real_pool.itertuples()
        ]
        real_pool = real_pool.sort_values(
            ["p_fake", "_stable_hash"], ascending=[False, True], kind="mergesort"
        )
        real_pool["_video_round"] = real_pool.groupby(
            ["p_fake", "video_id"], sort=False
        ).cumcount()
        real_pool = real_pool.sort_values(
            ["p_fake", "_video_round", "_stable_hash"],
            ascending=[False, True, True], kind="mergesort",
        )
        base_count = min(real_quota, len(real_pool))
        real_by_domain[domain] = real_pool.head(base_count).copy()
        real_pool_by_domain[domain] = real_pool
        shortfall = real_quota - base_count
        total_real_shortfall += shortfall
        audit[domain] = {
            "fake_below_threshold": int(len(fake)),
            "fake_selected": int(len(fake)),
            "real_total": int((group["label"] == 0).sum()),
            "real_hard_rule": f"p_real < {threshold} (p_fake > {real_hard_cutoff})",
            "real_hard_candidates": int(len(real_pool)),
            "real_target_quota": real_quota,
            "real_shortfall": shortfall,
            "real_fallback_extra": 0,
        }

    if total_real_shortfall:
        fallback_pool = real_pool_by_domain[fallback_domain]
        already_selected = len(real_by_domain[fallback_domain])
        extra = fallback_pool.iloc[
            already_selected:already_selected + total_real_shortfall
        ].copy()
        if len(extra) != total_real_shortfall:
            raise RuntimeError(
                f"{fallback_domain}: cannot supply real shortfall {total_real_shortfall}; "
                f"only {len(extra)} additional hard real frames are available"
            )
        real_by_domain[fallback_domain] = pd.concat(
            [real_by_domain[fallback_domain], extra], ignore_index=True
        )
        audit[fallback_domain]["real_fallback_extra"] = total_real_shortfall

    pieces = []
    for domain in domain_specs:
        real = real_by_domain[domain]
        real["selection_role"] = "selected_real"
        real["selection_is_hard"] = True
        audit[domain]["real_selected"] = int(len(real))
        audit[domain]["real_hard_selected"] = int(len(real))
        pieces.extend([fake_by_domain[domain], real])

    selected = pd.concat(pieces, ignore_index=True)
    selected = selected.drop(columns=["_stable_hash", "_video_round"], errors="ignore")
    keys = ["dataset", "video_id", "frame_id"]
    if selected.duplicated(keys).any():
        raise RuntimeError("Domain-matched selection contains duplicate frame keys")
    expected_total = sum(
        int(spec["expected_fake"]) + int(spec.get("real_quota", spec["expected_fake"]))
        for spec in domain_specs.values()
    )
    if len(selected) != expected_total:
        raise RuntimeError(f"Expected {expected_total} selected frames, got {len(selected)}")
    return selected, audit


def validate_selection(output: Path, recipe: dict, checkpoint: Path) -> None:
    filename = output / "selection.json"
    if not filename.is_file():
        raise FileNotFoundError(f"Run select first: {filename}")
    metadata = read_json(filename)
    if metadata.get("recipe") != recipe:
        raise RuntimeError(
            f"DevDet config changed after selection. Re-run select or restore the config: {filename}"
        )
    if Path(metadata["baseline_checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("Selection and current baseline checkpoint do not match; re-run select")


def validate_ffdev_manifest(records: pd.DataFrame, recipe: dict) -> None:
    selection = recipe.get("selection", {})
    if selection.get("mode", "legacy_global") != "domain_matched_threshold_v1":
        return
    required = {
        "dataset", "video_id", "frame_id", "label", "p_fake", "selection_role",
    }
    missing = required - set(records.columns)
    if missing:
        raise RuntimeError(f"FFDev selection manifest lacks columns: {sorted(missing)}")
    if records.duplicated(["dataset", "video_id", "frame_id"]).any():
        raise RuntimeError("FFDev selection manifest contains duplicate frame keys")

    threshold = float(selection["threshold"])
    fallback_domain = selection["real_shortfall_domain"]
    expected_real_counts = {}
    total_shortfall = 0
    for domain, spec in selection["domains"].items():
        quota = int(spec.get("real_quota", spec["expected_fake"]))
        available = int(spec["expected_hard_real"])
        expected_real_counts[domain] = min(quota, available)
        total_shortfall += max(quota - available, 0)
    expected_real_counts[fallback_domain] += total_shortfall

    expected_total = 0
    for domain, spec in selection["domains"].items():
        group = records[records["dataset"] == domain]
        fake = group[group["selection_role"] == "hard_fake"]
        real = group[group["selection_role"] == "selected_real"]
        expected_fake = int(spec["expected_fake"])
        expected_real = expected_real_counts[domain]
        if len(fake) != expected_fake or len(real) != expected_real:
            raise RuntimeError(
                f"{domain} manifest count mismatch: fake={len(fake)}/{expected_fake}, "
                f"real={len(real)}/{expected_real}"
            )
        if not (fake["label"] == 1).all() or not (fake["p_fake"] < threshold).all():
            raise RuntimeError(
                f"{domain} hard_fake rows must have label=1 and p_fake < {threshold}"
            )
        if not (real["label"] == 0).all():
            raise RuntimeError(f"{domain} selected_real rows must have label=0")
        if not (real["p_fake"] > 1.0 - threshold).all():
            raise RuntimeError(
                f"{domain} selected_real rows must satisfy p_real < {threshold} "
                f"(p_fake > {1.0 - threshold})"
            )
        expected_total += expected_fake + expected_real
    if len(records) != expected_total:
        unexpected = sorted(set(records["dataset"]) - set(selection["domains"]))
        raise RuntimeError(
            f"FFDev manifest has {len(records)} rows, expected {expected_total}; "
            f"unexpected domains={unexpected}"
        )


@torch.no_grad()
def score_records(model, records, image_size, args, device) -> pd.DataFrame:
    loader = data_loader(records, image_size, args, shuffle=False)
    scores = np.empty(len(records), dtype=np.float32)
    done = 0
    model.eval()
    progress = tqdm(loader, desc="Baseline scoring", unit="batch", dynamic_ncols=True)
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


def select_stage(args) -> None:
    recipe, cfg, baseline_cfg, checkpoint, output = resolve_setup(args)
    device = device_from(args)
    selection = recipe.get("selection", {})
    selection_mode = str(selection.get("mode", "legacy_global"))
    if selection_mode == "domain_matched_threshold_v1":
        records = discover_selection_records(cfg, recipe)
    elif selection_mode == "legacy_global":
        records = discover_training_records(cfg, recipe)
    else:
        raise ValueError(f"Unknown selection.mode={selection_mode!r}")
    model = build_model(cfg, checkpoint, device)
    scored = score_records(model, records, int(cfg["image_size"]), args, device)
    if selection_mode == "domain_matched_threshold_v1":
        selected, selection_audit = domain_matched_selection(
            scored, selection, int(recipe.get("seed", 0))
        )
        expected_samples = int(recipe["ffdev"]["samples"])
        if len(selected) != expected_samples:
            raise RuntimeError(
                f"ffdev.samples={expected_samples}, but selection produced {len(selected)} frames"
            )
        scored.to_csv(output / "selection_scores_all.csv", index=False)
        baseline_scored = baseline_training_subset(scored)
        hard_fake = selected[selected["selection_role"] == "hard_fake"]
        selected_real = selected[selected["selection_role"] == "selected_real"]
    else:
        ff = recipe["ffdev"]
        total = min(int(ff["samples"]), len(scored))
        fake_count = min(
            round(total * float(ff["hard_fake_fraction"])),
            int((scored.label == 1).sum()),
        )
        real_count = min(total - fake_count, int((scored.label == 0).sum()))
        hard_fake = scored[scored.label == 1].nsmallest(fake_count, "p_fake")
        selected_real = scored[scored.label == 0].nsmallest(real_count, "p_fake")
        selected = pd.concat([hard_fake, selected_real], ignore_index=True)
        baseline_scored = scored
        selection_audit = {}

    baseline_scored.to_csv(output / "baseline_scores.csv", index=False)
    selected.to_csv(output / "ffdev_samples.csv", index=False)
    metadata = {
        "baseline_config": str(baseline_cfg), "baseline_checkpoint": str(checkpoint),
        "selection_mode": selection_mode,
        "selection_candidate_samples": len(scored),
        "baseline_candidate_samples": len(baseline_scored),
        "selected_samples": len(selected),
        "hard_fake": len(hard_fake), "selected_real": len(selected_real),
        "nb2_real_scope": (
            "ffdev_only" if selection_mode == "domain_matched_threshold_v1" else "excluded"
        ),
        "scoring": {
            "amp": args.amp, "batch_size": int(args.batch_size),
            "image_size": int(cfg["image_size"]),
        },
        "selection_audit": selection_audit, "recipe": recipe,
    }
    (output / "selection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {output / 'baseline_scores.csv'}")
    if selection_mode == "domain_matched_threshold_v1":
        print(f"Saved {output / 'selection_scores_all.csv'}")
        for domain, row in selection_audit.items():
            print(
                f"{domain}: fake={row['fake_selected']} real={row['real_selected']} "
                f"(hard-real={row['real_hard_selected']}, "
                f"fallback-extra={row['real_fallback_extra']})"
            )
    print(
        f"Saved {output / 'ffdev_samples.csv'} "
        f"({len(hard_fake)} hard fake + {len(selected_real)} selected real)"
    )


def make_generator(ff_cfg: dict, device: torch.device) -> FFDevGenerator:
    return FFDevGenerator(
        base_channels=int(ff_cfg["base_channels"]), blocks=int(ff_cfg["blocks"]),
        dropout=float(ff_cfg["dropout"]),
    ).to(device)


def train_ffdev_stage(args) -> None:
    recipe, cfg, _, checkpoint, output = resolve_setup(args)
    device = device_from(args)
    sample_file = output / "ffdev_samples.csv"
    validate_selection(output, recipe, checkpoint)
    if not sample_file.is_file():
        raise FileNotFoundError(f"Selection manifest missing: {sample_file}")
    records = pd.read_csv(sample_file)
    validate_ffdev_manifest(records, recipe)
    detector = build_model(cfg, checkpoint, device).eval()
    detector.requires_grad_(False)
    ff = recipe["ffdev"]
    generator = make_generator(ff, device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(ff["learning_rate"]), betas=(0.5, 0.999))
    scaler = grad_scaler(device, args.amp)
    args.batch_size = int(ff["batch_size"])
    loader = data_loader(
        records, int(cfg["image_size"]), args, shuffle=True,
        augment=bool(recipe.get("training_augmentation", True)),
    )
    epsilon, lambda_tv = float(ff["epsilon"]), float(ff["lambda_tv"])
    history = []
    for epoch in range(int(ff["epochs"])):
        generator.train()
        total = 0.0
        seen = 0
        progress = tqdm(
            loader, desc=f"FFDev {epoch + 1:03d}/{int(ff['epochs']):03d}",
            unit="batch", dynamic_ncols=True,
        )
        for images, indices in progress:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(records.iloc[indices.numpy()]["label"].to_numpy(), device=device).long()
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp):
                developed = develop_image(images, generator(images), epsilon)
                logits = detector(developed)
                classification = F.cross_entropy(logits, labels)
                tv = total_variation(developed)
                loss = classification + lambda_tv * tv
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().float().item() * len(images)
            seen += len(images)
            progress.set_postfix(
                loss=f"{total / max(seen, 1):.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )
        mean_loss = total / len(records)
        history.append(mean_loss)
        print(f"FFDev epoch {epoch + 1:03d}/{int(ff['epochs']):03d} loss={mean_loss:.6f}")
        atomic_torch_save({
            "generator": generator.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "recipe": recipe, "history": history,
        }, output / "ffdev.pth")


@torch.no_grad()
def extract_features(model, records, image_size, args, device) -> torch.Tensor:
    loader = data_loader(records, image_size, args, shuffle=False)
    result = []
    model.eval()
    done = 0
    progress = tqdm(loader, desc="DoseDict features", unit="batch", dynamic_ncols=True)
    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        with amp_context(device, args.amp):
            feature = gated_dual_features(model, images)
        result.append(feature.float().cpu())
        done += len(images)
        progress.set_postfix(samples=f"{done}/{len(records)}", refresh=False)
    return torch.cat(result)


def fit_dict_stage(args) -> None:
    recipe, cfg, _, checkpoint, output = resolve_setup(args)
    device = device_from(args)
    scores_file = output / "baseline_scores.csv"
    validate_selection(output, recipe, checkpoint)
    if not scores_file.is_file():
        raise FileNotFoundError(f"Selection scores missing: {scores_file}")
    dc = recipe["dictionary"]
    sample_source = str(dc.get("sample_source", "baseline_hardest"))
    if sample_source == "ffdev_hard_fake":
        selected_file = output / "ffdev_samples.csv"
        if not selected_file.is_file():
            raise FileNotFoundError(f"FFDev selection manifest missing: {selected_file}")
        selected = pd.read_csv(selected_file)
        validate_ffdev_manifest(selected, recipe)
        if "selection_role" not in selected:
            raise RuntimeError(
                "dictionary.sample_source=ffdev_hard_fake requires a threshold-selection manifest"
            )
        hard_fake = selected[selected["selection_role"] == "hard_fake"].copy()
        expected = int(dc["hard_fake_samples"])
        if len(hard_fake) != expected:
            raise RuntimeError(
                f"DoseDict expected {expected} selected hard fakes, got {len(hard_fake)}"
            )
        hard_fake = hard_fake.sort_values("p_fake").reset_index(drop=True)
    elif sample_source == "baseline_hardest":
        scored = pd.read_csv(scores_file)
        hard_fake = scored[scored.label == 1].nsmallest(
            int(dc["hard_fake_samples"]), "p_fake"
        ).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown dictionary.sample_source={sample_source!r}")
    model = build_model(cfg, checkpoint, device)
    args.batch_size = int(dc.get("feature_batch_size", 64))
    features = extract_features(model, hard_fake, int(cfg["image_size"]), args, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dictionary, history = fit_dictionary(
        features, atoms=int(dc["atoms"]), lasso_lambda=float(dc["lasso_lambda"]),
        ista_iterations=int(dc["ista_iterations"]), epochs=int(dc["epochs"]),
        batch_size=int(dc["batch_size"]), learning_rate=float(dc["learning_rate"]),
        l2_normalize=bool(dc["l2_normalize"]), seed=int(recipe["seed"]), device=device,
    )
    atomic_torch_save({
        "dictionary": dictionary.state_dict(), "fit_history": history,
        "feature_dim": features.shape[1], "samples": len(features), "recipe": recipe,
    }, output / "dose_dictionary.pth")
    print(f"Saved {output / 'dose_dictionary.pth'} ({len(features)} x {features.shape[1]})")


def load_ffdev(output: Path, recipe: dict, device: torch.device) -> FFDevGenerator:
    filename = output / "ffdev.pth"
    if not filename.is_file():
        raise FileNotFoundError(f"Run train-ffdev first: {filename}")
    payload = torch.load(filename, map_location="cpu", weights_only=True)
    if payload.get("recipe") != recipe:
        raise RuntimeError("FFDev checkpoint was created with a different DevDet config")
    generator = make_generator(recipe["ffdev"], device)
    generator.load_state_dict(payload["generator"])
    return generator.eval().requires_grad_(False)


def load_dictionary(output: Path, recipe: dict) -> DoseDictionary:
    filename = output / "dose_dictionary.pth"
    if not filename.is_file():
        raise FileNotFoundError(f"Run fit-dict first: {filename}")
    payload = torch.load(filename, map_location="cpu", weights_only=True)
    if payload.get("recipe") != recipe:
        raise RuntimeError("DoseDict was created with a different DevDet config")
    return DoseDictionary.from_state_dict(payload["dictionary"])


def configure_daft_trainable(detector, scope: str):
    scope = str(scope).lower()
    if scope == "classifier":
        detector.requires_grad_(False)
        detector.fc.requires_grad_(True)
    elif scope == "all":
        # Preserve the trainability policy established by GatedDualDetector.
        pass
    else:
        raise ValueError(f"Unknown daft.train_scope={scope!r}; use 'classifier' or 'all'")
    trainable = [parameter for parameter in detector.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"DAFT train_scope={scope!r} produced no trainable parameters")
    return scope, trainable


def set_daft_train_mode(detector, scope: str) -> None:
    if scope == "classifier":
        # Keep frozen BN/dropout/stochastic-depth behavior identical to baseline.
        detector.eval()
        detector.fc.train()
    else:
        detector.train()


def train_daft_stage(args) -> None:
    recipe, cfg, _, checkpoint, output = resolve_setup(args)
    device = device_from(args)
    scores_file = output / "baseline_scores.csv"
    validate_selection(output, recipe, checkpoint)
    if not scores_file.is_file():
        raise FileNotFoundError(f"Selection scores missing: {scores_file}")
    records = pd.read_csv(scores_file)
    daft = recipe["daft"]
    if not bool(daft.get("enabled", True)):
        raise RuntimeError(
            "DAFT is disabled in this recipe. Use --detector-mode baseline for evaluation."
        )
    maximum = int(daft.get("max_samples", 0))
    if maximum and len(records) > maximum:
        # Stable class-balanced subset for a controlled first reconstruction.
        half = maximum // 2
        records = pd.concat([
            records[records.label == 0].sample(min(half, int((records.label == 0).sum())), random_state=recipe["seed"]),
            records[records.label == 1].sample(min(maximum - half, int((records.label == 1).sum())), random_state=recipe["seed"]),
        ], ignore_index=True)
    detector = build_model(cfg, checkpoint, device)
    generator = load_ffdev(output, recipe, device)
    dictionary = load_dictionary(output, recipe)
    dictionary.atoms = dictionary.atoms.to(device)
    train_scope, trainable = configure_daft_trainable(
        detector, daft.get("train_scope", "all")
    )
    trainable_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.SGD(
        trainable, lr=float(daft["learning_rate"]), momentum=float(daft["momentum"]),
        weight_decay=float(daft["weight_decay"]),
    )
    scaler = grad_scaler(device, args.amp)
    if args.additional_epochs is not None and not args.resume:
        raise ValueError("--additional-epochs requires --resume")
    start_epoch = 0
    history = []
    dose_history = []
    resume_rng_state = None
    if args.resume:
        resume_path = (
            output / "daft_detector.pth"
            if args.resume == "latest" else path(args.resume)
        )
        if not resume_path.is_file():
            raise FileNotFoundError(f"DAFT resume checkpoint not found: {resume_path}")
        # DAFT checkpoints contain optimizer and Python/NumPy/Torch RNG state,
        # so the PyTorch 2.6 weights-only loader cannot deserialize them.
        # Resume only accepts checkpoints generated by this local pipeline.
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if resume_payload.get("recipe") != recipe:
            raise RuntimeError("DAFT resume checkpoint was created with a different config")
        saved_scope = str(resume_payload.get("train_scope", "all")).lower()
        if saved_scope != train_scope:
            raise RuntimeError(
                f"DAFT resume train_scope={saved_scope!r}, current={train_scope!r}"
            )
        detector.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        if "scaler" in resume_payload:
            scaler.load_state_dict(resume_payload["scaler"])
        start_epoch = int(
            resume_payload.get("epoch_number", int(resume_payload["epoch"]) + 1)
        )
        history = list(resume_payload.get("history", []))
        dose_history = list(resume_payload.get("dose_history", []))
        resume_rng_state = resume_payload.get("rng_state")
        if len(history) != start_epoch or len(dose_history) != start_epoch:
            raise RuntimeError(
                f"DAFT resume history length mismatch: epoch={start_epoch}, "
                f"history={len(history)}, dose_history={len(dose_history)}"
            )
        additional_epochs = int(args.additional_epochs or 0)
        if additional_epochs <= 0:
            raise ValueError("Resuming DAFT requires --additional-epochs with a positive value")
        epochs_to_run = additional_epochs
        print(
            f"Resuming DAFT from {resume_path}: completed={start_epoch}, "
            f"additional={epochs_to_run}"
        )
        del resume_payload
    else:
        epochs_to_run = int(daft["epochs"])
    final_epoch = start_epoch + epochs_to_run
    if bool(daft.get("save_every_epoch", False)):
        conflicts = [
            output / f"daft_detector_epoch_{number:03d}.pth"
            for number in range(start_epoch + 1, final_epoch + 1)
            if (output / f"daft_detector_epoch_{number:03d}.pth").exists()
        ]
        if conflicts:
            raise FileExistsError(
                f"Refusing to overwrite existing DAFT epoch checkpoint: {conflicts[0]}"
            )
    class_counts = records["label"].value_counts().to_dict()
    if set(class_counts) != {0, 1}:
        raise ValueError(f"DAFT requires both classes, got counts={class_counts}")
    if bool(daft.get("class_balance", True)):
        class_weights = torch.tensor([
            len(records) / (2.0 * class_counts[0]),
            len(records) / (2.0 * class_counts[1]),
        ], device=device, dtype=torch.float32)
    else:
        class_weights = None
    print(
        f"DAFT samples real={class_counts[0]} fake={class_counts[1]} "
        f"class_weights={class_weights.tolist() if class_weights is not None else None} "
        f"train_scope={train_scope} trainable={trainable_count / 1e6:.3f}M"
    )
    args.batch_size = int(daft["batch_size"])
    loader = data_loader(
        records, int(cfg["image_size"]), args, shuffle=True,
        augment=bool(recipe.get("training_augmentation", True)),
    )
    restore_rng_state(resume_rng_state, device)
    dose_scale = resolve_dose_scale(daft, float(recipe["ffdev"]["epsilon"]))
    print(f"DAFT dose_mode={daft['dose_mode']} scale={dose_scale:.6f}")
    for epoch_offset in range(epochs_to_run):
        epoch_number = start_epoch + epoch_offset + 1
        total = 0.0
        seen = 0
        epoch_doses = []
        progress = tqdm(
            loader, desc=f"DAFT {epoch_number:03d}/{final_epoch:03d}",
            unit="batch", dynamic_ncols=True,
        )
        for images, indices in progress:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(records.iloc[indices.numpy()]["label"].to_numpy(), device=device).long()
            # Paper does not disclose whether dose features come from a frozen copy.
            # Use the live detector detached, which avoids a second 300M-param model.
            detector.eval()
            with torch.no_grad(), amp_context(device, args.amp):
                dose_feature = gated_dual_features(detector, images)
                dose = dictionary.dose(dose_feature) * dose_scale
                developed = develop_image(images, generator(images), dose)
            epoch_doses.append(dose.float().cpu())
            set_daft_train_mode(detector, train_scope)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp):
                logits = detector(developed)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach().float().item() * len(images)
            seen += len(images)
            progress.set_postfix(
                loss=f"{total / max(seen, 1):.6f}",
                dose=f"{dose.float().mean().item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )
        mean_loss = total / len(records)
        history.append(mean_loss)
        all_doses = torch.cat(epoch_doses)
        quantiles = torch.quantile(
            all_doses, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])
        ).tolist()
        dose_stats = dict(zip(("min", "p05", "p50", "p95", "max"), quantiles))
        dose_history.append(dose_stats)
        print(
            f"DAFT epoch {epoch_number:03d}/{final_epoch:03d} loss={mean_loss:.6f} "
            f"dose[p05/p50/p95]={dose_stats['p05']:.4f}/{dose_stats['p50']:.4f}/{dose_stats['p95']:.4f}"
        )
        # The model key is intentionally compatible with the existing evaluators.
        checkpoint_payload = {
            "model": detector.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch_number - 1, "epoch_number": epoch_number, "recipe": recipe,
            "train_scope": train_scope, "history": history,
            "dose_history": dose_history,
            "rng_state": capture_rng_state(device),
            "ffdev": str(output / "ffdev.pth"), "dictionary": str(output / "dose_dictionary.pth"),
        }
        latest = output / "daft_detector.pth"
        if bool(daft.get("save_every_epoch", False)):
            epoch_checkpoint = output / f"daft_detector_epoch_{epoch_number:03d}.pth"
            atomic_torch_save(checkpoint_payload, epoch_checkpoint)
            atomic_checkpoint_alias(epoch_checkpoint, latest)
            print(f"Saved DAFT epoch checkpoint: {epoch_checkpoint}")
        else:
            atomic_torch_save(checkpoint_payload, latest)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("stage", choices=("select", "train-ffdev", "fit-dict", "train-daft"))
    parser.add_argument("--devdet-config", default="configs/devdet_gated_dual.json")
    parser.add_argument("--baseline-config")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=64, help="select-stage batch size")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--resume", nargs="?", const="latest",
        help="train-daft only: checkpoint path, or omit the value to use latest",
    )
    parser.add_argument(
        "--additional-epochs", type=int,
        help="train-daft only: number of extra epochs to run after --resume",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage != "train-daft" and (args.resume or args.additional_epochs is not None):
        raise ValueError("--resume/--additional-epochs are only valid for train-daft")
    recipe = read_json(path(args.devdet_config))
    seed_everything(int(recipe.get("seed", 0)))
    dispatch = {
        "select": select_stage, "train-ffdev": train_ffdev_stage,
        "fit-dict": fit_dict_stage, "train-daft": train_daft_stage,
    }
    dispatch[args.stage](args)


if __name__ == "__main__":
    main()
