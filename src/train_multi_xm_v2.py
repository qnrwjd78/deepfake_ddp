"""Standalone synchronized-SAM + cross-method metric training in v2.

The existing v2 ``train_multi.py``, model, and dataset are intentionally not
modified. This entry point reproduces the original XM run using the normalized
v2 precrop paths.
"""

import os

XM_WORKSPACE_ROOT = os.environ.get("HOMEDIR", "/workspace")
os.environ.setdefault("HF_HOME", f"{XM_WORKSPACE_ROOT}/.cache/huggingface")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import argparse
import glob
import json
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model_xm import GatedDualDetectorXM
from utils.cross_method_xm import (
    build_pair_masks,
    cross_method_triplet_loss,
    gather_no_grad,
    gather_with_grad,
)
from utils.funcs import load_json
from utils.logs import log
from utils.sbi_xm_v2 import SBI_Multi_XM_V2_Dataset
from utils.scheduler import CosineDecayLR, LinearDecayLR


def _fmt_hms(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    import cv2

    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _stack_images(fake_images, real_images):
    fake = torch.as_tensor(np.asarray(fake_images), dtype=torch.float32).transpose(0, 1)
    real = torch.as_tensor(np.asarray(real_images), dtype=torch.float32).transpose(0, 1)
    stacked = torch.cat([real, fake], dim=0)
    sources, item_batch = stacked.shape[:2]
    return stacked.reshape(sources * item_batch, *stacked.shape[2:]), sources, item_batch


def _stack_meta(fake_meta, real_meta):
    fake = torch.as_tensor(np.asarray(fake_meta), dtype=torch.long).transpose(0, 1)
    real = torch.as_tensor(np.asarray(real_meta), dtype=torch.long).transpose(0, 1)
    stacked = torch.cat([real, fake], dim=0)
    return stacked.reshape(-1, stacked.shape[-1])


def _balanced_selection(
    sources,
    item_batch,
    target_batch,
    *,
    rank=0,
    world_size=1,
    generator=None,
):
    """Plan one source-aware *global* batch and return this rank's shard.

    A loader item contains one example from every source.  We first allocate
    ``global_batch // sources`` examples to every source, then assign one
    additional example to a random subset of sources for the remainder.
    Finally, the shared plan is shuffled and split evenly across DDP ranks.

    For the v2 recipe (global batch 40, 9 sources), every rank loads five
    candidate items.  The resulting global batch contains four examples from
    every source (36 total) plus one extra example from four randomly chosen
    sources, then each of four ranks receives ten examples.
    """
    sources = int(sources)
    item_batch = int(item_batch)
    target_batch = int(target_batch)
    rank = int(rank)
    world_size = int(world_size)
    if sources <= 0 or item_batch <= 0 or target_batch <= 0:
        raise ValueError("sources, item_batch, and target_batch must be positive")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid DDP rank/world pair: {rank}/{world_size}")
    if target_batch % world_size:
        raise ValueError(
            f"global batch {target_batch} must be divisible by world size {world_size}"
        )
    if target_batch < sources:
        raise ValueError(
            f"global batch {target_batch} cannot include all {sources} sources"
        )

    labels = torch.arange(sources).repeat_interleave(item_batch)
    total = sources * item_batch
    if total < target_batch:
        raise ValueError(
            f"candidate pool has {total} samples, fewer than global batch {target_batch}"
        )

    base = target_batch // sources
    remainder = target_batch % sources
    if base > item_batch:
        raise ValueError(
            f"need {base} candidates per source, but loader provides {item_batch}"
        )

    extra_sources = torch.randperm(sources, generator=generator)[:remainder]
    take_per_source = torch.full((sources,), base, dtype=torch.long)
    take_per_source[extra_sources] += 1
    selected = []
    for source in range(sources):
        source_indices = torch.arange(
            source * item_batch, (source + 1) * item_batch
        )
        source_indices = source_indices[
            torch.randperm(item_batch, generator=generator)
        ]
        selected.append(source_indices[: int(take_per_source[source])])
    selected = torch.cat(selected)
    selected = selected[
        torch.randperm(selected.numel(), generator=generator)
    ]
    local_batch = target_batch // world_size
    start = rank * local_batch
    return labels, selected[start : start + local_batch]


def make_collate_xm(batch_size, dual_view=False, *, rank=0, world_size=1):
    batch_counters = {}

    def collate(batch):
        if dual_view:
            img_fs, img_r, img_ws, img_rw, meta_fs, meta_r = zip(*batch)
        else:
            img_fs, img_r, meta_fs, meta_r = zip(*batch)

        images, sources, item_batch = _stack_images(img_fs, img_r)
        metadata = _stack_meta(meta_fs, meta_r)
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else -1
        batch_index = batch_counters.get(worker_id, 0)
        batch_counters[worker_id] = batch_index + 1
        # Keep the source plan identical across ranks even if data transforms
        # consume different amounts of the worker's default torch RNG.
        planner = torch.Generator(device="cpu")
        planner.manual_seed(
            (torch.initial_seed() + batch_index * 0x9E3779B1) % (2**63 - 1)
        )
        labels, selected = _balanced_selection(
            sources,
            item_batch,
            batch_size,
            rank=rank,
            world_size=world_size,
            generator=planner,
        )

        if images.shape[0] != metadata.shape[0] or images.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"collate alignment failure: images={images.shape[0]} "
                f"meta={metadata.shape[0]} labels={labels.shape[0]}"
            )

        result = {
            "img": images.index_select(0, selected),
            "label": labels.index_select(0, selected),
            "xm_meta": metadata.index_select(0, selected),
        }
        if dual_view:
            wide, wide_sources, wide_item_batch = _stack_images(img_ws, img_rw)
            if (wide_sources, wide_item_batch) != (sources, item_batch):
                raise RuntimeError("tight/wide source layout mismatch")
            result["img_wide"] = wide.index_select(0, selected)
        return result

    return collate


def _all_ranks_finite(value, ddp):
    finite = torch.isfinite(value.detach()).to(dtype=torch.int32)
    if ddp:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def _build_scheduler(optimizer, cfg, epochs):
    name = cfg["name"]
    if name == "LinearDecayLR":
        return LinearDecayLR(optimizer, epochs)
    if name == "CosineDecayLR":
        return CosineDecayLR(optimizer, epochs)
    raise ValueError(f"unsupported scheduler {name!r}")


def main(args):
    cfg = load_json(args.config)
    seed = args.seed
    cfg["seed"] = seed
    cfg["label_smoothing"] = args.label_smoothing
    cfg["xm_schema_version"] = 2
    cfg["xm_batching"] = "global_even_random_remainder_v1"
    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        local_rank = rank = 0
        device = torch.device("cuda", 0)
    is_main = rank == 0

    if cfg.get("model") != "gated_dual":
        raise ValueError("standalone XM trainer currently supports model='gated_dual' only")
    if cfg.get("dual_view", False):
        raise ValueError("standalone GatedDual XM trainer supports dual_view=false only")
    xm_cfg = cfg.get("cross_method_loss", {})
    if not xm_cfg.get("enabled", False):
        raise ValueError("cross_method_loss.enabled must be true for this trainer")
    xm_weight = float(xm_cfg.get("weight", 0.0))
    xm_margin = float(xm_cfg.get("margin", 0.2))
    known_methods = list(xm_cfg.get("known_methods", []))
    if xm_weight <= 0.0:
        raise ValueError(f"cross_method_loss.weight must be positive, got {xm_weight}")
    if not 0.0 <= xm_margin <= 2.0:
        raise ValueError(f"cross_method_loss.margin must be in [0,2], got {xm_margin}")
    if len(known_methods) < 2 or len(known_methods) != len(set(known_methods)):
        raise ValueError("known_methods must contain at least two unique method names")
    if not cfg.get("sam_first_pass_sync", False):
        raise ValueError("XM requires sam_first_pass_sync=true")

    global_batch = int(cfg["batch_size"])
    if global_batch % world_size != 0:
        raise ValueError(
            f"global batch {global_batch} must be divisible by world size {world_size}"
        )
    local_batch = global_batch // world_size
    if local_batch <= 1:
        raise ValueError(f"local batch must exceed one, got {local_batch}")

    device_name = torch.cuda.get_device_name(local_rank)
    major_cc = torch.cuda.get_device_capability(local_rank)[0]
    use_amp = major_cc >= 8 and torch.cuda.is_bf16_supported()
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if is_main:
        print(
            f"XM TRAIN | device={device_name} world={world_size} "
            f"global_batch={global_batch} local_batch={local_batch} "
            f"bf16={use_amp} margin={xm_margin} lambda={xm_weight}"
        )

    cfg["in_chans"] = 3
    fake_frame_paths = cfg["fake_frame_paths"]
    expected = cfg.get("expected_fake_domains")
    if expected is not None and len(fake_frame_paths) != expected:
        raise ValueError(
            f"expected {expected} fake domains exactly once, got {len(fake_frame_paths)}"
        )
    if not cfg.get("precropped", False):
        raise ValueError("v2 XM training requires precropped=true")
    if not cfg.get("merge_extra_fake_sources", False):
        raise ValueError("v2 XM training requires merge_extra_fake_sources=true")
    extra_sources = list(cfg.get("extra_train_sources") or [])
    if len(extra_sources) != 1:
        raise ValueError(
            "v2 XM must expose newbench+NB2 as one logical merged extra source"
        )
    merged_members = list(extra_sources[0].get("merged_labeled_sources", []))
    member_names = {str(member.get("name")) for member in merged_members}
    if member_names != {"newbench", "new_benchmark_2"}:
        raise ValueError(
            "The merged XM source must contain named newbench and new_benchmark_2 members"
        )
    members_by_name = {
        str(member.get("name")): member for member in merged_members
    }
    if members_by_name["newbench"].get("fake_only"):
        raise ValueError("XM must keep newbench real samples in the training pool")
    if not members_by_name["new_benchmark_2"].get("fake_only"):
        raise ValueError("XM must use new_benchmark_2 as a fake-only member")

    dataset = SBI_Multi_XM_V2_Dataset(
        phase="train",
        image_size=cfg["image_size"],
        fake_frame_paths=fake_frame_paths,
        fake_shift=cfg["fake_shift"],
        real_frame_path=cfg.get("real_frame_path"),
        align_crop=cfg.get("align_crop", False),
        extra_sources=extra_sources,
        dual_view=cfg.get("dual_view", False),
        wide_size=cfg.get("global_img_size", 224),
        precropped=cfg.get("precropped", False),
        xm_known_methods=known_methods,
        trusted_extra_real=xm_cfg.get("trusted_extra_real", True),
    )

    source_count = 2 + len(dataset.fake_image_lists)
    expected_layout = cfg.get("expected_xm_layout", {})
    expected_sources = int(expected_layout.get("source_count", 9))
    if source_count != expected_sources:
        raise RuntimeError(
            f"XM source layout changed: got {source_count}, expected {expected_sources}. "
            "newbench and NB2 fakes must remain one logical extra source."
        )
    expected_real = expected_layout.get("real_frames")
    if expected_real is not None and len(dataset) != int(expected_real):
        raise RuntimeError(
            f"XM real pool changed: got {len(dataset)}, expected {int(expected_real)}"
        )
    expected_fake_lengths = expected_layout.get("fake_source_frames")
    actual_fake_lengths = [len(items) for items in dataset.fake_image_lists]
    if expected_fake_lengths is not None:
        expected_fake_lengths = [int(value) for value in expected_fake_lengths]
        if actual_fake_lengths != expected_fake_lengths:
            raise RuntimeError(
                f"XM fake pools changed: got {actual_fake_lengths}, "
                f"expected {expected_fake_lengths}"
            )
    if is_main:
        print(
            f"XM v2 data verified: real={len(dataset)} "
            f"fake_sources={actual_fake_lengths} source_count={source_count}"
        )
    if ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
    else:
        sampler = None

    cpu_total = os.cpu_count() or 8
    jobs_per_node = max(1, int(os.environ.get("TRAIN_JOBS_PER_NODE", "1")))
    default_workers = max(4, min(12, (cpu_total - 4) // jobs_per_node))
    num_workers = int(os.environ.get("NUM_WORKERS", default_workers))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if is_main:
        print(
            f"DataLoader workers={num_workers}, sources={source_count}, "
            f"items_per_loader_batch={(global_batch - 1) // source_count + 1} "
            "(global source-aware selection)"
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=(global_batch - 1) // source_count + 1,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        collate_fn=make_collate_xm(
            global_batch,
            dual_view=cfg.get("dual_view", False),
            rank=rank,
            world_size=world_size,
        ),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )

    raw_model = GatedDualDetectorXM(cfg).to(device)
    total_params = sum(p.numel() for p in raw_model.parameters())
    if is_main:
        print(f"Total params: {total_params / 1e6:.2f}M")

    epochs = int(cfg["epoch"])
    scheduler = _build_scheduler(raw_model.optimizer, cfg["scheduler"], epochs)
    save_path = f'output/{args.session_name}_gated_dual_xm/'
    existing_checkpoints = glob.glob(os.path.join(save_path, "weights", "*.pth"))
    if existing_checkpoints and not args.resume:
        raise FileExistsError(
            f"{save_path} already contains checkpoints; choose a new -n name or use --resume"
        )
    if is_main:
        os.makedirs(os.path.join(save_path, "weights"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
        with open(os.path.join(save_path, "config.json"), "w") as handle:
            json.dump(cfg, handle, indent=4)
        logger = log(path=os.path.join(save_path, "logs") + "/", file="losses.txt")
    else:
        logger = None

    start_epoch = 0
    if args.resume:
        if args.resume == "auto":
            checkpoints = glob.glob(os.path.join(save_path, "weights", "*.pth"))
            if not checkpoints:
                raise FileNotFoundError(f"no checkpoint in {save_path}weights/")
            resume_path = max(
                checkpoints,
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
            )
        else:
            resume_path = args.resume
        checkpoint = torch.load(
            resume_path, map_location=device, weights_only=True
        )
        raw_model.load_state_dict(checkpoint["model"], strict=True)
        raw_model.optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if is_main:
            message = (
                f"RESUME {resume_path} -> epoch {start_epoch + 1}/{epochs}"
            )
            print(message)
            logger.info(message)

    model = raw_model
    if ddp:
        find_unused = os.environ.get("DDP_FIND_UNUSED", "1") == "1"
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )

    w_real = source_count / 2.0
    w_fake = source_count / (2.0 * (source_count - 1))
    class_weight = torch.tensor(
        [w_real, w_fake], device=device, dtype=torch.float32
    )
    if is_main:
        print(f"CE class weight: real={w_real:.6f}, fake={w_fake:.6f}")

    # ce_report, ce_opt, xm, total, acc, usable, real anchors, fake anchors,
    # triplets, active ratio, combined/real/fake positive distances, d_neg
    metric_names = (
        "ce_report",
        "ce_opt",
        "xm",
        "total",
        "acc",
        "usable",
        "real_anchors",
        "fake_anchors",
        "triplets",
        "active_ratio",
        "d_pos",
        "d_rr_pos",
        "d_ff_pos",
        "d_neg",
    )
    epoch_times = []
    saved = {
        path: int(os.path.splitext(os.path.basename(path))[0])
        for path in existing_checkpoints
    }
    smoke_steps = max(0, int(os.environ.get("XM_SMOKE_STEPS", "0")))

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        generator.manual_seed(seed + epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        if cfg["fake_shuffle"]:
            dataset.shuffle()
        model.train(True)

        metric_sum = torch.zeros(len(metric_names), device=device)
        method_count = len(known_methods)
        gate_sum = torch.zeros(
            method_count, raw_model.n_br, device=device, dtype=torch.float64
        )
        gate_sq_sum = torch.zeros_like(gate_sum)
        gate_count = torch.zeros(method_count, device=device, dtype=torch.float64)
        gate_entropy_sum = torch.zeros(
            method_count, device=device, dtype=torch.float64
        )

        progress = tqdm(loader, disable=not is_main)
        steps_run = 0
        for data in progress:
            image = data["img"].to(device, non_blocking=True).float()
            metadata = data["xm_meta"].to(device, non_blocking=True).long()
            source_label = data["label"].to(device, non_blocking=True).long()
            target = metadata[:, 0]
            expected_target = source_label.ne(0).long()
            if not torch.equal(target, expected_target):
                raise RuntimeError("collate image/metadata binary labels are misaligned")

            global_meta = gather_no_grad(metadata)
            positive, negative = build_pair_masks(global_meta)

            for sam_pass in range(2):
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, aux = model(image, return_aux=True)
                        ce_loss = F.cross_entropy(
                            logits,
                            target,
                            weight=class_weight,
                            label_smoothing=args.label_smoothing,
                        )
                else:
                    logits, aux = model(image, return_aux=True)
                    ce_loss = F.cross_entropy(
                        logits,
                        target,
                        weight=class_weight,
                        label_smoothing=args.label_smoothing,
                    )

                with torch.autocast(device_type="cuda", enabled=False):
                    fused_global = gather_with_grad(aux["fused"].float())
                    xm_loss, xm_stats = cross_method_triplet_loss(
                        fused_global,
                        global_meta,
                        positive,
                        negative,
                        margin=xm_margin,
                        require_real_and_fake=True,
                    )
                    total_loss = ce_loss.float() + xm_weight * xm_loss

                if not _all_ranks_finite(total_loss, ddp):
                    raise FloatingPointError(
                        f"non-finite loss at epoch={epoch + 1}"
                    )

                if sam_pass == 0:
                    first_logits = logits.detach()
                    first_gate = aux["gate"].detach().float()
                    first_ce = ce_loss.detach().float()
                    first_xm = xm_loss.detach().float()
                    first_total = total_loss.detach().float()
                    first_stats = {k: v.detach() for k, v in xm_stats.items()}

                # Both passes synchronize DDP gradients.  There is deliberately
                # no model.no_sync() in this standalone trainer.
                total_loss.backward()
                if sam_pass == 0:
                    raw_model.optimizer.first_step(zero_grad=True)
                else:
                    raw_model.optimizer.second_step(zero_grad=True)

            with torch.inference_mode():
                output = first_logits.float()
                ce_report = F.cross_entropy(
                    output, target, weight=class_weight, label_smoothing=0.0
                )
                prediction = output.argmax(dim=1)
                sample_weight = class_weight.gather(0, target)
                weighted_acc = (
                    (prediction == target).float().mul(sample_weight).sum()
                    / sample_weight.sum()
                )

                values = torch.stack(
                    [
                        ce_report,
                        first_ce,
                        first_xm,
                        first_total,
                        weighted_acc,
                        first_stats["usable"],
                        first_stats["valid_real_anchors"],
                        first_stats["valid_fake_anchors"],
                        first_stats["valid_triplets"],
                        first_stats["active_ratio"],
                        first_stats["mean_positive_distance"],
                        first_stats["mean_real_positive_distance"],
                        first_stats["mean_fake_positive_distance"],
                        first_stats["mean_negative_distance"],
                    ]
                )
                metric_sum += values

                methods = metadata[:, 1]
                valid = metadata[:, 3].bool()
                entropy = -(
                    first_gate.clamp_min(1e-12)
                    * first_gate.clamp_min(1e-12).log()
                ).sum(dim=1)
                for method_id in range(method_count):
                    mask = valid & target.eq(1) & methods.eq(method_id)
                    if bool(mask.any().item()):
                        selected_gate = first_gate[mask].double()
                        gate_sum[method_id] += selected_gate.sum(dim=0)
                        gate_sq_sum[method_id] += selected_gate.square().sum(dim=0)
                        gate_count[method_id] += mask.sum()
                        gate_entropy_sum[method_id] += entropy[mask].double().sum()

            del (
                first_logits,
                first_gate,
                first_ce,
                first_xm,
                first_total,
                first_stats,
                output,
                target,
                global_meta,
                positive,
                negative,
            )
            steps_run += 1
            if smoke_steps and steps_run >= smoke_steps:
                break

        steps = steps_run
        if steps == 0:
            raise RuntimeError("training loader produced zero steps")
        if ddp:
            dist.all_reduce(metric_sum, op=dist.ReduceOp.SUM)
            metric_sum /= world_size
            for tensor in (
                gate_sum,
                gate_sq_sum,
                gate_count,
                gate_entropy_sum,
            ):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        averages = metric_sum / steps
        metrics = {name: averages[i].item() for i, name in enumerate(metric_names)}

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        mean_epoch = sum(epoch_times) / len(epoch_times)
        eta = mean_epoch * (epochs - epoch - 1)

        if is_main:
            message = (
                f"Epoch {epoch + 1}/{epochs} | ce_report {metrics['ce_report']:.4f} "
                f"ce_opt {metrics['ce_opt']:.4f} xm {metrics['xm']:.4f} "
                f"total {metrics['total']:.4f} acc {metrics['acc']:.4f} | "
                f"usable {metrics['usable']:.3f} anchors R/F "
                f"{metrics['real_anchors']:.1f}/{metrics['fake_anchors']:.1f} "
                f"triplets {metrics['triplets']:.1f} active {metrics['active_ratio']:.3f} "
                f"dRR+ {metrics['d_rr_pos']:.3f} dFF+ {metrics['d_ff_pos']:.3f} "
                f"d- {metrics['d_neg']:.3f} "
                f"skipped {int(round(steps * (1.0 - metrics['usable'])))}/{steps} | "
                f"epoch {_fmt_hms(epoch_time)} ETA {_fmt_hms(eta)}"
            )
            logger.info(message)
            for method_id, method_name in enumerate(known_methods):
                count = gate_count[method_id].clamp_min(1.0)
                mean = gate_sum[method_id] / count
                variance = (gate_sq_sum[method_id] / count - mean.square()).clamp_min(0)
                std = variance.sqrt()
                entropy_mean = gate_entropy_sum[method_id] / count
                logger.info(
                    f"gate {method_name}: mean={mean.tolist()} "
                    f"std={std.tolist()} entropy={entropy_mean.item():.4f} "
                    f"n={int(gate_count[method_id].item())}"
                )

        catastrophe_threshold = float(os.environ.get("CATASTROPHIC_LOSS", "0"))
        catastrophe_after = int(os.environ.get("CATASTROPHIC_AFTER_EPOCH", "10"))
        if not np.isfinite(metrics["total"]) or (
            catastrophe_threshold > 0
            and epoch + 1 >= catastrophe_after
            and metrics["total"] > catastrophe_threshold
        ):
            if is_main:
                logger.warning(
                    f"CATASTROPHIC total={metrics['total']:.4f} "
                    f"threshold={catastrophe_threshold}"
                )
            raise SystemExit(3)

        if smoke_steps:
            if is_main:
                logger.info(
                    f"XM_SMOKE_STEPS={smoke_steps}: verified optimizer steps; "
                    "checkpoint intentionally not saved"
                )
            break

        scheduler.step()
        if is_main:
            checkpoint_path = os.path.join(
                save_path, "weights", f"{epoch + 1}.pth"
            )
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "optimizer": raw_model.optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            saved[checkpoint_path] = epoch + 1
            if len(saved) > 2:
                oldest = min(saved, key=saved.get)
                try:
                    os.remove(oldest)
                except OSError as exc:
                    logger.warning(f"could not remove {oldest}: {exc}")
                del saved[oldest]

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("-n", dest="session_name", required=True)
    parser.add_argument(
        "-r",
        "--resume",
        dest="resume",
        nargs="?",
        const="auto",
        default=None,
    )
    parser.add_argument("-s", dest="seed", type=int, default=906)
    parser.add_argument("-ls", dest="label_smoothing", type=float, default=0.0)
    main(parser.parse_args())
