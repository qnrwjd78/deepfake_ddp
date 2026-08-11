"""Cross-method batch-all metric regularization for the standalone XM run."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Rank-major all-gather whose backward reduce-scatters feature gradients.

    PyTorch 2.6's torch.distributed.nn.functional.all_gather uses SUM in its
    backward reduce-scatter.  When every rank evaluates the same global XM
    loss, DDP's parameter-gradient averaging supplies the matching 1/world
    factor.  Do not multiply or divide the returned loss by world size.
    """

    if not _distributed():
        return tensor
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(tensor.contiguous()), dim=0)


@torch.no_grad()
def gather_no_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Rank-major gather for fixed-shape integer metadata."""

    if not _distributed():
        return tensor
    outputs = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(outputs, tensor.contiguous())
    return torch.cat(outputs, dim=0)


def build_pair_masks(meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build [anchor, candidate] positive and negative masks.

    Metadata columns are: binary label, semantic method id, content id,
    contrastive-valid flag.
    """

    if meta.ndim != 2 or meta.shape[1] != 4:
        raise ValueError(f"XM metadata must be [N,4], got {tuple(meta.shape)}")

    labels, methods, contents, valid_i64 = meta.unbind(dim=1)
    valid = valid_i64.bool()
    real = labels.eq(0)
    fake = labels.eq(1)
    n = labels.numel()
    not_self = ~torch.eye(n, dtype=torch.bool, device=labels.device)
    valid_pair = valid[:, None] & valid[None, :] & not_self

    real_positive = (
        real[:, None]
        & real[None, :]
        & contents[:, None].ne(contents[None, :])
    )
    fake_positive = (
        fake[:, None]
        & fake[None, :]
        & methods[:, None].ne(methods[None, :])
        & contents[:, None].ne(contents[None, :])
    )
    positive = valid_pair & (real_positive | fake_positive)

    negative = valid_pair & (
        (real[:, None] & fake[None, :])
        | (fake[:, None] & real[None, :])
    )
    return positive, negative


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum()
    if bool(count.item()):
        return values.masked_select(mask).mean()
    return values.sum() * 0.0


def cross_method_triplet_loss(
    fused_global: torch.Tensor,
    meta_global: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    margin: float,
    require_real_and_fake: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute direct post-gate, L2-normalized, batch-all triplet loss.

    Reduction is anchor mean -> fake-method mean -> 1:1 real/fake mean.
    A batch lacking either side returns a graph-connected zero when strict
    1:1 reduction is requested.
    """

    if fused_global.ndim != 2:
        raise ValueError(f"fused feature must be [N,D], got {tuple(fused_global.shape)}")
    if fused_global.shape[0] != meta_global.shape[0]:
        raise ValueError("global feature/metadata lengths differ")
    if not 0.0 <= float(margin) <= 2.0:
        raise ValueError(f"normalized Euclidean margin must be in [0,2], got {margin}")

    # The caller disables autocast too; float() here is a second safety rail.
    z = F.normalize(fused_global.float(), dim=1, eps=1e-12)
    distance = torch.cdist(z, z, p=2)

    triplet_mask = positive[:, :, None] & negative[:, None, :]
    hinge = F.relu(
        distance[:, :, None] - distance[:, None, :] + float(margin)
    )
    per_anchor_count = triplet_mask.sum(dim=(1, 2))
    per_anchor_loss = (
        hinge.masked_fill(~triplet_mask, 0.0).sum(dim=(1, 2))
        / per_anchor_count.clamp_min(1)
    )
    eligible = per_anchor_count.gt(0)

    labels, methods, _, valid_i64 = meta_global.unbind(dim=1)
    valid = valid_i64.bool()
    real_anchor = eligible & valid & labels.eq(0)
    fake_anchor = eligible & valid & labels.eq(1) & methods.ge(0)

    real_available = bool(real_anchor.any().item())
    fake_method_losses = []
    if bool(fake_anchor.any().item()):
        for method_id in torch.unique(methods[fake_anchor]):
            method_anchor = fake_anchor & methods.eq(method_id)
            fake_method_losses.append(per_anchor_loss[method_anchor].mean())
    fake_available = bool(fake_method_losses)

    usable = real_available and fake_available
    if usable:
        real_loss = per_anchor_loss[real_anchor].mean()
        fake_loss = torch.stack(fake_method_losses).mean()
        loss = 0.5 * (real_loss + fake_loss)
    elif not require_real_and_fake and (real_available or fake_available):
        pieces = []
        if real_available:
            pieces.append(per_anchor_loss[real_anchor].mean())
        if fake_available:
            pieces.append(torch.stack(fake_method_losses).mean())
        loss = torch.stack(pieces).mean()
    else:
        loss = distance.sum() * 0.0

    valid_triplets = triplet_mask.sum()
    active_triplets = (triplet_mask & hinge.gt(0)).sum()
    pos_count = positive.sum()
    neg_count = negative.sum()
    real = labels.eq(0)
    fake = labels.eq(1)
    real_positive_mask = positive & real[:, None] & real[None, :]
    fake_positive_mask = positive & fake[:, None] & fake[None, :]

    stats = {
        "usable": torch.tensor(float(usable), device=distance.device),
        "valid_real_anchors": real_anchor.sum().detach().float(),
        "valid_fake_anchors": fake_anchor.sum().detach().float(),
        "valid_triplets": valid_triplets.detach().float(),
        "active_ratio": (
            active_triplets.float() / valid_triplets.clamp_min(1).float()
        ).detach(),
        "mean_positive_distance": _masked_mean(distance, positive).detach(),
        "mean_real_positive_distance": _masked_mean(
            distance, real_positive_mask
        ).detach(),
        "mean_fake_positive_distance": _masked_mean(
            distance, fake_positive_mask
        ).detach(),
        "mean_negative_distance": _masked_mean(distance, negative).detach(),
        "positive_pairs": pos_count.detach().float(),
        "negative_pairs": neg_count.detach().float(),
    }
    return loss, stats
