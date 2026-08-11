"""Metadata-carrying v2 precrop dataset for standalone XM training."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .sbi_precrop_xm import SBI_Multi_Precrop_XM_Dataset


REAL_METHOD_ID = -1
SBI_METHOD_ID = -2
UNKNOWN_FAKE_METHOD_ID = -3


def _stable_i63(key: str) -> int:
    raw = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(raw, byteorder="big", signed=False) & ((1 << 63) - 1)


def _normal(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _is_under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((_normal(path), root)) == root
    except ValueError:
        return False


def _method_name_from_ff_root(path: str) -> str:
    parts = Path(path).parts
    try:
        index = parts.index("manipulated_sequences")
        return parts[index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Cannot infer FF++ method from {path!r}; expected "
            ".../manipulated_sequences/<method>/..."
        ) from exc


class SBI_Multi_XM_V2_Dataset(SBI_Multi_Precrop_XM_Dataset):
    """Precrop SBI dataset plus deterministic cross-method metadata."""

    def __init__(
        self,
        *args,
        xm_known_methods: Sequence[str],
        trusted_extra_real: bool = True,
        **kwargs,
    ):
        fake_frame_paths = list(kwargs.get("fake_frame_paths", []))
        real_frame_path = kwargs.get("real_frame_path")
        if not fake_frame_paths:
            raise ValueError("XM dataset requires explicit fake_frame_paths")
        if not real_frame_path:
            raise ValueError("XM dataset requires explicit real_frame_path")

        super().__init__(*args, **kwargs)

        self._ff_real_root = _normal(real_frame_path)
        self._trusted_extra_real = bool(trusted_extra_real)
        self._known_fake_count = len(fake_frame_paths)
        self._method_to_id = {name: index for index, name in enumerate(xm_known_methods)}
        if len(self._method_to_id) != len(xm_known_methods):
            raise ValueError("cross_method_loss.known_methods contains duplicates")

        source_method_names = [
            _method_name_from_ff_root(path) for path in fake_frame_paths
        ]
        unknown = sorted(set(source_method_names) - set(self._method_to_id))
        missing = sorted(set(self._method_to_id) - set(source_method_names))
        if unknown or missing:
            raise ValueError(
                f"Known-method mismatch: unconfigured roots={unknown}, "
                f"missing roots={missing}"
            )
        self.fake_method_ids = [
            self._method_to_id[name] for name in source_method_names
        ]
        self.fake_method_ids.extend(
            [UNKNOWN_FAKE_METHOD_ID]
            * (len(self.fake_image_lists) - self._known_fake_count)
        )

        self._real_meta_by_path: Dict[str, np.ndarray] = {}
        self._fake_content_by_source: List[Dict[str, int]] = []
        collision_guard: Dict[int, str] = {}

        def content_id(key: str) -> int:
            value = _stable_i63(key)
            previous = collision_guard.setdefault(value, key)
            if previous != key:
                raise RuntimeError(
                    f"content-id collision: {previous!r} and {key!r} -> {value}"
                )
            return value

        extra_real_roots: List[Tuple[str, str]] = list(self.xm_extra_real_roots)
        for path in self.real_image_list:
            if _is_under(path, self._ff_real_root):
                folder = Path(path).parent.name
                key = f"ffpp:{folder.split('_', 1)[0]}"
                valid = True
            else:
                matches = [
                    (root, tag)
                    for root, tag in extra_real_roots
                    if _is_under(path, root)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Real sample {path!r} matched {len(matches)} XM extra roots"
                    )
                _, tag = matches[0]
                key = f"{tag}:{Path(path).parent.name}"
                valid = self._trusted_extra_real
            self._real_meta_by_path[path] = np.asarray(
                [0, REAL_METHOD_ID, content_id(key), int(valid)], dtype=np.int64
            )

        for source_index, paths in enumerate(self.fake_image_lists):
            source_content: Dict[str, int] = {}
            is_known_ff = source_index < self._known_fake_count
            for path in paths:
                folder = Path(path).parent.name
                if is_known_ff:
                    key = f"ffpp:{folder.split('_', 1)[0]}"
                else:
                    extra_index = source_index - self._known_fake_count
                    roots = self.xm_extra_fake_roots[extra_index]
                    matches = [tag for root, tag in roots if _is_under(path, root)]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Fake sample {path!r} matched {len(matches)} XM extra roots"
                        )
                    key = f"extra_fake{extra_index}:{matches[0]}:{folder}"
                source_content[path] = content_id(key)
            self._fake_content_by_source.append(source_content)

        print(
            "XM metadata: methods="
            + ",".join(
                f"{name}:{index}" for name, index in self._method_to_id.items()
            )
            + f" trusted_extra_real={self._trusted_extra_real}"
        )

    def __getitem__(self, index):
        fake_images = []
        wide_fake_images = []
        while True:
            try:
                real_path = self.real_image_list[index]
                if self.dual_view:
                    real_image, sbi_image, wide_real, wide_sbi = self.get_fake(
                        real_path, sbi=True
                    )
                else:
                    real_image, sbi_image = self.get_fake(real_path, sbi=True)
                break
            except Exception as exc:
                print(exc)
                index = torch.randint(low=0, high=len(self), size=(1,)).item()

        real_meta = self._real_meta_by_path[real_path]
        sbi_meta = np.asarray(
            [1, SBI_METHOD_ID, int(real_meta[2]), 0], dtype=np.int64
        )
        fake_images.append(sbi_image)
        fake_meta = [sbi_meta]
        if self.dual_view:
            wide_fake_images.append(wide_sbi)

        for source_index, image_list in enumerate(self.fake_image_lists):
            path = image_list[index % len(image_list)]
            if self.dual_view:
                fake_image, wide_fake = self.get_fake(path, sbi=False)
                fake_images.append(fake_image)
                wide_fake_images.append(wide_fake)
            else:
                fake_images.append(self.get_fake(path, sbi=False))

            is_known = source_index < self._known_fake_count
            fake_meta.append(
                np.asarray(
                    [
                        1,
                        self.fake_method_ids[source_index],
                        self._fake_content_by_source[source_index][path],
                        int(is_known),
                    ],
                    dtype=np.int64,
                )
            )

        if self.dual_view:
            return (
                fake_images,
                [real_image],
                wide_fake_images,
                [wide_real],
                fake_meta,
                [real_meta],
            )
        return fake_images, [real_image], fake_meta, [real_meta]

