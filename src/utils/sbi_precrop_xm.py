"""Isolated precropped SBI loader for the standalone XM training recipe.

This keeps the existing v2 ``utils/sbi.py`` untouched.  The completed XM
experiment used already-margin-cropped images with crop-local landmarks and no
RetinaFace requirement.  It also treated the normalized newbench and NB2
folders as one logical extra fake source; ``merged_labeled_sources`` preserves
that source layout after the workspaces are combined.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from . import sbi as base


def _modality_path(filename: str, directory: str) -> Path:
    value = filename.replace("/frames/", f"/{directory}/")
    return Path(value).with_suffix(".npy")


class SBI_Multi_Precrop_XM_Dataset(base.SBI_Multi_Dataset):
    """Original XM SBI data path with v2-normalized, Retina-free precrops."""

    def __init__(
        self,
        phase="train",
        image_size=224,
        n_frames=8,
        fake_frame_paths=None,
        fake_shift=False,
        real_frame_path=None,
        align_crop=False,
        extra_sources=None,
        dual_view=False,
        wide_size=224,
        precropped=True,
    ):
        if phase != "train":
            raise NotImplementedError("Standalone XM loader supports phase='train' only")
        if not precropped:
            raise ValueError("v2 XM data must use precropped=true")

        fake_frame_paths = list(fake_frame_paths or [])
        if not fake_frame_paths:
            raise ValueError("XM requires explicit FF++ fake_frame_paths")
        if len(fake_frame_paths) != len(set(fake_frame_paths)):
            raise ValueError("XM fake_frame_paths must contain six unique roots")

        self.precropped = True
        self.align_crop = bool(align_crop)
        self.dual_view = bool(dual_view)
        self.wide_size = int(wide_size)
        self.phase = phase
        self.path_lm = "/landmarks/"
        self.image_size = (int(image_size), int(image_size))

        resolution = int(image_size)
        destination = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        destination[:, 0] += 8.0
        destination *= resolution / 112.0
        margin = 0.3
        extra = resolution * margin / 2.0
        destination += extra
        destination *= resolution / (resolution + 2.0 * extra)
        self._dst5 = destination

        if real_frame_path:
            real_image_list = base.init_ff(
                dataset_path=real_frame_path, phase=phase, n_frames=n_frames
            )
        else:
            real_image_list = base.init_ff(phase=phase, n_frames=n_frames)

        def usable(filename: str) -> bool:
            return _modality_path(filename, "landmarks").is_file()

        real_image_list = [path for path in real_image_list if usable(path)]
        extra_fake_lists: list[tuple[str, list[str], list[tuple[str, str]]]] = []
        self.xm_extra_real_roots: list[tuple[str, str]] = []

        for source_index, source in enumerate(extra_sources or []):
            merged = list(source.get("merged_labeled_sources", []))
            if merged:
                if source.get("fake_only"):
                    include_real = False
                else:
                    include_real = True
                merged_real: list[str] = []
                merged_fake: list[str] = []
                fake_roots: list[tuple[str, str]] = []
                for member_index, member in enumerate(merged):
                    frames = str(member["frames"])
                    labels = str(member["labels"])
                    tag = str(member.get("name", f"member{member_index}"))
                    real_ids, fake_ids = self._labels_split(labels)
                    if include_real:
                        member_real = base.init_ff(
                            dataset_path=frames,
                            phase=phase,
                            n_frames=n_frames,
                            ff_split=False,
                            keep_ids=real_ids,
                        )
                        merged_real.extend(path for path in member_real if usable(path))
                        self.xm_extra_real_roots.append(
                            (os.path.realpath(os.path.abspath(frames)), f"extra{source_index}:{tag}")
                        )
                    member_fake = base.init_ff(
                        dataset_path=frames,
                        n_frames=n_frames,
                        shift=fake_shift,
                        ff_split=False,
                        keep_ids=fake_ids,
                    )
                    merged_fake.extend(path for path in member_fake if usable(path))
                    fake_roots.append(
                        (os.path.realpath(os.path.abspath(frames)), tag)
                    )
                real_image_list.extend(merged_real)
                label = "+".join(tag for _, tag in fake_roots)
                extra_fake_lists.append((label, merged_fake, fake_roots))
                print(
                    f"[extra-merged] {label}: real+{len(merged_real)} "
                    f"fake+{len(merged_fake)}"
                    f"{' (fake_only)' if source.get('fake_only') else ''}"
                )
                continue

            if "real_frames" in source or "fake_frames" in source:
                if "fake_frames" not in source:
                    raise ValueError("A direct XM extra source requires fake_frames")
                real_frames = source.get("real_frames")
                extra_real = []
                if real_frames and not source.get("fake_only"):
                    extra_real = [
                        path
                        for path in base.init_ff(
                            dataset_path=real_frames,
                            phase=phase,
                            n_frames=n_frames,
                            ff_split=False,
                        )
                        if usable(path)
                    ]
                    self.xm_extra_real_roots.append(
                        (
                            os.path.realpath(os.path.abspath(real_frames)),
                            f"extra{source_index}:direct",
                        )
                    )
                    real_image_list.extend(extra_real)
                fake_frames = str(source["fake_frames"])
                extra_fake = [
                    path
                    for path in base.init_ff(
                        dataset_path=fake_frames,
                        n_frames=n_frames,
                        shift=fake_shift,
                        ff_split=False,
                    )
                    if usable(path)
                ]
                extra_fake_lists.append(
                    (
                        f"direct{source_index}",
                        extra_fake,
                        [(os.path.realpath(os.path.abspath(fake_frames)), "direct")],
                    )
                )
                print(
                    f"[extra-direct] real+{len(extra_real)} fake+{len(extra_fake)}"
                )
                continue

            frames = str(source["frames"])
            real_ids, fake_ids = self._labels_split(source["labels"])
            extra_real = []
            if not source.get("fake_only"):
                extra_real = [
                    path
                    for path in base.init_ff(
                        dataset_path=frames,
                        phase=phase,
                        n_frames=n_frames,
                        ff_split=False,
                        keep_ids=real_ids,
                    )
                    if usable(path)
                ]
                real_image_list.extend(extra_real)
                self.xm_extra_real_roots.append(
                    (
                        os.path.realpath(os.path.abspath(frames)),
                        f"extra{source_index}:flat",
                    )
                )
            extra_fake = [
                path
                for path in base.init_ff(
                    dataset_path=frames,
                    n_frames=n_frames,
                    shift=fake_shift,
                    ff_split=False,
                    keep_ids=fake_ids,
                )
                if usable(path)
            ]
            extra_fake_lists.append(
                (
                    f"flat{source_index}",
                    extra_fake,
                    [(os.path.realpath(os.path.abspath(frames)), "flat")],
                )
            )
            print(f"[extra-flat] real+{len(extra_real)} fake+{len(extra_fake)}")

        print(f"Real: {len(real_image_list)}", end=" | ")
        print(f"SBI Fake: {len(real_image_list)}", end=" | ")

        fake_image_lists: list[list[str]] = []
        for source_index, fake_path in enumerate(fake_frame_paths):
            if not os.path.isdir(fake_path):
                raise FileNotFoundError(
                    f"fake_frame_paths[{source_index}] does not exist: {fake_path}"
                )
            fake_images = [
                path
                for path in base.init_ff(
                    dataset_path=fake_path,
                    n_frames=n_frames,
                    shift=fake_shift,
                )
                if usable(path)
            ]
            if len(real_image_list) < len(fake_images):
                fake_images = random.sample(fake_images, len(real_image_list))
            if not fake_images:
                raise RuntimeError(f"No usable fake frames in {fake_path}")
            fake_image_lists.append(fake_images)
            print(f"Added_Fake{source_index}: {len(fake_images)}", end=" | ")

        self.xm_extra_fake_roots: list[list[tuple[str, str]]] = []
        for source_index, (label, fake_images, fake_roots) in enumerate(extra_fake_lists):
            if len(real_image_list) < len(fake_images):
                fake_images = random.sample(fake_images, len(real_image_list))
            if not fake_images:
                raise RuntimeError(f"No usable fake frames in XM extra source {label}")
            fake_image_lists.append(fake_images)
            self.xm_extra_fake_roots.append(fake_roots)
            print(f"Added_ExtraFake{source_index}: {len(fake_images)}", end=" | ")
        print()

        self.real_image_list = real_image_list
        self.fake_image_lists = fake_image_lists
        self.transforms = self.get_dfdc_transforms()
        self.source_transforms = self.get_source_transforms()

    def get_fake(self, filename, sbi=True):
        image = np.asarray(Image.open(filename))
        landmark = np.load(_modality_path(filename, "landmarks"))
        if landmark.ndim == 3:
            landmark = landmark[0]
        elif landmark.ndim != 2 or landmark.shape[1] != 2:
            raise ValueError(f"Unexpected landmark shape {landmark.shape}: {filename}")

        bbox_lm = np.array(
            [
                landmark[:, 0].min(),
                landmark[:, 1].min(),
                landmark[:, 0].max(),
                landmark[:, 1].max(),
            ]
        )
        bbox = np.array(
            [[bbox_lm[0], bbox_lm[1]], [bbox_lm[2], bbox_lm[3]]]
        )
        landmark = self.reorder_landmark(landmark)
        if np.random.rand() < 0.5:
            image, _, landmark, bbox = self.hflip(image, None, landmark, bbox)

        if sbi:
            image_real, image_fake, _ = self.self_blending(
                image.copy(), landmark.copy()
            )
            transformed = self.transforms(
                image=image_fake.astype("uint8"),
                image1=image_real.astype("uint8"),
            )
            image_fake = transformed["image"]
            image_real = transformed["image1"]
            if self.dual_view:
                wide_real = cv2.resize(
                    image_real,
                    (self.wide_size, self.wide_size),
                    interpolation=cv2.INTER_LINEAR,
                ).astype("float32") / 255.0
                wide_fake = cv2.resize(
                    image_fake,
                    (self.wide_size, self.wide_size),
                    interpolation=cv2.INTER_LINEAR,
                ).astype("float32") / 255.0
                wide_real = wide_real.transpose((2, 0, 1))
                wide_fake = wide_fake.transpose((2, 0, 1))

            if self.align_crop:
                matrix = self._align_M(landmark)
                image_fake = cv2.warpAffine(
                    image_fake, matrix, self.image_size, flags=cv2.INTER_LINEAR
                )
                image_real = cv2.warpAffine(
                    image_real, matrix, self.image_size, flags=cv2.INTER_LINEAR
                )
            else:
                landmark_bbox = np.array(
                    [
                        [landmark[:, 0].min(), landmark[:, 1].min()],
                        [landmark[:, 0].max(), landmark[:, 1].max()],
                    ]
                )
                (
                    image_fake,
                    _,
                    _,
                    _,
                    y0,
                    y1,
                    x0,
                    x1,
                ) = base.crop_face(
                    image_fake,
                    landmark,
                    landmark_bbox,
                    margin=False,
                    crop_by_bbox=True,
                    abs_coord=True,
                    phase=self.phase,
                )
                image_real = image_real[y0:y1, x0:x1]
                image_fake = cv2.resize(
                    image_fake, self.image_size, interpolation=cv2.INTER_LINEAR
                )
                image_real = cv2.resize(
                    image_real, self.image_size, interpolation=cv2.INTER_LINEAR
                )

            self._cutout_pair(image_fake, image_real)
            image_fake = image_fake.astype("float32").transpose((2, 0, 1)) / 255.0
            image_real = image_real.astype("float32").transpose((2, 0, 1)) / 255.0
            if self.dual_view:
                return image_real, image_fake, wide_real, wide_fake
            return image_real, image_fake

        transformed = self.transforms(image=image.astype("uint8"))
        image_fake = transformed["image"]
        if self.dual_view:
            wide_fake = cv2.resize(
                image_fake,
                (self.wide_size, self.wide_size),
                interpolation=cv2.INTER_LINEAR,
            ).astype("float32") / 255.0
            wide_fake = wide_fake.transpose((2, 0, 1))

        if self.align_crop:
            image_fake = cv2.warpAffine(
                image_fake,
                self._align_M(landmark),
                self.image_size,
                flags=cv2.INTER_LINEAR,
            )
        else:
            landmark_bbox = np.array(
                [
                    [landmark[:, 0].min(), landmark[:, 1].min()],
                    [landmark[:, 0].max(), landmark[:, 1].max()],
                ]
            )
            image_fake, _, _, _, _, _, _, _ = base.crop_face(
                image_fake,
                landmark,
                landmark_bbox,
                margin=False,
                crop_by_bbox=True,
                abs_coord=True,
                phase=self.phase,
            )
            image_fake = cv2.resize(
                image_fake, self.image_size, interpolation=cv2.INTER_LINEAR
            )

        self._cutout_pair(image_fake)
        image_fake = image_fake.astype("float32").transpose((2, 0, 1)) / 255.0
        if self.dual_view:
            return image_fake, wide_fake
        return image_fake

    def _cutout_pair(self, first: np.ndarray, second: np.ndarray | None = None) -> None:
        length = 32
        height = random.randint(0, self.image_size[0] - length)
        width = random.randint(0, self.image_size[1] - length)
        first[height : height + length, width : width + length] = 0
        if second is not None:
            second[height : height + length, width : width + length] = 0

