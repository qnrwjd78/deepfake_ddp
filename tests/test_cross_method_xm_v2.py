import os
import random
import sys
import unittest

import numpy as np
import torch


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train_multi_xm_v2 import make_collate_xm
from utils.cross_method_xm import build_pair_masks, cross_method_triplet_loss


class CrossMethodMaskTest(unittest.TestCase):
    def setUp(self):
        # real-A, real-B, DF-C, DF-D, F2F-E, FS-F, SimSwap-C, NT-G
        self.meta = torch.tensor(
            [
                [0, -1, 10, 1],
                [0, -1, 11, 1],
                [1, 0, 12, 1],
                [1, 0, 13, 1],
                [1, 1, 14, 1],
                [1, 2, 15, 1],
                [1, 3, 12, 1],
                [1, 4, 16, 1],
            ],
            dtype=torch.long,
        )

    def test_documented_anchor_relations(self):
        positive, negative = build_pair_masks(self.meta)
        self.assertEqual(positive[2].nonzero().view(-1).tolist(), [4, 5, 7])
        self.assertEqual(negative[2].nonzero().view(-1).tolist(), [0, 1])
        self.assertEqual(
            int((positive[2, :, None] & negative[2, None, :]).sum()), 6
        )
        self.assertEqual(positive[0].nonzero().view(-1).tolist(), [1])
        self.assertEqual(negative[0].nonzero().view(-1).tolist(), [2, 3, 4, 5, 6, 7])

    def test_invalid_samples_are_never_used(self):
        meta = self.meta.clone()
        meta[7, 3] = 0
        positive, negative = build_pair_masks(meta)
        self.assertFalse(bool(positive[7].any()))
        self.assertFalse(bool(positive[:, 7].any()))
        self.assertFalse(bool(negative[7].any()))
        self.assertFalse(bool(negative[:, 7].any()))

    def test_loss_is_finite_and_backward_connected(self):
        torch.manual_seed(3)
        fused = torch.randn(8, 16, requires_grad=True)
        positive, negative = build_pair_masks(self.meta)
        loss, stats = cross_method_triplet_loss(
            fused, self.meta, positive, negative, margin=0.2
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(stats["usable"]), 1.0)
        loss.backward()
        self.assertIsNotNone(fused.grad)
        self.assertTrue(torch.isfinite(fused.grad).all())

    def test_empty_batch_returns_graph_zero(self):
        meta = torch.tensor(
            [[1, 0, 1, 1], [1, 0, 2, 1]], dtype=torch.long
        )
        fused = torch.randn(2, 8, requires_grad=True)
        positive, negative = build_pair_masks(meta)
        loss, stats = cross_method_triplet_loss(
            fused, meta, positive, negative, margin=0.2
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(stats["usable"]), 0.0)
        loss.backward()
        self.assertIsNotNone(fused.grad)


class CollateAlignmentTest(unittest.TestCase):
    @staticmethod
    def _image(value):
        return np.full((3, 2, 2), value, dtype=np.float32)

    def _item(self, offset):
        real_content = 10 + offset
        sbi_content = 20 + offset
        known_content = 30 + offset
        fake_images = [self._image(sbi_content), self._image(known_content)]
        real_images = [self._image(real_content)]
        fake_meta = [
            np.asarray([1, -2, sbi_content, 0], dtype=np.int64),
            np.asarray([1, 0, known_content, 1], dtype=np.int64),
        ]
        real_meta = [np.asarray([0, -1, real_content, 1], dtype=np.int64)]
        return fake_images, real_images, fake_meta, real_meta

    def test_balanced_selection_keeps_metadata_aligned(self):
        random.seed(7)
        torch.manual_seed(7)
        output = make_collate_xm(batch_size=4)([self._item(0), self._item(1)])
        self.assertEqual(output["img"].shape[0], 4)
        expected_binary = output["label"].ne(0).long()
        self.assertTrue(torch.equal(expected_binary, output["xm_meta"][:, 0]))
        pixels = output["img"][:, 0, 0, 0].long()
        self.assertTrue(torch.equal(pixels, output["xm_meta"][:, 2]))


if __name__ == "__main__":
    unittest.main()

