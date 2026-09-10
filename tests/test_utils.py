"""Tests for src/utils.py"""

import numpy as np
import pytest

from src.utils import (
    bbox_from_mask,
    compute_iou,
    compute_mask_iou,
    set_seed,
)


class TestComputeIoU:
    """Tests for bounding box IoU computation."""

    def test_perfect_overlap(self):
        box = [10, 20, 50, 60]
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        a = [0, 0, 10, 10]
        b = [20, 20, 30, 30]
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = [0, 0, 10, 10]
        b = [5, 5, 15, 15]
        # Intersection = 5*5 = 25, Union = 100 + 100 - 25 = 175
        expected = 25 / 175
        assert abs(compute_iou(a, b) - expected) < 1e-6

    def test_one_inside_other(self):
        outer = [0, 0, 100, 100]
        inner = [25, 25, 75, 75]
        # Intersection = 50*50 = 2500, Union = 10000 + 2500 - 2500 = 10000
        expected = 2500 / 10000
        assert abs(compute_iou(outer, inner) - expected) < 1e-6

    def test_touching_edges(self):
        a = [0, 0, 10, 10]
        b = [10, 0, 20, 10]
        assert compute_iou(a, b) == 0.0

    def test_symmetric(self):
        a = [10, 10, 50, 50]
        b = [30, 30, 70, 70]
        assert compute_iou(a, b) == compute_iou(b, a)


class TestComputeMaskIoU:
    """Tests for binary mask IoU computation."""

    def test_identical_masks(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        assert compute_mask_iou(mask, mask) == 1.0

    def test_no_overlap_masks(self):
        a = np.zeros((100, 100), dtype=np.uint8)
        b = np.zeros((100, 100), dtype=np.uint8)
        a[0:10, 0:10] = 1
        b[50:60, 50:60] = 1
        assert compute_mask_iou(a, b) == 0.0

    def test_empty_masks(self):
        a = np.zeros((100, 100), dtype=np.uint8)
        b = np.zeros((100, 100), dtype=np.uint8)
        assert compute_mask_iou(a, b) == 0.0


class TestBboxFromMask:
    """Tests for extracting bounding box from mask."""

    def test_simple_region(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 255
        bbox = bbox_from_mask(mask)
        assert bbox == [30, 20, 60, 40]

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert bbox_from_mask(mask) is None

    def test_single_pixel(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255
        bbox = bbox_from_mask(mask)
        assert bbox == [50, 50, 51, 51]


class TestSetSeed:
    """Tests for deterministic seeding."""

    def test_deterministic(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(99)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)
