"""Tests for src/tamper_generator.py"""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from src.document_template import DOC_HEIGHT, DOC_WIDTH, generate_document
from src.tamper_generator import (
    ATTACK_FUNCTIONS,
    apply_copy_move,
    apply_date_edit,
    apply_photo_swap,
    apply_text_edit,
    generate_tampered_dataset,
)


@pytest.fixture
def sample_document():
    """Generate a test document."""
    return generate_document("src_test", seed=42)


@pytest.fixture
def sample_image(sample_document):
    """Get the image from test document."""
    return sample_document["image"]


@pytest.fixture
def sample_bboxes(sample_document):
    """Get field bboxes from test document."""
    return sample_document["field_bboxes"]


class TestAttackTypes:
    """Tests for individual attack functions."""

    def test_date_edit(self, sample_image, sample_bboxes):
        result = apply_date_edit(sample_image, sample_bboxes, seed=42)
        assert isinstance(result["tampered_image"], Image.Image)
        assert result["tampered_image"].size == (DOC_WIDTH, DOC_HEIGHT)
        assert result["attack_type"] == "date_edit"
        assert result["target_field"] in ["dob", "issue_date", "expiry_date"]
        assert result["ground_truth_mask"].shape == (DOC_HEIGHT, DOC_WIDTH)
        assert result["ground_truth_mask"].max() == 255
        assert len(result["ground_truth_bbox"]) == 4

    def test_text_edit(self, sample_image, sample_bboxes):
        result = apply_text_edit(sample_image, sample_bboxes, seed=42)
        assert result["attack_type"] == "text_edit"
        assert result["target_field"] in ["name", "document_number"]
        assert result["original_value"] != result["tampered_value"]

    def test_photo_swap(self, sample_image, sample_bboxes):
        result = apply_photo_swap(sample_image, sample_bboxes, seed=42)
        assert result["attack_type"] == "photo_swap"
        assert result["target_field"] == "photo"

    def test_copy_move(self, sample_image, sample_bboxes):
        result = apply_copy_move(sample_image, sample_bboxes, seed=42)
        assert result["attack_type"] == "copy_move"
        assert "source_bbox" in result
        assert "destination_bbox" in result
        assert len(result["source_bbox"]) == 4
        assert len(result["destination_bbox"]) == 4

    def test_all_attacks_produce_masks(self, sample_image, sample_bboxes):
        for attack_name, attack_fn in ATTACK_FUNCTIONS.items():
            result = attack_fn(sample_image, sample_bboxes, seed=42)
            assert result["ground_truth_mask"].shape == (DOC_HEIGHT, DOC_WIDTH), \
                f"Mask shape mismatch for {attack_name}"
            assert result["ground_truth_mask"].dtype == np.uint8, \
                f"Mask dtype mismatch for {attack_name}"


class TestTamperedDataset:
    """Tests for full dataset generation."""

    def test_generates_all_variants(self, sample_document):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = generate_tampered_dataset(
                sample_document, tmpdir, jpeg_quality=85, seed=42,
            )
            # Should produce: 1 genuine + 4 attacks = 5
            assert len(results) == 5

            attack_types = {r["attack_type"] for r in results}
            assert "none" in attack_types
            assert "date_edit" in attack_types
            assert "text_edit" in attack_types
            assert "photo_swap" in attack_types
            assert "copy_move" in attack_types

    def test_jpeg_files_exist(self, sample_document):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = generate_tampered_dataset(
                sample_document, tmpdir, jpeg_quality=85, seed=42,
            )
            for r in results:
                assert os.path.exists(r["image_path"]), \
                    f"Missing image: {r['image_path']}"
                assert os.path.exists(r["mask_path"]), \
                    f"Missing mask: {r['mask_path']}"

    def test_metadata_correctness(self, sample_document):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = generate_tampered_dataset(
                sample_document, tmpdir, jpeg_quality=85, seed=42,
            )
            for r in results:
                assert "source_id" in r
                assert "attack_type" in r
                assert "label" in r
                assert "random_seed" in r
                assert r["source_id"] == "src_test"

    def test_genuine_has_no_bbox(self, sample_document):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = generate_tampered_dataset(
                sample_document, tmpdir, jpeg_quality=85, seed=42,
            )
            genuine = [r for r in results if r["label"] == "genuine"][0]
            assert genuine["bbox"] is None

    def test_tampered_have_bboxes(self, sample_document):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = generate_tampered_dataset(
                sample_document, tmpdir, jpeg_quality=85, seed=42,
            )
            tampered = [r for r in results if r["label"] == "tampered"]
            for r in tampered:
                assert r["bbox"] is not None
                assert len(r["bbox"]) == 4
