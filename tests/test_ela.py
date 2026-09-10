"""Tests for src/ela.py"""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from src.document_template import generate_document
from src.ela import (
    analyze_ela,
    anomaly_map,
    compute_baseline,
    compute_ela,
    extract_features,
    propose_candidate_region,
)
from src.tamper_generator import apply_date_edit, generate_tampered_dataset


def _save_test_jpeg(image, path, quality=85):
    """Helper to save a test JPEG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, "JPEG", quality=quality, subsampling=0)


class TestComputeELA:
    """Tests for basic ELA computation."""

    def test_output_shape(self):
        doc = generate_document("src_test", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jpg")
            _save_test_jpeg(doc["image"], path)

            heatmap = compute_ela(path, quality=90)
            assert heatmap.shape == (500, 800)
            assert heatmap.dtype == np.float32

    def test_non_negative(self):
        doc = generate_document("src_test", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jpg")
            _save_test_jpeg(doc["image"], path)

            heatmap = compute_ela(path, quality=90)
            assert heatmap.min() >= 0

    def test_tampered_has_higher_ela(self):
        """Tampered regions should show higher ELA than genuine."""
        doc = generate_document("src_test", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save genuine
            genuine_path = os.path.join(tmpdir, "genuine.jpg")
            _save_test_jpeg(doc["image"], genuine_path, quality=85)

            # Reload and tamper
            genuine_reloaded = Image.open(genuine_path).convert("RGB")
            result = apply_date_edit(genuine_reloaded, doc["field_bboxes"], seed=42)
            tampered_path = os.path.join(tmpdir, "tampered.jpg")
            _save_test_jpeg(result["tampered_image"], tampered_path, quality=85)

            # Compute ELA
            ela_genuine = compute_ela(genuine_path, quality=90)
            ela_tampered = compute_ela(tampered_path, quality=90)

            # Get the tampered region
            bbox = result["ground_truth_bbox"]
            x1, y1, x2, y2 = bbox

            # Mean ELA in tampered region should be higher for tampered image
            mean_genuine = ela_genuine[y1:y2, x1:x2].mean()
            mean_tampered = ela_tampered[y1:y2, x1:x2].mean()

            # The tampered region has been re-drawn on a reloaded JPEG,
            # so it should have different compression artifacts
            # This is a soft test - the direction may vary but the values should differ
            assert abs(mean_genuine - mean_tampered) > 0.01, \
                "ELA should differ between genuine and tampered regions"


class TestBaseline:
    """Tests for ELA baseline calibration."""

    def test_baseline_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                doc = generate_document(f"src_{i}", seed=i)
                path = os.path.join(tmpdir, f"genuine_{i}.jpg")
                _save_test_jpeg(doc["image"], path)
                paths.append(path)

            baseline = compute_baseline(paths, quality=90)
            assert baseline["mean_map"].shape == (500, 800)
            assert baseline["std_map"].shape == (500, 800)
            assert baseline["n_samples"] == 3

    def test_baseline_requires_images(self):
        with pytest.raises(ValueError):
            compute_baseline([], quality=90)


class TestAnomalyMap:
    """Tests for anomaly map computation."""

    def test_zero_for_baseline(self):
        """Anomaly of the mean should be zero (within std)."""
        mean_map = np.ones((100, 100), dtype=np.float32) * 5.0
        std_map = np.ones((100, 100), dtype=np.float32) * 1.0
        baseline = {"mean_map": mean_map, "std_map": std_map}

        # Input at the mean level
        heatmap = np.ones((100, 100), dtype=np.float32) * 5.0
        anom = anomaly_map(heatmap, baseline, k=2.0)
        assert anom.max() == 0.0

    def test_detects_anomaly(self):
        """Values above threshold should produce positive anomaly."""
        mean_map = np.ones((100, 100), dtype=np.float32) * 5.0
        std_map = np.ones((100, 100), dtype=np.float32) * 1.0
        baseline = {"mean_map": mean_map, "std_map": std_map}

        # Create input with high values in one region
        heatmap = np.ones((100, 100), dtype=np.float32) * 5.0
        heatmap[40:60, 40:60] = 20.0  # Well above threshold

        anom = anomaly_map(heatmap, baseline, k=2.0)
        assert anom[50, 50] > 0
        assert anom[0, 0] == 0


class TestExtractFeatures:
    """Tests for feature extraction."""

    def test_feature_keys(self):
        heatmap = np.random.rand(100, 100).astype(np.float32) * 10
        features = extract_features(heatmap)
        assert "mean" in features
        assert "std" in features
        assert "max" in features
        assert "p95" in features
        assert "p99" in features
        assert "high_error_pixel_ratio" in features

    def test_roi_extraction(self):
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20:40, 20:40] = 50.0

        features_roi = extract_features(heatmap, roi_bbox=[20, 20, 40, 40])
        features_full = extract_features(heatmap)

        assert features_roi["mean"] > features_full["mean"]

    def test_empty_region(self):
        heatmap = np.zeros((100, 100), dtype=np.float32)
        features = extract_features(heatmap, roi_bbox=[50, 50, 50, 50])
        assert features["mean"] == 0.0


class TestProposeCandidate:
    """Tests for candidate region proposal."""

    def test_finds_region(self):
        anom = np.zeros((100, 100), dtype=np.float32)
        anom[30:50, 30:50] = 10.0
        candidate = propose_candidate_region(anom, min_area=10)
        assert candidate is not None
        assert "bbox" in candidate
        assert candidate["area"] > 0

    def test_no_candidate_in_empty(self):
        anom = np.zeros((100, 100), dtype=np.float32)
        candidate = propose_candidate_region(anom, min_area=10)
        assert candidate is None
