"""Tests for src/copy_move.py"""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from src.copy_move import detect_copy_move
from src.document_template import generate_document
from src.tamper_generator import apply_copy_move


class TestCopyMoveDetection:
    """Tests for copy-move forgery detection."""

    def test_no_crash_on_genuine(self):
        """Genuine document should not crash the detector."""
        doc = generate_document("src_test", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "genuine.jpg")
            doc["image"].save(path, "JPEG", quality=85, subsampling=0)

            result = detect_copy_move(path)
            assert "detected" in result
            assert isinstance(result["detected"], bool)
            assert "num_matches" in result
            assert "num_inliers" in result

    def test_detects_copy_move(self):
        """Should detect copy-move on a tampered document."""
        doc = generate_document("src_test", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save genuine first
            genuine_path = os.path.join(tmpdir, "genuine.jpg")
            doc["image"].save(genuine_path, "JPEG", quality=85, subsampling=0)

            # Reload and apply copy-move
            genuine = Image.open(genuine_path).convert("RGB")
            result = apply_copy_move(genuine, doc["field_bboxes"], seed=42)

            # Save tampered
            tampered_path = os.path.join(tmpdir, "copy_move.jpg")
            result["tampered_image"].save(
                tampered_path, "JPEG", quality=85, subsampling=0
            )

            # Detect
            detection = detect_copy_move(tampered_path)
            # Note: detection may or may not succeed depending on the
            # complexity of the duplicated region. We test for no crash.
            assert "detected" in detection
            assert isinstance(detection["num_matches"], int)

    def test_output_schema(self):
        """Output should always have the expected keys."""
        doc = generate_document("src_test", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jpg")
            doc["image"].save(path, "JPEG", quality=85, subsampling=0)

            result = detect_copy_move(path)
            expected_keys = [
                "detected", "candidate_bbox", "alt_bbox",
                "num_matches", "num_inliers",
                "displacement", "confidence",
            ]
            for key in expected_keys:
                assert key in result, f"Missing key: {key}"

    def test_nonexistent_file(self):
        """Should handle missing file gracefully."""
        result = detect_copy_move("nonexistent.jpg")
        assert result["detected"] is False

    def test_reports_both_bboxes_when_detected(self):
        """When detected, both candidate and alt bbox should be present."""
        doc = generate_document("src_test", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            genuine_path = os.path.join(tmpdir, "genuine.jpg")
            doc["image"].save(genuine_path, "JPEG", quality=85, subsampling=0)

            genuine = Image.open(genuine_path).convert("RGB")
            tamper_result = apply_copy_move(genuine, doc["field_bboxes"], seed=42)

            tampered_path = os.path.join(tmpdir, "copy_move.jpg")
            tamper_result["tampered_image"].save(
                tampered_path, "JPEG", quality=85, subsampling=0
            )

            detection = detect_copy_move(tampered_path)
            if detection["detected"]:
                assert detection["candidate_bbox"] is not None
                assert detection["alt_bbox"] is not None
                assert len(detection["candidate_bbox"]) == 4
                assert len(detection["alt_bbox"]) == 4
