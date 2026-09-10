"""Tests for src/visualize.py"""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from src.document_template import generate_document
from src.visualize import create_forensic_card, visualize_batch


class TestForensicCard:
    """Tests for visual forensic card generation."""

    def test_creates_card_image(self):
        doc = generate_document("src_vis_test", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.jpg")
            doc["image"].save(img_path, "JPEG", quality=85)

            sample_data = {
                "source_id": "src_vis_test",
                "attack_type": "date_edit",
                "label": "tampered",
                "image_path": img_path,
                "ground_truth_bbox": [230, 158, 400, 178],
            }
            analysis_data = {
                "ela_detected": True,
                "ela_candidate_bbox": [240, 160, 390, 175],
                "ela_features": {"candidate_energy": 125.0},
                "copy_move_detected": False,
                "copy_move_num_inliers": 8,
                "copy_move_confidence": 0.05,
            }

            out_path = os.path.join(tmpdir, "card.png")
            card = create_forensic_card(sample_data, analysis_data, output_path=out_path)

            assert isinstance(card, Image.Image)
            assert os.path.exists(out_path)
            assert card.size[0] > 800  # Wide multi-panel canvas
            assert card.size[1] > 600

    def test_visualize_batch(self):
        doc = generate_document("src_batch_test", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.jpg")
            doc["image"].save(img_path, "JPEG", quality=85)

            samples = [
                {
                    "source_id": "src_batch_test",
                    "attack_type": "none",
                    "label": "genuine",
                    "image_path": img_path,
                    "ground_truth_bbox": None,
                },
                {
                    "source_id": "src_batch_test",
                    "attack_type": "date_edit",
                    "label": "tampered",
                    "image_path": img_path,
                    "ground_truth_bbox": [230, 158, 400, 178],
                },
            ]
            analysis = [
                {
                    "image_path": img_path,
                    "ela_detected": False,
                    "copy_move_detected": False,
                }
            ]

            out_dir = os.path.join(tmpdir, "visuals")
            saved = visualize_batch(samples, analysis, output_dir=out_dir, max_samples=2)

            assert len(saved) == 2
            for p in saved:
                assert os.path.exists(p)
