"""Tests for src/evaluate.py"""

import numpy as np
import pytest

from src.evaluate import evaluate_batch, evaluate_detection, evaluate_localization


class TestEvaluateDetection:
    """Tests for binary detection evaluation."""

    def test_perfect_detection(self):
        preds = [
            {"detected": True},
            {"detected": True},
            {"detected": False},
        ]
        gts = [
            {"label": "tampered"},
            {"label": "tampered"},
            {"label": "genuine"},
        ]
        result = evaluate_detection(preds, gts)
        assert result["true_positives"] == 2
        assert result["true_negatives"] == 1
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0
        assert result["detection_rate"] == 1.0

    def test_all_missed(self):
        preds = [{"detected": False}, {"detected": False}]
        gts = [{"label": "tampered"}, {"label": "tampered"}]
        result = evaluate_detection(preds, gts)
        assert result["detection_rate"] == 0.0
        assert result["false_negatives"] == 2

    def test_false_alarms(self):
        preds = [{"detected": True}]
        gts = [{"label": "genuine"}]
        result = evaluate_detection(preds, gts)
        assert result["false_positives"] == 1
        assert result["false_alarm_rate"] == 1.0


class TestEvaluateLocalization:
    """Tests for localization evaluation."""

    def test_perfect_localization(self):
        bbox = [10, 20, 50, 60]
        assert evaluate_localization(bbox, bbox) == 1.0

    def test_no_overlap(self):
        a = [0, 0, 10, 10]
        b = [50, 50, 60, 60]
        assert evaluate_localization(a, b) == 0.0

    def test_none_bbox(self):
        assert evaluate_localization(None, [10, 20, 50, 60]) == 0.0
        assert evaluate_localization([10, 20, 50, 60], None) == 0.0
        assert evaluate_localization(None, None) == 0.0


class TestEvaluateBatch:
    """Tests for batch evaluation."""

    def test_basic_batch(self):
        results = [
            {
                "source_id": "src_0001",
                "attack_type": "none",
                "label": "genuine",
                "ground_truth_bbox": None,
                "ela_detected": False,
                "ela_candidate_bbox": None,
                "copy_move_detected": False,
                "copy_move_bbox": None,
            },
            {
                "source_id": "src_0001",
                "attack_type": "date_edit",
                "label": "tampered",
                "ground_truth_bbox": [230, 158, 400, 178],
                "ela_detected": True,
                "ela_candidate_bbox": [225, 155, 405, 180],
                "copy_move_detected": False,
                "copy_move_bbox": None,
            },
        ]
        eval_result = evaluate_batch(results, iou_threshold=0.3)
        assert "ela" in eval_result
        assert "copy_move" in eval_result
        assert "n_samples" in eval_result
        assert eval_result["n_samples"] == 2
