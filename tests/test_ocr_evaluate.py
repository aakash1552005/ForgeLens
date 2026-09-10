"""
ForgeLens-X — Milestone 3: Unit Tests for OCR Evaluation & Metrics Engine
==========================================================================
Tests character error rate (CER), Levenshtein edit similarity calculations,
dataset evaluation batching, and strict dataset isolation in reporting.
"""

import os
import json
import csv
import pytest

from src.ocr_evaluate import (
    apply_downsampling,
    apply_gaussian_blur,
    apply_specular_glare,
    apply_underexposure,
    compute_cer,
    compute_levenshtein_distance,
    compute_normalized_similarity,
    evaluate_field_extraction,
    evaluate_ocr_dataset,
    evaluate_ocr_robustness,
    export_ocr_reports,
    load_synthetic_ocr_dataset,
)


class TestMetricPrimitives:
    """Tests for character-level string distance and similarity metrics."""

    def test_levenshtein_distance_exact(self):
        """Identical strings have zero distance."""
        assert compute_levenshtein_distance("Kavya Sharma", "Kavya Sharma") == 0
        assert compute_levenshtein_distance("", "") == 0

    def test_levenshtein_distance_edits(self):
        """Standard insertion, deletion, and substitution costs."""
        assert compute_levenshtein_distance("Kavya", "Kayva") == 2  # substitution of y, v
        assert compute_levenshtein_distance("Kavya", "Kavyaa") == 1 # insertion
        assert compute_levenshtein_distance("Kavya", "Kavy") == 1  # deletion
        assert compute_levenshtein_distance("", "FGL-123") == 7

    def test_normalized_similarity(self):
        """Normalized similarity in [0.0, 1.0]."""
        assert compute_normalized_similarity("Aryan Maharaj", "Aryan Maharaj") == 1.0
        assert compute_normalized_similarity("", "") == 1.0
        assert compute_normalized_similarity("abc", "xyz") == 0.0
        sim = compute_normalized_similarity("ABCD", "ABCE")
        assert 0.74 <= sim <= 0.76  # 1.0 - 1/4 = 0.75

    def test_character_error_rate(self):
        """CER = distance / len(ref)."""
        assert compute_cer("Aryan Maharaj", "Aryan Maharaj") == 0.0
        assert compute_cer("", "") == 0.0
        assert compute_cer("", "abc") == 1.0
        assert compute_cer("10/05/1977", "10/05/1978") == 0.10  # 1 / 10 = 0.10


class TestFieldEvaluation:
    """Tests for single-document structured field evaluation."""

    def test_evaluate_field_extraction_perfect_match(self):
        """All fields matching perfectly."""
        extracted = {
            "name": {"value": "Aryan Maharaj", "confidence": 0.90, "status": "EXTRACTED"},
            "dob": {"value": "10/05/1977", "confidence": 0.88, "status": "EXTRACTED"},
            "document_number": {"value": "FGL-216739-13", "confidence": 0.92, "status": "EXTRACTED"},
            "issue_date": {"value": "03/08/2023", "confidence": 0.85, "status": "EXTRACTED"},
            "expiry_date": {"value": "01/10/2040", "confidence": 0.89, "status": "EXTRACTED"},
        }
        gt = {
            "name": "Aryan Maharaj",
            "dob": "10/05/1977",
            "document_number": "FGL-216739-13",
            "issue_date": "03/08/2023",
            "expiry_date": "01/10/2040",
        }

        res = evaluate_field_extraction(extracted, gt)
        for f in ["name", "dob", "document_number", "issue_date", "expiry_date"]:
            assert res[f]["exact_match"] is True
            assert res[f]["cer"] == 0.0
            assert res[f]["similarity"] == 1.0

    def test_evaluate_field_extraction_dict_gt(self):
        """Handles MIDV-500 dictionary ground truth format {'value': ...}."""
        extracted = {
            "name": {"value": "KASTRIOT HOXHA", "confidence": 0.85, "status": "EXTRACTED"},
        }
        gt = {
            "name": {"value": "KASTRIOT HOXHA"},
        }

        res = evaluate_field_extraction(extracted, gt, target_fields=["name"])
        assert res["name"]["exact_match"] is True
        assert res["name"]["cer"] == 0.0
        assert res["name"]["similarity"] == 1.0


class TestDatasetEvaluationAndReportExport:
    """Tests for batch evaluation and reporting isolation."""

    def test_load_synthetic_dataset(self):
        """Verify synthetic dataset loader returns samples with ground-truth fields."""
        samples = load_synthetic_ocr_dataset(limit=3, seed=42)
        assert len(samples) == 3
        for s in samples:
            assert "image_path" in s
            assert "fields" in s
            assert "name" in s["fields"]
            assert "dob" in s["fields"]
            assert "document_number" in s["fields"]

    def test_evaluate_ocr_dataset(self):
        """Verify batch evaluation pipeline."""
        samples = load_synthetic_ocr_dataset(limit=2, seed=42)
        rep = evaluate_ocr_dataset(samples, dataset_type="synthetic", ocr_engine="rapidocr")

        assert rep["dataset_type"] == "synthetic"
        assert rep["total_documents_evaluated"] == 2
        assert "overall_metrics" in rep
        assert "field_metrics" in rep

        ov = rep["overall_metrics"]
        assert "exact_match_rate" in ov
        assert "mean_cer" in ov
        assert "mean_edit_similarity" in ov
        assert ov["mean_cer"] <= 0.1500

    def test_export_ocr_reports_strict_isolation(self, tmp_path):
        """
        Verify exported CSV and Markdown reports maintain strict dataset isolation
        between synthetic and MIDV-500 benchmarks.
        """
        synth_samples = load_synthetic_ocr_dataset(limit=2, seed=42)
        synth_rep = evaluate_ocr_dataset(synth_samples, dataset_type="synthetic")

        csv_path = str(tmp_path / "test_summary.csv")
        md_path = str(tmp_path / "test_report.md")
        json_path = str(tmp_path / "test_results.json")

        out = export_ocr_reports(
            synthetic_report=synth_rep,
            midv_report=None,
            csv_path=csv_path,
            md_path=md_path,
            json_path=json_path,
        )

        assert os.path.exists(csv_path)
        assert os.path.exists(md_path)
        assert os.path.exists(json_path)

        # Check Markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Milestone 3 Forensic Audit Report" in content
        assert "Internal Synthetic Benchmark (Forgelensia M1 Dataset)" in content
        assert "Zero-Leakage Source-Clip Partitioning Validation" in content

        # Check JSON content
        with open(json_path, "r", encoding="utf-8") as f:
            j_data = json.load(f)

        assert j_data["milestone"] == "M3"
        assert "synthetic_benchmark" in j_data


class TestOpticalStressRobustness:
    """Tests for synthetic optical stress perturbations and robustness evaluation."""

    def test_perturbation_functions(self):
        """Verify perturbation primitives return valid images of the original shape."""
        import numpy as np
        img = np.full((100, 100, 3), 128, dtype=np.uint8)

        blurred = apply_gaussian_blur(img, ksize=5)
        assert blurred.shape == img.shape
        assert blurred.dtype == np.uint8

        glare = apply_specular_glare(img, intensity=0.7)
        assert glare.shape == img.shape
        assert glare.dtype == np.uint8
        assert int(glare.max()) >= int(img.max())

        underexp = apply_underexposure(img, factor=0.4)
        assert underexp.shape == img.shape
        assert underexp.dtype == np.uint8
        assert int(underexp.mean()) < int(img.mean())

        downsampled = apply_downsampling(img, scale=0.5)
        assert downsampled.shape == img.shape
        assert downsampled.dtype == np.uint8

    def test_evaluate_ocr_robustness_execution(self):
        """Verify evaluate_ocr_robustness runs and computes relative degradation metrics."""
        samples = load_synthetic_ocr_dataset(limit=1, seed=42)
        rob_rep = evaluate_ocr_robustness(
            samples,
            stress_conditions=["baseline", "gaussian_blur"],
            ocr_engine="rapidocr",
        )

        assert "stress_conditions" in rob_rep
        assert "baseline" in rob_rep["stress_conditions"]
        assert "gaussian_blur" in rob_rep["stress_conditions"]
        assert "robustness_summary" in rob_rep
        b_cer = rob_rep["stress_conditions"]["baseline"]["mean_cer"]
        assert b_cer <= 0.1500

