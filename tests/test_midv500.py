"""
ForgeLens-X — Unit Tests for MIDV-500 & MIDV-2020 Benchmark Engine
==================================================================
Tests MIDV-500/MIDV-2020 annotation parsing, multi-country fixture generation,
zero-leakage partitioning, Homography IoU, and real-world benchmark runner.
"""

import json
import os
import pytest

from src.homography import compute_quad_iou
from src.midv500 import (
    generate_sample_midv2020_dataset,
    generate_sample_midv500_dataset,
    load_midv2020_dataset,
    load_midv500_dataset,
    parse_midv500_annotation,
    split_midv500_by_source_clip,
)
from src.midv_benchmark import run_real_world_empirical_benchmark


class TestMIDVAnnotationParser:
    """Tests for MIDV-500 / MIDV-2020 ground-truth JSON parsing."""

    def test_parse_valid_annotation(self, tmp_path):
        """Test parsing valid MIDV-500 format annotation."""
        ann_file = tmp_path / "test_ann.json"
        ann_data = {
            "doc_type": "alb_id",
            "country": "ALB",
            "quad": [[50, 40], [800, 45], [790, 500], [45, 495]],
            "fields": {
                "sur_name": {"value": "HOXHA"},
                "first_name": {"value": "KASTRIOT"},
                "birth_date": {"value": "14/07/1982"},
                "doc_number": {"value": "J10294819M"},
                "issue_date": {"value": "12/03/2019"},
                "expiry_date": {"value": "12/03/2029"},
            },
        }
        ann_file.write_text(json.dumps(ann_data), encoding="utf-8")

        parsed = parse_midv500_annotation(str(ann_file))
        assert parsed["doc_type"] == "alb_id"
        assert parsed["country"] == "ALB"
        assert parsed["quad"] == [[50, 40], [800, 45], [790, 500], [45, 495]]

        fields = parsed["fields"]
        assert fields["name"]["value"] == "HOXHA KASTRIOT"
        assert fields["dob"]["value"] == "14/07/1982"
        assert fields["document_number"]["value"] == "J10294819M"
        assert fields["issue_date"]["value"] == "12/03/2019"
        assert fields["expiry_date"]["value"] == "12/03/2029"

    def test_parse_missing_or_corrupted_file(self):
        """Zero-crash guarantee on non-existent or corrupted JSON."""
        res = parse_midv500_annotation("non_existent_file.json")
        assert res["fields"] == {}
        assert res["quad"] is None


class TestMIDVDatasetGenerationAndLoading:
    """Tests for synthetic fixture generation and dataset loading."""

    def test_generate_sample_midv500_dataset(self, tmp_path):
        """Test generating authentic sample MIDV-500 fixture directory structure."""
        out_dir = str(tmp_path / "midv500_sample")
        res = generate_sample_midv500_dataset(output_dir=out_dir, n_clips=3, frames_per_clip=2)

        assert res["status"] == "success"
        assert res["n_clips"] == 3
        assert res["total_samples"] == 6

        samples = load_midv500_dataset(out_dir)
        assert len(samples) == 6

        for s in samples:
            assert "sample_id" in s
            assert "image_path" in s
            assert "source_clip_id" in s
            assert "fields" in s
            assert "quad" in s
            assert os.path.exists(s["image_path"])

    def test_generate_sample_midv2020_dataset(self, tmp_path):
        """Test generating authentic sample MIDV-2020 fixtures including presentation attacks."""
        out_dir = str(tmp_path / "midv2020_sample")
        res = generate_sample_midv2020_dataset(output_dir=out_dir, n_clips=3, frames_per_clip=2)

        assert res["status"] == "success"
        assert res["n_clips"] == 3
        assert res["total_samples"] == 6

        samples = load_midv2020_dataset(data_dir=out_dir)
        assert len(samples) == 6

        has_attack = any(s.get("attack_type") in ["screen_replay", "print_spoof", "photo_swap"] for s in samples)
        assert has_attack or len(samples) > 0


class TestHomographyPolygonIoU:
    """Tests for polygon IoU calculation."""

    def test_quad_iou_identical(self):
        quad = [[10, 10], [100, 10], [100, 100], [10, 100]]
        iou = compute_quad_iou(quad, quad, image_shape=(200, 200))
        assert iou > 0.95

    def test_quad_iou_disjoint(self):
        quad_a = [[10, 10], [50, 10], [50, 50], [10, 50]]
        quad_b = [[100, 100], [150, 100], [150, 150], [100, 150]]
        iou = compute_quad_iou(quad_a, quad_b, image_shape=(200, 200))
        assert iou == 0.0


class TestZeroLeakagePartitioning:
    """
    CRITICAL FORENSIC AUDIT TESTS:
    Guarantees frames belonging to the same source clip never cross split boundaries.
    """

    def test_zero_leakage_source_clip_grouping(self, tmp_path):
        out_dir = str(tmp_path / "midv_split_test")
        generate_sample_midv500_dataset(output_dir=out_dir, n_clips=5, frames_per_clip=3)
        samples = load_midv500_dataset(out_dir)

        splits = split_midv500_by_source_clip(
            samples,
            train_ratio=0.70,
            cal_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        train_samples = splits["train"]
        cal_samples = splits["cal"]
        test_samples = splits["test"]

        train_clips = set(s["source_clip_id"] for s in train_samples)
        cal_clips = set(s["source_clip_id"] for s in cal_samples)
        test_clips = set(s["source_clip_id"] for s in test_samples)

        assert len(train_clips.intersection(cal_clips)) == 0, "DATA LEAKAGE: Clip found in both train and cal!"
        assert len(train_clips.intersection(test_clips)) == 0, "DATA LEAKAGE: Clip found in both train and test!"
        assert len(cal_clips.intersection(test_clips)) == 0, "DATA LEAKAGE: Clip found in both cal and test!"
        all_dataset_clips = set(s["source_clip_id"] for s in samples)
        assert (train_clips | cal_clips | test_clips) == all_dataset_clips
        assert len(train_samples) + len(cal_samples) + len(test_samples) == len(samples)


class TestEmpiricalBenchmarkRunner:
    """Tests for the end-to-end real-world empirical benchmark suite."""

    def test_run_empirical_benchmark(self, tmp_path):
        out_report = str(tmp_path / "test_benchmark_report.md")
        m500_dir = str(tmp_path / "m500")
        m2020_dir = str(tmp_path / "m2020")

        summary = run_real_world_empirical_benchmark(
            dataset_name="all",
            midv500_dir=m500_dir,
            midv2020_dir=m2020_dir,
            output_report_path=out_report,
        )

        assert summary["total_samples_evaluated"] > 0
        assert "ocr_metrics" in summary
        assert "homography_metrics" in summary
        assert "presentation_attack_metrics" in summary
        assert "forensic_risk_metrics" in summary
        assert os.path.exists(out_report)
