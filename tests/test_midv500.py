"""
ForgeLens-X — Milestone 3: Unit Tests for MIDV-500 Benchmark Loader
====================================================================
Tests MIDV-500 annotation parsing, sample fixture generation, and
verifies strict zero-leakage source-clip partitioning invariants.
"""

import json
import os
import shutil
import tempfile
import pytest

from src.midv500 import (
    generate_sample_midv500_dataset,
    load_midv500_dataset,
    parse_midv500_annotation,
    split_midv500_by_source_clip,
)


class TestMIDVAnnotationParser:
    """Tests for MIDV-500 ground-truth JSON parsing."""

    def test_parse_valid_annotation(self, tmp_path):
        """Test parsing valid MIDV-500 format annotation."""
        ann_file = tmp_path / "test_ann.json"
        ann_data = {
            "doc_type": "alb_id",
            "country": "ALB",
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


class TestMIDVDatasetGenerationAndLoading:
    """Tests for synthetic fixture generation and dataset loading."""

    def test_generate_sample_dataset(self, tmp_path):
        """Test generating authentic sample fixture directory structure."""
        out_dir = str(tmp_path / "midv_sample")
        res = generate_sample_midv500_dataset(output_dir=out_dir, n_clips=3, frames_per_clip=2)

        assert res["status"] == "success"
        assert res["n_clips"] == 3
        assert res["total_samples"] == 6

        samples = load_midv500_dataset(out_dir)
        assert len(samples) == 6

        # Check schema of each loaded sample
        for s in samples:
            assert "sample_id" in s
            assert "image_path" in s
            assert "source_clip_id" in s
            assert "fields" in s
            assert os.path.exists(s["image_path"])


class TestZeroLeakagePartitioning:
    """
    CRITICAL FORENSIC AUDIT TESTS:
    Guarantees frames belonging to the same source clip never cross split boundaries.
    """

    def test_zero_leakage_source_clip_grouping(self, tmp_path):
        """
        Verify mathematical disjointness of clips across train, cal, and test splits.
        """
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

        # Collect source clip IDs per partition
        train_clips = set(s["source_clip_id"] for s in train_samples)
        cal_clips = set(s["source_clip_id"] for s in cal_samples)
        test_clips = set(s["source_clip_id"] for s in test_samples)

        # 1. Zero Clip Leakage Invariant: Pairwise intersection must be empty
        assert len(train_clips.intersection(cal_clips)) == 0, "DATA LEAKAGE: Clip found in both train and cal!"
        assert len(train_clips.intersection(test_clips)) == 0, "DATA LEAKAGE: Clip found in both train and test!"
        assert len(cal_clips.intersection(test_clips)) == 0, "DATA LEAKAGE: Clip found in both cal and test!"

        # 2. Completeness Invariant: Total unique clips across splits equals original dataset
        all_dataset_clips = set(s["source_clip_id"] for s in samples)
        assert (train_clips | cal_clips | test_clips) == all_dataset_clips

        # 3. Sample Count Conservation: Total frames across splits equals total dataset frames
        assert len(train_samples) + len(cal_samples) + len(test_samples) == len(samples)

        # 4. Non-empty partitions when n_clips >= 3
        assert len(train_clips) >= 1
        assert len(cal_clips) >= 1
        assert len(test_clips) >= 1
