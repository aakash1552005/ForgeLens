"""
ForgeLens-X — Milestone 2: Face Verification Test Suite
=========================================================
Tests face detection, 5-point landmark alignment, verification schemas,
metric distances, cross-modality simulation, multi-model comparison,
evaluation metrics, visual forensic card generation, and CLI subcommands.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest
from PIL import Image

from src.cli import main
from src.face_dataset import (
    create_benchmark_pairs,
    load_benchmark_pairs,
    simulate_document_photo,
    simulate_live_selfie,
)
from src.face_evaluate import (
    calculate_verification_metrics,
    evaluate_face_pairs,
    export_evaluation_report,
)
from src.face_verify import (
    calculate_distance,
    calculate_similarity_pct,
    compare_models,
    extract_face,
    verify,
)
from src.face_visualize import create_face_forensic_card


@pytest.fixture(scope="session")
def face_fixtures():
    """Ensure face fixtures exist in data/test_faces and return paths."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_faces")
    d1 = os.path.join(base_dir, "david1.jpg")
    d2 = os.path.join(base_dir, "david2.jpg")
    diff = os.path.join(base_dir, "100032540_1.jpg")
    return {"david1": d1, "david2": d2, "diff": diff}


class TestFaceDetectionAndAlignment:
    """Tests for face extraction, bounding box detection, and 5-point landmark alignment."""

    def test_extract_face_success(self, face_fixtures):
        res = extract_face(face_fixtures["david1"])
        assert res is not None
        assert "face_crop" in res
        assert "bbox" in res
        assert "landmarks" in res
        assert res["face_crop"].shape == (112, 112, 3)
        assert len(res["bbox"]) == 4
        # 5 facial landmarks
        assert res["landmarks"] is not None
        assert len(res["landmarks"]) == 5

    def test_extract_face_non_face_blank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            blank_path = os.path.join(tmpdir, "blank.jpg")
            blank = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.imwrite(blank_path, blank)

            res = extract_face(blank_path)
            assert res is None


class TestFaceVerification:
    """Tests for core verification function, schema, and error handling."""

    def test_verify_schema_conformance(self, face_fixtures):
        res = verify(face_fixtures["david1"], face_fixtures["david2"])
        required_keys = [
            "verified", "distance", "threshold", "similarity_pct",
            "model", "detector", "distance_metric", "facial_areas",
            "landmarks", "verdict_tier", "time_seconds", "engine",
        ]
        for k in required_keys:
            assert k in res, f"Missing required key: {k}"

        assert isinstance(res["verified"], bool)
        assert isinstance(res["distance"], float)
        assert isinstance(res["threshold"], float)
        assert isinstance(res["similarity_pct"], float)
        assert res["verdict_tier"] in ["CONFIRMED_MATCH", "CONFIRMED_MISMATCH", "BORDERLINE"]

    def test_same_person_verified(self, face_fixtures):
        res = verify(face_fixtures["david1"], face_fixtures["david2"], model_name="ArcFace")
        assert res["verified"] is True
        assert res["distance"] < res["threshold"]
        assert res["similarity_pct"] >= 65.0
        assert res["verdict_tier"] == "CONFIRMED_MATCH"

    def test_different_person_rejected(self, face_fixtures):
        res = verify(face_fixtures["david1"], face_fixtures["diff"], model_name="ArcFace")
        assert res["verified"] is False
        assert res["distance"] > res["threshold"]
        assert res["similarity_pct"] < 50.0

    def test_invalid_input_file_error_handling(self):
        res = verify("non_existent_doc.jpg", "non_existent_live.jpg")
        assert res["verified"] is False
        assert res["error"] == "invalid_input"
        assert res["distance"] is None
        assert res["threshold"] is None
        assert res["similarity_pct"] == 0.0

    def test_no_face_detected_error_handling(self, face_fixtures):
        with tempfile.TemporaryDirectory() as tmpdir:
            blank_path = os.path.join(tmpdir, "noise.jpg")
            noise = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(blank_path, noise)

            res = verify(face_fixtures["david1"], blank_path)
            assert res["verified"] is False
            assert res["error"] == "no_face_detected"
            assert "detail" in res
            assert res["distance"] is None

    def test_calibrated_similarity_pct_properties(self):
        # When distance == threshold, similarity should be exactly 50.0%
        sim_boundary = calculate_similarity_pct(0.68, 0.68, beta=8.0)
        assert round(sim_boundary, 1) == 50.0

        # When distance < threshold, similarity should be > 50.0%
        sim_match = calculate_similarity_pct(0.20, 0.68, beta=8.0)
        assert sim_match > 50.0

        # When distance > threshold, similarity should be < 50.0%
        sim_mismatch = calculate_similarity_pct(1.00, 0.68, beta=8.0)
        assert sim_mismatch < 50.0


class TestMultiModelConsensus:
    """Tests for multi-model consensus and comparison."""

    def test_compare_models_agreement(self, face_fixtures):
        res = compare_models(
            face_fixtures["david1"],
            face_fixtures["david2"],
            models=["ArcFace", "SFace"],
        )
        assert "pair" in res
        assert "results" in res
        assert "consensus" in res
        assert "agreement_score" in res
        assert "total_models" in res
        assert res["total_models"] == 2
        assert len(res["results"]) == 2


class TestFaceDatasetAndPairs:
    """Tests for cross-modality simulation and benchmark pair generation."""

    def test_cross_modality_transforms(self, face_fixtures):
        img_bgr = cv2.imread(face_fixtures["david1"])
        assert img_bgr is not None

        doc_sim = simulate_document_photo(img_bgr, seed=42)
        assert doc_sim.shape == (240, 200, 3)

        live_sim = simulate_live_selfie(img_bgr, seed=42)
        assert live_sim.shape == (360, 300, 3)

    def test_create_and_load_benchmark_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = create_benchmark_pairs(output_dir=tmpdir, n_pairs=4, seed=123)
            assert res["n_pairs"] == 4
            assert res["n_genuine"] == 2
            assert res["n_imposter"] == 2
            assert os.path.exists(res["index_path"])

            # Test loading by dir
            pairs = load_benchmark_pairs(tmpdir)
            assert len(pairs) == 4
            for p in pairs:
                assert "doc_image_path" in p
                assert "live_image_path" in p
                assert os.path.exists(p["doc_image_path"])
                assert os.path.exists(p["live_image_path"])


class TestFaceEvaluation:
    """Tests for empirical metrics calculation and report generation."""

    def test_calculate_verification_metrics_perfect(self):
        preds = [True, True, False, False]
        gts = [True, True, False, False]
        dists = [0.2, 0.3, 0.9, 1.0]

        m = calculate_verification_metrics(preds, gts, dists)
        assert m["accuracy"] == 1.0
        assert m["far"] == 0.0
        assert m["frr"] == 0.0
        assert m["tar"] == 1.0
        assert m["separation_margin"] == pytest.approx(0.7, abs=0.01)

    def test_calculate_verification_metrics_imposter_leakage(self):
        preds = [True, True, True, False]
        gts = [True, True, False, False]
        dists = [0.2, 0.3, 0.5, 1.0]

        m = calculate_verification_metrics(preds, gts, dists)
        assert m["accuracy"] == 0.75
        assert m["far"] == 0.5  # 1 false positive out of 2 imposters
        assert m["frr"] == 0.0  # 0 false negatives out of 2 genuine

    def test_export_evaluation_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_report = {
                "model_name": "ArcFace",
                "distance_metric": "cosine",
                "n_pairs": 2,
                "metrics": {
                    "accuracy": 1.0,
                    "far": 0.0,
                    "frr": 0.0,
                    "tar": 1.0,
                    "mean_genuine_distance": 0.25,
                    "mean_imposter_distance": 0.85,
                    "separation_margin": 0.60,
                },
                "pair_results": [
                    {
                        "pair_id": "test_01",
                        "label": "genuine",
                        "is_same_person": True,
                        "verified": True,
                        "distance": 0.25,
                        "threshold": 0.68,
                        "similarity_pct": 92.0,
                        "verdict_tier": "CONFIRMED_MATCH",
                        "correct_decision": True,
                    }
                ],
            }
            csv_path = os.path.join(tmpdir, "test.csv")
            md_path = os.path.join(tmpdir, "test.md")
            json_path = os.path.join(tmpdir, "test.json")

            paths = export_evaluation_report(
                mock_report,
                csv_path=csv_path,
                md_path=md_path,
                json_path=json_path,
            )
            assert os.path.exists(paths["csv"])
            assert os.path.exists(paths["md"])
            assert os.path.exists(paths["json"])


class TestFaceVisualization:
    """Tests for visual explanation card generation."""

    def test_create_face_forensic_card(self, face_fixtures):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_card = os.path.join(tmpdir, "forensic_card.png")
            img = create_face_forensic_card(
                face_fixtures["david1"],
                face_fixtures["david2"],
                output_path=out_card,
            )
            assert isinstance(img, Image.Image)
            assert img.size == (1240, 860)
            assert os.path.exists(out_card)

    def test_create_face_forensic_card_no_face(self, face_fixtures):
        with tempfile.TemporaryDirectory() as tmpdir:
            blank_path = os.path.join(tmpdir, "blank.jpg")
            cv2.imwrite(blank_path, np.zeros((100, 100, 3), dtype=np.uint8))

            out_card = os.path.join(tmpdir, "error_card.png")
            img = create_face_forensic_card(
                face_fixtures["david1"],
                blank_path,
                output_path=out_card,
            )
            assert img.size == (1240, 860)
            assert os.path.exists(out_card)


class TestFaceCLI:
    """Tests for CLI subcommands: face-verify, face-compare, face-eval."""

    def test_cli_face_verify(self, face_fixtures, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["forgelens-m1", "face-verify", face_fixtures["david1"], face_fixtures["david2"]],
        )
        main()
        captured = capsys.readouterr()
        assert "VERDICT: [MATCH]" in captured.out
        assert "Similarity Score:" in captured.out

    def test_cli_face_compare(self, face_fixtures, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["forgelens-m1", "face-compare", face_fixtures["david1"], face_fixtures["david2"], "--models", "ArcFace", "SFace"],
        )
        main()
        captured = capsys.readouterr()
        assert "Multi-Model Face Comparison" in captured.out
        assert "Consensus Verdict:" in captured.out

    def test_cli_face_eval(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv",
            ["forgelens-m1", "face-eval", "--pairs", "4", "--cards", "1"],
        )
        main()
        captured = capsys.readouterr()
        assert "Benchmark Face Verification Evaluation" in captured.out
        assert "Verification Accuracy:" in captured.out
