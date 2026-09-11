"""Tests for src/cli.py"""

import os
import tempfile
import pytest

from src.cli import main
from src.document_template import generate_document
from src.tamper_generator import generate_tampered_dataset


class TestCLI:
    """Tests for CLI entry points."""

    def test_cli_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["forgelens-m1", "--help"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "screen" in captured.out
        assert "run-demo" in captured.out

    def test_screen_command(self, monkeypatch, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate a test document and sample
            doc = generate_document("src_test_cli", seed=101)
            results = generate_tampered_dataset(doc, tmpdir, seed=101)
            img_path = results[0]["image_path"]
            out_card = os.path.join(tmpdir, "screen_card.png")

            monkeypatch.setattr(
                "sys.argv",
                ["forgelens-m1", "screen", img_path, "--output", out_card],
            )
            main()

            captured = capsys.readouterr()
            assert "ForgeLens-X — Forensic Document Screening" in captured.out
            assert "VERDICT:" in captured.out
            assert os.path.exists(out_card)

    def test_screen_identity_cli(self, monkeypatch, capsys):
        test_faces_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_faces")
        doc_path = os.path.join(test_faces_dir, "doc_with_david.jpg")
        live_path = os.path.join(test_faces_dir, "david2.jpg")

        if not os.path.exists(doc_path):
            pytest.skip("doc_with_david fixture not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_card = os.path.join(tmpdir, "id_screen_card.png")
            monkeypatch.setattr(
                "sys.argv",
                ["forgelens-m1", "screen-identity", doc_path, live_path, "--output", out_card],
            )
            main()

            captured = capsys.readouterr()
            assert "End-to-End Identity Screening" in captured.out
            assert "OVERALL SCREENING VERDICT:" in captured.out
            assert os.path.exists(out_card)

    def test_train_fusion_cli(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["forgelens-m1", "train-fusion"])
        main()
        captured = capsys.readouterr()
        assert "Train Document-Risk Fusion Model" in captured.out
        assert "Milestone 6 Training Summary" in captured.out

    def test_evaluate_fusion_cli(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["forgelens-m1", "evaluate-fusion"])
        main()
        captured = capsys.readouterr()
        assert "Risk Fusion Test Evaluation" in captured.out
        assert "ROC-AUC:" in captured.out

    def test_score_risk_cli(self, monkeypatch, capsys):
        from src.utils import get_generated_dir
        sample = get_generated_dir() / "images" / "src_0000_genuine.jpg"
        if not sample.exists():
            pytest.skip("Sample genuine doc not generated")

        monkeypatch.setattr("sys.argv", ["forgelens-m1", "score-risk", str(sample)])
        main()
        captured = capsys.readouterr()
        assert "Calibrated Document Risk Scoring" in captured.out
        assert "Calibrated Fraud Probability" in captured.out
        assert "Operational Decision" in captured.out

    def test_system_audit_cli(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["forgelens-m1", "system-audit"])
        main()
        captured = capsys.readouterr()
        assert "Master System Diagnostic & Integration Health Audit" in captured.out
        assert "M6_Risk_Fusion_ML" in captured.out
        assert "Overall System Health : PASS" in captured.out

