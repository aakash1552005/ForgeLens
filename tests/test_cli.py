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
