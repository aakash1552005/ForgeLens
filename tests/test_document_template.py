"""Tests for src/document_template.py"""

import pytest
from PIL import Image

from src.document_template import DOC_HEIGHT, DOC_WIDTH, generate_document


class TestDocumentGeneration:
    """Tests for fictional document template generation."""

    def test_generates_image(self):
        result = generate_document("src_test", seed=42)
        assert isinstance(result["image"], Image.Image)
        assert result["image"].size == (DOC_WIDTH, DOC_HEIGHT)

    def test_field_bboxes_present(self):
        result = generate_document("src_test", seed=42)
        expected_fields = [
            "name", "dob", "document_number",
            "issue_date", "expiry_date", "photo", "stamp",
        ]
        for field in expected_fields:
            assert field in result["field_bboxes"], f"Missing field: {field}"
            assert "bbox" in result["field_bboxes"][field]
            assert "value" in result["field_bboxes"][field]
            assert len(result["field_bboxes"][field]["bbox"]) == 4

    def test_metadata_present(self):
        result = generate_document("src_0001", seed=42)
        assert result["metadata"]["source_id"] == "src_0001"
        assert result["metadata"]["seed"] == 42
        assert result["metadata"]["country"] == "FORGELENSIA"

    def test_deterministic(self):
        r1 = generate_document("src_0001", seed=42)
        r2 = generate_document("src_0001", seed=42)
        # Same seed should produce same field values
        assert r1["field_bboxes"]["name"]["value"] == r2["field_bboxes"]["name"]["value"]
        assert r1["field_bboxes"]["dob"]["value"] == r2["field_bboxes"]["dob"]["value"]

    def test_different_seeds_differ(self):
        r1 = generate_document("src_0001", seed=42)
        r2 = generate_document("src_0002", seed=99)
        # Different seeds should produce different names (with high probability)
        assert r1["field_bboxes"]["name"]["value"] != r2["field_bboxes"]["name"]["value"]

    def test_document_number_format(self):
        result = generate_document("src_test", seed=42)
        doc_num = result["field_bboxes"]["document_number"]["value"]
        assert doc_num.startswith("FGL-")
        parts = doc_num.split("-")
        assert len(parts) == 3

    def test_bbox_within_image(self):
        result = generate_document("src_test", seed=42)
        for field_name, field_data in result["field_bboxes"].items():
            bbox = field_data["bbox"]
            assert bbox[0] >= 0, f"{field_name} x1 out of bounds"
            assert bbox[1] >= 0, f"{field_name} y1 out of bounds"
            assert bbox[2] <= DOC_WIDTH, f"{field_name} x2 out of bounds"
            assert bbox[3] <= DOC_HEIGHT, f"{field_name} y2 out of bounds"
