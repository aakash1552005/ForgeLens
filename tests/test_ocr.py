"""
ForgeLens-X — Milestone 3: Unit Tests for OCR & Structured Field Extraction
=============================================================================
Tests optical character recognition pipeline, line extraction, structured
field extraction, and zero-crash exception handling policies.
"""

import os
import numpy as np
import pytest
from PIL import Image

from src.ocr import (
    _clean_field_value,
    extract_structured_fields,
    extract_text_lines,
    get_ocr_engine,
    load_image_for_ocr,
    preprocess_for_ocr,
)


class TestOCREngine:
    """Tests for OCR engine initialization and input loading."""

    def test_get_ocr_engine(self):
        """Test lazy initialization of RapidOCR engine."""
        engine = get_ocr_engine()
        assert engine is not None

    def test_load_image_for_ocr_valid_file(self):
        """Test loading image from path."""
        test_img = "data/test_faces/doc_with_david.jpg"
        if os.path.exists(test_img):
            arr = load_image_for_ocr(test_img)
            assert isinstance(arr, np.ndarray)
            assert arr.ndim == 3
            assert arr.shape[2] == 3

    def test_load_image_for_ocr_pil(self):
        """Test loading PIL Image input."""
        pil_img = Image.new("RGB", (100, 100), (255, 255, 255))
        arr = load_image_for_ocr(pil_img)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (100, 100, 3)

    def test_load_image_for_ocr_invalid(self):
        """Test zero-crash on invalid or missing inputs."""
        assert load_image_for_ocr(None) is None
        assert load_image_for_ocr("non_existent_image_12345.jpg") is None


class TestPreprocessing:
    """Tests for adaptive CLAHE image preprocessing."""

    def test_preprocess_for_ocr_shape_preservation(self):
        """Preprocessed image must match original dimensions."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[20:80, 50:150] = 180
        out = preprocess_for_ocr(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_preprocess_for_ocr_empty_input(self):
        """Empty or None input must not crash."""
        assert preprocess_for_ocr(None) is None
        empty = np.array([])
        assert preprocess_for_ocr(empty).size == 0


class TestCleaningRules:
    """Tests for field string normalization and artifact removal."""

    def test_clean_date_with_label_prefix(self):
        """Extracts date correctly when OCR includes label text."""
        raw = "ExpiryDate 01/10/2040"
        cleaned = _clean_field_value(raw, "date")
        assert cleaned == "01/10/2040"

    def test_clean_date_delimiters(self):
        """Normalizes dots and dashes to slashes."""
        assert _clean_field_value("14.07.1982", "date") == "14/07/1982"
        assert _clean_field_value("25-11-1988", "date") == "25/11/1988"

    def test_clean_document_number(self):
        """Strips whitespace and normalizes case."""
        assert _clean_field_value("fgl- 216739 - 13", "document_number") == "FGL-216739-13"
        assert _clean_field_value("j1029 4819m", "document_number") == "J10294819M"

    def test_clean_name_unsticking(self):
        """Inserts space if OCR concatenated camelCase words."""
        assert _clean_field_value("AryanMaharaj", "text") == "Aryan Maharaj"
        assert _clean_field_value("FullName: Aryan Maharaj", "text") == "Aryan Maharaj"


class TestStructuredFieldExtraction:
    """Tests for canonical structured identity extraction."""

    def test_extract_structured_fields_on_document(self):
        """Validate extraction on realistic test document."""
        test_path = "data/test_faces/doc_with_david.jpg"
        if not os.path.exists(test_path):
            pytest.skip(f"Test document not present: {test_path}")

        res = extract_structured_fields(test_path, engine="rapidocr")
        assert "fields" in res
        assert "time_seconds" in res
        assert res["time_seconds"] > 0

        fields = res["fields"]
        canonical = ["name", "dob", "document_number", "issue_date", "expiry_date"]

        for f in canonical:
            assert f in fields
            item = fields[f]
            assert "value" in item
            assert "confidence" in item
            assert "status" in item
            assert item["status"] in ["EXTRACTED", "LOW_CONFIDENCE", "UNKNOWN"]
            assert 0.0 <= item["confidence"] <= 1.0

        # Verify specific fields on the test document
        assert fields["name"]["value"] == "Aryan Maharaj"
        assert fields["dob"]["value"] == "10/05/1977"
        assert fields["document_number"]["value"] == "FGL-216739-13"

    def test_zero_crash_on_corrupted_or_blank_input(self):
        """OCR pipeline must NEVER crash on blank, black, or noise images."""
        blank = np.zeros((400, 600, 3), dtype=np.uint8)
        res = extract_structured_fields(blank)

        assert "fields" in res
        fields = res["fields"]
        for f in ["name", "dob", "document_number", "issue_date", "expiry_date"]:
            assert fields[f]["status"] in ["UNKNOWN", "LOW_CONFIDENCE"]
            assert fields[f]["confidence"] < 0.50

    def test_zero_crash_on_nonexistent_path(self):
        """OCR pipeline must return valid schema on missing file."""
        res = extract_structured_fields("does_not_exist_987654321.jpg")
        assert "fields" in res
        assert res.get("error") == "invalid_input"
        for f in ["name", "dob", "document_number", "issue_date", "expiry_date"]:
            assert res["fields"][f]["value"] is None
            assert res["fields"][f]["status"] == "UNKNOWN"
