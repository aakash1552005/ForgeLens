"""
Unit Tests for Milestone 4: Stroke Width Transform (SWT) & Typography Forensics
=============================================================================
Tests distance-transform stroke estimation, glyph aspect ratios,
and cross-field typography consistency Z-score auditing.
"""

import cv2
import numpy as np
import pytest

from src.font_forensics import audit_document_font_consistency, extract_field_stroke_metrics


def _create_synthetic_text_crop(text: str, thickness: int = 1, scale: float = 0.6) -> np.ndarray:
    """Helper to render synthetic text crop with specified stroke thickness."""
    crop = np.ones((50, 180, 3), dtype=np.uint8) * 245
    cv2.putText(crop, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return crop


def test_extract_field_stroke_metrics_blank_vs_text():
    """Verify stroke metric extraction on blank crops vs valid text crops."""
    blank = np.ones((50, 150, 3), dtype=np.uint8) * 255
    res_blank = extract_field_stroke_metrics(blank)
    assert res_blank["valid"] is False
    assert res_blank["glyph_count"] == 0

    text_crop = _create_synthetic_text_crop("FGL-104928-19", thickness=2)
    res_text = extract_field_stroke_metrics(text_crop)
    assert res_text["valid"] is True
    assert res_text["glyph_count"] >= 5
    assert res_text["stroke_mean"] > 1.0


def test_audit_font_consistency_insufficient_fields():
    """Verify handling when fewer than 2 text regions are present."""
    doc = np.ones((300, 500, 3), dtype=np.uint8) * 240
    fields = {"name": {"crop": _create_synthetic_text_crop("John Doe")}}
    res = audit_document_font_consistency(doc, fields)
    assert res["typography_verdict"] == "INSUFFICIENT_TEXT_REGIONS"
    assert res["is_consistent"] is True


def test_audit_font_consistency_uniform():
    """Verify that uniform text fonts yield clean TYPOGRAPHY_CONSISTENT verdict."""
    doc = np.ones((300, 500, 3), dtype=np.uint8) * 240
    fields = {
        "name": {"crop": _create_synthetic_text_crop("Alexander Mercer", thickness=2)},
        "document_number": {"crop": _create_synthetic_text_crop("FGL-102948-19", thickness=2)},
        "dob": {"crop": _create_synthetic_text_crop("15/05/1990", thickness=2)},
        "issue_date": {"crop": _create_synthetic_text_crop("10/01/2020", thickness=2)},
    }
    res = audit_document_font_consistency(doc, fields)
    assert res["typography_verdict"] == "TYPOGRAPHY_CONSISTENT"
    assert res["is_consistent"] is True
    assert res["max_stroke_zscore"] < 2.5
    assert len(res["anomalous_fields"]) == 0


def test_audit_font_consistency_spliced_outlier():
    """Verify detection of an anomalous inserted spliced font (significantly thicker stroke)."""
    doc = np.ones((300, 500, 3), dtype=np.uint8) * 240
    # 4 normal fields with thickness=1, and 1 spliced field with thickness=4
    fields = {
        "name": {"crop": _create_synthetic_text_crop("Alexander Mercer", thickness=1)},
        "document_number": {"crop": _create_synthetic_text_crop("FGL-102948-19", thickness=1)},
        "dob": {"crop": _create_synthetic_text_crop("15/05/1990", thickness=1)},
        "issue_date": {"crop": _create_synthetic_text_crop("10/01/2020", thickness=1)},
        "expiry_date": {"crop": _create_synthetic_text_crop("10/01/2030", thickness=4, scale=0.8)},
    }
    res = audit_document_font_consistency(doc, fields)
    assert res["typography_verdict"] == "SUSPECT_FONT_INCONSISTENCY"
    assert res["is_consistent"] is False
    assert res["max_stroke_zscore"] > 2.0
    anom_names = [a["field"] for a in res["anomalous_fields"]]
    assert "expiry_date" in anom_names
