"""
ForgeLens-X — Tests for OCR Forensic Cross-Modality Bridge (M1 x M3)
===================================================================
Tests for:
    - Bounding box intersection and overlap calculation
    - Spatial correlation between M1 physical tampering (ELA / Copy-Move) and M3 text fields
    - Automated flagging of field-specific alterations
"""

import pytest

from src.ocr_forensic_bridge import (
    compute_box_intersection_area,
    correlate_tamper_with_fields,
)


def test_compute_box_intersection_area_disjoint():
    """Verify non-overlapping boxes have zero intersection area."""
    box_a = [0, 0, 50, 50]
    box_b = [100, 100, 150, 150]
    assert compute_box_intersection_area(box_a, box_b) == 0.0


def test_compute_box_intersection_area_identical():
    """Verify identical boxes have intersection area equal to full box area."""
    box_a = [10, 20, 60, 70]  # width 50, height 50
    assert compute_box_intersection_area(box_a, box_a) == 2500.0


def test_compute_box_intersection_area_partial():
    """Verify partial overlap computes exact intersection rectangle."""
    box_a = [0, 0, 100, 100]
    box_b = [50, 50, 150, 150]
    # Overlap is [50, 50, 100, 100] -> 50 x 50 = 2500
    assert compute_box_intersection_area(box_a, box_b) == 2500.0


def test_compute_box_intersection_area_invalid():
    """Verify malformed or None bounding boxes return 0.0 without error."""
    assert compute_box_intersection_area(None, [0, 0, 10, 10]) == 0.0
    assert compute_box_intersection_area([0, 0, 10], [0, 0, 10, 10]) == 0.0
    assert compute_box_intersection_area([50, 50, 10, 10], [0, 0, 10, 10]) == 0.0


def test_correlate_tamper_clean_document():
    """Verify clean document without tamper candidates returns clean status."""
    fields = {
        "name": {"value": "John Doe", "bbox": [100, 50, 300, 80]},
        "dob": {"value": "01/01/1990", "bbox": [100, 100, 250, 130]},
    }
    tamper_signals = {
        "ela_detected": False,
        "ela_candidate_bbox": None,
        "copy_move_detected": False,
        "copy_move_bbox": None,
    }

    result = correlate_tamper_with_fields(fields, tamper_signals)
    assert result["status"] == "FIELD_ALTERATION_CLEAN"
    assert result["tampered_fields_count"] == 0
    assert len(result["tampered_field_names"]) == 0
    assert result["tampered_fields"]["name"]["is_tampered"] is False
    assert result["tampered_fields"]["dob"]["is_tampered"] is False


def test_correlate_tamper_ela_overlap():
    """Verify ELA anomaly overlapping an issue date flags that exact field."""
    fields = {
        "name": {"value": "John Doe", "bbox": [100, 50, 300, 80]},
        "issue_date": {"value": "15/05/2021", "bbox": [100, 150, 250, 180]},
    }
    tamper_signals = {
        "ela_detected": True,
        "ela_candidate_bbox": [95, 145, 255, 185],  # Encompasses issue_date
        "copy_move_detected": False,
        "copy_move_bbox": None,
    }

    result = correlate_tamper_with_fields(fields, tamper_signals, min_overlap_area=20.0)
    assert result["status"] == "FIELD_ALTERATION_DETECTED"
    assert result["tampered_fields_count"] == 1
    assert "issue_date" in result["tampered_field_names"]
    assert "name" not in result["tampered_field_names"]
    assert result["tampered_fields"]["issue_date"]["is_tampered"] is True
    assert result["tampered_fields"]["issue_date"]["overlap_type"] == "ELA_ANOMALY"


def test_correlate_tamper_copy_move_overlap():
    """Verify Copy-Move clone overlapping document_number flags that field."""
    fields = {
        "document_number": {"value": "FGL-123456", "bbox": [100, 200, 260, 235]},
    }
    tamper_signals = {
        "ela_detected": False,
        "ela_candidate_bbox": None,
        "copy_move_detected": True,
        "copy_move_bbox": [105, 205, 255, 230],
    }

    result = correlate_tamper_with_fields(fields, tamper_signals, min_overlap_area=20.0)
    assert result["status"] == "FIELD_ALTERATION_DETECTED"
    assert "document_number" in result["tampered_field_names"]
    assert result["tampered_fields"]["document_number"]["overlap_type"] == "COPY_MOVE_CLONE"


def test_correlate_tamper_non_overlapping_tamper():
    """Verify tampering in non-text areas (e.g., photo box or background) does not falsely flag text fields."""
    fields = {
        "name": {"value": "John Doe", "bbox": [200, 50, 400, 80]},
    }
    tamper_signals = {
        "ela_detected": True,
        "ela_candidate_bbox": [20, 50, 120, 180],  # Photo region on left
    }

    result = correlate_tamper_with_fields(fields, tamper_signals)
    assert result["status"] == "FIELD_ALTERATION_CLEAN"
    assert result["tampered_fields_count"] == 0
    assert result["tampered_fields"]["name"]["is_tampered"] is False
