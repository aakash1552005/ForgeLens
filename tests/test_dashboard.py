"""
ForgeLens-X — Milestone 7: Dashboard Unit & Integration Tests
=============================================================
Verifies dashboard utilities, dynamic vector overlays, heatmap blending,
crop extraction, plain-language synthesis, preset samples, and headless execution.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from dashboard.components.header import FACE_PRESETS, PRESET_SAMPLES
from dashboard.utils import (
    extract_region_crops,
    format_plain_language_explanations,
    generate_audit_certificate_dict,
    generate_copy_move_matches_bgr,
    generate_ela_heatmap_bgr,
    get_cached_baseline,
    get_cached_diagnostic_card,
    get_cached_fusion_model,
    render_dynamic_overlay,
    run_screening_pipeline,
)


def test_dashboard_preset_samples_exist():
    """Verify that all canonical preset benchmark samples exist on disk."""
    for key, pdata in PRESET_SAMPLES.items():
        path = pdata["path"]
        assert os.path.exists(path), f"Preset sample missing: {path} for key '{key}'"


def test_dashboard_face_presets_exist():
    """Verify that face pair presets exist on disk when specified."""
    for key, pdata in FACE_PRESETS.items():
        path = pdata["path"]
        if path is not None:
            assert os.path.exists(path), f"Face preset missing: {path} for key '{key}'"


def test_dashboard_caching_utilities():
    """Verify baseline and fusion model caching functions execute cleanly."""
    baseline = get_cached_baseline()
    assert baseline is None or isinstance(baseline, dict)

    fusion_bundle = get_cached_fusion_model()
    assert fusion_bundle is not None
    assert "base_model" in fusion_bundle
    assert "scaler" in fusion_bundle


def test_dashboard_screening_pipeline():
    """Verify that run_screening_pipeline produces complete forensic report."""
    test_img = "data/generated/images/src_0000_date_edit.jpg"
    assert os.path.exists(test_img)

    report = run_screening_pipeline(test_img)
    assert isinstance(report, dict)
    assert report.get("schema_version") == "1.0"
    assert "risk_score" in report
    assert "fraud_probability" in report
    assert "decision" in report
    assert "quality" in report
    assert "suspicious_regions" in report
    assert "risk_drivers" in report

    # Check risk score range
    assert 0.0 <= report["risk_score"] <= 100.0
    assert 0.0 <= report["fraud_probability"] <= 1.0


def test_dashboard_screening_pipeline_with_face():
    """Verify screening pipeline with optional live face selfie."""
    doc_img = "data/generated/images/src_0000_photo_swap.jpg"
    face_img = "data/face_pairs/images/pair_0001_genuine_live.jpg"
    if os.path.exists(doc_img) and os.path.exists(face_img):
        report = run_screening_pipeline(doc_img, reference_face_path=face_img)
        assert isinstance(report, dict)
        assert report.get("face_verification", {}).get("has_face_check") is True


def test_dashboard_dynamic_overlay_rendering():
    """Verify dynamic vector overlay generation across layer toggle configurations."""
    test_img = "data/generated/images/src_0000_date_edit.jpg"
    doc_bgr = cv2.imread(test_img)
    assert doc_bgr is not None
    h, w = doc_bgr.shape[:2]

    report = run_screening_pipeline(test_img)
    heat_bgr = generate_ela_heatmap_bgr(test_img)

    # 1. All layers ON with alpha blending
    overlay_rgb = render_dynamic_overlay(
        doc_bgr=doc_bgr,
        report=report,
        show_ocr=True,
        show_ela=True,
        show_copy_move=True,
        show_suspicious=True,
        ela_opacity=0.3,
        ela_heatmap_bgr=heat_bgr,
    )
    assert isinstance(overlay_rgb, np.ndarray)
    assert overlay_rgb.shape == (h, w, 3)
    assert overlay_rgb.dtype == np.uint8

    # 2. All layers OFF
    overlay_plain = render_dynamic_overlay(
        doc_bgr=doc_bgr,
        report=report,
        show_ocr=False,
        show_ela=False,
        show_copy_move=False,
        show_suspicious=False,
        ela_opacity=0.0,
    )
    assert overlay_plain.shape == (h, w, 3)


def test_dashboard_region_crop_extraction():
    """Verify deep zoom crop extractor generates valid sub-images."""
    test_img = "data/generated/images/src_0000_date_edit.jpg"
    doc_bgr = cv2.imread(test_img)
    heat_bgr = generate_ela_heatmap_bgr(test_img)

    bbox = [100, 100, 250, 200]
    orig_crop, heat_crop = extract_region_crops(doc_bgr, heat_bgr, bbox, padding=15)

    assert isinstance(orig_crop, np.ndarray)
    assert isinstance(heat_crop, np.ndarray)
    assert orig_crop.shape == heat_crop.shape
    assert orig_crop.shape[0] > 0 and orig_crop.shape[1] > 0


def test_dashboard_plain_language_generator():
    """Verify plain-language forensic synthesis produces readable examiner findings."""
    test_img = "data/generated/images/src_0000_date_edit.jpg"
    report = run_screening_pipeline(test_img)

    findings = format_plain_language_explanations(report)
    assert isinstance(findings, list)
    assert len(findings) > 0

    for finding in findings:
        assert "category" in finding
        assert "severity" in finding
        assert "text" in finding
        assert len(finding["text"]) > 10
        # Check that it's human language, not raw debug strings
        assert not finding["text"].startswith("{")
        assert "p99 =" not in finding["text"]


def test_dashboard_audit_certificate_export():
    """Verify formal signed examiner audit certificate generation."""
    test_img = "data/generated/images/src_0000_date_edit.jpg"
    report = run_screening_pipeline(test_img)

    cert = generate_audit_certificate_dict(
        report=report,
        examiner_notes="Watermark and security threads physically confirmed under UV.",
        examiner_id="EXAMINER-772",
        action_taken="ESCALATED_PHYSICAL_REVIEW",
    )

    assert cert["certificate_type"] == "FORGELENS_X_EXAMINER_AUDIT_CERTIFICATE"
    assert cert["examiner_id"] == "EXAMINER-772"
    assert cert["action_taken"] == "ESCALATED_PHYSICAL_REVIEW"
    assert "UV" in cert["examiner_notes"]
    assert "calibrated_risk_score" in cert
    assert "issued_at_utc" in cert


def test_dashboard_cached_diagnostic_card():
    """Verify 4-panel master diagnostic card generation."""
    test_img = "data/generated/images/src_0000_date_edit.jpg"
    card_rgb = get_cached_diagnostic_card(test_img)
    assert isinstance(card_rgb, np.ndarray)
    assert card_rgb.shape[0] > 700
    assert card_rgb.shape[1] > 1100


def test_dashboard_copy_move_generator():
    """Verify copy-move match visualization canvas generator."""
    test_img = "data/generated/images/src_0000_copy_move.jpg"
    cm_canvas = generate_copy_move_matches_bgr(test_img)
    assert isinstance(cm_canvas, np.ndarray)
    assert cm_canvas.shape[:2] == cv2.imread(test_img).shape[:2]
