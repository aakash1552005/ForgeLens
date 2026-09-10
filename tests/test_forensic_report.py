"""
Unit Tests for Milestone 5: Unified Forensic Report & Pipeline Integration
===========================================================================
Validates Schema 1.0 compliance, quality-aware gating, multi-source suspicious
regions correlation, heuristic attack classification, and turnkey M6 feature vector extraction.
"""

import json
import os
import tempfile
import cv2
import numpy as np
import pytest

from src.forensic_report import (
    analyze_document_quality,
    classify_attack_heuristic,
    correlate_suspicious_regions,
    export_unified_report,
    extract_m6_feature_vector,
    generate_unified_forensic_report,
)


def test_quality_analysis_sharp_and_blurry():
    """Verify Laplacian variance and resolution checking."""
    # 1. Sharp clean canvas with crisp text
    sharp_img = np.full((650, 1000, 3), 245, dtype=np.uint8)
    for y in range(50, 600, 40):
        cv2.line(sharp_img, (50, y), (950, y), (10, 10, 10), 2)

    sharp_qual = analyze_document_quality(sharp_img)
    assert sharp_qual["blur_score"] > 45.0
    assert sharp_qual["resolution_ok"] is True
    assert sharp_qual["analysis_reliability"] in ["HIGH", "MEDIUM"]

    # 2. Heavily blurred canvas
    blurry_img = cv2.GaussianBlur(sharp_img, (31, 31), 0)
    blurry_qual = analyze_document_quality(blurry_img)
    assert blurry_qual["blur_score"] < 45.0
    assert blurry_qual["analysis_reliability"] == "LOW"
    assert any("HEAVY_BLUR" in f for f in blurry_qual["quality_flags"])

    # 3. Small resolution canvas
    tiny_img = np.full((150, 200, 3), 200, dtype=np.uint8)
    tiny_qual = analyze_document_quality(tiny_img)
    assert tiny_qual["resolution_ok"] is False
    assert tiny_qual["analysis_reliability"] == "LOW"


def test_quality_gating_decision():
    """Verify that low-quality images route to INSUFFICIENT_EVIDENCE rather than false fraud."""
    low_qual = {
        "blur_score": 12.0,
        "resolution_ok": False,
        "analysis_reliability": "LOW",
        "quality_flags": ["HEAVY_BLUR", "LOW_RESOLUTION"],
    }
    tamper_signals = {
        "ela": {"features": {"candidate_energy": 25.0}},
        "copy_move": {"num_matches": 0},
    }
    attack, conf, basis, decision = classify_attack_heuristic(
        tamper_signals=tamper_signals,
        semantic_checks=[],
        font_audit=None,
        face_verification={"has_face_check": False},
        suspicious_regions=[],
        quality=low_qual,
    )
    assert decision == "INSUFFICIENT_EVIDENCE"
    assert attack == "none"
    assert conf == 0.0
    assert any("quality is degraded" in b for b in basis)


def test_correlate_suspicious_regions_ela_and_fields():
    """Verify spatial mapping of ELA candidates to named identity fields."""
    fields = {
        "name": {"bbox": [100, 100, 400, 140], "value": "John Doe"},
        "dob": {"bbox": [100, 160, 300, 200], "value": "15/05/1990"},
        "expiry_date": {"bbox": [100, 220, 300, 260], "value": "15/05/2030"},
    }

    # ELA anomaly directly over DOB field
    ela_cand = {
        "bbox": [95, 155, 305, 205],
        "energy": 120.0,
        "mean_anomaly": 2.4,
    }

    regions = correlate_suspicious_regions(fields=fields, ela_candidate=ela_cand)
    assert len(regions) == 1
    reg = regions[0]
    assert reg["source"] == "ela"
    assert reg["field"] == "dob"
    assert reg["confidence"] > 0.60
    assert "directly overlapping field 'dob'" in reg["evidence"]


def test_classify_attack_heuristic_variants():
    """Verify heuristic classification across date_edit, text_edit, copy_move, and photo_swap."""
    qual = {"analysis_reliability": "HIGH", "quality_flags": []}

    # 1. Date Edit: ELA over DOB + semantic date failure
    date_reg = [{"source": "ela", "field": "dob", "confidence": 0.85, "evidence": "ELA over DOB"}]
    date_sem = [{"check": "impossible_dates", "status": "FAIL", "detail": "Feb 30"}]
    att, conf, basis, dec = classify_attack_heuristic(
        tamper_signals={"copy_move": {"num_matches": 0}},
        semantic_checks=date_sem,
        font_audit=None,
        face_verification={"has_face_check": False},
        suspicious_regions=date_reg,
        quality=qual,
    )
    assert att == "date_edit"
    assert conf >= 0.70
    assert dec == "SUSPECT_TAMPERING"

    # 2. Text Edit: Typography outlier over name + schema failure
    text_reg = [{"source": "typography", "field": "name", "confidence": 0.85, "evidence": "Font Z-score"}]
    text_sem = [{"check": "name_structure_sanity", "status": "FAIL", "detail": "Dummy tokens"}]
    att_t, conf_t, basis_t, dec_t = classify_attack_heuristic(
        tamper_signals={"copy_move": {"num_matches": 0}},
        semantic_checks=text_sem,
        font_audit={"typography_verdict": "SUSPECT_FONT_INCONSISTENCY", "anomalous_fields": [{"field": "name"}]},
        face_verification={"has_face_check": False},
        suspicious_regions=text_reg,
        quality=qual,
    )
    assert att_t == "text_edit"
    assert dec_t == "SUSPECT_TAMPERING"

    # 3. Copy-Move: 25 matches with RANSAC
    cm_signals = {"copy_move": {"num_matches": 25, "confidence": 0.90}}
    att_cm, conf_cm, basis_cm, dec_cm = classify_attack_heuristic(
        tamper_signals=cm_signals,
        semantic_checks=[],
        font_audit=None,
        face_verification={"has_face_check": False},
        suspicious_regions=[],
        quality=qual,
    )
    assert att_cm == "copy_move"
    assert conf_cm >= 0.70

    # 4. Photo Swap: Face verification mismatch
    face_mismatch = {"has_face_check": True, "verified": False, "distance": 0.65, "threshold": 0.40}
    att_p, conf_p, basis_p, dec_p = classify_attack_heuristic(
        tamper_signals={"copy_move": {"num_matches": 0}},
        semantic_checks=[],
        font_audit=None,
        face_verification=face_mismatch,
        suspicious_regions=[{"source": "face", "field": "photo", "confidence": 0.90}],
        quality=qual,
    )
    assert att_p == "photo_swap"
    assert dec_p == "SUSPECT_TAMPERING"

    # 5. None (Genuine): Zero anomalies
    att_n, conf_n, basis_n, dec_n = classify_attack_heuristic(
        tamper_signals={"copy_move": {"num_matches": 0}},
        semantic_checks=[],
        font_audit=None,
        face_verification={"has_face_check": False},
        suspicious_regions=[],
        quality=qual,
    )
    assert att_n == "none"
    assert conf_n == 0.0
    assert dec_n == "CLEAR_AUTHENTIC"


def test_unified_report_schema_contract(tmp_path):
    """Verify that generate_unified_forensic_report returns full Schema 1.0 compliance."""
    # Create test image
    test_img_path = str(tmp_path / "test_doc.jpg")
    img = np.full((650, 1000, 3), 240, dtype=np.uint8)
    cv2.putText(img, "REPUBLIC OF FORGELENSIA", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "NAME: VALENTINA ROSSI", (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "DOB: 12/04/1988", (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "DOC NO: FGL-582910-18", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(test_img_path, img)

    report = generate_unified_forensic_report(test_img_path)

    # Required Schema 1.0 keys
    required_keys = [
        "schema_version", "document_id", "document_type", "quality", "fields",
        "semantic_checks", "tamper_signals", "face_verification",
        "suspicious_regions", "attack_type_guess", "attack_type_confidence",
        "attack_type_basis", "risk_score", "fraud_probability", "decision",
        "feature_vector"
    ]
    for k in required_keys:
        assert k in report, f"Missing required Schema 1.0 key: {k}"

    assert report["schema_version"] == "1.0"
    assert report["quality"]["resolution_ok"] is True
    assert "ela" in report["tamper_signals"]
    assert "copy_move" in report["tamper_signals"]
    assert report["risk_score"] is None
    assert report["fraud_probability"] is None


def test_m6_feature_vector_numeric():
    """Verify that extract_m6_feature_vector returns a valid dictionary of numbers."""
    dummy_report = {
        "tamper_signals": {
            "ela": {"features": {"mean": 5.2, "std": 1.1, "max": 25.0, "p95": 8.0, "p99": 14.0, "high_error_pixel_ratio": 0.01}},
            "copy_move": {"num_matches": 0, "confidence": 0.0},
        },
        "semantic_checks": [{"check": "chronology_order", "status": "PASS"}],
        "mrz": {"status": "PASS"},
        "face_verification": {"has_face_check": False, "distance": None},
        "font_forensics": {"max_stroke_zscore": 1.2, "typography_verdict": "UNIFORM_TYPOGRAPHY"},
        "metadata_forensics": {"is_tampered": False, "has_exif": True},
        "quality": {"blur_score": 120.0, "resolution_ok": True, "ocr_mean_confidence": 0.92, "analysis_reliability": "HIGH"},
        "suspicious_regions": [],
    }

    f_vec = extract_m6_feature_vector(dummy_report)
    assert isinstance(f_vec, dict)
    assert len(f_vec) >= 15
    for k, v in f_vec.items():
        assert isinstance(v, (int, float)), f"Feature {k} is not numeric: {v}"
        assert not np.isnan(v), f"Feature {k} is NaN"


def test_export_unified_report_serialization(tmp_path):
    """Verify JSON serialization and file writing."""
    out_json = str(tmp_path / "report.json")
    dummy_report = {
        "schema_version": "1.0",
        "document_id": "test_01",
        "num_val": np.int64(42),
        "float_val": np.float32(3.1415),
        "array_val": np.array([1, 2, 3]),
    }
    json_str = export_unified_report(dummy_report, json_path=out_json)
    assert os.path.exists(out_json)
    loaded = json.loads(json_str)
    assert loaded["num_val"] == 42
    assert loaded["array_val"] == [1, 2, 3]


def test_render_unified_forensic_card(tmp_path):
    """Verify master 4-panel diagnostic card generation."""
    from src.unified_visualize import render_unified_forensic_card

    card_path = str(tmp_path / "master_card.png")
    doc_bgr = np.full((650, 1000, 3), 230, dtype=np.uint8)
    dummy_report = {
        "document_id": "test_card_doc",
        "decision": "CLEAR_AUTHENTIC",
        "attack_type_guess": "none",
        "attack_type_confidence": 0.0,
        "attack_type_basis": ["Document verified authentic"],
        "quality": {"blur_score": 140.0, "analysis_reliability": "HIGH", "dimensions": [1000, 650]},
        "fields": {"name": {"bbox": [100, 100, 300, 140], "value": "ALICE SMITH", "confidence": 0.98}},
        "tamper_signals": {"ela": {}, "copy_move": {}},
        "face_verification": {"has_face_check": False},
        "suspicious_regions": [],
        "mrz": {"format": "TD1", "status": "PASS", "verdict": "VERIFIED"},
    }

    card_img, saved = render_unified_forensic_card(doc_bgr, dummy_report, output_path=card_path)
    assert card_img.shape == (870, 1280, 3)
    assert saved == card_path
    assert os.path.exists(card_path)


def test_nonexistent_image_zero_crash():
    """Verify zero-crash policy on nonexistent file paths."""
    report = generate_unified_forensic_report("nonexistent_path_file_999.jpg")
    assert report["schema_version"] == "1.0"
    assert report["decision"] == "INSUFFICIENT_EVIDENCE"
    assert report["quality"]["analysis_reliability"] == "LOW"
    assert "FILE_NOT_FOUND_OR_CORRUPT" in report["quality"]["quality_flags"]


def test_cli_unified_screen_command(tmp_path, monkeypatch, capsys):
    """Verify CLI unified-screen command dispatch."""
    from src.cli import main

    test_img = str(tmp_path / "cli_test_doc.jpg")
    img = np.full((650, 1000, 3), 240, dtype=np.uint8)
    cv2.putText(img, "FORGELENSIA TEST DOC", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(test_img, img)

    out_card = str(tmp_path / "cli_card.png")
    out_json = str(tmp_path / "cli_report.json")

    monkeypatch.setattr(
        "sys.argv",
        ["forgelens-m1", "unified-screen", test_img, "--output", out_card, "--json", out_json],
    )
    main()

    captured = capsys.readouterr()
    assert "Milestone 5: Unified Forensic Screening" in captured.out
    assert "Decision Tier" in captured.out
    assert os.path.exists(out_card)
    assert os.path.exists(out_json)


def test_cli_unified_screen_selfie_flag(tmp_path, monkeypatch, capsys):
    """Verify CLI unified-screen command supports --selfie alias alongside --face."""
    from src.cli import main

    test_img = str(tmp_path / "cli_test_doc.jpg")
    img = np.full((650, 1000, 3), 240, dtype=np.uint8)
    cv2.putText(img, "FORGELENSIA TEST DOC", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(test_img, img)

    test_selfie = str(tmp_path / "cli_selfie.jpg")
    selfie = np.full((300, 300, 3), 180, dtype=np.uint8)
    cv2.imwrite(test_selfie, selfie)

    monkeypatch.setattr(
        "sys.argv",
        ["forgelens-m1", "unified-screen", test_img, "--selfie", test_selfie, "--no-card"],
    )
    main()

    captured = capsys.readouterr()
    assert "Milestone 5: Unified Forensic Screening" in captured.out
    assert "Reference Face:" in captured.out

