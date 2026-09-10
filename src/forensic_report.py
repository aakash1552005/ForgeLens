"""
ForgeLens-X — Milestone 5: Unified Forensic Report Engine
=========================================================
Integrates Milestone 1 (ELA & Copy-Move physical tamper detection),
Milestone 2 (Biometric face verification & portrait extraction),
Milestone 3 (Optical character recognition & field localization), and
Milestone 4 (Semantic rules, ICAO Doc 9303 MRZ, EXIF/XMP, & Typography forensics)
into ONE unified, stable, schema-validated forensic evidence contract.

Schema Version: 1.0 (Canonical ForgeLens-X Contract)
"""

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

from src.copy_move import detect_copy_move
from src.ela import analyze_ela, compute_baseline
from src.face_verify import verify as verify_document_face
from src.font_forensics import audit_document_font_consistency
from src.identity_screener import extract_face_from_document
from src.metadata_forensics import audit_metadata_provenance, extract_image_metadata
from src.mrz import cross_validate_viz_and_mrz, parse_mrz
from src.ocr import extract_structured_fields
from src.ocr_forensic_bridge import compute_box_intersection_area, correlate_tamper_with_fields
from src.semantic_checks import run_semantic_rule_battery
from src.utils import bbox_from_mask, compute_iou, ensure_dirs, get_reports_dir


def _load_m5_config() -> Dict[str, Any]:
    """Load configuration from configs/m5_config.yaml."""
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "m5_config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "quality": {
            "min_blur_score": 45.0,
            "medium_blur_score": 80.0,
            "min_width": 600,
            "min_height": 400,
            "min_ocr_confidence": 0.40,
            "high_ocr_confidence": 0.75,
            "enable_quality_gating": True,
        },
        "spatial_correlation": {
            "min_iofa_overlap": 0.22,
            "min_iou_overlap": 0.08,
            "photo_region_bbox": [30, 80, 200, 260],
        },
        "attack_heuristics": {
            "ela_anomaly_energy_threshold": 75.0,
            "copy_move_min_matches": 15,
            "face_mismatch_distance_threshold": 0.40,
            "font_zscore_threshold": 2.50,
        },
    }


_DEFAULT_BASELINE: Optional[Dict[str, Any]] = None


def get_default_baseline() -> Optional[Dict[str, Any]]:
    """Retrieve or lazily compute baseline from genuine document samples."""
    global _DEFAULT_BASELINE
    if _DEFAULT_BASELINE is not None:
        return _DEFAULT_BASELINE

    gen_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "generated", "images")
    if os.path.exists(gen_dir):
        gen_paths = sorted([
            os.path.join(gen_dir, f) for f in os.listdir(gen_dir)
            if f.endswith("_genuine.jpg") or f.endswith("_none.jpg")
        ])[:15]
        if gen_paths:
            try:
                _DEFAULT_BASELINE = compute_baseline(gen_paths, quality=90)
            except Exception:
                _DEFAULT_BASELINE = None
    return _DEFAULT_BASELINE


# ---------------------------------------------------------------------------
# 1. Quality & Reliability Analysis Layer
# ---------------------------------------------------------------------------

def analyze_document_quality(
    image_bgr: np.ndarray,
    ocr_fields: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assess optical document scan quality to prevent false fraud accusations.
    Evaluates sharpness (Laplacian variance), dimensional resolution, and OCR confidence.

    Returns:
        {
            "blur_score": float,
            "resolution_ok": bool,
            "dimensions": [width, height],
            "ocr_mean_confidence": float or null,
            "analysis_reliability": "HIGH" | "MEDIUM" | "LOW",
            "quality_flags": list of str
        }
    """
    if config is None:
        config = _load_m5_config()
    q_cfg = config.get("quality", {})
    min_blur = q_cfg.get("min_blur_score", 45.0)
    med_blur = q_cfg.get("medium_blur_score", 80.0)
    min_w = q_cfg.get("min_width", 600)
    min_h = q_cfg.get("min_height", 400)
    min_ocr = q_cfg.get("min_ocr_confidence", 0.40)
    high_ocr = q_cfg.get("high_ocr_confidence", 0.75)

    if image_bgr is None or image_bgr.size == 0:
        return {
            "blur_score": 0.0,
            "resolution_ok": False,
            "dimensions": [0, 0],
            "ocr_mean_confidence": 0.0,
            "analysis_reliability": "LOW",
            "quality_flags": ["EMPTY_OR_UNREADABLE_IMAGE"],
        }

    h, w = image_bgr.shape[:2]
    resolution_ok = bool(w >= min_w and h >= min_h)

    # Compute sharpness via Laplacian variance
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = round(laplacian_var, 2)

    # Compute OCR confidence across recognized fields
    ocr_confs = []
    if ocr_fields:
        for f_name, f_data in ocr_fields.items():
            if f_name.startswith("_") or not isinstance(f_data, dict):
                continue
            conf = f_data.get("confidence")
            if conf is not None:
                ocr_confs.append(float(conf))

    ocr_mean = round(float(np.mean(ocr_confs)), 3) if ocr_confs else None

    flags = []
    if blur_score < min_blur:
        flags.append(f"HEAVY_BLUR (score={blur_score} < {min_blur})")
    elif blur_score < med_blur:
        flags.append(f"MODERATE_BLUR (score={blur_score})")

    if not resolution_ok:
        flags.append(f"LOW_RESOLUTION ({w}x{h} < {min_w}x{min_h})")

    if ocr_mean is not None and ocr_mean < min_ocr:
        flags.append(f"LOW_OCR_CONFIDENCE ({ocr_mean} < {min_ocr})")

    # Determine analysis reliability tier
    if blur_score < min_blur or not resolution_ok or (ocr_mean is not None and ocr_mean < min_ocr):
        reliability = "LOW"
    elif blur_score < med_blur or (ocr_mean is not None and ocr_mean < high_ocr):
        reliability = "MEDIUM"
    else:
        reliability = "HIGH"

    return {
        "blur_score": blur_score,
        "resolution_ok": resolution_ok,
        "dimensions": [int(w), int(h)],
        "ocr_mean_confidence": ocr_mean,
        "analysis_reliability": reliability,
        "quality_flags": flags,
    }


# ---------------------------------------------------------------------------
# 2. Multi-Source Suspicious Region Correlation
# ---------------------------------------------------------------------------

def correlate_suspicious_regions(
    fields: Dict[str, Any],
    ela_candidate: Optional[Dict[str, Any]] = None,
    copy_move_res: Optional[Dict[str, Any]] = None,
    semantic_audit: Optional[Dict[str, Any]] = None,
    font_audit: Optional[Dict[str, Any]] = None,
    face_res: Optional[Dict[str, Any]] = None,
    doc_shape: Optional[Tuple[int, int]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Cross-reference detected physical and semantic anomalies with OCR field bboxes.
    Constructs standardized suspicious regions with exact bounding boxes, associated fields,
    confidence metrics, and human-readable evidence.
    """
    if config is None:
        config = _load_m5_config()
    sp_cfg = config.get("spatial_correlation", {})
    min_iofa = sp_cfg.get("min_iofa_overlap", 0.15)
    default_photo_box = sp_cfg.get("photo_region_bbox", [30, 80, 200, 260])

    suspicious_regions = []

    # Helper: Find overlapping field for a given bounding box
    def _find_best_field(target_box: List[int]) -> Tuple[Optional[str], float]:
        if not target_box or len(target_box) < 4:
            return None, 0.0
        best_f = None
        best_ratio = 0.0

        for fname, fdata in fields.items():
            if fname.startswith("_") or not isinstance(fdata, dict):
                continue
            fbox = fdata.get("bbox")
            if not fbox or len(fbox) < 4:
                continue

            inter_area = compute_box_intersection_area(target_box, fbox)
            f_area = max(1, (fbox[2] - fbox[0]) * (fbox[3] - fbox[1]))
            t_area = max(1, (target_box[2] - target_box[0]) * (target_box[3] - target_box[1]))
            ratio_f = inter_area / f_area
            ratio_t = inter_area / t_area
            is_match = (ratio_f >= min_iofa) or (ratio_t >= 0.50 and ratio_f >= 0.10)
            if is_match and ratio_f > best_ratio:
                best_ratio = ratio_f
                best_f = fname

        return best_f, best_ratio

    # 1. ELA Physical Tamper Region
    if ela_candidate and ela_candidate.get("bbox"):
        e_box = ela_candidate["bbox"]
        matched_f, overlap = _find_best_field(e_box)
        energy = float(ela_candidate.get("energy", 0.0))
        mean_anom = float(ela_candidate.get("mean_anomaly", 0.0))
        conf = min(0.98, round(0.40 + (energy / 200.0) * 0.50, 2))

        # Check if overlaps portrait area only if no named OCR field matched
        if matched_f is None:
            ec_x = (e_box[0] + e_box[2]) / 2.0
            ec_y = (e_box[1] + e_box[3]) / 2.0
            inside_photo = (
                default_photo_box[0] <= ec_x < default_photo_box[2]
                and default_photo_box[1] <= ec_y < default_photo_box[3]
            )
            p_inter = compute_box_intersection_area(e_box, default_photo_box)
            p_area = max(1, (default_photo_box[2] - default_photo_box[0]) * (default_photo_box[3] - default_photo_box[1]))
            e_area = max(1, (e_box[2] - e_box[0]) * (e_box[3] - e_box[1]))
            if inside_photo or (p_inter / e_area >= 0.20) or (p_inter / p_area >= 0.15):
                matched_f = "photo"

        field_name = matched_f if matched_f else "unassigned_canvas"
        evidence_str = (
            f"Physical JPEG compression discontinuity (energy={energy:.1f}, mean={mean_anom:.2f})"
            + (f" directly overlapping field '{matched_f}' ({overlap*100:.1f}% coverage)" if matched_f else "")
        )

        suspicious_regions.append({
            "source": "ela",
            "bbox": [int(x) for x in e_box],
            "field": field_name,
            "confidence": conf,
            "evidence": evidence_str,
        })

    # 2. Copy-Move Forgery Motif
    if copy_move_res and copy_move_res.get("detected"):
        c_box = copy_move_res.get("candidate_bbox")
        alt_box = copy_move_res.get("alt_bbox")
        n_matches = copy_move_res.get("num_matches", 0)
        conf = min(0.99, round(float(copy_move_res.get("confidence", 0.85)), 2))

        if c_box:
            matched_f, _ = _find_best_field(c_box)
            suspicious_regions.append({
                "source": "copy_move",
                "bbox": [int(x) for x in c_box],
                "field": matched_f if matched_f else "cloned_motif_receiver",
                "confidence": conf,
                "evidence": f"ORB-RANSAC duplicated motif candidate with {n_matches} inlier keypoint correspondences",
            })
        if alt_box:
            matched_f, _ = _find_best_field(alt_box)
            suspicious_regions.append({
                "source": "copy_move",
                "bbox": [int(x) for x in alt_box],
                "field": matched_f if matched_f else "cloned_motif_source",
                "confidence": conf,
                "evidence": f"Corresponding donor region for duplicated motif ({n_matches} keypoint matches)",
            })

    # 3. Typography Font Splicing Outliers
    if font_audit:
        for anom in font_audit.get("anomalous_fields", []):
            fname = anom.get("field")
            z_score = float(anom.get("stroke_zscore", 0.0))
            fbox = fields.get(fname, {}).get("bbox") if isinstance(fields.get(fname), dict) else None
            if fbox and len(fbox) == 4:
                conf = min(0.95, round(0.50 + min(abs(z_score) / 5.0, 0.45), 2))
                suspicious_regions.append({
                    "source": "typography",
                    "bbox": [int(x) for x in fbox],
                    "field": fname,
                    "confidence": conf,
                    "evidence": f"Stroke-width transform outlier (|Z|={abs(z_score):.2f} > 2.50, mean={anom.get('field_mean_stroke', 0.0):.2f}px)",
                })

    # 4. Semantic Rule Failures Mapped to Fields
    if semantic_audit:
        for chk_name in semantic_audit.get("failed_checks", []):
            chk_obj = semantic_audit.get("checks", {}).get(chk_name, {})
            detail = chk_obj.get("detail", f"Semantic check {chk_name} failed")

            # Map semantic check to fields
            target_fields = []
            if chk_name in ["impossible_dates", "chronology_order", "age_at_issue_sanity", "validity_window_sanity", "anachronism_check"]:
                target_fields = ["dob", "issue_date", "expiry_date"]
            elif chk_name == "document_number_format":
                target_fields = ["document_number"]
            elif chk_name == "name_structure_sanity":
                target_fields = ["name"]
            elif chk_name == "country_code_sanity":
                target_fields = ["country", "nationality"]

            for tf in target_fields:
                if tf in fields and isinstance(fields[tf], dict):
                    tf_box = fields[tf].get("bbox")
                    if tf_box and len(tf_box) == 4:
                        suspicious_regions.append({
                            "source": "semantic",
                            "bbox": [int(x) for x in tf_box],
                            "field": tf,
                            "confidence": 0.90,
                            "evidence": f"Logical rule failure ({chk_name}): {detail}",
                        })

    # 5. Face Verification Mismatch (Photo Swap)
    if face_res and face_res.get("has_face_check"):
        if face_res.get("verified") is False and face_res.get("distance") is not None:
            dist = float(face_res["distance"])
            thresh = float(face_res.get("threshold", 0.40))
            p_box = face_res.get("facial_areas", {}).get("document_face") or default_photo_box
            conf = min(0.99, round(0.50 + min((dist - thresh) / 0.50, 0.45), 2))
            suspicious_regions.append({
                "source": "face",
                "bbox": [int(x) for x in p_box] if p_box else default_photo_box,
                "field": "photo",
                "confidence": conf,
                "evidence": f"Biometric face mismatch (distance={dist:.3f} > threshold={thresh:.3f}, similarity={face_res.get('similarity_pct', 0.0)}%)",
            })

    return suspicious_regions


# ---------------------------------------------------------------------------
# 3. Heuristic Attack Classification
# ---------------------------------------------------------------------------

def classify_attack_heuristic(
    tamper_signals: Dict[str, Any],
    semantic_checks: List[Dict[str, Any]],
    font_audit: Optional[Dict[str, Any]],
    face_verification: Dict[str, Any],
    suspicious_regions: List[Dict[str, Any]],
    quality: Dict[str, Any],
    metadata_audit: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[float], List[str], str]:
    """
    Classify the primary tampering attack heuristic.

    Mutually exclusive hypotheses:
        - date_edit
        - text_edit
        - photo_swap
        - copy_move
        - none

    Returns:
        (attack_type_guess, attack_type_confidence, attack_type_basis, decision)
    """
    if config is None:
        config = _load_m5_config()
    q_cfg = config.get("quality", {})
    enable_q_gating = q_cfg.get("enable_quality_gating", True)

    basis: List[str] = []
    scores = {
        "date_edit": 0.0,
        "text_edit": 0.0,
        "photo_swap": 0.0,
        "copy_move": 0.0,
        "none": 0.1,  # Baseline prior for authentic
    }

    # 1. Quality-Aware Gating
    if enable_q_gating and quality.get("analysis_reliability") == "LOW":
        # Check if there is an overwhelming physical anomaly (energy > 150 or copy-move > 30)
        ela_energy = 0.0
        if tamper_signals.get("ela", {}).get("features"):
            ela_energy = float(tamper_signals["ela"]["features"].get("candidate_energy", 0.0))
        cm_matches = int(tamper_signals.get("copy_move", {}).get("num_matches", 0))

        if ela_energy < 150.0 and cm_matches < 30:
            basis.append(
                f"Image scan quality is degraded ({', '.join(quality.get('quality_flags', ['LOW_QUALITY']))}). "
                "Evidence is insufficient to reliably confirm intentional tampering."
            )
            return "none", 0.0, basis, "INSUFFICIENT_EVIDENCE"

    # 2. Inspect Copy-Move Signals
    cm_data = tamper_signals.get("copy_move", {})
    cm_detected = cm_data.get("detected")
    if cm_detected is None:
        cm_detected = (cm_data.get("num_matches", 0) >= config.get("attack_heuristics", {}).get("copy_move_min_matches", 15))

    if cm_detected and cm_data.get("num_matches", 0) >= config.get("attack_heuristics", {}).get("copy_move_min_matches", 15):
        n_m = cm_data["num_matches"]
        cm_conf = float(cm_data.get("confidence") or 0.85)
        scores["copy_move"] += 0.85 + 0.15 * min(cm_conf / 0.50, 1.0)
        basis.append(f"Copy-Move cloning detected with {n_m} verified keypoint correspondences (conf={cm_conf:.2f})")

    # 3. Inspect Suspicious Regions for Field Associations
    for s_reg in suspicious_regions:
        field = s_reg.get("field", "")
        src = s_reg.get("source", "")
        conf = float(s_reg.get("confidence", 0.70))

        if field in ["dob", "issue_date", "expiry_date"]:
            scores["date_edit"] += 0.65 * conf
            basis.append(f"Suspicious region ({src}) identified on date field '{field}': {s_reg.get('evidence')}")
        elif field in ["name", "document_number", "country", "nationality"]:
            if field == "name":
                has_name_tamper = (
                    (font_audit and font_audit.get("typography_verdict") == "SUSPECT_FONT_INCONSISTENCY")
                    or any(chk.get("check") in ["name_structure_sanity", "duplicate_field_contradiction"] and chk.get("status") == "FAIL" for chk in semantic_checks)
                    or (s_reg.get("bbox") and s_reg["bbox"][0] < 280)
                )
                weight = 0.65 if has_name_tamper else 0.15
            else:
                weight = 0.65
            scores["text_edit"] += weight * conf
            if weight >= 0.50:
                basis.append(f"Suspicious region ({src}) identified on text field '{field}': {s_reg.get('evidence')}")
        elif field == "photo":
            scores["photo_swap"] += 0.75 * conf
            basis.append(f"Suspicious region ({src}) identified over photo portrait region: {s_reg.get('evidence')}")

    # 4. Inspect Semantic Rule Failures
    for chk in semantic_checks:
        if chk.get("status") == "FAIL":
            c_name = chk.get("check")
            c_det = chk.get("detail", "")
            if c_name in ["impossible_dates", "chronology_order", "age_at_issue_sanity", "validity_window_sanity", "anachronism_check"]:
                scores["date_edit"] += 0.60
                basis.append(f"Semantic date validation rule failed ({c_name}): {c_det}")
            elif c_name in ["document_number_format", "name_structure_sanity", "country_code_sanity", "duplicate_field_contradiction"]:
                scores["text_edit"] += 0.60
                basis.append(f"Semantic identity text rule failed ({c_name}): {c_det}")

    # 5. Inspect Typography Font Splicing
    if font_audit and font_audit.get("typography_verdict") == "SUSPECT_FONT_INCONSISTENCY":
        for a in font_audit.get("anomalous_fields", []):
            fname = a.get("field")
            if fname in ["dob", "issue_date", "expiry_date"]:
                scores["date_edit"] += 0.50
            else:
                scores["text_edit"] += 0.50
        basis.append(f"Typographic font inconsistency detected (max stroke Z-score = {font_audit.get('max_stroke_zscore', 0.0):.2f})")

    # 6. Inspect Biometric Face Verification
    if face_verification.get("has_face_check"):
        if face_verification.get("verified") is False:
            scores["photo_swap"] += 0.75
            basis.append(
                f"Biometric face mismatch (cosine distance = {face_verification.get('distance')}, "
                f"threshold = {face_verification.get('threshold')})"
            )
        elif face_verification.get("verified") is True:
            # High-confidence face match reduces likelihood of photo swap
            scores["photo_swap"] = max(0.0, scores["photo_swap"] - 0.40)

    # 7. Metadata Provenance
    if metadata_audit and metadata_audit.get("is_tampered"):
        sw = metadata_audit.get("software") or metadata_audit.get("software_detected")
        basis.append(f"Image metadata provenance indicates editing software signature: {sw}")
        # Metadata tamper boosts top non-zero attack
        top_curr = max(scores, key=scores.get)
        if top_curr != "none":
            scores[top_curr] += 0.20

    # Determine highest scoring attack type
    max_attack = max(scores, key=scores.get)
    max_score = scores[max_attack]

    # Check for corroborating evidence across independent modalities
    has_corroboration = (
        any(chk.get("status") == "FAIL" for chk in semantic_checks)
        or (font_audit and font_audit.get("typography_verdict") == "SUSPECT_FONT_INCONSISTENCY")
        or (cm_data.get("detected") is True)
        or (face_verification.get("has_face_check") and face_verification.get("verified") is False)
        or (metadata_audit and metadata_audit.get("is_tampered"))
    )

    # Physical ELA energy check
    ela_energy = 0.0
    if tamper_signals.get("ela", {}).get("features"):
        ela_energy = float(tamper_signals["ela"]["features"].get("candidate_energy", 0.0))

    # For date_edit, photo_swap, copy_move: highly specific to attack modalities
    # For text_edit: require corroboration, or document_number field, or decisive energy
    is_text_tamper = (
        max_attack == "text_edit"
        and (has_corroboration or any(s.get("field") in ["document_number", "name"] for s in suspicious_regions))
    )

    is_supported = (
        max_score >= 0.40 and (
            max_attack in ["date_edit", "photo_swap", "copy_move"]
            or is_text_tamper
        )
    )

    if not is_supported or max_attack == "none":
        attack_guess = "none"
        attack_conf = 0.0
        decision = "CLEAR_AUTHENTIC"
        if not basis:
            basis.append("All physical, typographic, biometric, and semantic validation checks strictly verified.")
    else:
        attack_guess = max_attack
        attack_conf = min(0.99, round(min(max_score, 1.0) * 0.95, 2))
        decision = "SUSPECT_TAMPERING"

    return attack_guess, attack_conf, basis, decision


# ---------------------------------------------------------------------------
# 4. Turnkey Feature Vector for Milestone 6 Machine Learning
# ---------------------------------------------------------------------------

def extract_m6_feature_vector(report: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract a clean, numeric feature vector for the Milestone 6 Logistic Regression model.
    Handles missing values via explicit imputation rules (e.g. has_face_check = 0).
    """
    f_vec: Dict[str, float] = {}

    # 1. ELA Features
    ela_feat = report.get("tamper_signals", {}).get("ela", {}).get("features", {})
    f_vec["ela_mean"] = float(ela_feat.get("mean", 0.0))
    f_vec["ela_std"] = float(ela_feat.get("std", 0.0))
    f_vec["ela_max"] = float(ela_feat.get("max", 0.0))
    f_vec["ela_p95"] = float(ela_feat.get("p95", 0.0))
    f_vec["ela_p99"] = float(ela_feat.get("p99", 0.0))
    f_vec["ela_high_error_ratio"] = float(ela_feat.get("high_error_pixel_ratio", 0.0))
    f_vec["ela_candidate_confidence"] = float(report.get("tamper_signals", {}).get("ela", {}).get("confidence") or 0.0)

    # 2. Copy-Move Features
    cm = report.get("tamper_signals", {}).get("copy_move", {})
    f_vec["copy_move_num_matches"] = float(cm.get("num_matches", 0))
    f_vec["copy_move_confidence"] = float(cm.get("confidence") or 0.0)
    f_vec["copy_move_detected"] = 1.0 if cm.get("num_matches", 0) >= 15 else 0.0

    # 3. Semantic Rule Flags
    sem_checks = report.get("semantic_checks", [])
    failed_checks = {c.get("check") for c in sem_checks if c.get("status") == "FAIL"}
    f_vec["semantic_failed_count"] = float(len(failed_checks))
    f_vec["semantic_date_order_flag"] = 1.0 if "chronology_order" in failed_checks else 0.0
    f_vec["semantic_impossible_date_flag"] = 1.0 if "impossible_dates" in failed_checks else 0.0
    f_vec["semantic_age_sanity_flag"] = 1.0 if "age_at_issue_sanity" in failed_checks else 0.0
    f_vec["semantic_doc_number_flag"] = 1.0 if "document_number_format" in failed_checks else 0.0
    f_vec["semantic_contradiction_flag"] = 1.0 if "duplicate_field_contradiction" in failed_checks else 0.0
    f_vec["semantic_country_code_flag"] = 1.0 if "country_code_sanity" in failed_checks else 0.0

    # 4. MRZ Checksum Status
    mrz_obj = report.get("mrz", {})
    mrz_status = mrz_obj.get("status")
    f_vec["mrz_checksum_pass"] = 1.0 if mrz_status == "PASS" else 0.0
    f_vec["mrz_checksum_fail"] = 1.0 if mrz_status == "FAIL" else 0.0
    f_vec["mrz_has_data"] = 1.0 if mrz_status is not None else 0.0

    # 5. Face Verification
    face = report.get("face_verification", {})
    has_face = bool(face.get("has_face_check"))
    f_vec["has_face_check"] = 1.0 if has_face else 0.0
    # Impute missing face distance with default uninformative distance 0.50 (not 0.0!)
    f_vec["face_distance"] = float(face.get("distance", 0.50)) if has_face and face.get("distance") is not None else 0.50
    f_vec["face_verified"] = 1.0 if has_face and face.get("verified") is True else 0.0
    f_vec["face_mismatch"] = 1.0 if has_face and face.get("verified") is False else 0.0

    # 6. Typography Font Metrics
    font_audit = report.get("font_forensics", {})
    f_vec["font_max_stroke_zscore"] = float(font_audit.get("max_stroke_zscore", 0.0))
    f_vec["font_inconsistency_flag"] = 1.0 if font_audit.get("typography_verdict") == "SUSPECT_FONT_INCONSISTENCY" else 0.0

    # 7. Metadata Provenance
    meta = report.get("metadata_forensics", {})
    f_vec["metadata_is_tampered"] = 1.0 if meta.get("is_tampered") else 0.0
    f_vec["metadata_has_exif"] = 1.0 if meta.get("has_exif") else 0.0

    # 8. Quality Metrics
    qual = report.get("quality", {})
    f_vec["quality_blur_score"] = float(qual.get("blur_score", 0.0))
    f_vec["quality_resolution_ok"] = 1.0 if qual.get("resolution_ok") else 0.0
    f_vec["quality_ocr_confidence"] = float(qual.get("ocr_mean_confidence") or 0.0)
    f_vec["quality_is_low_reliability"] = 1.0 if qual.get("analysis_reliability") == "LOW" else 0.0

    # 9. Suspicious Regions Count
    f_vec["suspicious_regions_count"] = float(len(report.get("suspicious_regions", [])))

    return f_vec


# ---------------------------------------------------------------------------
# 5. Master Pipeline: Generate Unified Forensic Report
# ---------------------------------------------------------------------------

def generate_unified_forensic_report(
    image_path: str,
    reference_face_path: Optional[str] = None,
    doc_type: str = "forgelensia",
    baseline: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run complete end-to-end multi-modal forensic screening across M1, M2, M3, M4.

    Produces ONE standardized JSON-serializable forensic report strictly adhering
    to the Milestone 5 Schema 1.0 specification.
    """
    if config is None:
        config = _load_m5_config()

    doc_id = os.path.splitext(os.path.basename(image_path))[0]
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        # Return graceful zero-crash error report
        return {
            "schema_version": "1.0",
            "document_id": doc_id,
            "document_type": doc_type,
            "quality": {
                "blur_score": 0.0,
                "resolution_ok": False,
                "dimensions": [0, 0],
                "ocr_mean_confidence": None,
                "analysis_reliability": "LOW",
                "quality_flags": ["FILE_NOT_FOUND_OR_CORRUPT"],
            },
            "fields": {},
            "semantic_checks": [],
            "tamper_signals": {
                "ela": {"candidate_bbox": [], "confidence": None, "features": {}},
                "copy_move": {"bbox": [], "alt_bbox": [], "num_matches": 0, "confidence": None},
            },
            "face_verification": {"has_face_check": False, "verified": None, "distance": None},
            "suspicious_regions": [],
            "attack_type_guess": "none",
            "attack_type_confidence": 0.0,
            "attack_type_basis": ["Document image file could not be read."],
            "risk_score": None,
            "fraud_probability": None,
            "decision": "INSUFFICIENT_EVIDENCE",
            "feature_vector": {},
        }

    # -----------------------------------------------------------------------
    # Step A: Milestone 3 — Structured Field Extraction & MRZ
    # -----------------------------------------------------------------------
    ocr_result = extract_structured_fields(img_bgr)
    raw_fields = ocr_result.get("fields", {})

    # Format fields for clean M5 contract: {name: {value, bbox, confidence}}
    formatted_fields = {}
    for fname, fdata in raw_fields.items():
        if fname.startswith("_") or not isinstance(fdata, dict):
            continue
        formatted_fields[fname] = {
            "value": fdata.get("value"),
            "bbox": [int(x) for x in fdata.get("bbox", [])] if fdata.get("bbox") else None,
            "confidence": round(float(fdata.get("confidence", 0.0)), 3) if fdata.get("confidence") is not None else None,
            "raw_text": fdata.get("raw_text"),
        }

    # -----------------------------------------------------------------------
    # Step B: Quality Assessment Layer
    # -----------------------------------------------------------------------
    quality = analyze_document_quality(img_bgr, ocr_fields=formatted_fields, config=config)

    # -----------------------------------------------------------------------
    # Step C: Milestone 1 — Physical Tamper Signals (ELA & Copy-Move)
    # -----------------------------------------------------------------------
    if baseline is None:
        baseline = get_default_baseline()
    if baseline is not None:
        b_mean = baseline.get("mean_map")
        if b_mean is not None and b_mean.shape != img_bgr.shape[:2]:
            baseline = None

    ela_energy_th = config.get("attack_heuristics", {}).get("ela_anomaly_energy_threshold", 75.0)
    ela_res = analyze_ela(image_path, baseline=baseline, energy_threshold=ela_energy_th)
    ela_cand = ela_res.get("candidate")
    ela_cand_bbox = [int(x) for x in ela_cand["bbox"]] if ela_cand and ela_cand.get("bbox") else []
    ela_conf = round(float(min(0.98, 0.40 + (ela_cand.get("energy", 0.0) / 200.0) * 0.50)), 2) if ela_cand else None
    ela_features = ela_res.get("features", {})

    copy_move_res = detect_copy_move(image_path)
    cm_detected = copy_move_res.get("detected", False)
    cm_bbox = [int(x) for x in copy_move_res.get("candidate_bbox", [])] if copy_move_res.get("candidate_bbox") else []
    cm_alt_bbox = [int(x) for x in copy_move_res.get("alt_bbox", [])] if copy_move_res.get("alt_bbox") else []
    cm_matches = int(copy_move_res.get("num_matches", 0))
    cm_conf = round(float(copy_move_res.get("confidence", 0.85)), 2) if cm_detected else None

    tamper_signals = {
        "ela": {
            "candidate_bbox": ela_cand_bbox,
            "confidence": ela_conf,
            "features": {
                "mean": round(float(ela_features.get("mean", 0.0)), 2),
                "std": round(float(ela_features.get("std", 0.0)), 2),
                "max": round(float(ela_features.get("max", 0.0)), 2),
                "p95": round(float(ela_features.get("p95", 0.0)), 2),
                "p99": round(float(ela_features.get("p99", 0.0)), 2),
                "high_error_pixel_ratio": round(float(ela_features.get("high_error_pixel_ratio", 0.0)), 4),
                "candidate_energy": round(float(ela_cand.get("energy", 0.0)), 1) if ela_cand else 0.0,
            },
        },
        "copy_move": {
            "bbox": cm_bbox,
            "alt_bbox": cm_alt_bbox,
            "num_matches": cm_matches,
            "confidence": cm_conf,
            "detected": bool(cm_detected),
        },
    }

    # -----------------------------------------------------------------------
    # Step D: Milestone 2 — Biometric Face Verification
    # -----------------------------------------------------------------------
    doc_face_info = extract_face_from_document(image_path)
    has_face_check = bool(reference_face_path and os.path.exists(reference_face_path))

    if has_face_check:
        face_audit = verify_document_face(image_path, reference_face_path)
        face_verification = {
            "has_face_check": True,
            "verified": face_audit.get("verified"),
            "distance": face_audit.get("distance"),
            "threshold": face_audit.get("threshold"),
            "similarity_pct": face_audit.get("similarity_pct"),
            "model": face_audit.get("model"),
            "face_detected": bool(doc_face_info is not None),
            "face_bbox": [int(x) for x in doc_face_info.get("bbox", [])] if doc_face_info and doc_face_info.get("bbox") else None,
            "warnings": face_audit.get("warnings", []),
        }
    else:
        face_verification = {
            "has_face_check": False,
            "verified": None,
            "distance": None,
            "threshold": None,
            "similarity_pct": None,
            "model": None,
            "face_detected": bool(doc_face_info is not None),
            "face_bbox": [int(x) for x in doc_face_info.get("bbox", [])] if doc_face_info and doc_face_info.get("bbox") else None,
            "warnings": [],
        }

    # -----------------------------------------------------------------------
    # Step E: Milestone 4 — Semantic Rules, MRZ, Typography, & Provenance
    # -----------------------------------------------------------------------
    semantic_battery = run_semantic_rule_battery(formatted_fields, doc_type=doc_type)
    semantic_checks = semantic_battery.get("checks_list", [])

    # MRZ Audit
    mrz_lines = ocr_result.get("mrz_lines", [])
    mrz_data = parse_mrz(mrz_lines) if mrz_lines else None
    mrz_viz_cross = cross_validate_viz_and_mrz(formatted_fields, mrz_data) if mrz_data else None

    # Typography & Font Audit
    font_audit = audit_document_font_consistency(img_bgr, formatted_fields)

    # Metadata & EXIF Audit
    raw_meta = extract_image_metadata(image_path)
    metadata_audit = audit_metadata_provenance(raw_meta)

    # -----------------------------------------------------------------------
    # Step F: Cross-Modal Suspicious Region Correlation
    # -----------------------------------------------------------------------
    suspicious_regions = correlate_suspicious_regions(
        fields=formatted_fields,
        ela_candidate=ela_cand,
        copy_move_res=copy_move_res,
        semantic_audit=semantic_battery,
        font_audit=font_audit,
        face_res=face_verification,
        doc_shape=img_bgr.shape[:2],
        config=config,
    )

    # -----------------------------------------------------------------------
    # Step G: Heuristic Attack Classification & Operational Decision
    # -----------------------------------------------------------------------
    attack_guess, attack_conf, attack_basis, decision = classify_attack_heuristic(
        tamper_signals=tamper_signals,
        semantic_checks=semantic_checks,
        font_audit=font_audit,
        face_verification=face_verification,
        suspicious_regions=suspicious_regions,
        quality=quality,
        metadata_audit=metadata_audit,
        config=config,
    )

    # Assemble Canonical Milestone 5 Unified Report
    report = {
        "schema_version": "1.0",
        "document_id": doc_id,
        "document_type": doc_type,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "quality": quality,
        "fields": formatted_fields,
        "semantic_checks": semantic_checks,
        "tamper_signals": tamper_signals,
        "face_verification": face_verification,
        "suspicious_regions": suspicious_regions,
        "attack_type_guess": attack_guess,
        "attack_type_confidence": attack_conf,
        "attack_type_basis": attack_basis,
        "risk_score": None,          # Reserved for Milestone 6 ML regression
        "fraud_probability": None,   # Reserved for Milestone 6 ML regression
        "decision": decision,
        "font_forensics": {
            "typography_verdict": font_audit.get("typography_verdict"),
            "max_stroke_zscore": font_audit.get("max_stroke_zscore"),
            "anomalous_fields_count": len(font_audit.get("anomalous_fields", [])),
        },
        "metadata_forensics": {
            "provenance_verdict": metadata_audit.get("provenance_verdict"),
            "is_tampered": metadata_audit.get("is_tampered"),
            "software": metadata_audit.get("software"),
            "has_exif": metadata_audit.get("has_exif"),
        },
        "mrz": {
            "format": mrz_data.get("format") if mrz_data else None,
            "status": mrz_data.get("status") if mrz_data else None,
            "verdict": mrz_data.get("verdict") if mrz_data else None,
            "viz_cross": mrz_viz_cross,
        },
    }

    # Step H: Pre-extract turnkey M6 feature vector
    report["feature_vector"] = extract_m6_feature_vector(report)

    return report


# ---------------------------------------------------------------------------
# 6. JSON Export Helper
# ---------------------------------------------------------------------------

class _NumpySafeEncoder(json.JSONEncoder):
    """Custom JSON encoder converting NumPy types to standard Python primitives."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, Path)):
            return str(obj)
        return super().default(obj)


def export_unified_report(
    report: Dict[str, Any],
    json_path: Optional[str] = None,
    indent: int = 2,
) -> str:
    """
    Serialize unified forensic report to JSON string and optionally save to disk.
    """
    json_str = json.dumps(report, cls=_NumpySafeEncoder, indent=indent)
    if json_path:
        ensure_dirs(os.path.dirname(os.path.abspath(json_path)))
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_str)
    return json_str
