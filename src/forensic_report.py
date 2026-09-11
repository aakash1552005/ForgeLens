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
            "min_brightness": 20.0,
            "max_brightness": 245.0,
            "min_contrast": 15.0,
            "max_color_cast": 65.0,
            "min_aspect_ratio": 1.0,
            "max_aspect_ratio": 2.2,
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
    Evaluates sharpness (Laplacian variance), dimensional resolution, illumination
    (mean brightness), contrast standard deviation, color cast, aspect ratio plausibility,
    and OCR recognition confidence.

    Returns:
        {
            "blur_score": float,
            "resolution_ok": bool,
            "dimensions": [width, height],
            "aspect_ratio": float,
            "mean_brightness": float,
            "contrast_std": float,
            "color_cast_score": float,
            "ocr_mean_confidence": float or null,
            "field_completeness_ratio": float,
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
    min_bright = q_cfg.get("min_brightness", 20.0)
    max_bright = q_cfg.get("max_brightness", 245.0)
    min_contrast = q_cfg.get("min_contrast", 15.0)
    max_cast = q_cfg.get("max_color_cast", 65.0)
    min_ar = q_cfg.get("min_aspect_ratio", 1.0)
    max_ar = q_cfg.get("max_aspect_ratio", 2.2)

    if image_bgr is None or image_bgr.size == 0:
        return {
            "blur_score": 0.0,
            "resolution_ok": False,
            "dimensions": [0, 0],
            "aspect_ratio": 0.0,
            "mean_brightness": 0.0,
            "contrast_std": 0.0,
            "color_cast_score": 0.0,
            "ocr_mean_confidence": 0.0,
            "field_completeness_ratio": 0.0,
            "analysis_reliability": "LOW",
            "quality_flags": ["EMPTY_OR_UNREADABLE_IMAGE"],
        }

    h, w = image_bgr.shape[:2]
    resolution_ok = bool(w >= min_w and h >= min_h)

    # Compute geometric aspect ratio (width / height)
    aspect_ratio = round(float(w / max(1, h)), 3)

    # Compute sharpness via Laplacian variance
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = round(laplacian_var, 2)

    # Compute illumination and contrast metrics
    mean_bright = round(float(np.mean(gray)), 2)
    contrast_std = round(float(np.std(gray)), 2)

    # Compute color cast / chromatic imbalance (channel delta)
    if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
        ch_means = [float(np.mean(image_bgr[:, :, c])) for c in range(3)]
        color_cast_score = round(float(max(ch_means) - min(ch_means)), 2)
    else:
        color_cast_score = 0.0

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

    # Compute field completeness ratio across standard credential fields
    field_comp_ratio = 0.0
    if ocr_fields:
        mandatory = ["name", "document_number", "dob", "expiry_date"]
        found = sum(1 for m in mandatory if ocr_fields.get(m, {}).get("value"))
        field_comp_ratio = round(float(found / len(mandatory)), 2)

    flags = []
    if blur_score < min_blur:
        flags.append(f"HEAVY_BLUR (score={blur_score} < {min_blur})")
    elif blur_score < med_blur:
        flags.append(f"MODERATE_BLUR (score={blur_score})")

    if not resolution_ok:
        flags.append(f"LOW_RESOLUTION ({w}x{h} < {min_w}x{min_h})")

    if aspect_ratio < min_ar or aspect_ratio > max_ar:
        flags.append(f"NON_STANDARD_ASPECT_RATIO (ratio={aspect_ratio} outside [{min_ar}, {max_ar}])")

    if mean_bright < min_bright:
        flags.append(f"EXTREME_UNDEREXPOSURE (brightness={mean_bright} < {min_bright})")
    elif mean_bright > max_bright:
        flags.append(f"EXTREME_OVEREXPOSURE (brightness={mean_bright} > {max_bright})")

    if contrast_std < min_contrast:
        flags.append(f"INSUFFICIENT_CONTRAST (std={contrast_std} < {min_contrast})")

    if color_cast_score > max_cast:
        flags.append(f"STRONG_COLOR_CAST (imbalance={color_cast_score} > {max_cast})")

    if ocr_mean is not None and ocr_mean < min_ocr:
        flags.append(f"LOW_OCR_CONFIDENCE ({ocr_mean} < {min_ocr})")

    # Determine analysis reliability tier
    is_critically_degraded = (
        blur_score < min_blur
        or not resolution_ok
        or (ocr_mean is not None and ocr_mean < min_ocr)
        or mean_bright < min_bright
        or mean_bright > max_bright
        or contrast_std < min_contrast
        or aspect_ratio < 0.60
        or aspect_ratio > 3.0
    )

    if is_critically_degraded:
        reliability = "LOW"
    elif blur_score < med_blur or (ocr_mean is not None and ocr_mean < high_ocr) or contrast_std < 25.0:
        reliability = "MEDIUM"
    else:
        reliability = "HIGH"

    return {
        "blur_score": blur_score,
        "resolution_ok": resolution_ok,
        "dimensions": [int(w), int(h)],
        "aspect_ratio": aspect_ratio,
        "mean_brightness": mean_bright,
        "contrast_std": contrast_std,
        "color_cast_score": color_cast_score,
        "ocr_mean_confidence": ocr_mean,
        "field_completeness_ratio": field_comp_ratio,
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
    mrz_data: Optional[Dict[str, Any]] = None,
    mrz_viz_cross: Optional[Dict[str, Any]] = None,
    metadata_audit: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Cross-reference detected physical and semantic anomalies with OCR field bboxes.
    Constructs standardized suspicious regions with exact bounding boxes, associated fields,
    all overlapping secondary fields, confidence metrics, and human-readable evidence.
    """
    if config is None:
        config = _load_m5_config()
    sp_cfg = config.get("spatial_correlation", {})
    min_iofa = sp_cfg.get("min_iofa_overlap", 0.15)
    default_photo_box = sp_cfg.get("photo_region_bbox", [30, 80, 200, 260])

    suspicious_regions = []

    # Helper: Find overlapping fields for a given bounding box
    def _find_best_field(target_box: List[int]) -> Tuple[Optional[str], float, List[str]]:
        if not target_box or len(target_box) < 4:
            return None, 0.0, []
        best_f = None
        best_ratio = 0.0
        all_overlapping = []

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
            if is_match:
                all_overlapping.append(fname)
                if ratio_f > best_ratio:
                    best_ratio = ratio_f
                    best_f = fname

        return best_f, best_ratio, all_overlapping

    # 1. ELA Physical Tamper Region
    if ela_candidate and ela_candidate.get("bbox"):
        e_box = ela_candidate["bbox"]
        matched_f, overlap, all_ov = _find_best_field(e_box)
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
                all_ov.append("photo")

        field_name = matched_f if matched_f else "unassigned_canvas"
        ov_list = all_ov if all_ov else ([field_name] if field_name != "unassigned_canvas" else [])
        evidence_str = (
            f"Physical JPEG compression discontinuity (energy={energy:.1f}, mean={mean_anom:.2f})"
            + (f" directly overlapping field '{matched_f}' ({overlap*100:.1f}% coverage)" if matched_f else "")
        )

        suspicious_regions.append({
            "source": "ela",
            "bbox": [int(x) for x in e_box],
            "field": field_name,
            "overlapping_fields": ov_list,
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
            matched_f, _, all_ov = _find_best_field(c_box)
            suspicious_regions.append({
                "source": "copy_move",
                "bbox": [int(x) for x in c_box],
                "field": matched_f if matched_f else "cloned_motif_receiver",
                "overlapping_fields": all_ov if all_ov else [matched_f or "cloned_motif_receiver"],
                "confidence": conf,
                "evidence": f"ORB-RANSAC duplicated motif candidate with {n_matches} inlier keypoint correspondences",
            })
        if alt_box:
            matched_f, _, all_ov = _find_best_field(alt_box)
            suspicious_regions.append({
                "source": "copy_move",
                "bbox": [int(x) for x in alt_box],
                "field": matched_f if matched_f else "cloned_motif_source",
                "overlapping_fields": all_ov if all_ov else [matched_f or "cloned_motif_source"],
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
                    "overlapping_fields": [fname],
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
                            "overlapping_fields": [tf],
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
                "overlapping_fields": ["photo"],
                "confidence": conf,
                "evidence": f"Biometric face mismatch (distance={dist:.3f} > threshold={thresh:.3f}, similarity={face_res.get('similarity_pct', 0.0)}%)",
            })

    # 6. MRZ Validation Anomaly
    if mrz_data and mrz_data.get("status") == "FAIL":
        h_dim = doc_shape[0] if doc_shape else 500
        w_dim = doc_shape[1] if doc_shape else 800
        mrz_box = [int(w_dim * 0.10), int(h_dim * 0.82), int(w_dim * 0.90), int(h_dim * 0.96)]
        suspicious_regions.append({
            "source": "mrz",
            "bbox": mrz_box,
            "field": "mrz",
            "overlapping_fields": ["mrz"],
            "confidence": 0.98,
            "evidence": f"ICAO Doc 9303 Modulo-10 checksum failure: {mrz_data.get('verdict', 'Checksum Forgery')}",
        })
    elif mrz_viz_cross and mrz_viz_cross.get("verdict") == "VIZ_MRZ_CONTRADICTION_FRAUD":
        h_dim = doc_shape[0] if doc_shape else 500
        w_dim = doc_shape[1] if doc_shape else 800
        mrz_box = [int(w_dim * 0.10), int(h_dim * 0.82), int(w_dim * 0.90), int(h_dim * 0.96)]
        disc_str = "; ".join(mrz_viz_cross.get("discrepancies", ["Identity contradiction between VIZ and MRZ"]))
        suspicious_regions.append({
            "source": "mrz",
            "bbox": mrz_box,
            "field": "mrz",
            "overlapping_fields": ["mrz"],
            "confidence": 0.98,
            "evidence": f"VIZ/MRZ Identity Contradiction: {disc_str}",
        })

    # 7. Metadata Provenance Editing Signature
    if metadata_audit and metadata_audit.get("is_tampered"):
        sw = metadata_audit.get("software") or metadata_audit.get("software_detected") or "Editing Software"
        suspicious_regions.append({
            "source": "metadata",
            "bbox": [0, 0, int(doc_shape[1]) if doc_shape else 800, 30],
            "field": "file_provenance",
            "overlapping_fields": ["provenance"],
            "confidence": 0.85,
            "evidence": f"Image metadata provenance indicates editing software signature: {sw}",
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
        elif field == "mrz" or src == "mrz":
            ev_str = str(s_reg.get("evidence", "")).lower()
            if any(k in ev_str for k in ["date", "dob", "expiry", "birth"]):
                scores["date_edit"] += 0.70 * conf
                basis.append(f"MRZ discrepancy indicates date manipulation: {s_reg.get('evidence')}")
            elif any(k in ev_str for k in ["doc", "number", "name", "id"]):
                scores["text_edit"] += 0.70 * conf
                basis.append(f"MRZ discrepancy indicates identity text manipulation: {s_reg.get('evidence')}")
            else:
                scores["text_edit"] += 0.50 * conf
                scores["date_edit"] += 0.50 * conf
                basis.append(f"MRZ checksum or format anomaly detected: {s_reg.get('evidence')}")

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
            # Physical-Biometric Synergy: Corroborate biometric mismatch with ELA boundary anomaly
            has_photo_ela = any(
                s.get("field") == "photo" and s.get("source") == "ela"
                for s in suspicious_regions
            )
            if has_photo_ela:
                scores["photo_swap"] += 0.40
                basis.append("Dual corroboration: Biometric face mismatch aligns with physical ELA compression anomaly on portrait.")
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
        or any(s.get("source") == "mrz" or s.get("field") == "mrz" for s in suspicious_regions)
    )

    # Physical ELA energy check
    ela_energy = 0.0
    if tamper_signals.get("ela", {}).get("features"):
        ela_energy = float(tamper_signals["ela"]["features"].get("candidate_energy", 0.0))

    # For date_edit, photo_swap, copy_move: highly specific to attack modalities
    # For text_edit: require corroboration, or document_number field, or decisive energy
    is_text_tamper = (
        max_attack == "text_edit"
        and (
            has_corroboration
            or any(s.get("field") in ["document_number", "name", "mrz"] or s.get("source") == "mrz" for s in suspicious_regions)
        )
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


def compute_attack_hypotheses_and_severity(
    tamper_signals: Dict[str, Any],
    semantic_checks: List[Dict[str, Any]],
    font_audit: Optional[Dict[str, Any]],
    face_verification: Dict[str, Any],
    suspicious_regions: List[Dict[str, Any]],
    attack_guess: str,
    attack_conf: float,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Rank candidate attack hypotheses, detect multi-attack scenarios,
    and establish forensic fraud severity (CRITICAL, HIGH, MODERATE, NONE).
    """
    if config is None:
        config = _load_m5_config()
    ah_cfg = config.get("attack_heuristics", {})
    sec_th = ah_cfg.get("multi_attack_threshold", 0.35)

    scores = {
        "date_edit": 0.0,
        "text_edit": 0.0,
        "photo_swap": 0.0,
        "copy_move": 0.0,
    }

    # 1. Copy-Move
    cm = tamper_signals.get("copy_move", {})
    if cm.get("detected") and cm.get("num_matches", 0) >= ah_cfg.get("copy_move_min_matches", 15):
        scores["copy_move"] += 0.85 + 0.15 * min(float(cm.get("confidence") or 0.85) / 0.50, 1.0)

    # 2. Suspicious regions
    for s_reg in suspicious_regions:
        f = s_reg.get("field", "")
        conf = float(s_reg.get("confidence", 0.70))
        if f in ["dob", "issue_date", "expiry_date"]:
            scores["date_edit"] += 0.65 * conf
        elif f in ["name", "document_number", "country", "nationality"]:
            scores["text_edit"] += 0.65 * conf
        elif f == "photo":
            scores["photo_swap"] += 0.75 * conf
        elif f == "mrz" or s_reg.get("source") == "mrz":
            ev_str = str(s_reg.get("evidence", "")).lower()
            if any(k in ev_str for k in ["date", "dob", "expiry", "birth"]):
                scores["date_edit"] += 0.70 * conf
            elif any(k in ev_str for k in ["doc", "number", "name", "id"]):
                scores["text_edit"] += 0.70 * conf
            else:
                scores["text_edit"] += 0.50 * conf
                scores["date_edit"] += 0.50 * conf

    # 3. Semantic checks
    for chk in semantic_checks:
        if chk.get("status") == "FAIL":
            cn = chk.get("check")
            if cn in ["impossible_dates", "chronology_order", "age_at_issue_sanity", "validity_window_sanity", "anachronism_check"]:
                scores["date_edit"] += 0.60
            elif cn in ["document_number_format", "name_structure_sanity", "country_code_sanity", "duplicate_field_contradiction"]:
                scores["text_edit"] += 0.60

    # 4. Typography
    if font_audit and font_audit.get("typography_verdict") == "SUSPECT_FONT_INCONSISTENCY":
        scores["text_edit"] += 0.50

    # 5. Face verification
    if face_verification.get("has_face_check") and face_verification.get("verified") is False:
        scores["photo_swap"] += 0.80
        # Physical-Biometric Synergy boost
        has_photo_ela = any(
            s.get("field") == "photo" and s.get("source") == "ela"
            for s in suspicious_regions
        )
        if has_photo_ela:
            scores["photo_swap"] += 0.40

    if attack_guess == "none":
        ranked = []
        sec_guess = None
        sec_conf = None
        multi_attack = False
        severity = "NONE"
    else:
        ranked = [
            {"attack_type": k, "score": round(float(v), 3)}
            for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if v >= 0.15
        ]

        other_attacks = [item for item in ranked if item["attack_type"] != attack_guess]
        secondary = other_attacks[0] if other_attacks else None
        sec_guess = secondary["attack_type"] if secondary and secondary["score"] >= sec_th else None
        sec_conf = round(float(min(0.95, secondary["score"] * 0.90)), 2) if sec_guess else None
        multi_attack = bool(sec_guess and secondary["score"] >= (sec_th + 0.05))

        # Fraud severity
        has_corroboration = (
            any(chk.get("status") == "FAIL" for chk in semantic_checks)
            or (font_audit and font_audit.get("typography_verdict") == "SUSPECT_FONT_INCONSISTENCY")
            or (cm.get("detected") is True)
            or (face_verification.get("has_face_check") and face_verification.get("verified") is False)
            or any(s.get("source") == "mrz" or s.get("field") == "mrz" for s in suspicious_regions)
        )

        has_photo_corroboration = (
            face_verification.get("has_face_check")
            and face_verification.get("verified") is False
            and any(s.get("field") == "photo" and s.get("source") == "ela" for s in suspicious_regions)
        )

        if (
            (attack_guess == "photo_swap" and (
                (face_verification.get("has_face_check") and face_verification.get("verified") is False and (face_verification.get("distance") or 0.0) >= 0.45)
                or has_photo_corroboration
            ))
            or (attack_guess == "copy_move" and cm.get("num_matches", 0) >= 30)
            or (attack_guess in ["date_edit", "text_edit"] and has_corroboration and attack_conf >= 0.80)
        ):
            severity = "CRITICAL"
        elif attack_conf >= 0.60 or has_corroboration:
            severity = "HIGH"
        else:
            severity = "MODERATE"

    return {
        "secondary_attack_guess": sec_guess,
        "secondary_attack_confidence": sec_conf,
        "multi_attack_detected": multi_attack,
        "attack_hypotheses_ranked": ranked,
        "fraud_severity": severity,
    }


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
    f_vec["quality_mean_brightness"] = float(qual.get("mean_brightness", 128.0)) / 255.0
    f_vec["quality_contrast_std"] = min(1.0, float(qual.get("contrast_std", 50.0)) / 100.0)
    f_vec["quality_aspect_ratio"] = float(qual.get("aspect_ratio", 1.54))
    f_vec["quality_field_completeness"] = float(qual.get("field_completeness_ratio", 1.0))

    # 9. Multi-Attack & Hypothesis Ranking
    f_vec["multi_attack_detected"] = 1.0 if report.get("multi_attack_detected") else 0.0
    f_vec["secondary_attack_confidence"] = float(report.get("secondary_attack_confidence") or 0.0)
    f_vec["primary_attack_confidence"] = float(report.get("attack_type_confidence") or 0.0)

    # 10. MRZ VIZ Cross-Validation Contradiction
    mrz_obj = report.get("mrz", {})
    viz_c = mrz_obj.get("viz_cross", {}) if isinstance(mrz_obj, dict) else {}
    f_vec["mrz_viz_contradiction_flag"] = 1.0 if viz_c and viz_c.get("verdict") == "VIZ_MRZ_CONTRADICTION_FRAUD" else 0.0

    # 11. Suspicious Regions Count
    f_vec["suspicious_regions_count"] = float(len(report.get("suspicious_regions", [])))

    # 12. Face Area Ratio
    face_area = report.get("face_verification", {}).get("face_area_ratio")
    f_vec["face_area_ratio"] = float(face_area) if face_area is not None else 0.0

    return f_vec


# ---------------------------------------------------------------------------
# 5. Master Pipeline: Generate Unified Forensic Report
# ---------------------------------------------------------------------------

def generate_executive_summary(
    decision: str,
    attack_guess: str,
    attack_conf: float,
    fraud_severity: str,
    quality: Dict[str, Any],
    secondary_attack: Optional[str] = None,
    suspicious_regions_count: int = 0,
) -> str:
    """Produce concise, unambiguous one-line executive verdict for audits."""
    if decision == "CLEAR_AUTHENTIC":
        return (
            f"AUTHENTIC: Document verified with high scan fidelity "
            f"(sharpness={quality.get('blur_score', 0.0):.1f}) and zero corroborated "
            f"physical, typographic, biometric, or semantic anomalies."
        )
    elif decision == "INSUFFICIENT_EVIDENCE":
        q_reasons = ", ".join(quality.get("quality_flags", ["degraded scan"]))
        return (
            f"INSUFFICIENT EVIDENCE: Scan quality is degraded ({q_reasons}); "
            f"recommend requesting a high-resolution optical rescan before definitive fraud disposition."
        )
    elif decision in ["SUSPECT_TAMPERING", "CRITICAL_FRAUD"]:
        sec_clause = f" with co-occurring {secondary_attack.upper()} manipulation" if secondary_attack else ""
        return (
            f"{fraud_severity} FRAUD: Corroborated {attack_guess.upper()} tampering detected "
            f"({attack_conf * 100:.1f}% confidence, {suspicious_regions_count} suspicious region(s)){sec_clause}."
        )
    return "REVIEW RECOMMENDED: Borderline forensic indicators require secondary examiner inspection."


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
            "executive_summary": "INSUFFICIENT EVIDENCE: Document image file could not be read or opened.",
            "quality": {
                "blur_score": 0.0,
                "resolution_ok": False,
                "dimensions": [0, 0],
                "aspect_ratio": 0.0,
                "mean_brightness": 0.0,
                "contrast_std": 0.0,
                "color_cast_score": 0.0,
                "ocr_mean_confidence": None,
                "field_completeness_ratio": 0.0,
                "analysis_reliability": "LOW",
                "quality_flags": ["FILE_NOT_FOUND_OR_CORRUPT"],
            },
            "fields": {},
            "semantic_checks": [],
            "tamper_signals": {
                "ela": {"candidate_bbox": [], "confidence": None, "features": {}},
                "copy_move": {"bbox": [], "alt_bbox": [], "num_matches": 0, "confidence": None},
            },
            "face_verification": {"has_face_check": False, "verified": None, "distance": None, "face_area_ratio": 0.0},
            "suspicious_regions": [],
            "attack_type_guess": "none",
            "attack_type_confidence": 0.0,
            "attack_type_basis": ["Document image file could not be read."],
            "secondary_attack_guess": None,
            "secondary_attack_confidence": None,
            "multi_attack_detected": False,
            "attack_hypotheses_ranked": [],
            "fraud_severity": "NONE",
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

    face_area_ratio = 0.0
    if doc_face_info and doc_face_info.get("bbox"):
        fb = doc_face_info["bbox"]
        fb_area = max(0, (fb[2] - fb[0]) * (fb[3] - fb[1]))
        total_doc_area = max(1, img_bgr.shape[0] * img_bgr.shape[1])
        face_area_ratio = round(float(fb_area / total_doc_area), 4)

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
            "face_area_ratio": face_area_ratio,
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
            "face_area_ratio": face_area_ratio,
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
        mrz_data=mrz_data,
        mrz_viz_cross=mrz_viz_cross,
        metadata_audit=metadata_audit,
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

    hypo_info = compute_attack_hypotheses_and_severity(
        tamper_signals=tamper_signals,
        semantic_checks=semantic_checks,
        font_audit=font_audit,
        face_verification=face_verification,
        suspicious_regions=suspicious_regions,
        attack_guess=attack_guess,
        attack_conf=attack_conf,
        config=config,
    )

    exec_summary = generate_executive_summary(
        decision=decision,
        attack_guess=attack_guess,
        attack_conf=attack_conf,
        fraud_severity=hypo_info["fraud_severity"],
        quality=quality,
        secondary_attack=hypo_info["secondary_attack_guess"] if hypo_info["multi_attack_detected"] else None,
        suspicious_regions_count=len(suspicious_regions),
    )

    # Assemble Canonical Milestone 5 Unified Report
    report = {
        "schema_version": "1.0",
        "document_id": doc_id,
        "document_type": doc_type,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "executive_summary": exec_summary,
        "quality": quality,
        "fields": formatted_fields,
        "semantic_checks": semantic_checks,
        "tamper_signals": tamper_signals,
        "face_verification": face_verification,
        "suspicious_regions": suspicious_regions,
        "attack_type_guess": attack_guess,
        "attack_type_confidence": attack_conf,
        "attack_type_basis": attack_basis,
        "secondary_attack_guess": hypo_info["secondary_attack_guess"],
        "secondary_attack_confidence": hypo_info["secondary_attack_confidence"],
        "multi_attack_detected": hypo_info["multi_attack_detected"],
        "attack_hypotheses_ranked": hypo_info["attack_hypotheses_ranked"],
        "fraud_severity": hypo_info["fraud_severity"],
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


def validate_report_schema(report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that a report dictionary strictly adheres to Milestone 5 Schema 1.0.

    Returns:
        (is_valid, error_messages)
    """
    required_keys = [
        "schema_version",
        "document_id",
        "document_type",
        "executive_summary",
        "quality",
        "fields",
        "semantic_checks",
        "tamper_signals",
        "face_verification",
        "suspicious_regions",
        "attack_type_guess",
        "attack_type_confidence",
        "attack_type_basis",
        "secondary_attack_guess",
        "secondary_attack_confidence",
        "multi_attack_detected",
        "attack_hypotheses_ranked",
        "fraud_severity",
        "risk_score",
        "fraud_probability",
        "decision",
        "feature_vector",
    ]
    errors = []
    for k in required_keys:
        if k not in report:
            errors.append(f"Missing required Schema 1.0 key: '{k}'")

    if report.get("schema_version") != "1.0":
        errors.append(f"Invalid schema_version '{report.get('schema_version')}', expected '1.0'")

    return len(errors) == 0, errors
