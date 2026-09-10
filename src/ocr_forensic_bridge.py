"""
ForgeLens-X — Milestone 3: OCR Forensic Cross-Modality Bridge
=============================================================
Bridges physical document tamper forensics (Milestone 1: ELA & Copy-Move)
with structured optical character recognition fields (Milestone 3).

Maps localized tamper anomalies directly onto specific identity fields
(e.g., "The Issue Date field has been altered via JPEG ELA compression anomaly").
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_box_intersection_area(boxA: List[int], boxB: List[int]) -> int:
    """Compute pixel intersection area between two [x1, y1, x2, y2] bounding boxes."""
    if not boxA or not boxB or len(boxA) < 4 or len(boxB) < 4:
        return 0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)

    return inter_w * inter_h


def correlate_tamper_with_fields(
    extracted_fields: Dict[str, Any],
    tamper_signals: Optional[Dict[str, Any]] = None,
    tamper_mask: Optional[np.ndarray] = None,
    overlap_threshold: float = 0.15,
    min_overlap_area: float = 0.0,
) -> Dict[str, Any]:
    """
    Spatially correlate detected tamper regions with extracted OCR identity fields.
    """
    if tamper_signals is None:
        tamper_signals = {}

    field_results = {}
    tampered_field_names = []

    # Extract potential tamper bounding boxes from M1 signals
    tamper_boxes = []

    # 1. ELA Anomaly Region
    ela_cand = tamper_signals.get("ela_candidate")
    ela_cand_bbox = tamper_signals.get("ela_candidate_bbox")
    if ela_cand and isinstance(ela_cand, dict) and ela_cand.get("bbox"):
        tamper_boxes.append({
            "source": "ELA Anomaly (Compression Inconsistency)",
            "type": "ELA_ANOMALY",
            "bbox": ela_cand["bbox"],
            "energy": ela_cand.get("energy", 0.0),
        })
    elif ela_cand_bbox and len(ela_cand_bbox) >= 4:
        tamper_boxes.append({
            "source": "ELA Anomaly (Compression Inconsistency)",
            "type": "ELA_ANOMALY",
            "bbox": ela_cand_bbox,
            "energy": 0.0,
        })
    elif tamper_signals.get("ela_bbox"):
        tamper_boxes.append({
            "source": "ELA Anomaly (Compression Inconsistency)",
            "type": "ELA_ANOMALY",
            "bbox": tamper_signals["ela_bbox"],
            "energy": 0.0,
        })

    # 2. Copy-Move Clone Region
    cm_bbox = tamper_signals.get("copy_move_bbox") or tamper_signals.get("candidate_bbox")
    if cm_bbox and len(cm_bbox) >= 4:
        tamper_boxes.append({
            "source": "Copy-Move Cloning (Duplicated Motif)",
            "type": "COPY_MOVE_CLONE",
            "bbox": cm_bbox,
            "inliers": tamper_signals.get("copy_move_inliers", 0),
        })

    # 3. Iterate each extracted OCR field
    for fname, f_info in extracted_fields.items():
        if not isinstance(f_info, dict):
            continue

        f_bbox = f_info.get("bbox")
        if not f_bbox or len(f_bbox) < 4:
            field_results[fname] = {
                "field": fname,
                "is_tampered": False,
                "tamper_source": None,
                "overlap_type": None,
                "tamper_overlap_area": 0.0,
                "overlap_ratio": 0.0,
                "severity": "NONE",
                "explanation": "No bounding box available for spatial correlation.",
            }
            continue

        f_x1, f_y1, f_x2, f_y2 = f_bbox
        field_area = max(1, (f_x2 - f_x1) * (f_y2 - f_y1))

        max_overlap = 0.0
        max_inter_area = 0.0
        best_source = None
        best_type = None

        # A. Bounding Box Containment Overlap
        for t_entry in tamper_boxes:
            t_box = t_entry["bbox"]
            inter_area = compute_box_intersection_area(f_bbox, t_box)
            overlap = inter_area / field_area
            if overlap > max_overlap or inter_area > max_inter_area:
                max_overlap = overlap
                max_inter_area = inter_area
                best_source = t_entry["source"]
                best_type = t_entry.get("type", "UNKNOWN")

        # B. Direct Pixel Mask Overlap (if binary mask provided)
        if tamper_mask is not None and tamper_mask.size > 0:
            mh, mw = tamper_mask.shape[:2]
            cx1 = max(0, min(mw - 1, f_x1))
            cy1 = max(0, min(mh - 1, f_y1))
            cx2 = max(0, min(mw, f_x2))
            cy2 = max(0, min(mh, f_y2))

            if cx2 > cx1 and cy2 > cy1:
                crop_mask = tamper_mask[cy1:cy2, cx1:cx2]
                mask_pixels = np.count_nonzero(crop_mask)
                m_overlap = mask_pixels / field_area
                if m_overlap > max_overlap:
                    max_overlap = m_overlap
                    max_inter_area = float(mask_pixels)
                    best_source = "Pixel-Level Manipulation Mask"
                    best_type = "PIXEL_MASK"

        is_tampered = (max_overlap >= overlap_threshold) or (min_overlap_area > 0 and max_inter_area >= min_overlap_area)

        if is_tampered:
            tampered_field_names.append(fname)
            severity = "CRITICAL" if max_overlap >= 0.40 else "WARNING"
            explanation = (
                f"Field '{fname}' exhibits spatial overlap ({max_overlap*100:.1f}%) with {best_source}. "
                "Text content has likely been rewritten or altered."
            )
        else:
            severity = "NONE"
            explanation = "Field region shows no significant spatial intersection with detected tamper anomalies."

        field_results[fname] = {
            "field": fname,
            "is_tampered": is_tampered,
            "tamper_source": best_source if is_tampered else None,
            "overlap_type": best_type if is_tampered else None,
            "tamper_overlap_area": round(max_inter_area, 1),
            "overlap_ratio": round(max_overlap, 4),
            "severity": severity,
            "explanation": explanation,
        }

    # Determine overall document integrity verdict
    if tampered_field_names:
        verdict = "FIELD_ALTERATION_DETECTED"
        status = "FIELD_ALTERATION_DETECTED"
        summary = (
            f"Physical tampering detected on {len(tampered_field_names)} field(s): "
            f"{', '.join(tampered_field_names)}. Text values in these fields should be treated as fraudulent."
        )
    elif tamper_signals.get("document_tampered"):
        verdict = "DOCUMENT_TAMPERED_OUTSIDE_TEXT_FIELDS"
        status = "FIELD_ALTERATION_CLEAN"
        summary = "Document tampering detected, but alterations are outside structured identity text fields."
    else:
        verdict = "TEXT_FIELDS_AUTHENTIC"
        status = "FIELD_ALTERATION_CLEAN"
        summary = "All structured text fields show clean spatial correlation with zero tamper overlap."

    return {
        "status": status,
        "tampered_fields_count": len(tampered_field_names),
        "tampered_field_names": tampered_field_names,
        "tampered_fields": field_results,
        "integrity_verdict": verdict,
        "summary": summary,
    }


def fuse_forensic_modalities(
    spatial_bridge: Optional[Dict[str, Any]] = None,
    semantic_audit: Optional[Dict[str, Any]] = None,
    font_audit: Optional[Dict[str, Any]] = None,
    metadata_audit: Optional[Dict[str, Any]] = None,
    mrz_audit: Optional[Dict[str, Any]] = None,
    mrz_viz_cross: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Cross-Milestone Forensic Fusion Bridge (Milestone 1 x Milestone 3 x Milestone 4).
    Correlates physical pixel tamper signals, logical semantic rules, typography metrics,
    EXIF provenance, and ICAO MRZ checksums into a unified, explainable threat verdict.
    """
    red_flags: List[str] = []
    elevated_correlations: List[Dict[str, Any]] = []

    # 1. Spatial Field Tampering
    spatially_tampered_fields = set()
    if spatial_bridge:
        spatially_tampered_fields = set(spatial_bridge.get("tampered_field_names", []))
        if spatially_tampered_fields:
            red_flags.append(f"Spatial physical tampering detected in fields: {list(spatially_tampered_fields)}")

    # 2. Semantic Rule Failures
    semantic_failed_checks = []
    if semantic_audit:
        semantic_failed_checks = semantic_audit.get("failed_checks", [])
        if semantic_failed_checks:
            red_flags.append(f"Semantic rule failures detected: {semantic_failed_checks}")

    # 3. Typography Font Anomalies
    font_anomalous_fields = set()
    if font_audit:
        for a in font_audit.get("anomalous_fields", []):
            fname = a.get("field")
            if fname:
                font_anomalous_fields.add(fname)
        if font_anomalous_fields:
            red_flags.append(f"Typographic font inconsistencies (Z > 2.5) in fields: {list(font_anomalous_fields)}")

    # 4. Metadata Provenance
    metadata_tampered = False
    if metadata_audit:
        if metadata_audit.get("is_tampered"):
            metadata_tampered = True
            red_flags.append(f"Metadata provenance alert: {metadata_audit.get('provenance_verdict')} ({metadata_audit.get('software_detected')})")

    # 5. MRZ Checksum and Cross-Validation
    mrz_tampered = False
    if mrz_audit:
        if mrz_audit.get("status") == "FAIL":
            mrz_tampered = True
            red_flags.append(f"MRZ checksum forgery detected: {mrz_audit.get('unrepaired_failures')}")
    if mrz_viz_cross:
        if not mrz_viz_cross.get("is_concordant") and mrz_viz_cross.get("comparable_fields", 0) > 0:
            mrz_tampered = True
            red_flags.append(f"VIZ-to-MRZ contradiction detected: {mrz_viz_cross.get('mismatches')}")

    # Cross-Modality Convergence Analysis
    # Check if a field is flagged across multiple independent modalities
    all_suspicious_fields = spatially_tampered_fields.union(font_anomalous_fields)
    for fname in all_suspicious_fields:
        modalities = []
        if fname in spatially_tampered_fields:
            modalities.append("Spatial ELA/Copy-Move")
        if fname in font_anomalous_fields:
            modalities.append("Stroke Width Typography")
        # Check if semantic failure relates to this field
        if semantic_failed_checks:
            for s_chk in semantic_failed_checks:
                if fname in s_chk or (fname in ["dob", "issue_date", "expiry_date"] and ("chronology" in s_chk or "date" in s_chk or "age" in s_chk)):
                    modalities.append(f"Semantic Rule ({s_chk})")

        if len(modalities) >= 2:
            elevated_correlations.append({
                "field": fname,
                "converging_modalities": modalities,
                "detail": f"Field '{fname}' exhibits simultaneous independent anomalies across {len(modalities)} forensic modalities: {modalities}",
            })

    # Composite Threat Verdict Determination
    if elevated_correlations:
        threat_level = "CRITICAL_CONFIRMED_FRAUD"
        is_authentic = False
        verdict_summary = (
            f"CRITICAL FRAUD CONFIRMED: Multi-modality convergence detected on {len(elevated_correlations)} field(s). "
            f"Physical tampering, typography anomalies, and/or semantic contradictions coincide directly."
        )
    elif mrz_tampered:
        threat_level = "CRITICAL_MRZ_TAMPERING"
        is_authentic = False
        verdict_summary = "CRITICAL FRAUD: Machine Readable Zone checksum failure or VIZ contradiction cannot be reconciled."
    elif spatially_tampered_fields:
        threat_level = "HIGH_SPATIAL_TAMPERING"
        is_authentic = False
        verdict_summary = f"HIGH ALERT: Spatial compression or clone tampering identified across fields: {list(spatially_tampered_fields)}."
    elif semantic_failed_checks:
        threat_level = "SUSPECT_LOGICAL_CONTRADICTION"
        is_authentic = False
        verdict_summary = f"SUSPECT: Logical semantic rules violated ({len(semantic_failed_checks)} failures), suggesting fabricated or modified data."
    elif font_anomalous_fields:
        threat_level = "SUSPECT_FONT_INCONSISTENCY"
        is_authentic = False
        verdict_summary = f"SUSPECT: Typographic font outlier detected in fields {list(font_anomalous_fields)} (different stroke weight / font insertion)."
    elif metadata_tampered:
        threat_level = "SUSPECT_METADATA_PROVENANCE"
        is_authentic = False
        verdict_summary = "SUSPECT: Image headers disclose photo-manipulation software signatures or timestamp inversions."
    else:
        threat_level = "AUTHENTIC_MULTI_MODAL_CLEARANCE"
        is_authentic = True
        verdict_summary = "AUTHENTIC: Clean clearance across all physical, logical, typographic, metadata, and MRZ forensic modalities."

    return {
        "threat_level": threat_level,
        "is_authentic": is_authentic,
        "red_flag_count": len(red_flags),
        "red_flags": red_flags,
        "elevated_correlations": elevated_correlations,
        "verdict_summary": verdict_summary,
        "modality_breakdown": {
            "spatial_tamper_count": len(spatially_tampered_fields),
            "semantic_failures_count": len(semantic_failed_checks),
            "font_anomalies_count": len(font_anomalous_fields),
            "metadata_tampered": metadata_tampered,
            "mrz_tampered": mrz_tampered,
        },
    }

