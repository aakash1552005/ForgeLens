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
