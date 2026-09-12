"""
ForgeLens-X — End-to-End Identity Screening Bridge
=====================================================
Bridges Milestone 1 (Physical Document Tamper Forensics: ELA + Copy-Move)
with Milestone 2 (Biometric Face Verification & Quality Pre-screening).

Forensic Core:
    Detects the critical fraud scenario where an imposter splices their own photo
    onto a stolen genuine identity credential (photo_swap + matching live face).
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.copy_move import detect_copy_move
from src.ela import analyze_ela, compute_baseline
from src.face_quality import assess_face_quality
from src.face_verify import extract_face, verify
from src.utils import compute_iou, ensure_dirs, get_reports_dir, load_config


# ---------------------------------------------------------------------------
# Document Face Extraction Helper
# ---------------------------------------------------------------------------

def extract_face_from_document(
    document_path: str,
    candidate_bbox: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Localize and extract the face portrait from an identity document image.
    First inspects the candidate photo region (default [30, 80, 200, 260]) with
    a 15% safety margin, then falls back to full document canvas detection.

    Returns:
        Face extraction dictionary with bounding box and landmarks mapped
        to the full document coordinate space.
    """
    if isinstance(document_path, np.ndarray):
        img_bgr = document_path.copy()
    elif isinstance(document_path, str):
        if not os.path.exists(document_path):
            return None
        img_bgr = cv2.imread(document_path)
    else:
        return None

    if img_bgr is None or img_bgr.size == 0:
        return None

    ih, iw = img_bgr.shape[:2]

    # Target photo region on standard Forgelensia credential
    if candidate_bbox is None:
        candidate_bbox = [30, 80, 200, 260]

    cx1, cy1, cx2, cy2 = candidate_bbox
    pad_x = int((cx2 - cx1) * 0.15)
    pad_y = int((cy2 - cy1) * 0.15)
    x1 = max(0, cx1 - pad_x)
    y1 = max(0, cy1 - pad_y)
    x2 = min(iw, cx2 + pad_x)
    y2 = min(ih, cy2 + pad_y)

    crop = img_bgr[y1:y2, x1:x2]

    # Run in-memory face extraction directly on candidate crop
    crop_face = extract_face(crop, align=True) if crop.size > 0 else None

    if crop_face is not None:
        bx, by, bw, bh = crop_face["bbox"]
        global_bbox = [x1 + bx, y1 + by, bw, bh]

        global_landmarks = None
        if crop_face.get("landmarks"):
            global_landmarks = [
                [float(x1 + pt[0]), float(y1 + pt[1])]
                for pt in crop_face["landmarks"]
            ]

        # Re-assess quality with global coordinates
        quality = assess_face_quality(img_bgr, landmarks=global_landmarks, bbox=global_bbox)

        return {
            "face_crop": crop_face["face_crop"],
            "bbox": global_bbox,
            "landmarks": global_landmarks,
            "confidence": crop_face["confidence"],
            "detector": crop_face["detector"],
            "original_size": (iw, ih),
            "quality": quality,
        }

    # Fallback: Detect directly on the full document image in memory
    full_face = extract_face(img_bgr, align=True)
    return full_face


# ---------------------------------------------------------------------------
# Master Identity Screening Function
# ---------------------------------------------------------------------------

def screen_identity(
    document_path: str,
    live_face_path: str,
    model_name: str = "ArcFace",
    distance_metric: str = "cosine",
    baseline_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Perform end-to-end forensic screening:
    1. Check document physical authenticity (ELA anomaly + Copy-Move cloning).
    2. Localize and extract ID portrait photo.
    3. Assess biometric quality on document photo and presented live selfie.
    4. Verify facial match across modalities.
    5. Check for physical photo-swap overlap with ELA anomaly proposals.
    6. Synthesize explainable multi-signal screening verdict.

    Args:
        document_path: path to full identity document image
        live_face_path: path to presented live selfie photo
        model_name: facial recognition model ('ArcFace', 'Facenet512', 'SFace')
        distance_metric: distance metric ('cosine' or 'euclidean_l2')
        baseline_stats: optional precomputed ELA baseline statistics

    Returns:
        Structured dictionary with combined forensic findings, signals, and verdict.
    """
    start_time = time.time()
    config = load_config()

    # 1. Input validation
    for path, label in [(document_path, "document"), (live_face_path, "live")]:
        if not path or not os.path.exists(path):
            err_msg = f"{label.capitalize()} image file not found: {path}"
            return {
                "verdict": "ERROR",
                "verdict_tier": "INVALID_INPUT",
                "error": err_msg,
                "summary": err_msg,
                "document_tampered": False,
                "face_verified": False,
                "photo_swap_detected": False,
                "flags": [f"INVALID_{label.upper()}_PATH"],
                "signals": {},
                "time_seconds": round(time.time() - start_time, 3),
            }

    # 2. Run M1 Physical Document Forensics
    ela_cfg = config.get("ela", {})
    ela_res = analyze_ela(
        document_path,
        baseline=baseline_stats,
        quality=ela_cfg.get("recompress_quality", 85),
        k=ela_cfg.get("baseline_k", 2.0),
        min_std=ela_cfg.get("std_floor", 1.5),
        min_area=ela_cfg.get("min_candidate_area", 400),
        closing_ksize=tuple(ela_cfg.get("closing_ksize", [11, 7])),
    )

    cm_cfg = config.get("copy_move", {})
    cm_res = detect_copy_move(
        document_path,
        n_features=cm_cfg.get("n_features", 5000),
        match_threshold=cm_cfg.get("match_threshold", 0.70),
        min_spatial_distance=cm_cfg.get("min_spatial_distance", 50.0),
        ransac_threshold=cm_cfg.get("ransac_threshold", 5.0),
        min_inliers=cm_cfg.get("min_inliers", 35),
        min_confidence=cm_cfg.get("min_confidence", 0.15),
    )

    ela_candidate = ela_res.get("candidate")
    ela_tampered = ela_candidate is not None
    cm_detected = cm_res.get("detected", False)
    document_tampered = bool(ela_tampered or cm_detected)

    # 3. Localize Document ID Photo
    doc_face_data = extract_face_from_document(document_path)
    if doc_face_data is None:
        summary_msg = "No facial portrait could be localized on the identity document."
        return {
            "verdict": "FLAGGED",
            "verdict_tier": "NO_DOCUMENT_FACE",
            "summary": summary_msg,
            "explanation": summary_msg,
            "document_tampered": document_tampered,
            "face_verified": False,
            "photo_swap_detected": False,
            "flags": ["NO_DOCUMENT_FACE"] + (["DOCUMENT_TAMPERED"] if document_tampered else []),
            "signals": {
                "ela_detected": ela_tampered,
                "copy_move_detected": cm_detected,
            },
            "time_seconds": round(time.time() - start_time, 3),
        }

    doc_face_bbox = doc_face_data["bbox"]  # [x, y, w, h]
    doc_face_rect = [
        doc_face_bbox[0],
        doc_face_bbox[1],
        doc_face_bbox[0] + doc_face_bbox[2],
        doc_face_bbox[1] + doc_face_bbox[3],
    ]

    # 4. Check for Photo Splicing / Swap Overlap with ELA Anomaly
    photo_swap_detected = False
    photo_swap_iou = 0.0
    if ela_candidate and ela_candidate.get("bbox"):
        ela_rect = ela_candidate["bbox"]  # [x1, y1, x2, y2]
        photo_swap_iou = compute_iou(ela_rect, doc_face_rect)
        ec_x = (ela_rect[0] + ela_rect[2]) / 2.0
        ec_y = (ela_rect[1] + ela_rect[3]) / 2.0
        inside_face = (
            doc_face_rect[0] <= ec_x <= doc_face_rect[2]
            and doc_face_rect[1] <= ec_y <= doc_face_rect[3]
        )
        if photo_swap_iou > 0.10 or inside_face:
            photo_swap_detected = True

    # 5. Run M2 Face Verification
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_crop:
        tmp_crop_path = tmp_crop.name
    cv2.imwrite(tmp_crop_path, doc_face_data["face_crop"])

    try:
        f_res = verify(
            document_face_path=tmp_crop_path,
            live_face_path=live_face_path,
            model_name=model_name,
            distance_metric=distance_metric,
        )
    finally:
        if os.path.exists(tmp_crop_path):
            try:
                os.remove(tmp_crop_path)
            except Exception:
                pass

    face_verified = f_res.get("verified", False)
    face_sim_pct = f_res.get("similarity_pct", 0.0)
    face_dist = f_res.get("distance")
    face_thresh = f_res.get("threshold")

    # 6. Synthesize Combined Forensic Verdict
    flags = []
    if ela_tampered:
        flags.append(f"ELA Anomaly Energy Spike (Score={ela_candidate['energy']:.1f})")
    if cm_detected:
        flags.append(f"Copy-Move Cloning ({cm_res.get('num_inliers', 0)} inliers)")
    if photo_swap_detected:
        flags.append("Spliced Photo Swap Overlap (ELA anomaly overlaps document portrait)")
    if not face_verified:
        dist_str = f"{face_dist:.4f}" if face_dist is not None else "N/A"
        thresh_str = f"{face_thresh:.4f}" if face_thresh is not None else "N/A"
        flags.append(f"Biometric Facial Mismatch (Distance={dist_str} > Threshold={thresh_str})")

    # Forensic Decision Matrix
    if not document_tampered and face_verified:
        verdict = "VERIFIED_AUTHENTIC"
        tier = "PASS"
        summary = "Document compression is authentic and biometric presentation matches identity portrait."
    elif photo_swap_detected and face_verified:
        verdict = "CRITICAL_PHOTO_SWAP_FRAUD"
        tier = "HIGH_ALERT"
        summary = "CRITICAL: Live face matches ID photo, but photo has been physically spliced into document canvas!"
    elif not document_tampered and not face_verified:
        verdict = "IMPOSTER_MISMATCH"
        tier = "REJECT"
        summary = "Document is genuine, but presented live individual does not match credential portrait."
    elif document_tampered and face_verified:
        verdict = "TAMPERED_DOCUMENT_ALTERATION"
        tier = "REJECT"
        summary = "Physical document has altered fields (dates/text), although live face matches portrait."
    else:
        verdict = "TOTAL_FRAUD_REJECTED"
        tier = "HIGH_ALERT"
        summary = "Document is physically forged AND presented live individual is an imposter."

    elapsed = round(time.time() - start_time, 3)

    return {
        "verdict": verdict,
        "verdict_tier": tier,
        "summary": summary,
        "explanation": summary,
        "document_path": document_path,
        "live_face_path": live_face_path,
        "document_tampered": document_tampered,
        "face_verified": face_verified,
        "photo_swap_detected": photo_swap_detected,
        "photo_swap_iou": round(float(photo_swap_iou), 3),
        "flags": flags,
        "signals": {
            "ela_detected": ela_tampered,
            "ela_energy": ela_candidate["energy"] if ela_candidate else 0.0,
            "copy_move_detected": cm_detected,
            "copy_move_inliers": cm_res.get("num_inliers", 0),
            "face_distance": face_dist,
            "face_threshold": face_thresh,
            "face_similarity_pct": face_sim_pct,
            "face_model": model_name,
        },
        "document_face_bbox": doc_face_bbox,
        "quality": f_res.get("quality", {}),
        "warnings": f_res.get("warnings", []),
        "time_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Visual Diagnostic Card for Identity Screening
# ---------------------------------------------------------------------------

def create_identity_screening_card(
    document_path: Union[str, Dict[str, Any]],
    live_face_path: Optional[str] = None,
    screening_result: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> Image.Image:
    """
    Generate a 1240 x 860 publication-grade forensic explanation card
    combining M1 document physical forensics with M2 face verification.

    Supports both calling conventions:
        create_identity_screening_card(screening_result, output_path=...)
        create_identity_screening_card(doc_path, live_path, screening_result, output_path=...)
    """
    from src.face_visualize import create_face_forensic_card
    from src.visualize import create_forensic_card

    if isinstance(document_path, dict):
        screening_result = document_path
        if live_face_path and isinstance(live_face_path, str) and not output_path:
            output_path = live_face_path
        document_path = screening_result.get("document_path", "")
        live_face_path = screening_result.get("live_face_path", "")
    elif screening_result is None:
        screening_result = {}

    # 1. Prepare sample data for document
    doc_stem = Path(document_path).stem if document_path else "doc"
    sample_data = {
        "image_path": document_path,
        "attack_type": "photo_swap" if screening_result.get("photo_swap_detected") else "screening",
        "label": "tampered" if screening_result.get("document_tampered") else "genuine",
        "source_id": doc_stem,
        "ground_truth_bbox": None,
    }

    sig = screening_result.get("signals", {})
    analysis_data = {
        "source_id": Path(document_path).stem,
        "attack_type": "photo_swap" if screening_result.get("photo_swap_detected") else "screening",
        "label": "tampered" if screening_result.get("document_tampered") else "genuine",
        "image_path": document_path,
        "ela_detected": sig.get("ela_detected", False),
        "ela_candidate_bbox": screening_result.get("document_face_bbox"),
        "ela_features": {"energy": sig.get("ela_energy", 0.0)},
        "copy_move_detected": sig.get("copy_move_detected", False),
        "copy_move_bbox": None,
        "copy_move_inliers": sig.get("copy_move_inliers", 0),
        "copy_move_confidence": 0.0,
    }

    # Extract temporary document crop for face card
    doc_face_data = extract_face_from_document(document_path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_crop:
        tmp_crop_path = tmp_crop.name

    if doc_face_data:
        cv2.imwrite(tmp_crop_path, doc_face_data["face_crop"])
    else:
        cv2.imwrite(tmp_crop_path, np.zeros((112, 112, 3), dtype=np.uint8))

    card_img = None
    try:
        card_img = create_face_forensic_card(
            document_face_path=tmp_crop_path,
            live_face_path=live_face_path,
            output_path=output_path,
            model_name=sig.get("face_model", "ArcFace"),
        )
    finally:
        if os.path.exists(tmp_crop_path):
            try:
                os.remove(tmp_crop_path)
            except Exception:
                pass

    return card_img
