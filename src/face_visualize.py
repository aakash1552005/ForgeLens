"""
ForgeLens-X — Milestone 2: Face Verification Visual Forensic Card
==================================================================
Renders high-resolution multi-panel diagnostic explanation cards for human review.

Architecture:
    Panel 1: Document Face Crop + 5-point Landmark Annotations
    Panel 2: Presented Live Face Crop + 5-point Landmark Annotations
    Panel 3: Calibrated Metric Distance & Similarity Gauge
    Panel 4: Forensic Verdict & Audit Decision Summary
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.document_template import _get_bold_font, _get_font
from src.face_verify import extract_face, verify
from src.utils import ensure_dirs


# ---------------------------------------------------------------------------
# Visual Styling Constants
# ---------------------------------------------------------------------------

CANVAS_WIDTH = 1240
CANVAS_HEIGHT = 860
PANEL_WIDTH = 580
PANEL_HEIGHT = 360

BG_COLOR = (15, 23, 42)          # Deep Slate (#0f172a)
PANEL_BG = (30, 41, 59)          # Slate Card (#1e293b)
PANEL_BORDER = (51, 65, 85)      # Border Slate (#334155)
TEXT_WHITE = (248, 250, 252)     # White (#f8fafc)
TEXT_MUTED = (148, 163, 184)     # Muted grey (#94a3b8)
ACCENT_BLUE = (56, 189, 248)     # Sky Blue (#38bdf8)
ACCENT_CYAN = (6, 182, 212)      # Cyan (#06b6d4)
MATCH_GREEN = (34, 197, 94)      # Emerald (#22c55e)
MATCH_BG = (22, 101, 52)         # Deep Green (#166534)
MISMATCH_RED = (239, 68, 68)     # Red (#ef4444)
MISMATCH_BG = (153, 27, 27)      # Deep Red (#991b1b)
BORDERLINE_AMBER = (245, 158, 11)# Amber (#f59e0b)
BORDERLINE_BG = (180, 83, 9)     # Deep Amber (#b45309)


# ---------------------------------------------------------------------------
# Helper: Overlay Landmarks & Crop on OpenCV Image
# ---------------------------------------------------------------------------

def _render_face_panel(
    image_path: str,
    bbox: Optional[List[int]],
    landmarks: Optional[List[List[float]]],
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
    title: str = "Face Analysis",
    quality: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Render a face panel showing cropped/focused face with 5 landmarks.
    """
    pw, ph = target_size
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    canvas[:] = (30, 41, 59)  # Slate background

    if not os.path.exists(image_path):
        cv2.putText(
            canvas, f"Image not found: {Path(image_path).name}",
            (30, ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1, cv2.LINE_AA
        )
        return canvas

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        cv2.putText(
            canvas, "Failed to read image",
            (30, ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1, cv2.LINE_AA
        )
        return canvas

    ih, iw = img_bgr.shape[:2]

    # If bbox exists, show face crop on the left and full image context on the right
    if bbox is not None and len(bbox) >= 4:
        bx, by, bw, bh = bbox
        # Add 20% margin around face
        pad_x = int(bw * 0.25)
        pad_y = int(bh * 0.25)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(iw, bx + bw + pad_x)
        y2 = min(ih, by + bh + pad_y)

        face_crop = img_bgr[y1:y2, x1:x2].copy()
        if face_crop.size > 0:
            # Draw landmarks on face crop
            if landmarks:
                lm_names = ["R.Eye", "L.Eye", "Nose", "R.Mouth", "L.Mouth"]
                lm_colors = [
                    (255, 200, 0),   # Cyan/Yellow
                    (255, 200, 0),
                    (0, 255, 255),   # Yellow
                    (0, 220, 0),     # Green
                    (0, 220, 0),
                ]
                for idx, pt in enumerate(landmarks):
                    lx, ly = pt[0] - x1, pt[1] - y1
                    cv2.circle(face_crop, (int(lx), int(ly)), 4, lm_colors[idx % 5], -1, cv2.LINE_AA)
                    cv2.circle(face_crop, (int(lx), int(ly)), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # Resize face crop to fit left half of panel
            crop_display_w = 260
            crop_display_h = 260
            face_resized = cv2.resize(face_crop, (crop_display_w, crop_display_h), interpolation=cv2.INTER_AREA)

            # Border around crop
            cv2.rectangle(face_resized, (0, 0), (crop_display_w - 1, crop_display_h - 1), (56, 189, 248), 2)

            # Paste face crop into canvas
            start_y = 50
            start_x = 25
            canvas[start_y:start_y + crop_display_h, start_x:start_x + crop_display_w] = face_resized

            # Thumbnail of full context on the right
            thumb_w = 230
            thumb_h = int(ih * (thumb_w / iw))
            if thumb_h > 180:
                thumb_h = 180
                thumb_w = int(iw * (thumb_h / ih))

            full_thumb = cv2.resize(img_bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            # Draw bbox on thumb
            sx = thumb_w / float(iw)
            sy = thumb_h / float(ih)
            cv2.rectangle(
                full_thumb,
                (int(bx * sx), int(by * sy)),
                (int((bx + bw) * sx), int((by + bh) * sy)),
                (0, 255, 0), 2
            )
            thumb_x = start_x + crop_display_w + 25
            thumb_y = start_y + 20
            canvas[thumb_y:thumb_y + thumb_h, thumb_x:thumb_x + thumb_w] = full_thumb
            cv2.rectangle(
                canvas,
                (thumb_x - 1, thumb_y - 1),
                (thumb_x + thumb_w, thumb_y + thumb_h),
                (71, 85, 105), 1
            )

            # Labels and metrics below thumbnail
            text_y = thumb_y + thumb_h + 30
            cv2.putText(
                canvas, f"BBox: [{bx}, {by}, {bw}, {bh}]",
                (thumb_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (203, 213, 225), 1, cv2.LINE_AA
            )
            cv2.putText(
                canvas, f"Crop Size: {bw}x{bh} px",
                (thumb_x, text_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA
            )
            if quality:
                tier = quality.get("quality_tier", "ACCEPTABLE")
                q_score = quality.get("quality_score", 0.0)
                sh = quality.get("sharpness", 0.0)
                iod = quality.get("iod", 0.0)
                q_col = (34, 197, 94) if tier in ["EXCELLENT", "ACCEPTABLE"] else ((245, 158, 11) if tier == "DEGRADED" else (239, 68, 68))
                cv2.putText(
                    canvas, f"ISO Quality: {tier} ({q_score:.0f}/100)",
                    (thumb_x, text_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, q_col, 1, cv2.LINE_AA
                )
                cv2.putText(
                    canvas, f"Sharpness: {sh:.0f} | IOD: {iod:.0f}px",
                    (thumb_x, text_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA
                )
            elif landmarks:
                cv2.putText(
                    canvas, "5-Pt Landmarks: Detected",
                    (thumb_x, text_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (52, 211, 153), 1, cv2.LINE_AA
                )
    else:
        # No face detected
        # Show full image dimmed with warning
        thumb_w = 400
        thumb_h = int(ih * (thumb_w / iw))
        if thumb_h > 240:
            thumb_h = 240
            thumb_w = int(iw * (thumb_h / ih))
        thumb = cv2.resize(img_bgr, (thumb_w, thumb_h))
        # Dim it
        thumb = (thumb * 0.4).astype(np.uint8)
        tx = (pw - thumb_w) // 2
        ty = 60
        canvas[ty:ty + thumb_h, tx:tx + thumb_w] = thumb
        cv2.rectangle(canvas, (tx, ty), (tx + thumb_w, ty + thumb_h), (239, 68, 68), 2)
        cv2.putText(
            canvas, "NO FACE DETECTED",
            (tx + 40, ty + thumb_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (239, 68, 68), 2, cv2.LINE_AA
        )

    # Panel Title Header
    cv2.rectangle(canvas, (0, 0), (pw, 36), (20, 30, 48), -1)
    cv2.putText(
        canvas, title.upper(),
        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 1, cv2.LINE_AA
    )

    return canvas


# ---------------------------------------------------------------------------
# Helper: Render Panel 3 (Distance & Similarity Gauge)
# ---------------------------------------------------------------------------

def _render_gauge_panel(
    verification_res: Dict[str, Any],
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> np.ndarray:
    """
    Render distance vs calibrated threshold visual meter and similarity scale.
    """
    pw, ph = target_size
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    canvas[:] = (30, 41, 59)

    # Header
    cv2.rectangle(canvas, (0, 0), (pw, 36), (20, 30, 48), -1)
    cv2.putText(
        canvas, "PANEL 3: METRIC DISTANCE & SIMILARITY GAUGE",
        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (248, 250, 252), 1, cv2.LINE_AA
    )

    if "error" in verification_res and verification_res.get("distance") is None:
        cv2.putText(
            canvas, "Verification could not be computed:",
            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (239, 68, 68), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, verification_res.get("detail", "Error"),
            (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (148, 163, 184), 1, cv2.LINE_AA
        )
        return canvas

    dist = float(verification_res.get("distance", 1.0))
    thresh = float(verification_res.get("threshold", 0.68))
    sim_pct = float(verification_res.get("similarity_pct", 0.0))
    metric = verification_res.get("distance_metric", "cosine")

    # 1. Distance Bar
    # Scale: 0.0 to max_scale (e.g. 1.20 for cosine, or thresh * 2.0)
    max_scale = max(1.20, thresh * 1.6)
    bar_x = 50
    bar_y = 100
    bar_w = 480
    bar_h = 24

    cv2.putText(
        canvas, f"{metric.upper()} DISTANCE vs THRESHOLD",
        (bar_x, bar_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1, cv2.LINE_AA
    )

    # Background track: Green for match side (left of threshold), Red for mismatch (right of threshold)
    thresh_pixel_x = int(bar_x + (thresh / max_scale) * bar_w)
    thresh_pixel_x = min(max(bar_x, thresh_pixel_x), bar_x + bar_w)

    # Green zone
    cv2.rectangle(canvas, (bar_x, bar_y), (thresh_pixel_x, bar_y + bar_h), (22, 101, 52), -1)
    # Red zone
    cv2.rectangle(canvas, (thresh_pixel_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (153, 27, 27), -1)
    # Border
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (71, 85, 105), 1)

    # Threshold divider line
    cv2.line(canvas, (thresh_pixel_x, bar_y - 6), (thresh_pixel_x, bar_y + bar_h + 6), (255, 255, 255), 2)
    cv2.putText(
        canvas, f"Threshold tau={thresh:.3f}",
        (thresh_pixel_x - 45, bar_y + bar_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (248, 250, 252), 1, cv2.LINE_AA
    )

    # Measured distance pin
    dist_pixel_x = int(bar_x + (dist / max_scale) * bar_w)
    dist_pixel_x = min(max(bar_x, dist_pixel_x), bar_x + bar_w)

    pin_color = (34, 197, 94) if dist <= thresh else (239, 68, 68)
    # Draw indicator needle / circle
    cv2.circle(canvas, (dist_pixel_x, bar_y + bar_h // 2), 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (dist_pixel_x, bar_y + bar_h // 2), 7, pin_color, -1, cv2.LINE_AA)
    cv2.putText(
        canvas, f"d={dist:.4f}",
        (max(bar_x, dist_pixel_x - 25), bar_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pin_color, 1, cv2.LINE_AA
    )

    # Scale tick marks (0.0, 0.5, 1.0)
    for tick_val in [0.0, 0.4, 0.8, 1.2]:
        if tick_val <= max_scale:
            tx = int(bar_x + (tick_val / max_scale) * bar_w)
            cv2.line(canvas, (tx, bar_y + bar_h), (tx, bar_y + bar_h + 4), (100, 116, 139), 1)
            cv2.putText(
                canvas, f"{tick_val:.1f}", (tx - 10, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA
            )

    # 2. Similarity Percentage Gauge
    sim_y = 220
    cv2.putText(
        canvas, "CALIBRATED CONFIDENCE / SIMILARITY SCORE",
        (bar_x, sim_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1, cv2.LINE_AA
    )

    # Gradient fill from red to green
    sim_fill_w = int((sim_pct / 100.0) * bar_w)
    cv2.rectangle(canvas, (bar_x, sim_y), (bar_x + bar_w, sim_y + bar_h), (15, 23, 42), -1)

    # Color of similarity bar
    if sim_pct >= 65.0:
        bar_fill_color = (34, 197, 94)  # Green
    elif sim_pct >= 40.0:
        bar_fill_color = (245, 158, 11) # Amber
    else:
        bar_fill_color = (239, 68, 68)  # Red

    if sim_fill_w > 0:
        cv2.rectangle(canvas, (bar_x, sim_y), (bar_x + sim_fill_w, sim_y + bar_h), bar_fill_color, -1)
    cv2.rectangle(canvas, (bar_x, sim_y), (bar_x + bar_w, sim_y + bar_h), (71, 85, 105), 1)

    # 50% boundary mark
    mid_x = bar_x + bar_w // 2
    cv2.line(canvas, (mid_x, sim_y), (mid_x, sim_y + bar_h), (255, 255, 255), 1)
    cv2.putText(
        canvas, "50% Boundary (tau)",
        (mid_x - 50, sim_y + bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA
    )

    # Score value text
    cv2.putText(
        canvas, f"{sim_pct:.1f}%",
        (bar_x + bar_w + 12, sim_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bar_fill_color, 2, cv2.LINE_AA
    )

    # Calibration explanation note
    note_y = 310
    cv2.putText(
        canvas, "Calibration formula: S = 100 / (1 + exp(beta * (d - tau)))",
        (bar_x, note_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 116, 139), 1, cv2.LINE_AA
    )
    margin = thresh - dist
    margin_text = f"Decision Margin: {margin:+.4f} ({'PASS' if margin >= 0 else 'FAIL'})"
    cv2.putText(
        canvas, margin_text,
        (bar_x, note_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA
    )

    return canvas


# ---------------------------------------------------------------------------
# Helper: Render Panel 4 (Forensic Verdict Summary Card)
# ---------------------------------------------------------------------------

def _render_verdict_panel(
    verification_res: Dict[str, Any],
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> np.ndarray:
    """
    Render structured decision summary card with verdict tier badge.
    """
    pw, ph = target_size
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    canvas[:] = (30, 41, 59)

    # Header
    cv2.rectangle(canvas, (0, 0), (pw, 36), (20, 30, 48), -1)
    cv2.putText(
        canvas, "PANEL 4: FORENSIC VERDICT & AUDIT DECISION",
        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (248, 250, 252), 1, cv2.LINE_AA
    )

    # Determine Verdict Status
    verified = verification_res.get("verified", False)
    error = verification_res.get("error")
    tier = verification_res.get("verdict_tier", "BORDERLINE")

    # Big Badge Rect
    badge_x, badge_y = 35, 55
    badge_w, badge_h = 510, 56

    if error:
        badge_bg = (71, 85, 105)
        badge_border = (148, 163, 184)
        badge_text = f"VERIFICATION ERROR: {error.upper()}"
        text_color = (248, 250, 252)
    elif tier == "CONFIRMED_MATCH":
        badge_bg = MATCH_BG
        badge_border = MATCH_GREEN
        badge_text = "MATCH CONFIRMED (SAME IDENTITY)"
        text_color = (255, 255, 255)
    elif tier == "CONFIRMED_MISMATCH":
        badge_bg = MISMATCH_BG
        badge_border = MISMATCH_RED
        badge_text = "MISMATCH CONFIRMED (IMPOSTER)"
        text_color = (255, 255, 255)
    else:
        badge_bg = BORDERLINE_BG
        badge_border = BORDERLINE_AMBER
        badge_text = "BORDERLINE REVIEW REQUIRED"
        text_color = (255, 255, 255)

    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_bg, -1)
    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_border, 2)
    cv2.putText(
        canvas, badge_text,
        (badge_x + 30, badge_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA
    )

    # Key-Value Audit Table
    start_y = 140
    row_h = 24
    kx = 45
    vx = 260

    dist = verification_res.get("distance")
    thresh = verification_res.get("threshold")
    sim_pct = verification_res.get("similarity_pct", 0.0)
    model = verification_res.get("model", "ArcFace")
    engine = verification_res.get("engine", "native_opencv")
    detector = verification_res.get("detector", "yunet")
    latency = verification_res.get("time_seconds", 0.0)

    rows = [
        ("Similarity Confidence:", f"{sim_pct:.1f}%", MATCH_GREEN if sim_pct >= 65 else (MISMATCH_RED if sim_pct <= 35 else BORDERLINE_AMBER)),
        ("Measured Distance:", f"{dist:.4f}" if dist is not None else "N/A", TEXT_WHITE),
        ("Calibrated Threshold (tau):", f"{thresh:.4f}" if thresh is not None else "N/A", TEXT_WHITE),
        ("Recognition Model:", f"{model} ({'512-D' if model in ['ArcFace', 'Facenet512'] else '128-D'})", ACCENT_CYAN),
        ("Face Alignment Engine:", f"{detector.capitalize()} 5-Pt Affine (112x112)", TEXT_WHITE),
        ("Execution Engine / Latency:", f"{engine} ({latency:.3f}s)", TEXT_MUTED),
        ("Fusion Policy Status:", "Independent Gate (Outside Learned Fusion)", (251, 191, 36)),
    ]

    for label, val, color in rows:
        cv2.putText(
            canvas, label,
            (kx, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, val,
            (vx, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA
        )
        start_y += row_h

    # Policy Warning Box at the bottom
    warn_y = 315
    cv2.rectangle(canvas, (badge_x, warn_y), (badge_x + badge_w, warn_y + 32), (15, 23, 42), -1)
    cv2.rectangle(canvas, (badge_x, warn_y), (badge_x + badge_w, warn_y + 32), (51, 65, 85), 1)
    cv2.putText(
        canvas, "NOTE: Face match verifies identity link only; does NOT certify document authenticity.",
        (badge_x + 12, warn_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (203, 213, 225), 1, cv2.LINE_AA
    )

    return canvas


# ---------------------------------------------------------------------------
# Main Function: Create Multi-Panel Face Forensic Card
# ---------------------------------------------------------------------------

def create_face_forensic_card(
    document_face_path: str,
    live_face_path: str,
    verification_result: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    model_name: str = "ArcFace",
) -> Image.Image:
    """
    Render a 2x2 multi-panel diagnostic forensic card for a face verification pair.

    Args:
        document_face_path: path to document face photo
        live_face_path: path to presented live selfie photo
        verification_result: optional pre-computed verification dictionary.
                             If None, will be computed on-demand.
        output_path: optional path to save the resulting PNG image.
        model_name: model to use if computing verification on-demand.

    Returns:
        PIL.Image of the rendered card (1240 x 860 px).
    """
    # 1. Compute verification if not provided
    if verification_result is None:
        verification_result = verify(
            document_face_path=document_face_path,
            live_face_path=live_face_path,
            model_name=model_name,
        )

    # 2. Extract bounding boxes and landmarks
    fa = verification_result.get("facial_areas", {})
    doc_bbox = fa.get("document_face")
    live_bbox = fa.get("live_face")

    lm = verification_result.get("landmarks", {})
    doc_lm = lm.get("document_face")
    live_lm = lm.get("live_face")

    # If bboxes/landmarks missing (e.g. if deepface was used), attempt extraction for visualization
    if doc_bbox is None and os.path.exists(document_face_path):
        d_data = extract_face(document_face_path)
        if d_data:
            doc_bbox = d_data.get("bbox")
            doc_lm = d_data.get("landmarks")

    if live_bbox is None and os.path.exists(live_face_path):
        l_data = extract_face(live_face_path)
        if l_data:
            live_bbox = l_data.get("bbox")
            live_lm = l_data.get("landmarks")

    qual = verification_result.get("quality", {})
    doc_q = qual.get("document_face")
    live_q = qual.get("live_face")

    # 3. Render 4 individual panels
    p1 = _render_face_panel(
        document_face_path, doc_bbox, doc_lm,
        target_size=(PANEL_WIDTH, PANEL_HEIGHT),
        title="Panel 1: Document Face Crop & Landmarks",
        quality=doc_q,
    )
    p2 = _render_face_panel(
        live_face_path, live_bbox, live_lm,
        target_size=(PANEL_WIDTH, PANEL_HEIGHT),
        title="Panel 2: Presented Live Face & Landmarks",
        quality=live_q,
    )
    p3 = _render_gauge_panel(
        verification_result,
        target_size=(PANEL_WIDTH, PANEL_HEIGHT),
    )
    p4 = _render_verdict_panel(
        verification_result,
        target_size=(PANEL_WIDTH, PANEL_HEIGHT),
    )

    # 4. Assemble into master canvas (1240 x 860)
    card = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    card[:] = BG_COLOR

    # Header Bar
    header_h = 70
    cv2.rectangle(card, (0, 0), (CANVAS_WIDTH, header_h), (15, 23, 42), -1)

    cv2.putText(
        card, "ForgeLens-X — Face Verification Forensic Audit Card",
        (30, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (248, 250, 252), 2, cv2.LINE_AA
    )
    cv2.putText(
        card, "Milestone 2: Cross-Modality Document-to-Live Face Verification Engine",
        (30, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA
    )

    # Model pill badge on top-right
    pill_text = f"Model: {verification_result.get('model', model_name)}"
    pill_w = 180
    pill_x = CANVAS_WIDTH - pill_w - 30
    cv2.rectangle(card, (pill_x, 22), (pill_x + pill_w, 52), (30, 41, 59), -1)
    cv2.rectangle(card, (pill_x, 22), (pill_x + pill_w, 52), (56, 189, 248), 1)
    cv2.putText(
        card, pill_text,
        (pill_x + 16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA
    )

    # Place Panels
    # Panel 1: Top-Left
    # Panel 2: Top-Right
    # Panel 3: Bottom-Left
    # Panel 4: Bottom-Right
    gap = 20
    x_left = 30
    x_right = x_left + PANEL_WIDTH + gap
    y_top = header_h + 10
    y_bottom = y_top + PANEL_HEIGHT + gap

    card[y_top:y_top + PANEL_HEIGHT, x_left:x_left + PANEL_WIDTH] = p1
    cv2.rectangle(card, (x_left, y_top), (x_left + PANEL_WIDTH, y_top + PANEL_HEIGHT), PANEL_BORDER, 1)

    card[y_top:y_top + PANEL_HEIGHT, x_right:x_right + PANEL_WIDTH] = p2
    cv2.rectangle(card, (x_right, y_top), (x_right + PANEL_WIDTH, y_top + PANEL_HEIGHT), PANEL_BORDER, 1)

    card[y_bottom:y_bottom + PANEL_HEIGHT, x_left:x_left + PANEL_WIDTH] = p3
    cv2.rectangle(card, (x_left, y_bottom), (x_left + PANEL_WIDTH, y_bottom + PANEL_HEIGHT), PANEL_BORDER, 1)

    card[y_bottom:y_bottom + PANEL_HEIGHT, x_right:x_right + PANEL_WIDTH] = p4
    cv2.rectangle(card, (x_right, y_bottom), (x_right + PANEL_WIDTH, y_bottom + PANEL_HEIGHT), PANEL_BORDER, 1)

    # Footer note
    footer_text = "ForgeLens-X Forensic Screening System | Research & Audit Artifact | Zero-Crash Diagnostic Output"
    cv2.putText(
        card, footer_text,
        (30, CANVAS_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (71, 85, 105), 1, cv2.LINE_AA
    )

    # Convert to PIL RGB
    card_rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(card_rgb)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        pil_img.save(output_path)

    return pil_img
