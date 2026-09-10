"""
ForgeLens-X — Milestone 3: OCR Visual Forensic Diagnostic Card
==============================================================
Renders high-resolution 4-panel diagnostic explanation cards for human forensic review:
    Panel 1: Document Canvas & Field Spatial Bounding Box Annotations
    Panel 2: Cropped Character Snippets & OCR Recognition Gallery
    Panel 3: Field Confidence Gauges & String Distance / CER Comparison
    Panel 4: Forensic Audit Decision & Extraction Health Summary
"""

import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.document_template import _get_bold_font, _get_font
from src.ocr import extract_structured_fields, load_image_for_ocr
from src.ocr_evaluate import (
    compute_cer,
    compute_levenshtein_distance,
    compute_normalized_similarity,
    evaluate_field_extraction,
)
from src.utils import ensure_dirs, get_reports_dir


# ---------------------------------------------------------------------------
# Visual Styling Constants (Consistent with M1 and M2)
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

# Field bounding box color scheme
FIELD_COLORS = {
    "name": (56, 189, 248),           # Cyan/Sky Blue
    "dob": (168, 85, 247),            # Purple
    "document_number": (34, 197, 94), # Green
    "issue_date": (245, 158, 11),     # Amber
    "expiry_date": (239, 68, 68),     # Red
}


# ---------------------------------------------------------------------------
# Panel 1: Document Canvas & Bounding Box Spatial Layout
# ---------------------------------------------------------------------------

def _render_document_canvas_panel(
    doc_bgr: np.ndarray,
    extracted_fields: Dict[str, Any],
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> np.ndarray:
    """Render document canvas with spatial overlay of extracted field bounding boxes."""
    pw, ph = target_size
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    canvas[:] = PANEL_BG

    # Header bar
    cv2.rectangle(canvas, (0, 0), (pw, 36), (20, 30, 48), -1)
    cv2.putText(
        canvas, "PANEL 1: DOCUMENT FIELD SPATIAL LAYOUT",
        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEXT_WHITE, 1, cv2.LINE_AA
    )

    if doc_bgr is None or doc_bgr.size == 0:
        cv2.putText(
            canvas, "No document image available",
            (30, ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_MUTED, 1, cv2.LINE_AA
        )
        return canvas

    dh, dw = doc_bgr.shape[:2]
    annotated = doc_bgr.copy()

    # Draw bboxes for each field
    for field_name, f_info in extracted_fields.items():
        bbox = f_info.get("bbox")
        val = f_info.get("value")
        status = f_info.get("status", "UNKNOWN")

        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(dw - 1, int(x1)))
        y1 = max(0, min(dh - 1, int(y1)))
        x2 = max(0, min(dw, int(x2)))
        y2 = max(0, min(dh, int(y2)))

        color = FIELD_COLORS.get(field_name, (200, 200, 200))
        if status == "UNKNOWN":
            color = (100, 100, 100)

        # Draw bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Small label tag
        lbl = f"{field_name[:4].upper()}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tag_y1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 6, tag_y1 + th + 6), (20, 30, 48), -1)
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 6, tag_y1 + th + 6), color, 1)
        cv2.putText(
            annotated, lbl, (x1 + 3, tag_y1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA
        )

    # Scale annotated doc to fit available panel area (leaving space for header & margins)
    margin_x, margin_y = 15, 45
    avail_w = pw - (margin_x * 2)
    avail_h = ph - margin_y - 15

    scale = min(avail_w / dw, avail_h / dh)
    new_w = int(dw * scale)
    new_h = int(dh * scale)

    resized = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in panel
    off_x = margin_x + (avail_w - new_w) // 2
    off_y = margin_y + (avail_h - new_h) // 2

    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    cv2.rectangle(canvas, (off_x, off_y), (off_x + new_w, off_y + new_h), PANEL_BORDER, 1)

    return canvas


# ---------------------------------------------------------------------------
# Panel 2: Field Crops & Character Recognition Gallery
# ---------------------------------------------------------------------------

def _render_crops_gallery_panel(
    doc_bgr: np.ndarray,
    extracted_fields: Dict[str, Any],
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> np.ndarray:
    """Render cropped image snippets of the 5 fields alongside recognized text."""
    pw, ph = target_size
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    canvas[:] = PANEL_BG

    # Header bar
    cv2.rectangle(canvas, (0, 0), (pw, 36), (20, 30, 48), -1)
    cv2.putText(
        canvas, "PANEL 2: FIELD OPTICAL CROPS & RECOGNITION",
        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEXT_WHITE, 1, cv2.LINE_AA
    )

    dh, dw = doc_bgr.shape[:2] if (doc_bgr is not None and doc_bgr.size > 0) else (1, 1)

    fields_order = ["name", "dob", "document_number", "issue_date", "expiry_date"]
    field_labels = {
        "name": "Name",
        "dob": "DOB",
        "document_number": "Doc No",
        "issue_date": "Issue Date",
        "expiry_date": "Expiry Date",
    }

    start_y = 48
    row_height = 58
    crop_w = 160
    crop_h = 42

    for idx, f_name in enumerate(fields_order):
        f_info = extracted_fields.get(f_name, {})
        val = f_info.get("value")
        bbox = f_info.get("bbox")
        status = f_info.get("status", "UNKNOWN")
        conf = float(f_info.get("confidence", 0.0))

        row_y = start_y + (idx * row_height)
        color = FIELD_COLORS.get(f_name, (200, 200, 200))

        # Field tag
        lbl = field_labels.get(f_name, f_name)
        cv2.putText(
            canvas, lbl, (15, row_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
        )

        # Optical Crop Box
        box_x = 105
        box_y = row_y + 2
        cv2.rectangle(canvas, (box_x, box_y), (box_x + crop_w, box_y + crop_h), (20, 30, 48), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + crop_w, box_y + crop_h), PANEL_BORDER, 1)

        has_crop = False
        if doc_bgr is not None and bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(dw - 1, int(x1)))
            y1 = max(0, min(dh - 1, int(y1)))
            x2 = max(0, min(dw, int(x2)))
            y2 = max(0, min(dh, int(y2)))

            if x2 > x1 and y2 > y1:
                crop = doc_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    ch, cw = crop.shape[:2]
                    scale = min((crop_w - 4) / cw, (crop_h - 4) / ch)
                    rw, rh = max(1, int(cw * scale)), max(1, int(ch * scale))
                    r_crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)

                    cx_off = box_x + 2 + (crop_w - 4 - rw) // 2
                    cy_off = box_y + 2 + (crop_h - 4 - rh) // 2
                    canvas[cy_off:cy_off + rh, cx_off:cx_off + rw] = r_crop
                    has_crop = True

        if not has_crop:
            cv2.putText(
                canvas, "[No Crop]", (box_x + 40, box_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA
            )

        # Recognized String & Confidence
        text_x = box_x + crop_w + 14
        display_val = str(val) if val else "[NOT EXTRACTED]"
        val_color = TEXT_WHITE if status == "EXTRACTED" else (BORDERLINE_AMBER if status == "LOW_CONFIDENCE" else TEXT_MUTED)

        # Truncate if long
        if len(display_val) > 24:
            display_val = display_val[:22] + "..."

        cv2.putText(
            canvas, display_val, (text_x, row_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, val_color, 1, cv2.LINE_AA
        )

        status_str = f"Status: {status} | Conf: {conf:.2f}"
        cv2.putText(
            canvas, status_str, (text_x, row_y + 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA
        )

    return canvas


# ---------------------------------------------------------------------------
# Panel 3: Field Confidence Gauges & Edit Metrics
# ---------------------------------------------------------------------------

def _render_metrics_panel(
    extracted_fields: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> np.ndarray:
    """Render confidence bar gauges and CER/edit distance metrics."""
    pw, ph = target_size
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    canvas[:] = PANEL_BG

    # Header bar
    cv2.rectangle(canvas, (0, 0), (pw, 36), (20, 30, 48), -1)
    title = "PANEL 3: CONFIDENCE GAUGES & METRICS" if not ground_truth else "PANEL 3: CONFIDENCE & CER AUDIT"
    cv2.putText(
        canvas, title,
        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEXT_WHITE, 1, cv2.LINE_AA
    )

    fields_order = ["name", "dob", "document_number", "issue_date", "expiry_date"]
    field_labels = {
        "name": "Name",
        "dob": "DOB",
        "document_number": "Doc No",
        "issue_date": "Issue Date",
        "expiry_date": "Expiry Date",
    }

    eval_results = {}
    if ground_truth:
        eval_results = evaluate_field_extraction(extracted_fields, ground_truth)

    start_y = 52
    row_height = 56
    gauge_x = 110
    gauge_w = 260
    gauge_h = 16

    for idx, f_name in enumerate(fields_order):
        f_info = extracted_fields.get(f_name, {})
        conf = float(f_info.get("confidence", 0.0))
        status = f_info.get("status", "UNKNOWN")
        color = FIELD_COLORS.get(f_name, (200, 200, 200))

        row_y = start_y + (idx * row_height)

        # Label
        lbl = field_labels.get(f_name, f_name)
        cv2.putText(
            canvas, lbl, (15, row_y + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA
        )

        # Background bar
        cv2.rectangle(canvas, (gauge_x, row_y), (gauge_x + gauge_w, row_y + gauge_h), (20, 30, 48), -1)
        cv2.rectangle(canvas, (gauge_x, row_y), (gauge_x + gauge_w, row_y + gauge_h), PANEL_BORDER, 1)

        # Filled portion
        fill_w = int(gauge_w * min(1.0, max(0.0, conf)))
        fill_color = MATCH_GREEN if conf >= 0.70 else (BORDERLINE_AMBER if conf >= 0.50 else MISMATCH_RED)
        if fill_w > 0:
            cv2.rectangle(canvas, (gauge_x, row_y), (gauge_x + fill_w, row_y + gauge_h), fill_color, -1)

        # Threshold mark (0.50)
        thresh_x = gauge_x + int(gauge_w * 0.50)
        cv2.line(canvas, (thresh_x, row_y - 2), (thresh_x, row_y + gauge_h + 2), (255, 255, 255), 1)

        # Confidence percentage text
        pct_text = f"{conf * 100:.1f}%"
        cv2.putText(
            canvas, pct_text, (gauge_x + gauge_w + 10, row_y + 13),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_WHITE, 1, cv2.LINE_AA
        )

        # Secondary row info (CER / Match or Extraction details)
        info_y = row_y + 30
        if ground_truth and f_name in eval_results:
            ev = eval_results[f_name]
            cer = ev.get("cer", 0.0)
            sim = ev.get("similarity", 0.0)
            em = ev.get("exact_match", False)

            em_str = "MATCH: YES" if em else "MATCH: NO"
            em_col = MATCH_GREEN if em else MISMATCH_RED

            cv2.putText(
                canvas, em_str, (gauge_x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, em_col, 1, cv2.LINE_AA
            )
            cv2.putText(
                canvas, f"CER: {cer:.3f} | Sim: {sim * 100:.1f}%", (gauge_x + 95, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA
            )
        else:
            cv2.putText(
                canvas, f"Threshold: 50.0% | Status: {status}", (gauge_x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_MUTED, 1, cv2.LINE_AA
            )

    # Threshold indicator note at bottom
    cv2.putText(
        canvas, "White line indicates acceptance threshold (0.50). Green = High, Amber = Mid, Red = Low.",
        (15, ph - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_MUTED, 1, cv2.LINE_AA
    )

    return canvas


# ---------------------------------------------------------------------------
# Panel 4: Forensic OCR Audit Verdict & Summary
# ---------------------------------------------------------------------------

def _render_verdict_panel(
    extracted_res: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> np.ndarray:
    """Render overall forensic triage decision badge and audit metadata table."""
    pw, ph = target_size
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    canvas[:] = PANEL_BG

    # Header bar
    cv2.rectangle(canvas, (0, 0), (pw, 36), (20, 30, 48), -1)
    cv2.putText(
        canvas, "PANEL 4: FORENSIC OCR AUDIT VERDICT",
        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, TEXT_WHITE, 1, cv2.LINE_AA
    )

    extracted_fields = extracted_res.get("fields", {})
    engine = extracted_res.get("engine", "rapidocr")
    doc_type = extracted_res.get("document_type", "identity_card")

    # Assess overall extraction state
    total_fields = 5
    extracted_count = sum(1 for f in extracted_fields.values() if f.get("status") == "EXTRACTED")
    low_conf_count = sum(1 for f in extracted_fields.values() if f.get("status") == "LOW_CONFIDENCE")
    unknown_count = sum(1 for f in extracted_fields.values() if f.get("status") == "UNKNOWN")

    confs = [f.get("confidence", 0.0) for f in extracted_fields.values()]
    mean_conf = float(np.mean(confs)) if confs else 0.0

    # Decision Badge
    badge_x, badge_y = 35, 52
    badge_w, badge_h = 510, 56

    if extracted_count == total_fields and mean_conf >= 0.70:
        badge_bg = MATCH_BG
        badge_border = MATCH_GREEN
        badge_text = "ALL FIELDS EXTRACTED (HIGH FIDELITY)"
        text_color = (255, 255, 255)
    elif extracted_count >= 3:
        badge_bg = BORDERLINE_BG
        badge_border = BORDERLINE_AMBER
        badge_text = f"PARTIAL EXTRACTION ({extracted_count}/{total_fields} FIELDS VERIFIED)"
        text_color = (255, 255, 255)
    else:
        badge_bg = MISMATCH_BG
        badge_border = MISMATCH_RED
        badge_text = f"EXTRACTION DEFICIENT ({unknown_count} FIELDS UNKNOWN)"
        text_color = (255, 255, 255)

    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_bg, -1)
    cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_border, 2)
    cv2.putText(
        canvas, badge_text,
        (badge_x + 20, badge_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_color, 2, cv2.LINE_AA
    )

    # Key-Value Audit Table
    start_y = 135
    row_h = 24
    kx = 45
    vx = 260

    rows = [
        ("OCR Engine:", f"{engine.upper()} (PaddleOCR ONNX / PyTesseract)"),
        ("Document Schema:", doc_type.replace("_", " ").title()),
        ("Target Fields:", f"{total_fields} Canonical Fields"),
        ("Extracted Successfully:", f"{extracted_count} / {total_fields}"),
        ("Low Confidence / Ambiguous:", f"{low_conf_count}"),
        ("Missing / Unknown:", f"{unknown_count}"),
        ("Mean Field Confidence:", f"{mean_conf * 100:.2f}%"),
    ]

    if ground_truth:
        ev_map = evaluate_field_extraction(extracted_fields, ground_truth)
        cers = [ev["cer"] for ev in ev_map.values() if ev["ground_truth"]]
        mean_cer = float(np.mean(cers)) if cers else 0.0
        exact_matches = sum(1 for ev in ev_map.values() if ev["exact_match"] and ev["ground_truth"])
        rows.append(("Mean Character Error Rate (CER):", f"{mean_cer:.4f} (Target <= 0.1500)"))
        rows.append(("Exact Match Accuracy:", f"{exact_matches} / {len(cers)} fields"))

    for i, (k, v) in enumerate(rows):
        cy = start_y + (i * row_h)
        cv2.putText(canvas, k, (kx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_MUTED, 1, cv2.LINE_AA)
        val_color = TEXT_WHITE
        if "CER" in k:
            val_color = MATCH_GREEN if "Target <=" in v and float(v.split()[0]) <= 0.15 else BORDERLINE_AMBER
        cv2.putText(canvas, v, (vx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, val_color, 1, cv2.LINE_AA)

    # Forensic Compliance Note
    note_y = ph - 18
    cv2.putText(
        canvas, "Forensic screening rule: Missing fields route to human inspection, not automatic fraud.",
        (15, note_y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, TEXT_MUTED, 1, cv2.LINE_AA
    )

    return canvas


# ---------------------------------------------------------------------------
# Top Header & Bottom Banner Assembly
# ---------------------------------------------------------------------------

def _render_header_banner(
    canvas: np.ndarray,
    doc_name: str,
    engine: str,
    timestamp_str: str,
):
    """Render top branding banner."""
    cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, 48), (20, 30, 48), -1)
    cv2.line(canvas, (0, 48), (CANVAS_WIDTH, 48), PANEL_BORDER, 1)

    cv2.putText(
        canvas, "FORGELENS-X", (25, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, ACCENT_BLUE, 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas, "— FORENSIC OCR & STRUCTURED FIELD AUDIT CARD", (195, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_WHITE, 1, cv2.LINE_AA
    )

    right_text = f"Doc: {doc_name} | Engine: {engine} | {timestamp_str}"
    (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(
        canvas, right_text, (CANVAS_WIDTH - tw - 25, 31),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA
    )


def _render_footer_banner(canvas: np.ndarray):
    """Render bottom forensic compliance footer."""
    cv2.rectangle(canvas, (0, CANVAS_HEIGHT - 32), (CANVAS_WIDTH, CANVAS_HEIGHT), (20, 30, 48), -1)
    cv2.line(canvas, (0, CANVAS_HEIGHT - 32), (CANVAS_WIDTH, CANVAS_HEIGHT - 32), PANEL_BORDER, 1)

    footer_text = (
        "ForgeLens-X M3 Forensic Screening System | PaddleOCR PP-OCRv4 ONNX / RapidOCR | "
        "Strict Dataset Isolation | Non-Punitive Missing Field Protocol"
    )
    cv2.putText(
        canvas, footer_text, (25, CANVAS_HEIGHT - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_MUTED, 1, cv2.LINE_AA
    )


# ---------------------------------------------------------------------------
# Main Public Interface
# ---------------------------------------------------------------------------

def generate_ocr_diagnostic_card(
    image_input: Union[str, np.ndarray, Image.Image],
    extracted_res: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    engine: str = "rapidocr",
) -> str:
    """
    Generate high-resolution 4-panel visual explanation card for OCR extraction.

    Args:
        image_input: path to document image, numpy array, or PIL Image
        extracted_res: output from extract_structured_fields (if None, will run extraction)
        ground_truth: optional dictionary of ground truth field values
        output_path: file destination path (.png)
        engine: OCR engine to run if extracted_res is None

    Returns:
        Absolute path to the saved diagnostic card image.
    """
    # 1. Load document image
    doc_bgr = load_image_for_ocr(image_input)
    if doc_bgr is None:
        raise ValueError(f"Unable to load document image for visual card: {image_input}")

    # 2. Extract fields if not provided
    if extracted_res is None:
        extracted_res = extract_structured_fields(doc_bgr, engine=engine)

    extracted_fields = extracted_res.get("fields", {})
    actual_engine = extracted_res.get("engine", engine)

    # 3. Derive doc name and timestamp
    if isinstance(image_input, str):
        doc_name = Path(image_input).name
    else:
        doc_name = "Document_Scan"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 4. Initialize main canvas
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    canvas[:] = BG_COLOR

    # 5. Render top header
    _render_header_banner(canvas, doc_name, actual_engine, now_str)

    # 6. Render individual panels
    p1 = _render_document_canvas_panel(doc_bgr, extracted_fields)
    p2 = _render_crops_gallery_panel(doc_bgr, extracted_fields)
    p3 = _render_metrics_panel(extracted_fields, ground_truth=ground_truth)
    p4 = _render_verdict_panel(extracted_res, ground_truth=ground_truth)

    # 7. Position panels on canvas (2x2 grid)
    # Layout Coordinates:
    #   Panel 1: x = 25,  y = 65
    #   Panel 2: x = 635, y = 65
    #   Panel 3: x = 25,  y = 445
    #   Panel 4: x = 635, y = 445
    x1, x2 = 25, 635
    y1, y2 = 65, 445

    canvas[y1:y1 + PANEL_HEIGHT, x1:x1 + PANEL_WIDTH] = p1
    canvas[y1:y1 + PANEL_HEIGHT, x2:x2 + PANEL_WIDTH] = p2
    canvas[y2:y2 + PANEL_HEIGHT, x1:x1 + PANEL_WIDTH] = p3
    canvas[y2:y2 + PANEL_HEIGHT, x2:x2 + PANEL_WIDTH] = p4

    # Border outlines for each panel
    cv2.rectangle(canvas, (x1, y1), (x1 + PANEL_WIDTH, y1 + PANEL_HEIGHT), PANEL_BORDER, 1)
    cv2.rectangle(canvas, (x2, y1), (x2 + PANEL_WIDTH, y1 + PANEL_HEIGHT), PANEL_BORDER, 1)
    cv2.rectangle(canvas, (x1, y2), (x1 + PANEL_WIDTH, y2 + PANEL_HEIGHT), PANEL_BORDER, 1)
    cv2.rectangle(canvas, (x2, y2), (x2 + PANEL_WIDTH, y2 + PANEL_HEIGHT), PANEL_BORDER, 1)

    # 8. Render bottom footer
    _render_footer_banner(canvas)

    # 9. Save image
    if output_path is None:
        vis_dir = os.path.join(get_reports_dir(), "visuals")
        ensure_dirs(vis_dir)
        stem = Path(doc_name).stem
        output_path = os.path.join(vis_dir, f"ocr_card_{stem}.png")

    ensure_dirs(os.path.dirname(output_path))
    cv2.imwrite(output_path, canvas)

    return os.path.abspath(output_path)


def batch_generate_ocr_cards(
    sample_evaluations: List[Dict[str, Any]],
    output_dir: str,
    max_cards: int = 5,
) -> List[str]:
    """Generate a batch of diagnostic cards from sample evaluation results."""
    ensure_dirs(output_dir)
    generated_paths = []

    for item in sample_evaluations[:max_cards]:
        img_path = item.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue

        raw_ext = item.get("raw_extracted", {})
        gt_dict = {}
        for f, res in item.get("evaluations", {}).items():
            gt_dict[f] = res.get("ground_truth", "")

        stem = Path(img_path).stem
        out_path = os.path.join(output_dir, f"ocr_card_{stem}.png")

        try:
            saved = generate_ocr_diagnostic_card(
                image_input=img_path,
                extracted_res={"fields": raw_ext, "engine": "rapidocr", "document_type": "identity_card"},
                ground_truth=gt_dict,
                output_path=out_path,
            )
            generated_paths.append(saved)
        except Exception as e:
            print(f"[Warning] Failed to generate visual card for {stem}: {e}")

    return generated_paths
