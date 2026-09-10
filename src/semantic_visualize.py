"""
ForgeLens-X — Milestone 4: Semantic Visual Forensic Diagnostic Card
===================================================================
Renders high-resolution 4-panel diagnostic explanation cards for human forensic review:
    Panel 1: Document Canvas & Semantic / Spatial Field Highlights
    Panel 2: Optical Field Crops & Decoded MRZ / Schema Concordance
    Panel 3: Canonical 8-Rule Semantic Integrity Matrix
    Panel 4: Multi-Modal Forensic Fusion Verdict & Gauges (SWT / EXIF / MRZ)
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.document_template import _get_bold_font, _get_font
from src.utils import ensure_dirs, get_reports_dir


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
NA_GRAY = (100, 116, 139)        # Slate 500 (#64748b)
NA_BG = (51, 65, 85)             # Slate 700 (#334155)


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    status: str,
    font: ImageFont.ImageFont,
    pad_x: int = 8,
    pad_y: int = 3,
) -> int:
    """Draw status pill badge and return badge width."""
    if status in ["PASS", "VERIFIED", "AUTHENTIC", "CLEAN"]:
        bg_col = MATCH_BG
        txt_col = MATCH_GREEN
    elif status in ["FAIL", "SUSPECT", "CRITICAL", "FORGERY", "TAMPERED"]:
        bg_col = MISMATCH_BG
        txt_col = MISMATCH_RED
    elif status in ["UNKNOWN", "WARNING", "BORDERLINE", "REPAIRED"]:
        bg_col = BORDERLINE_BG
        txt_col = BORDERLINE_AMBER
    else:  # NOT_APPLICABLE / STRIPPED / N/A
        bg_col = NA_BG
        txt_col = TEXT_MUTED

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    bx1, by1 = x, y
    bx2, by2 = x + tw + pad_x * 2, y + th + pad_y * 2
    draw.rounded_rectangle([bx1, by1, bx2, by2], radius=4, fill=bg_col)
    draw.text((bx1 + pad_x, by1 + pad_y - 1), text, font=font, fill=txt_col)
    return (bx2 - bx1)


# ---------------------------------------------------------------------------
# Panel 1: Document Canvas & Spatial Highlights
# ---------------------------------------------------------------------------

def _render_document_canvas_panel(
    doc_bgr: np.ndarray,
    fields: Dict[str, Any],
    semantic_audit: Dict[str, Any],
    font_audit: Optional[Dict[str, Any]] = None,
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> np.ndarray:
    """Render document canvas with spatial overlays and tamper highlighting."""
    p_w, p_h = target_size
    panel = np.full((p_h, p_w, 3), PANEL_BG, dtype=np.uint8)

    # Header bar
    cv2.rectangle(panel, (0, 0), (p_w, 38), (38, 51, 73), -1)
    cv2.putText(panel, "PANEL 1: DOCUMENT CANVAS & FIELD SPATIAL LAYOUT", (16, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 1, cv2.LINE_AA)

    if doc_bgr is None or doc_bgr.size == 0:
        cv2.putText(panel, "[No Document Image Available]", (p_w // 4, p_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1, cv2.LINE_AA)
        return panel

    dh, dw = doc_bgr.shape[:2]
    canvas_area_w = p_w - 32
    canvas_area_h = p_h - 60

    scale = min(canvas_area_w / dw, canvas_area_h / dh)
    new_w, new_h = int(dw * scale), int(dh * scale)
    resized_doc = cv2.resize(doc_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    off_x = 16 + (canvas_area_w - new_w) // 2
    off_y = 48 + (canvas_area_h - new_h) // 2

    # Draw scaled document
    panel[off_y:off_y + new_h, off_x:off_x + new_w] = resized_doc

    # Identify suspect fields from font audit or semantic audit
    font_anomalous_fields = set()
    if font_audit:
        for a in font_audit.get("anomalous_fields", []):
            if a.get("field"):
                font_anomalous_fields.add(a["field"])

    # Highlight fields
    for fname, fdata in fields.items():
        if fname.startswith("_") or not isinstance(fdata, dict):
            continue
        bbox = fdata.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            sx1 = int(off_x + x1 * scale)
            sy1 = int(off_y + y1 * scale)
            sx2 = int(off_x + x2 * scale)
            sy2 = int(off_y + y2 * scale)

            is_anomalous = (fname in font_anomalous_fields)
            color = (68, 68, 239) if is_anomalous else (248, 189, 56)  # Red BGR vs Cyan BGR
            thickness = 2 if is_anomalous else 1

            cv2.rectangle(panel, (sx1, sy1), (sx2, sy2), color, thickness)
            tag = f"{fname} [!]" if is_anomalous else fname
            cv2.putText(panel, tag, (sx1, max(12, sy1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Border
    cv2.rectangle(panel, (0, 0), (p_w - 1, p_h - 1), PANEL_BORDER, 1)
    return panel


# ---------------------------------------------------------------------------
# Panel 2: Field Optical Crops & Decoded MRZ / Schema Concordance
# ---------------------------------------------------------------------------

def _render_optical_crops_panel(
    fields: Dict[str, Any],
    mrz_data: Optional[Dict[str, Any]] = None,
    doc_bgr: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> np.ndarray:
    """Render extracted text crops and parsed MRZ / schema details."""
    p_w, p_h = target_size
    img_pil = Image.new("RGB", (p_w, p_h), PANEL_BG)
    draw = ImageDraw.ImageDraw(img_pil)

    font_title = _get_bold_font(13)
    font_bold = _get_bold_font(11)
    font_reg = _get_font(10)
    font_mono = _get_font(9)

    # Header bar
    draw.rectangle([0, 0, p_w, 38], fill=(38, 51, 73))
    draw.text((16, 11), "PANEL 2: OPTICAL CROPS & DECODED CREDENTIAL FIELDS", font=font_title, fill=TEXT_WHITE)

    curr_y = 48
    target_fields = ["document_number", "dob", "expiry_date", "name"]

    for fname in target_fields:
        fdata = fields.get(fname, {})
        val = fdata.get("value") if isinstance(fdata, dict) else str(fdata or "")
        crop = fdata.get("crop") if isinstance(fdata, dict) else None

        # If crop missing but doc_bgr available, crop it
        if crop is None and isinstance(fdata, dict) and fdata.get("bbox") and doc_bgr is not None:
            bx1, by1, bx2, by2 = fdata["bbox"]
            dh, dw = doc_bgr.shape[:2]
            bx1, by1 = max(0, int(bx1)), max(0, int(by1))
            bx2, by2 = min(dw, int(bx2)), min(dh, int(by2))
            if bx2 > bx1 and by2 > by1:
                crop = doc_bgr[by1:by2, bx1:bx2]

        label = fname.upper().replace("_", " ")
        draw.text((16, curr_y), f"{label}:", font=font_bold, fill=ACCENT_BLUE)

        val_display = str(val)[:30] if val else "[NOT EXTRACTED]"
        draw.text((170, curr_y), val_display, font=font_reg, fill=TEXT_WHITE)

        # Place miniature crop thumbnail if available
        if crop is not None and isinstance(crop, np.ndarray) and crop.size > 0:
            ch, cw = crop.shape[:2]
            thumb_h = 24
            thumb_w = min(160, int(cw * (thumb_h / max(1, ch))))
            if thumb_w > 10:
                resized_crop = cv2.resize(crop, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                crop_pil = Image.fromarray(cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB))
                img_pil.paste(crop_pil, (380, curr_y - 3))
                draw.rectangle([379, curr_y - 4, 380 + thumb_w + 1, curr_y + thumb_h], outline=PANEL_BORDER, width=1)

        curr_y += 32

    # Divider line
    draw.line([16, curr_y, p_w - 16, curr_y], fill=PANEL_BORDER, width=1)
    curr_y += 10

    # Decoded MRZ Section
    draw.text((16, curr_y), "MACHINE READABLE ZONE (ICAO Doc 9303):", font=font_bold, fill=ACCENT_CYAN)
    curr_y += 20

    if mrz_data and mrz_data.get("format"):
        fmt = mrz_data.get("format")
        mrz_verdict = mrz_data.get("verdict", "UNKNOWN")
        _draw_badge(draw, 340, curr_y - 20, fmt, "PASS", font_mono)
        _draw_badge(draw, 410, curr_y - 20, mrz_verdict[:22], mrz_data.get("status", "PASS"), font_mono)

        raw_lines = mrz_data.get("raw_lines", [])
        for r_line in raw_lines[:3]:
            draw.text((16, curr_y), r_line, font=font_mono, fill=(186, 230, 253))
            curr_y += 18

        # Repairs or Checksum info
        repairs = mrz_data.get("repairs", [])
        if repairs:
            draw.text((16, curr_y), f"Hypothesis Repair Applied ({len(repairs)} flips):", font=font_bold, fill=BORDERLINE_AMBER)
            curr_y += 16
            for rep in repairs[:2]:
                draw.text((24, curr_y), f"• {rep['field']}: {rep['detail']}", font=font_mono, fill=TEXT_MUTED)
                curr_y += 16
        else:
            chk_sum = mrz_data.get("checksum_summary", {})
            valid_count = sum(1 for v in chk_sum.values() if v)
            draw.text((16, curr_y), f"Checksums: {valid_count}/{len(chk_sum)} fields verified with cyclic [7, 3, 1] weights.", font=font_reg, fill=MATCH_GREEN if mrz_data.get("all_checksums_valid") else MISMATCH_RED)
    else:
        draw.text((16, curr_y), "No MRZ block detected (Standard National ID / Synthetic Card without TD1/TD3).", font=font_reg, fill=TEXT_MUTED)
        curr_y += 20
        draw.text((16, curr_y), "Semantic integrity evaluated via Visual Inspection Zone (VIZ) rules.", font=font_reg, fill=TEXT_MUTED)

    panel_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel_bgr, (0, 0), (p_w - 1, p_h - 1), PANEL_BORDER, 1)
    return panel_bgr


# ---------------------------------------------------------------------------
# Panel 3: Canonical 8-Rule Semantic Integrity Matrix
# ---------------------------------------------------------------------------

def _render_semantic_rules_panel(
    semantic_audit: Dict[str, Any],
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> np.ndarray:
    """Render canonical 8-rule status checklist with badge indicators."""
    p_w, p_h = target_size
    img_pil = Image.new("RGB", (p_w, p_h), PANEL_BG)
    draw = ImageDraw.ImageDraw(img_pil)

    font_title = _get_bold_font(13)
    font_bold = _get_bold_font(11)
    font_reg = _get_font(10)
    font_badge = _get_bold_font(9)

    # Header bar
    draw.rectangle([0, 0, p_w, 38], fill=(38, 51, 73))
    draw.text((16, 11), "PANEL 3: CANONICAL 8-RULE SEMANTIC INTEGRITY MATRIX", font=font_title, fill=TEXT_WHITE)

    checks_list = semantic_audit.get("checks_list", [])
    if not checks_list and "checks" in semantic_audit:
        checks_list = list(semantic_audit["checks"].values())

    curr_y = 48
    rule_labels = {
        "impossible_dates": "1. Calendar Sanity (e.g. 30 Feb)",
        "chronology_order": "2. Chronology (DOB < Issue < Exp)",
        "age_at_issue_sanity": "3. Age-at-Issue Sanity",
        "validity_window_sanity": "4. Validity Window (<= 25 yrs)",
        "anachronism_check": "5. Future Anachronism Buffer",
        "document_number_format": "6. Doc Number Schema Regex",
        "duplicate_field_contradiction": "7. Duplicate Contradiction",
        "name_structure_sanity": "8. Name Alpha Structure Sanity",
    }

    for chk_data in checks_list[:8]:
        c_name = chk_data.get("check", "unknown")
        c_status = chk_data.get("status", "UNKNOWN")
        c_detail = chk_data.get("detail", "")

        label = rule_labels.get(c_name, c_name.replace("_", " ").title())
        draw.text((16, curr_y + 2), label, font=font_bold, fill=TEXT_WHITE)

        # Draw Status Pill Badge
        bw = _draw_badge(draw, 240, curr_y, c_status, c_status, font_badge)

        # Truncate detail to fit
        short_detail = c_detail[:45] + "..." if len(c_detail) > 48 else c_detail
        detail_color = MISMATCH_RED if c_status == "FAIL" else TEXT_MUTED
        draw.text((240 + bw + 12, curr_y + 2), short_detail, font=font_reg, fill=detail_color)

        curr_y += 36

    panel_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel_bgr, (0, 0), (p_w - 1, p_h - 1), PANEL_BORDER, 1)
    return panel_bgr


# ---------------------------------------------------------------------------
# Panel 4: Multi-Modal Forensic Fusion Verdict & Gauges
# ---------------------------------------------------------------------------

def _render_multi_modal_gauges_panel(
    fusion_res: Dict[str, Any],
    metadata_audit: Optional[Dict[str, Any]] = None,
    font_audit: Optional[Dict[str, Any]] = None,
    mrz_data: Optional[Dict[str, Any]] = None,
    target_size: Tuple[int, int] = (PANEL_WIDTH, PANEL_HEIGHT),
) -> np.ndarray:
    """Render multi-modal gauges (SWT font, EXIF provenance, MRZ) and composite threat badge."""
    p_w, p_h = target_size
    img_pil = Image.new("RGB", (p_w, p_h), PANEL_BG)
    draw = ImageDraw.ImageDraw(img_pil)

    font_title = _get_bold_font(13)
    font_bold = _get_bold_font(11)
    font_reg = _get_font(10)
    font_verdict = _get_bold_font(14)
    font_badge = _get_bold_font(9)

    # Header bar
    draw.rectangle([0, 0, p_w, 38], fill=(38, 51, 73))
    draw.text((16, 11), "PANEL 4: FORENSIC FUSION VERDICT & MULTI-MODAL GAUGES", font=font_title, fill=TEXT_WHITE)

    curr_y = 50

    # Big Composite Threat Badge
    threat_level = fusion_res.get("threat_level", "UNKNOWN")
    is_auth = fusion_res.get("is_authentic", False)
    badge_status = "PASS" if is_auth else "FAIL"

    draw.text((16, curr_y), "COMPOSITE THREAT LEVEL:", font=font_bold, fill=TEXT_MUTED)
    curr_y += 20
    _draw_badge(draw, 16, curr_y, threat_level, badge_status, font_verdict, pad_x=14, pad_y=6)
    curr_y += 45

    # 1. Stroke Width Typography Gauge
    draw.text((16, curr_y), "TYPOGRAPHY FORENSICS (SWT):", font=font_bold, fill=ACCENT_BLUE)
    curr_y += 18
    if font_audit:
        f_verdict = font_audit.get("typography_verdict", "UNKNOWN")
        max_z = font_audit.get("max_stroke_zscore", 0.0)
        _draw_badge(draw, 16, curr_y, f_verdict, "PASS" if f_verdict == "TYPOGRAPHY_CONSISTENT" else "FAIL", font_badge)
        draw.text((220, curr_y + 2), f"Max Z-Score: {max_z:.2f} (Threshold: 2.50)", font=font_reg, fill=TEXT_WHITE)
    else:
        draw.text((16, curr_y), "No font metrics computed.", font=font_reg, fill=TEXT_MUTED)
    curr_y += 32

    # 2. JPEG EXIF & Provenance Gauge
    draw.text((16, curr_y), "EXIF METADATA PROVENANCE:", font=font_bold, fill=ACCENT_CYAN)
    curr_y += 18
    if metadata_audit:
        p_verdict = metadata_audit.get("provenance_verdict", "UNKNOWN")
        sw = metadata_audit.get("software_detected")
        _draw_badge(draw, 16, curr_y, p_verdict, "FAIL" if metadata_audit.get("is_tampered") else "PASS", font_badge)
        sw_detail = f"Software: '{sw}'" if sw else "No blacklisted editing signatures."
        draw.text((220, curr_y + 2), sw_detail, font=font_reg, fill=TEXT_WHITE)
    else:
        draw.text((16, curr_y), "No EXIF provenance audit available.", font=font_reg, fill=TEXT_MUTED)
    curr_y += 32

    # 3. MRZ & VIZ Concordance Gauge
    draw.text((16, curr_y), "ICAO MRZ & VIZ CONCORDANCE:", font=font_bold, fill=BORDERLINE_AMBER)
    curr_y += 18
    if mrz_data:
        mrz_status = mrz_data.get("status", "UNKNOWN")
        mrz_verdict = mrz_data.get("verdict", "UNKNOWN")
        _draw_badge(draw, 16, curr_y, mrz_verdict, mrz_status, font_badge)
        unrepaired = mrz_data.get("unrepaired_failures", [])
        detail_txt = f"Failures: {unrepaired}" if unrepaired else "All check digits verified."
        draw.text((240, curr_y + 2), detail_txt, font=font_reg, fill=TEXT_WHITE)
    else:
        draw.text((16, curr_y), "MRZ absent; visual credential format.", font=font_reg, fill=TEXT_MUTED)
    curr_y += 32

    # Divider & Inspector Narrative
    draw.line([16, curr_y, p_w - 16, curr_y], fill=PANEL_BORDER, width=1)
    curr_y += 10

    summary_text = fusion_res.get("verdict_summary", "")
    draw.text((16, curr_y), "Inspector Summary:", font=font_bold, fill=TEXT_WHITE)
    curr_y += 18
    # Wrap text into 2 lines
    if len(summary_text) > 75:
        line1 = summary_text[:75]
        split_idx = line1.rfind(" ")
        if split_idx > 40:
            draw.text((16, curr_y), line1[:split_idx], font=font_reg, fill=TEXT_MUTED)
            curr_y += 16
            draw.text((16, curr_y), summary_text[split_idx:].strip()[:75], font=font_reg, fill=TEXT_MUTED)
        else:
            draw.text((16, curr_y), line1, font=font_reg, fill=TEXT_MUTED)
    else:
        draw.text((16, curr_y), summary_text, font=font_reg, fill=TEXT_MUTED)

    panel_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel_bgr, (0, 0), (p_w - 1, p_h - 1), PANEL_BORDER, 1)
    return panel_bgr


# ---------------------------------------------------------------------------
# Master 4-Panel Canvas Assembly
# ---------------------------------------------------------------------------

def generate_semantic_diagnostic_card(
    doc_bgr: np.ndarray,
    fields: Dict[str, Any],
    semantic_audit: Dict[str, Any],
    metadata_audit: Optional[Dict[str, Any]] = None,
    font_audit: Optional[Dict[str, Any]] = None,
    mrz_data: Optional[Dict[str, Any]] = None,
    mrz_viz_cross: Optional[Dict[str, Any]] = None,
    spatial_bridge: Optional[Dict[str, Any]] = None,
    doc_id: str = "doc_sample",
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """
    Assemble high-resolution 1240x860 4-panel semantic explanation card.
    """
    from src.ocr_forensic_bridge import fuse_forensic_modalities

    fusion_res = fuse_forensic_modalities(
        spatial_bridge=spatial_bridge,
        semantic_audit=semantic_audit,
        font_audit=font_audit,
        metadata_audit=metadata_audit,
        mrz_audit=mrz_data,
        mrz_viz_cross=mrz_viz_cross,
    )

    p1 = _render_document_canvas_panel(doc_bgr, fields, semantic_audit, font_audit)
    p2 = _render_optical_crops_panel(fields, mrz_data, doc_bgr)
    p3 = _render_semantic_rules_panel(semantic_audit)
    p4 = _render_multi_modal_gauges_panel(fusion_res, metadata_audit, font_audit, mrz_data)

    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), BG_COLOR, dtype=np.uint8)

    # Master Header Bar (Height 50)
    cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, 50), (20, 29, 47), -1)
    cv2.putText(canvas, "FORGELENS-X — MILESTONE 4: SEMANTIC, MRZ & TYPOGRAPHY DIAGNOSTIC CARD",
                (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (248, 250, 252), 2, cv2.LINE_AA)
    doc_tag = f"DOC: {doc_id}  |  STATUS: {fusion_res['threat_level']}"
    cv2.putText(canvas, doc_tag, (CANVAS_WIDTH - 420, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

    # Layout 2x2 Grid
    # Row 1: y = 65..425
    canvas[65:65 + PANEL_HEIGHT, 24:24 + PANEL_WIDTH] = p1
    canvas[65:65 + PANEL_HEIGHT, 636:636 + PANEL_WIDTH] = p2

    # Row 2: y = 445..805
    canvas[445:445 + PANEL_HEIGHT, 24:24 + PANEL_WIDTH] = p3
    canvas[445:445 + PANEL_HEIGHT, 636:636 + PANEL_WIDTH] = p4

    # Footer Bar
    cv2.rectangle(canvas, (0, CANVAS_HEIGHT - 35), (CANVAS_WIDTH, CANVAS_HEIGHT), (20, 29, 47), -1)
    footer_text = "ForgeLens-X Multi-Modal Forensic Engine  |  ICAO Doc 9303 Compliant  |  Distance-Transform SWT Typography Analysis"
    cv2.putText(canvas, footer_text, (24, CANVAS_HEIGHT - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 116, 139), 1, cv2.LINE_AA)

    # Determine save destination
    if output_path is None:
        vis_dir = os.path.join(os.getcwd(), "reports", "visuals")
        ensure_dirs([vis_dir])
        output_path = os.path.join(vis_dir, f"semantic_card_{doc_id}.png")
    else:
        out_parent = os.path.dirname(output_path)
        if out_parent:
            ensure_dirs([out_parent])

    cv2.imwrite(output_path, canvas)
    return canvas, output_path
