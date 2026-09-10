"""
ForgeLens-X — Milestone 5: Unified Forensic Card Visualization Engine
======================================================================
Renders publication-grade 1280x860 4-panel diagnostic inspection cards
synthesizing physical pixel tamper signals (ELA/Copy-Move), biometric face
verification, structured OCR crops, semantic validation, and attack hypotheses.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils import ensure_dirs, get_reports_dir

# ---------------------------------------------------------------------------
# Visual Styling Constants
# ---------------------------------------------------------------------------
CARD_WIDTH = 1280
CARD_HEIGHT = 870
PANEL_W = 625
PANEL_H = 400
PAD = 10

BG_COLOR = (23, 17, 13)         # Dark slate BGR (#0d1117)
PANEL_BG = (36, 28, 22)         # Charcoal card BGR (#161c24)
BORDER_COLOR = (55, 45, 38)     # Subtle border
TEXT_WHITE = (240, 240, 240)    # Primary text
TEXT_MUTED = (160, 160, 160)    # Secondary captions
CYAN = (248, 189, 56)           # OCR fields
GREEN = (78, 201, 78)           # Authentic / Pass
RED = (68, 68, 239)             # Fraud / Fail
AMBER = (11, 158, 245)          # Warning / Moderate
PURPLE = (200, 70, 180)         # Copy-move
ORANGE = (30, 140, 255)         # Review / Insufficient


def _draw_header(card: np.ndarray, doc_id: str, decision: str, attack_type: str) -> None:
    """Draw card top banner."""
    # Top header bar
    cv2.rectangle(card, (0, 0), (CARD_WIDTH, 42), (30, 22, 18), -1)
    cv2.line(card, (0, 42), (CARD_WIDTH, 42), BORDER_COLOR, 1)

    title = "FORGELENS-X :: UNIFIED FORENSIC SCREENING CARD"
    cv2.putText(card, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_WHITE, 2, cv2.LINE_AA)

    doc_str = f"DOC ID: {doc_id[:25]}"
    cv2.putText(card, doc_str, (540, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_MUTED, 1, cv2.LINE_AA)

    # Decision badge
    dec_color = GREEN if decision == "CLEAR_AUTHENTIC" else RED if decision == "SUSPECT_TAMPERING" else ORANGE
    cv2.rectangle(card, (920, 8), (1260, 34), dec_color, -1)
    cv2.putText(card, f"DECISION: {decision[:18]}", (930, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)


def _render_document_canvas_panel(
    doc_bgr: np.ndarray,
    report: Dict[str, Any],
    target_size: Tuple[int, int] = (PANEL_W, PANEL_H),
) -> np.ndarray:
    """Panel 1: Document canvas with overlaid OCR fields, ELA candidate, and copy-move links."""
    pw, ph = target_size
    panel = np.full((ph, pw, 3), PANEL_BG, dtype=np.uint8)

    # Panel Title
    cv2.putText(panel, "PANEL 1: MULTI-MODAL SPATIAL CANVAS", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, CYAN, 1, cv2.LINE_AA)

    if doc_bgr is None or doc_bgr.size == 0:
        cv2.putText(panel, "[NO IMAGE AVAILABLE]", (pw // 4, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_MUTED, 1, cv2.LINE_AA)
        return panel

    dh, dw = doc_bgr.shape[:2]
    avail_w = pw - 30
    avail_h = ph - 45
    scale = min(avail_w / dw, avail_h / dh)
    nw, nh = int(dw * scale), int(dh * scale)
    off_x = 15 + (avail_w - nw) // 2
    off_y = 35 + (avail_h - nh) // 2

    resized = cv2.resize(doc_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = resized.copy()

    fields = report.get("fields", {})
    tamper = report.get("tamper_signals", {})
    sus_regs = report.get("suspicious_regions", [])
    sus_fields = {s.get("field"): s for s in sus_regs if s.get("field")}

    # Draw OCR field bounding boxes
    for fname, fdata in fields.items():
        if fname.startswith("_") or not isinstance(fdata, dict):
            continue
        fbox = fdata.get("bbox")
        if fbox and len(fbox) == 4:
            x1 = int(fbox[0] * scale)
            y1 = int(fbox[1] * scale)
            x2 = int(fbox[2] * scale)
            y2 = int(fbox[3] * scale)

            is_suspect = fname in sus_fields
            box_col = RED if is_suspect else CYAN
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_col, 2 if is_suspect else 1)

            lbl = f"{fname} [!]" if is_suspect else fname
            cv2.putText(canvas, lbl, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_col, 1, cv2.LINE_AA)

    # Overlay ELA Candidate region
    ela_box = tamper.get("ela", {}).get("candidate_bbox")
    if ela_box and len(ela_box) == 4:
        ex1 = int(ela_box[0] * scale)
        ey1 = int(ela_box[1] * scale)
        ex2 = int(ela_box[2] * scale)
        ey2 = int(ela_box[3] * scale)
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), RED, 2)
        cv2.putText(canvas, "ELA ANOMALY", (ex1, min(nh - 5, ey2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED, 1, cv2.LINE_AA)

    # Overlay Copy-Move Bboxes
    cm_box = tamper.get("copy_move", {}).get("bbox")
    alt_box = tamper.get("copy_move", {}).get("alt_bbox")
    if cm_box and len(cm_box) == 4:
        cx1, cy1, cx2, cy2 = [int(v * scale) for v in cm_box]
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), PURPLE, 2)
        cv2.putText(canvas, "CLONE RCVR", (cx1, max(12, cy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, PURPLE, 1, cv2.LINE_AA)

    if alt_box and len(alt_box) == 4:
        ax1, ay1, ax2, ay2 = [int(v * scale) for v in alt_box]
        cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), PURPLE, 2)
        cv2.putText(canvas, "CLONE DONOR", (ax1, max(12, ay1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, PURPLE, 1, cv2.LINE_AA)

    panel[off_y:off_y + nh, off_x:off_x + nw] = canvas
    return panel


def _render_evidence_crops_panel(
    doc_bgr: np.ndarray,
    report: Dict[str, Any],
    target_size: Tuple[int, int] = (PANEL_W, PANEL_H),
) -> np.ndarray:
    """Panel 2: Optical crops, face portrait, and quality metrics."""
    pw, ph = target_size
    panel = np.full((ph, pw, 3), PANEL_BG, dtype=np.uint8)

    cv2.putText(panel, "PANEL 2: BIOMETRIC & OPTICAL EVIDENCE CROPS", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, CYAN, 1, cv2.LINE_AA)

    # 1. Quality summary badge
    qual = report.get("quality", {})
    b_score = qual.get("blur_score", 0.0)
    rel = qual.get("analysis_reliability", "UNKNOWN")
    rel_col = GREEN if rel == "HIGH" else AMBER if rel == "MEDIUM" else RED

    cv2.putText(panel, f"Sharpness (Laplacian): {b_score:.1f}", (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Reliability: {rel}", (250, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, rel_col, 1, cv2.LINE_AA)

    dims = qual.get("dimensions", [0, 0])
    cv2.putText(panel, f"Resolution: {dims[0]}x{dims[1]}", (420, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUTED, 1, cv2.LINE_AA)

    # 2. Face Portrait Verification
    cv2.line(panel, (15, 68), (pw - 15, 68), BORDER_COLOR, 1)
    cv2.putText(panel, "FACE BIOMETRIC SCREENING:", (15, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)

    face = report.get("face_verification", {})
    has_face = face.get("has_face_check")
    f_box = face.get("face_bbox") or [30, 80, 200, 260]

    # Crop document photo
    doc_photo_crop = None
    if doc_bgr is not None and len(f_box) == 4:
        px1, py1, px2, py2 = f_box
        dh, dw = doc_bgr.shape[:2]
        px1 = max(0, min(dw - 1, px1))
        py1 = max(0, min(dh - 1, py1))
        px2 = max(px1 + 1, min(dw, px2))
        py2 = max(py1 + 1, min(dh, py2))
        crop = doc_bgr[py1:py2, px1:px2]
        if crop.size > 0:
            doc_photo_crop = cv2.resize(crop, (95, 120), interpolation=cv2.INTER_AREA)

    if doc_photo_crop is not None:
        panel[102:102 + 120, 20:20 + 95] = doc_photo_crop
        cv2.rectangle(panel, (20, 102), (20 + 95, 102 + 120), CYAN, 1)
        cv2.putText(panel, "ID Photo", (35, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)

    # Face verification details
    if has_face:
        verified = face.get("verified")
        dist = face.get("distance")
        thresh = face.get("threshold")
        sim = face.get("similarity_pct")
        f_col = GREEN if verified else RED
        f_txt = "CONFIRMED MATCH" if verified else "FACE MISMATCH FRAUD"

        cv2.putText(panel, f"Verdict: {f_txt}", (135, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, f_col, 1, cv2.LINE_AA)
        cv2.putText(panel, f"Cosine Distance: {dist} (Threshold: {thresh})", (135, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(panel, f"Similarity Confidence: {sim}%", (135, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_MUTED, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Status: UNCHECKED (No reference selfie supplied)", (135, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(panel, "Document portrait localized for physical tamper audit.", (135, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)

    # 3. Optical Field Crops
    cv2.line(panel, (15, 250), (pw - 15, 250), BORDER_COLOR, 1)
    cv2.putText(panel, "EXTRACTED FIELD VALUES:", (15, 272),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)

    fields = report.get("fields", {})
    disp_fields = ["name", "document_number", "dob", "expiry_date"]
    y_pos = 295
    for df in disp_fields:
        fdata = fields.get(df, {})
        val = str(fdata.get("value") or "NOT DETECTED")[:28]
        conf = fdata.get("confidence")
        conf_str = f"(conf: {conf:.2f})" if conf is not None else ""
        cv2.putText(panel, f"{df.upper()}:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CYAN, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{val} {conf_str}", (160, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_WHITE, 1, cv2.LINE_AA)
        y_pos += 26

    return panel


def _render_suspicious_regions_panel(
    report: Dict[str, Any],
    target_size: Tuple[int, int] = (PANEL_W, PANEL_H),
) -> np.ndarray:
    """Panel 3: Itemized suspicious regions table & MRZ checksums."""
    pw, ph = target_size
    panel = np.full((ph, pw, 3), PANEL_BG, dtype=np.uint8)

    cv2.putText(panel, "PANEL 3: SUSPICIOUS REGIONS & MRZ FORENSICS", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, CYAN, 1, cv2.LINE_AA)

    sus_regs = report.get("suspicious_regions", [])

    if not sus_regs:
        cv2.putText(panel, "ZERO SUSPICIOUS REGIONS IDENTIFIED", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
        cv2.putText(panel, "All fields demonstrate smooth spatial and physical continuity.", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, f"Identified {len(sus_regs)} anomalous region(s):", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 1, cv2.LINE_AA)

        y_pos = 78
        for i, reg in enumerate(sus_regs[:4]):
            src = reg.get("source", "unknown").upper()
            fld = reg.get("field", "canvas")
            conf = float(reg.get("confidence", 0.0))
            evid = reg.get("evidence", "")[:45]

            tag_col = RED if src in ["ELA", "SEMANTIC", "FACE"] else AMBER
            cv2.rectangle(panel, (18, y_pos - 12), (90, y_pos + 4), tag_col, -1)
            cv2.putText(panel, src[:7], (22, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(panel, f"Field: {fld} (conf: {conf:.2f})", (98, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(panel, f"{evid}...", (20, y_pos + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, TEXT_MUTED, 1, cv2.LINE_AA)

            y_pos += 38

    # Bottom: MRZ Summary Section
    cv2.line(panel, (15, 245), (pw - 15, 245), BORDER_COLOR, 1)
    cv2.putText(panel, "ICAO DOC 9303 MRZ VALIDATION:", (15, 268),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)

    mrz = report.get("mrz", {})
    if not mrz or not mrz.get("status"):
        cv2.putText(panel, "No Machine Readable Zone present (Credential is VIZ-only format).", (20, 295),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)
    else:
        status = mrz.get("status")
        stat_col = GREEN if status == "PASS" else RED
        fmt = mrz.get("format", "TD3")
        verdict = mrz.get("verdict", "N/A")

        cv2.putText(panel, f"Format: {fmt}  |  Checksum Status: {status}", (20, 295),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, stat_col, 1, cv2.LINE_AA)
        cv2.putText(panel, f"Verdict: {verdict}", (20, 318),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_WHITE, 1, cv2.LINE_AA)

    return panel


def _render_executive_summary_panel(
    report: Dict[str, Any],
    target_size: Tuple[int, int] = (PANEL_W, PANEL_H),
) -> np.ndarray:
    """Panel 4: Attack classification hypothesis, confidence bar, and evidentiary basis."""
    pw, ph = target_size
    panel = np.full((ph, pw, 3), PANEL_BG, dtype=np.uint8)

    cv2.putText(panel, "PANEL 4: ATTACK CLASSIFICATION & RATIONALE", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, CYAN, 1, cv2.LINE_AA)

    attack_guess = str(report.get("attack_type_guess", "none")).upper()
    attack_conf = float(report.get("attack_type_confidence") or 0.0)
    decision = report.get("decision", "CLEAR_AUTHENTIC")
    basis = report.get("attack_type_basis", [])

    # Attack Type Badge
    att_col = GREEN if attack_guess == "NONE" else RED
    cv2.rectangle(panel, (18, 42), (230, 80), att_col, -1)
    cv2.putText(panel, f"ATTACK: {attack_guess}", (26, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # Confidence Bar
    cv2.putText(panel, f"Confidence: {attack_conf * 100:.1f}%", (250, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_WHITE, 1, cv2.LINE_AA)
    # Background bar
    cv2.rectangle(panel, (250, 62), (480, 76), (50, 50, 50), -1)
    # Filled bar
    fill_w = int(230 * attack_conf)
    cv2.rectangle(panel, (250, 62), (250 + fill_w, 76), att_col, -1)

    # Evidentiary Basis
    cv2.line(panel, (15, 96), (pw - 15, 96), BORDER_COLOR, 1)
    cv2.putText(panel, "HEURISTIC EVIDENTIARY BASIS:", (15, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_WHITE, 1, cv2.LINE_AA)

    y_pos = 142
    if not basis:
        basis = ["No tampering detected. Physical, typographic, and semantic properties match authentic baseline."]

    for b in basis[:6]:
        bullet_txt = f"- {b}"[:68]
        cv2.putText(panel, bullet_txt, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_WHITE, 1, cv2.LINE_AA)
        y_pos += 22

    # Final Recommendation
    cv2.line(panel, (15, 280), (pw - 15, 280), BORDER_COLOR, 1)
    rec_title = "SYSTEM ACTION RECOMMENDATION:"
    cv2.putText(panel, rec_title, (15, 302), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUTED, 1, cv2.LINE_AA)

    if decision == "CLEAR_AUTHENTIC":
        rec_str = "[APPROVE] Credential cleared. No physical or logical fraud detected."
        r_col = GREEN
    elif decision == "SUSPECT_TAMPERING":
        rec_str = "[REJECT / ESCALATE] Credential shows corroborated tampering evidence."
        r_col = RED
    elif decision == "INSUFFICIENT_EVIDENCE":
        rec_str = "[REQUEST RESCAN] Image quality insufficient to confirm authenticity."
        r_col = ORANGE
    else:
        rec_str = "[SECONDARY REVIEW] Borderline anomaly requires human examiner review."
        r_col = AMBER

    cv2.putText(panel, rec_str, (20, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.42, r_col, 1, cv2.LINE_AA)

    return panel


# ---------------------------------------------------------------------------
# Master Rendering Function
# ---------------------------------------------------------------------------

def render_unified_forensic_card(
    doc_bgr: np.ndarray,
    report: Dict[str, Any],
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Generate master 1280x860 4-panel diagnostic card and save to disk.
    """
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), BG_COLOR, dtype=np.uint8)

    doc_id = report.get("document_id", "unnamed_document")
    decision = report.get("decision", "CLEAR_AUTHENTIC")
    attack_type = report.get("attack_type_guess", "none")

    # Draw Header
    _draw_header(card, doc_id, decision, attack_type)

    # Render Panels
    p1 = _render_document_canvas_panel(doc_bgr, report)
    p2 = _render_evidence_crops_panel(doc_bgr, report)
    p3 = _render_suspicious_regions_panel(report)
    p4 = _render_executive_summary_panel(report)

    # Layout:
    # Row 1: Panel 1 (left) | Panel 2 (right)
    # Row 2: Panel 3 (left) | Panel 4 (right)
    x_left = PAD
    x_right = PAD + PANEL_W + PAD
    y_row1 = 48
    y_row2 = 48 + PANEL_H + PAD

    card[y_row1:y_row1 + PANEL_H, x_left:x_left + PANEL_W] = p1
    card[y_row1:y_row1 + PANEL_H, x_right:x_right + PANEL_W] = p2
    card[y_row2:y_row2 + PANEL_H, x_left:x_left + PANEL_W] = p3
    card[y_row2:y_row2 + PANEL_H, x_right:x_right + PANEL_W] = p4

    # Save to disk if requested
    saved_path = None
    if output_path:
        ensure_dirs(os.path.dirname(os.path.abspath(output_path)))
        cv2.imwrite(output_path, card)
        saved_path = output_path
    else:
        vis_dir = os.path.join(get_reports_dir(), "visuals", "unified")
        ensure_dirs(vis_dir)
        def_path = os.path.join(vis_dir, f"unified_card_{doc_id}.png")
        cv2.imwrite(def_path, card)
        saved_path = def_path

    return card, saved_path
