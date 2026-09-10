"""
ForgeLens-X — Visual Forensic Explanation Generator
=====================================================
Renders multi-panel forensic explanation cards for human review.

Core objective:
    EXPLAIN WHY → HUMAN REVIEW

Each explanation card consists of:
    Panel 1: Input Document + Ground Truth Annotation
    Panel 2: Error Level Analysis (ELA) Heatmap + Anomaly Proposal
    Panel 3: Copy-Move Feature Matching & Inlier Vectors
    Panel 4: Forensic Evidence & Calibrated Risk Decision Card
"""

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.copy_move import detect_copy_move
from src.document_template import _get_bold_font, _get_font
from src.ela import compute_ela
from src.utils import ensure_dirs, get_reports_dir


def create_forensic_card(
    sample_data: dict,
    analysis_data: dict,
    output_path: str = None,
) -> Image.Image:
    """
    Render a 2x2 forensic explanation card for a document sample.

    Args:
        sample_data: dict with metadata (image_path, attack_type, bbox, etc.)
        analysis_data: dict with detector outputs (ela_detected, copy_move_detected, etc.)
        output_path: optional path to save the resulting image

    Returns:
        PIL.Image of the rendered explanation card (1240 x 860 px)
    """
    image_path = sample_data["image_path"]
    attack_type = sample_data.get("attack_type", "none")
    label = sample_data.get("label", "genuine")
    source_id = sample_data.get("source_id", "unknown")
    gt_bbox = sample_data.get("bbox") or sample_data.get("ground_truth_bbox")

    # Dimensions for sub-panels
    pw, ph = 580, 360

    # -----------------------------------------------------------------------
    # Panel 1: Original Document with GT & Detection overlays
    # -----------------------------------------------------------------------
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        p1 = np.zeros((ph, pw, 3), dtype=np.uint8)
    else:
        p1 = cv2.resize(orig_img, (pw, ph))

    scale_x = pw / 800.0
    scale_y = ph / 500.0

    # Draw Ground Truth bbox in green
    if gt_bbox:
        gx1 = int(gt_bbox[0] * scale_x)
        gy1 = int(gt_bbox[1] * scale_y)
        gx2 = int(gt_bbox[2] * scale_x)
        gy2 = int(gt_bbox[3] * scale_y)
        cv2.rectangle(p1, (gx1, gy1), (gx2, gy2), (0, 220, 0), 2)
        cv2.putText(
            p1, f"GT: {attack_type}", (gx1, max(18, gy1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA
        )

    # Draw ELA predicted bbox in cyan
    ela_bbox = analysis_data.get("ela_candidate_bbox")
    if ela_bbox:
        ex1 = int(ela_bbox[0] * scale_x)
        ey1 = int(ela_bbox[1] * scale_y)
        ex2 = int(ela_bbox[2] * scale_x)
        ey2 = int(ela_bbox[3] * scale_y)
        cv2.rectangle(p1, (ex1, ey1), (ex2, ey2), (255, 200, 0), 2)

    _add_panel_title(p1, "1. Document & Ground Truth Overlay")

    # -----------------------------------------------------------------------
    # Panel 2: ELA Difference Heatmap
    # -----------------------------------------------------------------------
    ela_heatmap = compute_ela(image_path, quality=90)
    # Normalize heatmap for display (clip top 99.5% for contrast)
    norm_ela = np.clip(ela_heatmap * 8.0, 0, 255).astype(np.uint8)
    p2_color = cv2.applyColorMap(norm_ela, cv2.COLORMAP_JET)
    p2 = cv2.resize(p2_color, (pw, ph))

    if ela_bbox:
        cv2.rectangle(p2, (ex1, ey1), (ex2, ey2), (255, 255, 255), 2)
        energy = analysis_data.get("ela_features", {}).get("candidate_energy", 0.0)
        cv2.putText(
            p2, f"Anomaly Energy: {energy:.1f}", (ex1, max(18, ey1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

    _add_panel_title(p2, "2. Error Level Analysis (ELA Heatmap)")

    # -----------------------------------------------------------------------
    # Panel 3: Copy-Move Feature Matches
    # -----------------------------------------------------------------------
    p3 = _render_copy_move_panel(image_path, pw, ph, scale_x, scale_y)
    _add_panel_title(p3, "3. Copy-Move Keypoint Correlation")

    # -----------------------------------------------------------------------
    # Panel 4: Forensic Verdict Summary Card (rendered with PIL)
    # -----------------------------------------------------------------------
    p4_pil = _render_verdict_card(pw, ph, source_id, attack_type, label, analysis_data)
    p4 = cv2.cvtColor(np.array(p4_pil), cv2.COLOR_RGB2BGR)
    _add_panel_title(p4, "4. Forensic Assessment & Calibrated Evidence")

    # -----------------------------------------------------------------------
    # Assemble 2x2 Canvas
    # -----------------------------------------------------------------------
    margin = 20
    header_h = 60
    total_w = pw * 2 + margin * 3
    total_h = ph * 2 + margin * 3 + header_h

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = (26, 28, 36)  # Dark slate background

    # Title header
    cv2.putText(
        canvas,
        f"FORGELENS-X FORENSIC SCREENING REPORT — SAMPLE: {source_id.upper()}",
        (margin, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y1_top = header_h + margin
    y1_bot = y1_top + ph
    y2_top = y1_bot + margin
    y2_bot = y2_top + ph

    x1_l = margin
    x1_r = x1_l + pw
    x2_l = x1_r + margin
    x2_r = x2_l + pw

    canvas[y1_top:y1_bot, x1_l:x1_r] = p1
    canvas[y1_top:y1_bot, x2_l:x2_r] = p2
    canvas[y2_top:y2_bot, x1_l:x1_r] = p3
    canvas[y2_top:y2_bot, x2_l:x2_r] = p4

    result_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    if output_path:
        ensure_dirs(os.path.dirname(output_path))
        result_image.save(output_path)

    return result_image


def _add_panel_title(panel: np.ndarray, title: str) -> None:
    """Add a semi-transparent title bar on top of a panel."""
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (panel.shape[1], 28), (15, 17, 23), -1)
    cv2.addWeighted(overlay, 0.75, panel, 0.25, 0, panel)
    cv2.putText(
        panel,
        title,
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def _render_copy_move_panel(
    image_path: str,
    pw: int,
    ph: int,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Detect and render copy-move matches."""
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return np.zeros((ph, pw, 3), dtype=np.uint8)

    panel = cv2.cvtColor(cv2.resize(img_gray, (pw, ph)), cv2.COLOR_GRAY2BGR)

    # Detect ORB
    orb = cv2.ORB_create(nfeatures=2000)
    kp, des = orb.detectAndCompute(img_gray, None)

    if des is not None and len(kp) >= 10:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        try:
            matches = bf.knnMatch(des, des, k=3)
            good_matches = []
            for match_group in matches:
                for m in match_group:
                    if m.queryIdx == m.trainIdx:
                        continue
                    if m.distance > 30:
                        continue
                    pt1 = np.array(kp[m.queryIdx].pt)
                    pt2 = np.array(kp[m.trainIdx].pt)
                    if np.linalg.norm(pt1 - pt2) < 50.0:
                        continue
                    good_matches.append(m)
                    break

            # Draw lines between matches
            for m in good_matches[:60]:  # Cap at 60 to prevent visual clutter
                pt1 = kp[m.queryIdx].pt
                pt2 = kp[m.trainIdx].pt
                p1_s = (int(pt1[0] * scale_x), int(pt1[1] * scale_y))
                p2_s = (int(pt2[0] * scale_x), int(pt2[1] * scale_y))
                cv2.circle(panel, p1_s, 3, (0, 255, 255), -1)
                cv2.circle(panel, p2_s, 3, (255, 0, 255), -1)
                cv2.line(panel, p1_s, p2_s, (0, 200, 255), 1, cv2.LINE_AA)

            count_str = f"Matches: {len(good_matches)}"
            cv2.putText(panel, count_str, (pw - 130, ph - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        except cv2.error:
            pass

    return panel


def _render_verdict_card(
    pw: int,
    ph: int,
    source_id: str,
    attack_type: str,
    label: str,
    analysis_data: dict,
) -> Image.Image:
    """Render crisp text card for forensic decision."""
    card = Image.new("RGB", (pw, ph), (18, 22, 32))
    draw = ImageDraw.Draw(card)

    font_title = _get_bold_font(15)
    font_body = _get_font(13)
    font_bold = _get_bold_font(13)

    ela_det = analysis_data.get("ela_detected", False)
    cm_det = analysis_data.get("copy_move_detected", False)
    is_tampered_detected = ela_det or cm_det

    y = 40
    line_h = 24

    # Sample info
    draw.text((20, y), f"Sample ID:", fill=(150, 160, 180), font=font_body)
    draw.text((160, y), f"{source_id}", fill=(240, 240, 240), font=font_bold)
    y += line_h

    draw.text((20, y), f"Ground Truth:", fill=(150, 160, 180), font=font_body)
    gt_color = (240, 80, 80) if label == "tampered" else (80, 220, 120)
    draw.text((160, y), f"{label.upper()} ({attack_type})", fill=gt_color, font=font_bold)
    y += line_h

    # Divider
    draw.line([(20, y), (pw - 20, y)], fill=(45, 50, 68), width=1)
    y += 10

    # ELA Section
    ela_status = "TAMPER DETECTED" if ela_det else "CLEAN"
    ela_col = (255, 180, 0) if ela_det else (80, 220, 120)
    draw.text((20, y), "ELA Forensic Signal:", fill=(150, 160, 180), font=font_body)
    draw.text((180, y), ela_status, fill=ela_col, font=font_bold)
    y += line_h

    ela_features = analysis_data.get("ela_features", {})
    energy = ela_features.get("candidate_energy", 0.0)
    draw.text((40, y), f"• Anomaly Energy: {energy:.1f}", fill=(180, 190, 210), font=font_body)
    y += line_h

    # Copy-Move Section
    cm_status = "CLONING DETECTED" if cm_det else "NO REPLICATION"
    cm_col = (255, 100, 100) if cm_det else (80, 220, 120)
    draw.text((20, y), "Copy-Move Signal:", fill=(150, 160, 180), font=font_body)
    draw.text((180, y), cm_status, fill=cm_col, font=font_bold)
    y += line_h

    inliers = analysis_data.get("copy_move_num_inliers", 0)
    conf = analysis_data.get("copy_move_confidence", 0.0) or 0.0
    draw.text((40, y), f"• Inliers: {inliers} (Ratio: {conf:.1%})", fill=(180, 190, 210), font=font_body)
    y += line_h

    # Divider
    draw.line([(20, y), (pw - 20, y)], fill=(45, 50, 68), width=1)
    y += 12

    # Overall Verdict
    draw.text((20, y), "Screening Verdict:", fill=(150, 160, 180), font=font_body)
    if is_tampered_detected:
        verdict = "FLAGGED FOR HUMAN REVIEW"
        verdict_color = (255, 70, 70)
    else:
        verdict = "LOW FORENSIC RISK (CLEAR)"
        verdict_color = (60, 220, 120)

    draw.text((180, y), verdict, fill=verdict_color, font=font_bold)
    y += line_h

    recommendation = (
        "Isolate flagged field for manual forensic audit."
        if is_tampered_detected
        else "No anomalous recompression or cloned motifs found."
    )
    draw.text((20, y), f"Action: {recommendation}", fill=(160, 170, 190), font=font_body)

    return card


def visualize_batch(
    samples: list,
    analysis_results: list,
    output_dir: str = None,
    max_samples: int = 5,
) -> list:
    """
    Generate explanation cards for representative samples across attack types.

    Args:
        samples: list of sample metadata dicts
        analysis_results: list of forensic analysis dicts
        output_dir: directory to save images (default: reports/visuals/)
        max_samples: max visual cards to generate

    Returns:
        List of saved image paths
    """
    if output_dir is None:
        output_dir = str(get_reports_dir() / "visuals")
    ensure_dirs(output_dir)

    # Index analysis by image_path
    analysis_by_path = {
        r["image_path"]: r for r in analysis_results if "image_path" in r
    }

    # Group by attack_type to ensure diverse coverage
    from collections import defaultdict
    by_attack = defaultdict(list)
    for s in samples:
        by_attack[s.get("attack_type", "none")].append(s)

    selected = []
    # Take at least 1 from each attack category
    for atk, items in by_attack.items():
        if items:
            selected.append(items[0])
            if len(selected) >= max_samples:
                break

    # If still below max_samples, add remaining
    if len(selected) < max_samples:
        for s in samples:
            if s not in selected:
                selected.append(s)
                if len(selected) >= max_samples:
                    break

    saved_paths = []
    for s in selected:
        img_path = s["image_path"]
        analysis = analysis_by_path.get(img_path, {})
        source_id = s.get("source_id", "sample")
        attack_type = s.get("attack_type", "none")

        card_path = os.path.join(output_dir, f"{source_id}_{attack_type}_forensic.png")
        create_forensic_card(s, analysis, output_path=card_path)
        saved_paths.append(card_path)

    return saved_paths
