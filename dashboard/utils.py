"""
ForgeLens-X — Dashboard Utilities & Caching Infrastructure
==========================================================
Resource caching (@st.cache_resource), data caching (@st.cache_data),
vector canvas overlays, ELA heatmap blending, deep crop extraction,
and plain-language forensic synthesis.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from src.copy_move import detect_copy_move
from src.ela import analyze_ela, compute_baseline, compute_ela
from src.forensic_report import (
    generate_unified_forensic_report,
    get_default_baseline,
)
from src.risk_fusion import load_fusion_model
from src.unified_visualize import render_unified_forensic_card
from src.utils import get_reports_dir


# ---------------------------------------------------------------------------
# 1. Resource Caching (@st.cache_resource)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initializing forensic baseline...")
def get_cached_baseline() -> Optional[Dict[str, Any]]:
    """Retrieve and cache the precomputed ELA genuine baseline."""
    return get_default_baseline()


@st.cache_resource(show_spinner="Loading risk calibration model...")
def get_cached_fusion_model() -> Optional[Dict[str, Any]]:
    """Load and cache the trained Milestone 6 Risk Fusion model bundle."""
    try:
        return load_fusion_model()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. Pipeline Execution Caching (@st.cache_data)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Executing unified multi-modal forensic screening...", ttl=3600)
def run_screening_pipeline(
    image_path: str,
    reference_face_path: Optional[str] = None,
    file_mtime: float = 0.0,
) -> Dict[str, Any]:
    """Execute the full ForgeLens-X unified forensic pipeline and cache report."""
    report = generate_unified_forensic_report(
        image_path=image_path,
        reference_face_path=reference_face_path,
        baseline=None,
    )
    return report


@st.cache_data(show_spinner="Rendering master diagnostic card...", ttl=3600)
def get_cached_diagnostic_card(
    image_path: str,
    reference_face_path: Optional[str] = None,
    file_mtime: float = 0.0,
) -> np.ndarray:
    """Render and cache the 1280x870 4-panel screening card in RGB format."""
    report = run_screening_pipeline(image_path, reference_face_path, file_mtime=file_mtime)
    doc_bgr = cv2.imread(image_path)
    if doc_bgr is None:
        return np.zeros((870, 1280, 3), dtype=np.uint8)
    card_bgr, _ = render_unified_forensic_card(doc_bgr, report)
    return cv2.cvtColor(card_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# 2B. Multi-Layer Memory Cache & SVG Icon Infrastructure
# ---------------------------------------------------------------------------

_LAYER_CACHE: Dict[Tuple[str, float, Optional[str]], Dict[str, np.ndarray]] = {}


def get_cached_view_layers(
    image_path: str,
    report: Dict[str, Any],
    reference_face_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Pre-renders and caches all 4 visual evidence layers in memory.
    Enables instant (< 10ms) zero-latency layer switching in the UI.
    """
    try:
        mtime = os.path.getmtime(image_path)
    except Exception:
        mtime = 0.0

    cache_key = (image_path, mtime, reference_face_path)
    if cache_key in _LAYER_CACHE:
        return _LAYER_CACHE[cache_key]

    doc_bgr = cv2.imread(image_path)
    if doc_bgr is None:
        dummy = np.zeros((400, 600, 3), dtype=np.uint8)
        return {"original": dummy, "overlay": dummy, "heatmap": dummy, "card": dummy}

    # 1. Clean original RGB
    orig_rgb = cv2.cvtColor(doc_bgr, cv2.COLOR_BGR2RGB)

    # 2. Dynamic bounding box & tamper overlay
    overlay_rgb = render_dynamic_overlay(
        doc_bgr=doc_bgr,
        report=report,
        show_ocr=True,
        show_ela=True,
        show_copy_move=True,
        show_suspicious=True,
        ela_opacity=0.0,
    )

    # 3. ELA compression thermal heatmap
    heat_bgr = generate_ela_heatmap_bgr(image_path, report=report)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    # 4. Master 4-panel diagnostic card
    try:
        card_rgb = get_cached_diagnostic_card(image_path, reference_face_path, file_mtime=mtime)
    except Exception:
        card_rgb = orig_rgb

    layers = {
        "original": orig_rgb,
        "overlay": overlay_rgb,
        "heatmap": heat_rgb,
        "card": card_rgb,
    }

    _LAYER_CACHE[cache_key] = layers
    return layers


def get_svg_icon(name: str, color: str = "currentColor", size: int = 16) -> str:
    """Return crisp inline SVG vector icons (21st.dev / Lucide style)."""
    icons = {
        "shield": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
        "shield_check": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
        "alert_triangle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "check": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
        "layers": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "fingerprint": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12C2 6.5 6.5 2 12 2a10 10 0 0 1 8 4"/><path d="M5 19.5C5.5 18 6 15 6 12c0-.7.12-1.37.34-2"/><path d="M17.29 21.02c.12-.6.18-1.23.18-1.87 0-3-1-4-2-6"/><path d="M12 10a2 2 0 0 0-2 2c0 1.02-.1 2.51-.26 4"/><path d="M8.65 22c.21-.66.45-1.32.75-2"/><path d="M14 13.1a14 14 0 0 0 .5 4.9"/></svg>',
        "download": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
        "file_text": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>',
        "eye": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>',
        "cpu": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 14h3"/><path d="M1 9h3"/><path d="M1 14h3"/></svg>',
        "crosshair": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>',
    }
    return icons.get(name, f'<span style="font-size: {size}px;">•</span>')


# ---------------------------------------------------------------------------
# 3. Dynamic Vector Overlay Renderer
# ---------------------------------------------------------------------------

def render_dynamic_overlay(
    doc_bgr: np.ndarray,
    report: Dict[str, Any],
    show_ocr: bool = True,
    show_ela: bool = True,
    show_copy_move: bool = True,
    show_suspicious: bool = True,
    ela_opacity: float = 0.0,
    ela_heatmap_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Renders an interactive multi-layer forensic canvas with real-time toggles.
    Colors:
      - OCR Fields: Electric Cyan (#38bdf8 / (248, 189, 56) BGR)
      - ELA Anomaly: Solar Amber (#f59e0b / (11, 158, 245) BGR)
      - Copy-Move: Cyber Violet (#8b5cf6 / (246, 92, 139) BGR)
      - Suspicious Regions: Crimson Alert (#ef4444 / (68, 68, 239) BGR)
    """
    canvas = doc_bgr.copy()
    h, w = canvas.shape[:2]

    # Layer 0: Alpha-blend ELA heatmap over canvas
    if ela_opacity > 0.0 and ela_heatmap_bgr is not None:
        resized_heat = cv2.resize(ela_heatmap_bgr, (w, h))
        canvas = cv2.addWeighted(canvas, 1.0 - ela_opacity, resized_heat, ela_opacity, 0.0)

    # Layer 1: OCR Structured Field Boxes
    if show_ocr:
        fields = report.get("fields", {})
        for fname, fval in fields.items():
            if isinstance(fval, dict) and "bbox" in fval:
                box = fval["bbox"]
                if box and len(box) == 4:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (248, 189, 56), 1, cv2.LINE_AA)
                    # Field tag badge
                    tag_text = fname[:12]
                    cv2.rectangle(canvas, (x1, max(0, y1 - 14)), (x1 + len(tag_text) * 7 + 6, max(0, y1)), (30, 22, 18), -1)
                    cv2.putText(canvas, tag_text, (x1 + 3, max(10, y1 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (248, 189, 56), 1, cv2.LINE_AA)

    # Layer 2: ELA Candidate Splicing Box
    if show_ela:
        tamper = report.get("tamper_signals", {})
        ela_data = tamper.get("ela", {})
        ela_box = ela_data.get("candidate_bbox")
        if ela_box and len(ela_box) == 4:
            x1, y1, x2, y2 = [int(v) for v in ela_box]
            # Draw dashed/bright amber rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (11, 158, 245), 2, cv2.LINE_AA)
            energy = ela_data.get("anomaly_energy", 0.0)
            lbl = f"ELA ANOMALY ({energy:.0f})"
            cv2.rectangle(canvas, (x1, max(0, y1 - 18)), (x1 + len(lbl) * 8 + 8, max(0, y1)), (11, 158, 245), -1)
            cv2.putText(canvas, lbl, (x1 + 4, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

    # Layer 3: Copy-Move Verified Keypoint Correspondence Links
    if show_copy_move:
        cm_data = report.get("tamper_signals", {}).get("copy_move", {})
        if cm_data.get("detected") and cm_data.get("num_matches", 0) > 0:
            clusters = cm_data.get("cluster_boxes", [])
            for cbox in clusters:
                if len(cbox) == 4:
                    cx1, cy1, cx2, cy2 = [int(v) for v in cbox]
                    cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (246, 92, 139), 2, cv2.LINE_AA)
            if len(clusters) >= 2:
                # Draw connecting arrow between centers
                c1_mid = ((clusters[0][0] + clusters[0][2]) // 2, (clusters[0][1] + clusters[0][3]) // 2)
                c2_mid = ((clusters[1][0] + clusters[1][2]) // 2, (clusters[1][1] + clusters[1][3]) // 2)
                cv2.arrowedLine(canvas, c1_mid, c2_mid, (246, 92, 139), 2, tipLength=0.08)
                cv2.putText(canvas, f"CLONED MOTIF ({cm_data.get('num_matches')} pts)", (min(c1_mid[0], c2_mid[0]), max(20, min(c1_mid[1], c2_mid[1]) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (246, 92, 139), 1, cv2.LINE_AA)

    # Layer 4: Suspicious Regions (High-Priority Alert Badges)
    if show_suspicious:
        susp_regions = report.get("suspicious_regions", [])
        for i, sreg in enumerate(susp_regions):
            box = sreg.get("bbox")
            if box and len(box) == 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                # High-contrast double border
                cv2.rectangle(canvas, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (68, 68, 239), 2, cv2.LINE_AA)

                # Glowing tag badge
                source_tag = sreg.get("source", "anom").upper()
                field_tag = sreg.get("field", "")
                conf = sreg.get("confidence", 0.0)
                tag_label = f"[{source_tag}: {field_tag} {conf*100:.0f}%]" if field_tag else f"[{source_tag} {conf*100:.0f}%]"

                # Badge background
                badge_w = len(tag_label) * 8 + 12
                cv2.rectangle(canvas, (x1, min(h - 5, y2)), (x1 + badge_w, min(h, y2 + 18)), (68, 68, 239), -1)
                cv2.putText(canvas, tag_label, (x1 + 4, min(h - 4, y2 + 13)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# 4. Heatmap & Feature View Generators
# ---------------------------------------------------------------------------

def generate_ela_heatmap_bgr(image_path: str, quality: int = 90, report: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Generate normalized ELA error difference map with COLORMAP_JET."""
    baseline = get_cached_baseline()
    ela_res = analyze_ela(image_path, baseline=baseline, quality=quality)
    diff = ela_res.get("ela_heatmap")
    if diff is None:
        try:
            diff = compute_ela(image_path, quality=quality)
        except Exception:
            diff = None

    if diff is None:
        orig = cv2.imread(image_path)
        return orig if orig is not None else np.zeros((400, 600, 3), dtype=np.uint8)

    # Normalize heatmap for contrast (scale by 8.0 for clear thermal visualization) and apply JET colormap
    norm_ela = np.clip(diff * 8.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_ela, cv2.COLORMAP_JET)
    return heatmap


def generate_copy_move_matches_bgr(image_path: str) -> np.ndarray:
    """Generate copy-move keypoint match visualization canvas."""
    cm_res = detect_copy_move(image_path)
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((400, 600, 3), dtype=np.uint8)

    canvas = img.copy()
    if cm_res.get("detected") and "matched_keypoints" in cm_res:
        matches = cm_res["matched_keypoints"]
        for p1, p2 in matches:
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.circle(canvas, pt1, 3, (246, 92, 139), -1)
            cv2.circle(canvas, pt2, 3, (246, 92, 139), -1)
            cv2.line(canvas, pt1, pt2, (200, 70, 180), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# 5. Deep Zoom Crop Inspector
# ---------------------------------------------------------------------------

def extract_region_crops(
    doc_bgr: np.ndarray,
    ela_heatmap_bgr: np.ndarray,
    bbox: List[int],
    padding: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts zoomed sub-regions for deep inspection:
      Returns (original_crop_rgb, ela_crop_rgb)
    """
    h, w = doc_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Add padding and clamp
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(w, x2 + padding)
    py2 = min(h, y2 + padding)

    orig_crop = doc_bgr[py1:py2, px1:px2]
    if ela_heatmap_bgr.shape[:2] != (h, w):
        heat_resized = cv2.resize(ela_heatmap_bgr, (w, h))
    else:
        heat_resized = ela_heatmap_bgr
    heat_crop = heat_resized[py1:py2, px1:px2]

    # Draw bounding box inside the crop for focus
    box_crop_orig = orig_crop.copy()
    cv2.rectangle(box_crop_orig, (x1 - px1, y1 - py1), (x2 - px1, y2 - py1), (68, 68, 239), 2, cv2.LINE_AA)

    return (
        cv2.cvtColor(box_crop_orig, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(heat_crop, cv2.COLOR_BGR2RGB),
    )


# ---------------------------------------------------------------------------
# 6. Plain-Language Examiner Explanations
# ---------------------------------------------------------------------------

def format_plain_language_explanations(report: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Synthesizes natural-language explanations tailored for document examiners.
    Translates raw statistics into human-understandable findings.
    """
    explanations: List[Dict[str, str]] = []

    # 1. Physical Splicing & ELA
    tamper = report.get("tamper_signals", {})
    ela_data = tamper.get("ela", {})
    if ela_data.get("detected"):
        ov_field = ela_data.get("overlapping_field")
        energy = ela_data.get("anomaly_energy", 0.0)
        if ov_field:
            explanations.append({
                "category": "Compression Discontinuity",
                "severity": "HIGH",
                "text": f"Localized JPEG recompression anomaly identified directly overlapping the '{ov_field}' field (residual anomaly energy {energy:.1f}). This indicates the text was pasted or altered after document generation.",
            })
        else:
            explanations.append({
                "category": "Compression Discontinuity",
                "severity": "MODERATE",
                "text": f"Unusual JPEG compression residue detected across an unlabelled image region (energy {energy:.1f}).",
            })

    # 2. Copy-Move Cloning
    cm_data = tamper.get("copy_move", {})
    if cm_data.get("detected"):
        matches = cm_data.get("num_matches", 0)
        explanations.append({
            "category": "Duplicated Graphic Motif",
            "severity": "HIGH",
            "text": f"Duplicated visual motif identified across distinct document coordinates with {matches} verified ORB keypoint correspondences, characteristic of digital stamp or seal cloning.",
        })

    # 3. Biometric Face Verification
    face_res = report.get("face_verification", {})
    if face_res.get("has_face_check"):
        if face_res.get("verified"):
            explanations.append({
                "category": "Biometric Identity",
                "severity": "CLEAR",
                "text": f"Biometric face verification confirmed identity match between the document portrait and presented live selfie ({face_res.get('similarity_pct', 0.0):.1f}% similarity).",
            })
        elif face_res.get("verified") is False and face_res.get("distance") is not None:
            explanations.append({
                "category": "Biometric Identity",
                "severity": "CRITICAL",
                "text": f"Biometric mismatch: The person presenting the document does not match the portrait embedded on the credential (cosine distance {face_res['distance']:.3f} exceeds threshold {face_res.get('threshold', 0.40):.3f}). Potential imposter or photo swap.",
            })

    # 4. Semantic Rules & Chronology
    semantic_data = report.get("semantic_checks", [])
    if isinstance(semantic_data, list):
        for check_item in semantic_data:
            if isinstance(check_item, dict) and check_item.get("status") == "FAIL":
                cname = check_item.get("check", "Semantic Rule")
                detail = check_item.get("detail", "Logical inconsistency")
                explanations.append({
                    "category": "Document Logic Conflict",
                    "severity": "HIGH",
                    "text": f"Logical inconsistency flagged ({cname.replace('_', ' ').title()}): {detail}.",
                })
    elif isinstance(semantic_data, dict):
        failed_checks = semantic_data.get("failed_checks", [])
        checks_dict = semantic_data.get("checks", {})
        for fcheck in failed_checks:
            detail = checks_dict.get(fcheck, {}).get("detail", "Logical inconsistency")
            explanations.append({
                "category": "Document Logic Conflict",
                "severity": "HIGH",
                "text": f"Logical inconsistency flagged ({fcheck.replace('_', ' ').title()}): {detail}.",
            })

    # 5. Typography Font Forensics
    font_res = report.get("font_forensics", {})
    if font_res.get("is_anomalous"):
        anom_fields = [a.get("field") for a in font_res.get("anomalous_fields", [])]
        explanations.append({
            "category": "Typography Splicing",
            "severity": "MODERATE",
            "text": f"Font stroke-width and aspect ratio inconsistency detected in field(s) {', '.join(anom_fields)}. Typography characteristics deviate noticeably from genuine template baseline.",
        })

    # 6. Metadata / Provenance
    meta_res = report.get("metadata_forensics", {})
    if meta_res.get("is_tampered"):
        sw = meta_res.get("software") or meta_res.get("software_detected") or "Image Editing Tool"
        explanations.append({
            "category": "Digital Provenance",
            "severity": "HIGH",
            "text": f"Digital file provenance audit revealed software manipulation signatures ('{sw}'). The file was edited using graphic software prior to screening.",
        })

    # 7. Quality & Clean State Fallback
    if not explanations:
        quality = report.get("quality", {})
        explanations.append({
            "category": "Forensic Cleanliness",
            "severity": "CLEAR",
            "text": f"No physical compression anomalies, clone vectors, typography outliers, or semantic conflicts detected. Document scan quality is {quality.get('analysis_reliability', 'HIGH')} (sharpness: {quality.get('blur_score', 0):.0f}).",
        })

    return explanations


# ---------------------------------------------------------------------------
# 7. Audit Certificate Export
# ---------------------------------------------------------------------------

def generate_audit_certificate_dict(
    report: Dict[str, Any],
    examiner_notes: str = "",
    examiner_id: str = "EXAMINER-409",
    action_taken: str = "REVIEW_PENDING",
) -> Dict[str, Any]:
    """Generate a formal signed audit certificate dict for record keeping."""
    cert = {
        "certificate_type": "FORGELENS_X_EXAMINER_AUDIT_CERTIFICATE",
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        "examiner_id": examiner_id,
        "action_taken": action_taken,
        "examiner_notes": examiner_notes.strip(),
        "document_id": report.get("document_id"),
        "calibrated_risk_score": report.get("risk_score"),
        "fraud_probability": report.get("fraud_probability"),
        "operational_decision": report.get("decision"),
        "risk_tier": report.get("risk_tier"),
        "attack_hypothesis": report.get("attack_type_guess"),
        "top_risk_drivers": report.get("risk_drivers", []),
        "scan_quality": report.get("quality"),
        "full_forensic_report": report,
    }
    return cert
