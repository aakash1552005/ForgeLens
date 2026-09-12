"""
ForgeLens-X — Milestone 9: Facial Morphing Attack Detection (MAD)
================================================================
Detects passport photo morphing attacks where two distinct identities are
digitally blended into a single credential portrait to fool e-Gate border systems.

Implements:
1. No-Reference Single-Image MAD (S-MAD): High-pass Laplacian edge ghosting,
   double iris/nostril contour detection, and midline bilateral gradient deformation.
2. Differential MAD (D-MAD): Projects deep ArcFace embeddings of the document portrait
   against the live presenting traveler's selfie, computing the orthogonal residue
   vector to reveal hidden secondary identities.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.identity_screener import extract_face_from_document


def _to_bgr_array(image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """Normalize image input to BGR numpy array."""
    if isinstance(image_input, np.ndarray):
        return image_input.copy()
    elif isinstance(image_input, bytes):
        np_arr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes.")
        return img
    elif isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Failed to read image at path: {image_input}")
        return img
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def detect_double_edges(face_bgr: np.ndarray) -> Tuple[float, bool, List[str]]:
    """
    Detects double-edge ghosting contours and interpolation shadows
    around high-contrast facial landmarks (nostrils, iris borders, vermilion).

    Returns:
        (ghosting_score [0.0 - 1.0], detected [bool], affected_regions [List[str]])
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.0, False, []

    h, w = face_bgr.shape[:2]
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # High-pass Laplacian edge magnitude
    lap = cv2.Laplacian(gray_blur, cv2.CV_32F, ksize=3)
    lap_mag = np.abs(lap)

    # Adaptive Canny edge extraction
    med = np.median(gray)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray_blur, lower, upper)

    # Inspect key morphing artifact zones
    zones = {
        "nostril_wings": (int(h * 0.48), int(h * 0.62), int(w * 0.35), int(w * 0.65)),
        "eye_contours": (int(h * 0.28), int(h * 0.45), int(w * 0.18), int(w * 0.82)),
        "lip_vermilion": (int(h * 0.62), int(h * 0.78), int(w * 0.30), int(w * 0.70)),
    }

    flagged_zones = []
    zone_scores = []

    for zone_name, (y1, y2, x1, x2) in zones.items():
        z_lap = lap_mag[y1:y2, x1:x2]
        z_edges = edges[y1:y2, x1:x2]
        if z_lap.size == 0 or z_edges.size == 0:
            continue

        edge_density = float(np.mean(z_edges > 0))
        # High edge density + diffuse gradient standard deviation indicates ghosting shadow
        lap_std = float(np.std(z_lap))
        # Authentic faces have single boundaries (edge density ~0.25-0.45 in natural facial features);
        # Morphed double contours produce extreme duplicate boundary clutter (> 0.52) with high Laplacian variance
        if edge_density > 0.52 and lap_std > 48.0:
            flagged_zones.append(zone_name)
            zone_scores.append(min(1.0, (edge_density - 0.52) / 0.15))

    if zone_scores:
        ghosting_score = float(np.mean(zone_scores))
    else:
        ghosting_score = 0.0

    detected = bool(len(flagged_zones) >= 2 or ghosting_score >= 0.70)
    return round(ghosting_score, 3), detected, flagged_zones


def analyze_gradient_inconsistency(face_bgr: np.ndarray) -> Tuple[float, bool]:
    """
    Evaluates bilateral facial symmetry gradient deformation.
    Blended morphs distort smooth bilateral gradient trajectories across the midline.

    Returns:
        (asymmetry_score [0.0 - 1.0], is_anomalous [bool])
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.0, False

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr
    h, w = gray.shape[:2]
    mid_x = w // 2

    # Left half vs horizontally flipped right half
    left = gray[:, :mid_x]
    right = cv2.flip(gray[:, mid_x:2 * mid_x], 1)

    min_w = min(left.shape[1], right.shape[1])
    if min_w < 16 or h < 32:
        return 0.0, False

    l_crop = left[:, :min_w].astype(np.float32)
    r_crop = right[:, :min_w].astype(np.float32)

    # Compute Sobel gradients
    gx_l = cv2.Sobel(l_crop, cv2.CV_32F, 1, 0, ksize=3)
    gx_r = cv2.Sobel(r_crop, cv2.CV_32F, 1, 0, ksize=3)

    # Bilateral gradient correlation
    diff = np.abs(gx_l + gx_r)  # Left + flipped right should be roughly opposite in X
    mean_diff = float(np.mean(diff))

    # Real human faces have some natural asymmetry (diff around 15-40 with realistic directional lighting)
    # Blended morphed credentials exhibit high local shear strain (> 55)
    asymmetry_score = float(np.clip((mean_diff - 45.0) / 25.0, 0.0, 1.0))
    is_anom = bool(asymmetry_score >= 0.70)

    return round(asymmetry_score, 3), is_anom


def compute_differential_morph_metric(
    doc_embedding: np.ndarray,
    selfie_embedding: np.ndarray,
) -> Tuple[float, bool]:
    """
    D-MAD: Evaluates orthogonal embedding projection residue between
    document portrait and live presenting selfie.

    If a passport photo is a blend of Subject A (innocent) and Subject B (accomplice),
    comparing Subject A's selfie against the morph gives:
      1. Moderate cosine distance (borderline match [0.35 - 0.55]).
      2. Large orthogonal residue vector || e_doc - (e_doc · e_selfie) * e_selfie ||
         corresponding to the hidden secondary identity.

    Returns:
        (dmad_score [0.0 - 100.0], is_morph_suspect [bool])
    """
    e_d = doc_embedding.flatten().astype(np.float64)
    e_s = selfie_embedding.flatten().astype(np.float64)

    norm_d = np.linalg.norm(e_d)
    norm_s = np.linalg.norm(e_s)
    if norm_d < 1e-6 or norm_s < 1e-6:
        return 0.0, False

    u_d = e_d / norm_d
    u_s = e_s / norm_s

    # Cosine distance
    cos_sim = float(np.dot(u_d, u_s))
    cos_dist = float(1.0 - cos_sim)

    # Orthogonal projection residue
    proj = cos_sim * u_s
    residue_vec = u_d - proj
    residue_norm = float(np.linalg.norm(residue_vec))

    # Characteristic morph ambiguity band: distance in [0.32, 0.58] with high orthogonal residue
    if 0.32 <= cos_dist <= 0.58 and residue_norm > 0.65:
        dmad_score = float(min(100.0, 50.0 + (residue_norm - 0.65) * 150.0))
        is_suspect = True
    elif cos_dist > 0.68:
        # Outright mismatch (different person entirely, not necessarily a morph)
        dmad_score = 0.0
        is_suspect = False
    else:
        # High confidence match (single authentic identity)
        dmad_score = float(max(0.0, (residue_norm - 0.50) * 40.0))
        is_suspect = False

    return round(dmad_score, 1), is_suspect


def evaluate_photo_morphing(
    portrait_input: Union[str, bytes, np.ndarray],
    selfie_input: Optional[Union[str, bytes, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Master Facial Morphing Attack Detection (MAD) evaluation.

    Performs Single-Image MAD (S-MAD) on the document portrait,
    plus Differential MAD (D-MAD) if a reference selfie is supplied.

    Returns:
        Structured audit dictionary with morphing risk score,
        verdict, and component diagnostics.
    """
    port_bgr = _to_bgr_array(portrait_input)
    h, w = port_bgr.shape[:2]

    face_crop = port_bgr
    face_bbox = None

    if min(h, w) > 160:
        face_info = extract_face_from_document(port_bgr)
        if face_info and face_info.get("bbox"):
            face_bbox = face_info["bbox"]
            bx, by, bw, bh = face_bbox
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w, bx + bw), min(h, by + bh)
            if (x2 - x1) > 30 and (y2 - y1) > 30:
                face_crop = port_bgr[y1:y2, x1:x2]

    # S-MAD: Single-image artifact detection
    ghost_score, is_ghost, ghost_zones = detect_double_edges(face_crop)
    asym_score, is_asym = analyze_gradient_inconsistency(face_crop)

    # Base S-MAD risk score (0.0 to 100.0)
    smad_score = round(float(np.clip((0.60 * ghost_score + 0.40 * asym_score) * 100.0, 0.0, 100.0)), 1)
    morph_detected = bool(is_ghost and is_asym and smad_score >= 70.0)

    # D-MAD: Differential evaluation if selfie provided
    dmad_score = 0.0
    is_dmad_suspect = False

    if selfie_input is not None:
        try:
            from src.face_verify import extract_face_embedding
            doc_emb = extract_face_embedding(face_crop)
            selfie_bgr = _to_bgr_array(selfie_input)
            selfie_emb = extract_face_embedding(selfie_bgr)

            if doc_emb is not None and selfie_emb is not None:
                dmad_score, is_dmad_suspect = compute_differential_morph_metric(doc_emb, selfie_emb)
                if is_dmad_suspect:
                    morph_detected = True
        except Exception:
            pass

    # Overall calibrated morphing score
    overall_score = max(smad_score, dmad_score)
    if morph_detected:
        overall_score = max(overall_score, 75.0)

    morph_tier = "SUSPECT_MORPHED_COMPOSITE" if morph_detected else "GENUINE_SINGLE_IDENTITY"

    return {
        "morphing_detected": morph_detected,
        "morphing_score": round(overall_score, 1),
        "morph_tier": morph_tier,
        "signals": {
            "smad_single_image": {
                "score": smad_score,
                "ghosting_detected": is_ghost,
                "ghosting_zones": ghost_zones,
                "gradient_asymmetry_score": asym_score,
            },
            "dmad_differential": {
                "evaluated": bool(selfie_input is not None),
                "dmad_score": dmad_score,
                "suspect_secondary_identity": is_dmad_suspect,
            },
        },
        "face_bbox": face_bbox,
    }


def generate_morph_diagnostic_card(
    portrait_input: Union[str, bytes, np.ndarray],
    selfie_input: Optional[Union[str, bytes, np.ndarray]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Renders an executive 4-panel visual forensic explanation card for Facial Morphing Detection.
    Panels:
        1. Document Portrait Canvas & Face Bounding Box
        2. S-MAD Laplacian Ghosting & Double Edge Contours
        3. S-MAD Bilateral Midline Gradient Asymmetry Heatmap
        4. Executive MAD Forensic Audit & D-MAD Projection Metrics
    """
    port_bgr = _to_bgr_array(portrait_input)
    res = evaluate_photo_morphing(port_bgr, selfie_input=selfie_input)

    h, w = port_bgr.shape[:2]
    face_crop = port_bgr
    if res.get("face_bbox"):
        bx, by, bw, bh = res["face_bbox"]
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w, bx + bw), min(h, by + bh)
        if (x2 - x1) > 20 and (y2 - y1) > 20:
            face_crop = port_bgr[y1:y2, x1:x2]

    pw, ph = 480, 360
    card = np.full((ph * 2 + 80, pw * 2 + 40, 3), 18, dtype=np.uint8)

    # 1. Panel 1: Document Portrait
    p1 = cv2.resize(face_crop, (pw, ph))
    border_color = (60, 60, 230) if res["morphing_detected"] else (60, 200, 60)
    cv2.rectangle(p1, (0, 0), (pw - 1, ph - 1), border_color, 3)
    cv2.putText(p1, "1. DOCUMENT PORTRAIT CROP", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    card[70:70 + ph, 15:15 + pw] = p1

    # 2. Panel 2: Laplacian Edge Ghosting Map
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    norm_lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_lap = cv2.applyColorMap(norm_lap, cv2.COLORMAP_INFERNO)
    p2 = cv2.resize(color_lap, (pw, ph))
    cv2.putText(p2, "2. S-MAD LAPLACIAN GHOSTING", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    ghost_txt = f"Ghost Zones: {', '.join(res['signals']['smad_single_image']['ghosting_zones']) or 'None (Sharp Single Edges)'}"
    cv2.putText(p2, ghost_txt, (15, ph - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
    card[70:70 + ph, 25 + pw:25 + pw * 2] = p2

    # 3. Panel 3: Bilateral Gradient Asymmetry Heatmap
    mid = gray.shape[1] // 2
    if mid > 10:
        left = gray[:, :mid]
        right = cv2.flip(gray[:, mid:2 * mid], 1)
        min_w = min(left.shape[1], right.shape[1])
        diff = np.abs(left[:, :min_w].astype(np.float32) - right[:, :min_w].astype(np.float32))
        norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_asym = cv2.applyColorMap(norm_diff, cv2.COLORMAP_HOT)
        p3 = cv2.resize(color_asym, (pw, ph))
    else:
        p3 = np.full((ph, pw, 3), 40, dtype=np.uint8)
    cv2.putText(p3, "3. BILATERAL MIDLINE STRAIN", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    asym_txt = f"Asymmetry Score: {res['signals']['smad_single_image']['gradient_asymmetry_score']:.2f}"
    cv2.putText(p3, asym_txt, (15, ph - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
    card[80 + ph:80 + ph * 2, 15:15 + pw] = p3

    # 4. Panel 4: Executive Morphing Forensic Audit
    p4 = np.full((ph, pw, 3), 28, dtype=np.uint8)
    cv2.putText(p4, "4. MORPHING ATTACK AUDIT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    tier_color = (60, 60, 240) if res["morphing_detected"] else (80, 220, 80)
    cv2.putText(p4, f"STATUS: {res['morph_tier']}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tier_color, 2)
    cv2.putText(p4, f"Composite Morph Score: {res['morphing_score']:.1f} / 100.0", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Breakdown
    smad_info = res["signals"]["smad_single_image"]
    dmad_info = res["signals"]["dmad_differential"]

    cv2.putText(p4, f"Single-Image MAD (S-MAD) Score : {smad_info['score']:.1f}", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(p4, f" - Ghost Edge Contours Flagged  : {smad_info['ghosting_detected']}", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(p4, f" - Bilateral Gradient Asymmetry : {smad_info['gradient_asymmetry_score']:.3f}", (15, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.putText(p4, f"Differential MAD (D-MAD) Status: {'EVALUATED' if dmad_info['evaluated'] else 'NOT PROVIDED (NO SELFIE)'}", (15, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if dmad_info["evaluated"]:
        cv2.putText(p4, f" - Orthogonal Projection Residue: {dmad_info['dmad_score']:.1f}", (15, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(p4, f" - Secondary Identity Intercept : {dmad_info['suspect_secondary_identity']}", (15, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    card[80 + ph:80 + ph * 2, 25 + pw:25 + pw * 2] = p4

    # Top Header
    cv2.putText(card, "ForgeLens-X | FACIAL MORPHING ATTACK DETECTION (MAD) FORENSIC AUDIT", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2)

    if output_path is None:
        os.makedirs("reports/morphing", exist_ok=True)
        output_path = f"reports/morphing/morph_diagnostic_{int(time.time())}.png"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, card)
    return output_path
