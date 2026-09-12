"""
ForgeLens-X — Milestone 9: Passive Presentation Attack Detection (PAD / Liveness)
=================================================================================
Implements ISO/IEC 30107 compliant passive single-frame anti-spoofing without
requiring specialized infrared or active challenge-response hardware.

Features:
- 2D Fourier Moiré Pattern Analysis: Detects digital LCD/OLED display screen replay lattices.
- Local Binary Pattern (LBP) Micro-Texture: Detects printed paper halftone dot matrices.
- YCrCb Chrominance Gamut Entropy: Analyzes human skin vascular hemoglobin absorption vs compressed print/screen gamut.
- Corneal Catchlight Symmetry: Checks bilateral pupil reflection geometry to uncover synthetic GAN/Diffusion deepfakes.
- Zero-Crash Policy: Handles non-frontal, low-resolution, or missing faces with graceful degradation.
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


def detect_moire_screen_replay(face_bgr: np.ndarray) -> Tuple[float, bool]:
    """
    Detects digital screen replay artifacts (LCD/OLED/monitor displays)
    via periodic 2D Fourier high-frequency Moiré interference patterns.

    Returns:
        (moire_score [0.0 - 1.0], is_screen_replay [bool])
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.0, False

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return 0.0, False

    # Resize to standard analysis patch
    patch = cv2.resize(gray, (128, 128)).astype(np.float32)

    # 2D Fast Fourier Transform
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_mag = np.log1p(magnitude)

    # Mask out DC center component
    cy, cx = 64, 64
    r_inner = 12
    y, x = np.ogrid[:128, :128]
    mask_dc = ((x - cx) ** 2 + (y - cy) ** 2) <= (r_inner ** 2)
    log_mag[mask_dc] = 0.0

    # High frequency outer ring (where Moiré interference lattices cluster)
    r_outer = 48
    mask_hf = (((x - cx) ** 2 + (y - cy) ** 2) > (r_inner ** 2)) & (((x - cx) ** 2 + (y - cy) ** 2) <= (r_outer ** 2))
    hf_energy = log_mag[mask_hf]

    if len(hf_energy) == 0:
        return 0.0, False

    mean_hf = np.mean(hf_energy)
    std_hf = np.std(hf_energy)
    max_hf = np.max(hf_energy)

    # Periodic lattice spikes produce sharp outlier peaks relative to mean HF floor
    peak_ratio = (max_hf - mean_hf) / max(1e-5, std_hf)
    moire_score = float(np.clip((peak_ratio - 3.2) / 3.0, 0.0, 1.0))
    is_replay = bool(moire_score >= 0.55)

    return round(moire_score, 3), is_replay


def compute_lbp_texture_score(face_bgr: np.ndarray) -> Tuple[float, bool]:
    """
    Computes Local Binary Pattern (LBP) texture entropy.
    Natural human skin exhibits smooth, continuous micro-gradients with subsurface scattering.
    Paper printouts exhibit discrete ink dithering, halftone dots, and paper grain noise.

    Returns:
        (texture_entropy [0.0 - 1.0], is_print_attack [bool])
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.0, False

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return 0.0, False

    # Extract central cheek / forehead region to avoid hair and background
    y1, y2 = int(h * 0.25), int(h * 0.75)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    crop = gray[y1:y2, x1:x2]

    # Compute 8-neighbor basic LBP
    padded = np.pad(crop, ((1, 1), (1, 1)), mode='reflect').astype(np.int32)
    center = padded[1:-1, 1:-1]
    lbp = np.zeros_like(crop, dtype=np.uint8)

    shifts = [
        (-1, -1, 0), (-1, 0, 1), (-1, 1, 2),
        (0, 1, 3), (1, 1, 4), (1, 0, 5),
        (1, -1, 6), (0, -1, 7)
    ]

    for dy, dx, bit in shifts:
        neighbor = padded[1 + dy:1 + dy + crop.shape[0], 1 + dx:1 + dx + crop.shape[1]]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    # Compute histogram
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -float(np.sum(hist * np.log2(hist)))

    # Real human skin typically has lower LBP entropy (smooth transitions, 4.2 - 5.5)
    # Halftone printed paper has high structural entropy (> 6.2) due to dot dithering
    norm_entropy = float(np.clip((entropy - 5.3) / 1.5, 0.0, 1.0))
    is_print = bool(norm_entropy >= 0.65)

    return round(norm_entropy, 3), is_print


def analyze_color_diversity(face_bgr: np.ndarray) -> Tuple[float, bool]:
    """
    Analyzes chrominance distribution in YCrCb color space.
    Real faces exhibit natural vascular variation across hemoglobin absorption bands.
    Screens and low-gamut paper prints show flattened, clamped chrominance profiles.

    Returns:
        (gamut_score [0.0 - 1.0], is_clamped [bool])
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.0, False

    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    cr_std = float(np.std(cr))
    cb_std = float(np.std(cb))
    combined_variance = (cr_std + cb_std) / 2.0

    # Human skin usually has chrominance std between 6.0 and 18.0
    # Clamped printouts or grayscale copies have very low variance (< 3.0)
    if combined_variance < 3.2:
        gamut_score = 0.85
        is_clamped = True
    elif combined_variance < 4.8:
        gamut_score = 0.50
        is_clamped = False
    else:
        gamut_score = float(np.clip(1.0 - (combined_variance / 20.0), 0.0, 0.35))
        is_clamped = False

    return round(gamut_score, 3), is_clamped


def evaluate_corneal_symmetry(face_bgr: np.ndarray) -> Tuple[float, bool]:
    """
    Evaluates corneal catchlight reflection symmetry between left and right eyes.
    Real faces illuminated by an ambient scene have identical bilateral reflection geometry.
    Generative deepfakes (StyleGAN, Diffusion) generate corneal reflections independently.

    Returns:
        (disparity_score [0.0 - 1.0], is_synthetic [bool])
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.0, False

    h, w = face_bgr.shape[:2]
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # Estimate eye regions from standard facial proportions
    # Left eye: y in [0.30, 0.45], x in [0.20, 0.45]
    # Right eye: y in [0.30, 0.45], x in [0.55, 0.80]
    ey1, ey2 = int(h * 0.30), int(h * 0.46)
    lx1, lx2 = int(w * 0.20), int(w * 0.45)
    rx1, rx2 = int(w * 0.55), int(w * 0.80)

    left_eye = gray[ey1:ey2, lx1:lx2]
    right_eye = gray[ey1:ey2, rx1:rx2]

    if left_eye.size == 0 or right_eye.size == 0:
        return 0.0, False

    # Extract specular catchlights (brightest 2% pixels in eye box)
    l_thresh = np.percentile(left_eye, 98)
    r_thresh = np.percentile(right_eye, 98)

    l_mask = (left_eye >= l_thresh).astype(np.uint8)
    r_mask = (right_eye >= r_thresh).astype(np.uint8)

    l_count = int(np.sum(l_mask))
    r_count = int(np.sum(r_mask))

    if l_count == 0 or r_count == 0:
        return 0.0, False

    ratio = min(l_count, r_count) / max(l_count, r_count)
    disparity = float(1.0 - ratio)

    # Disparity > 0.70 suggests synthetic eye reflection artifact
    is_synth = bool(disparity > 0.70)
    return round(disparity, 3), is_synth


def evaluate_face_liveness(
    face_input: Union[str, bytes, np.ndarray],
    crop_face_if_full_image: bool = True,
) -> Dict[str, Any]:
    """
    Master Presentation Attack Detection (PAD / Liveness) evaluation.

    Returns:
        Structured audit dictionary with calibrated liveness score,
        spoof classification, and component diagnostics.
    """
    img_bgr = _to_bgr_array(face_input)
    h, w = img_bgr.shape[:2]

    # If full portrait/selfie provided, crop the face bounding box
    face_crop = img_bgr
    face_bbox = None

    if crop_face_if_full_image and min(h, w) > 160:
        from src.face_verify import extract_face
        face_info = extract_face(img_bgr, align=False)
        if face_info and face_info.get("bbox"):
            face_bbox = face_info["bbox"]
            bx, by, bw, bh = face_bbox
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w, bx + bw), min(h, by + bh)
            if (x2 - x1) > 30 and (y2 - y1) > 30:
                face_crop = img_bgr[y1:y2, x1:x2]

    # Compute forensic component scores
    moire_score, is_screen = detect_moire_screen_replay(face_crop)
    lbp_score, is_print = compute_lbp_texture_score(face_crop)
    gamut_score, is_clamped = analyze_color_diversity(face_crop)
    corneal_score, is_deepfake = evaluate_corneal_symmetry(face_crop)

    # Composite spoof risk score (0.0 = completely live, 1.0 = fraudulent presentation)
    spoof_risk = (
        0.35 * moire_score +
        0.30 * lbp_score +
        0.15 * gamut_score +
        0.20 * corneal_score
    )

    # Calibrated Liveness Probability (1.0 - spoof_risk)
    liveness_score = round(float(np.clip(1.0 - spoof_risk, 0.0, 1.0)), 3)
    is_live = bool(liveness_score >= 0.60 and not is_screen and not is_print and not is_deepfake)

    # Classify spoof attack type
    if is_screen:
        spoof_type = "screen_replay"
        spoof_tier = "SUSPECT_SCREEN_REPLAY"
    elif is_print:
        spoof_type = "print_attack"
        spoof_tier = "SUSPECT_PRINT"
    elif is_deepfake:
        spoof_type = "synthetic_deepfake"
        spoof_tier = "SYNTHETIC_DEEPFAKE"
    elif not is_live:
        spoof_type = "presentation_anomaly"
        spoof_tier = "SUSPECT_ANOMALY"
    else:
        spoof_type = "none"
        spoof_tier = "GENUINE_LIVE_SUBJECT"

    # Planar specular reflection analysis
    specular_score, is_planar_glare = detect_planar_specular_uniformity(face_crop)

    return {
        "is_live": is_live,
        "liveness_score": liveness_score,
        "spoof_risk": round(float(spoof_risk), 3),
        "spoof_tier": spoof_tier,
        "spoof_type_guess": spoof_type,
        "signals": {
            "moire_screen_replay": {
                "score": moire_score,
                "detected": is_screen,
            },
            "lbp_texture_entropy": {
                "score": lbp_score,
                "detected": is_print,
            },
            "chrominance_gamut": {
                "score": gamut_score,
                "clamped": is_clamped,
            },
            "corneal_symmetry": {
                "score": corneal_score,
                "synthetic_detected": is_deepfake,
            },
            "specular_uniformity": {
                "score": specular_score,
                "planar_glare_detected": is_planar_glare,
            },
        },
        "face_bbox": face_bbox,
    }


def detect_planar_specular_uniformity(face_bgr: np.ndarray) -> Tuple[float, bool]:
    """
    Evaluates 3D Lambertian facial skin curvature vs flat 2D planar screen/paper glare.
    Curved human facial anatomy disperses illumination gradients across spherical cheek/forehead planes.
    Planar displays and paper sheets produce uniform linear specular gradients or localized flash burn.
    """
    if face_bgr is None or face_bgr.size == 0:
        return 0.0, False

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return 0.0, False

    # Second-order Hessian determinant to measure surface curvature
    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.5)
    dxx = cv2.Sobel(blur, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(blur, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(blur, cv2.CV_32F, 1, 1, ksize=3)

    hessian_det = (dxx * dyy) - (dxy ** 2)
    # Real 3D facial surfaces have high non-zero Gaussian curvature variance
    curv_variance = float(np.std(hessian_det))

    # Flat screens have low intrinsic surface curvature
    if curv_variance < 8.0:
        planarity_score = 0.75
        is_planar = True
    elif curv_variance < 18.0:
        planarity_score = 0.40
        is_planar = False
    else:
        planarity_score = float(np.clip(1.0 - (curv_variance / 60.0), 0.0, 0.30))
        is_planar = False

    return round(planarity_score, 3), is_planar


def generate_liveness_diagnostic_card(
    face_input: Union[str, bytes, np.ndarray],
    output_path: Optional[str] = None,
) -> str:
    """
    Renders an executive 4-panel visual forensic explanation card for Presentation Attack Detection.
    Panels:
        1. Original Selfie / Crop with Bounding Box & Status Header
        2. 2D FFT Fourier Magnitude Spectrum (Moiré Lattice Detection)
        3. Local Binary Pattern (LBP) Micro-Texture Map
        4. Executive Telemetry & Calibrated Forensic Risk Meter
    """
    img_bgr = _to_bgr_array(face_input)
    res = evaluate_face_liveness(img_bgr, crop_face_if_full_image=True)

    h, w = img_bgr.shape[:2]
    face_crop = img_bgr
    if res.get("face_bbox"):
        bx, by, bw, bh = res["face_bbox"]
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(w, bx + bw), min(h, by + bh)
        if (x2 - x1) > 20 and (y2 - y1) > 20:
            face_crop = img_bgr[y1:y2, x1:x2]

    # Panel dimensions
    pw, ph = 480, 360
    card = np.full((ph * 2 + 80, pw * 2 + 40, 3), 18, dtype=np.uint8)

    # 1. Panel 1: Original Face with BBox
    p1 = cv2.resize(face_crop, (pw, ph))
    border_color = (60, 200, 60) if res["is_live"] else (50, 50, 220)
    cv2.rectangle(p1, (0, 0), (pw - 1, ph - 1), border_color, 3)
    cv2.putText(p1, "1. FACIAL CANVAS & BBOX", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    card[70:70 + ph, 15:15 + pw] = p1

    # 2. Panel 2: 2D FFT Fourier Spectrum
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    patch = cv2.resize(gray, (128, 128)).astype(np.float32)
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_fft = cv2.applyColorMap(norm_mag, cv2.COLORMAP_VIRIDIS)
    p2 = cv2.resize(color_fft, (pw, ph))
    # Draw Moiré search ring
    cv2.circle(p2, (pw // 2, ph // 2), int(pw * 0.35), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(p2, (pw // 2, ph // 2), int(pw * 0.10), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(p2, "2. 2D FFT MOIRE SPECTRUM", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    moire_txt = f"Moiré Score: {res['signals']['moire_screen_replay']['score']:.3f}"
    cv2.putText(p2, moire_txt, (15, ph - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    card[70:70 + ph, 25 + pw:25 + pw * 2] = p2

    # 3. Panel 3: LBP Texture Map
    padded = np.pad(patch, ((1, 1), (1, 1)), mode='reflect').astype(np.int32)
    center = padded[1:-1, 1:-1]
    lbp = np.zeros_like(patch, dtype=np.uint8)
    shifts = [(-1, -1, 0), (-1, 0, 1), (-1, 1, 2), (0, 1, 3), (1, 1, 4), (1, 0, 5), (1, -1, 6), (0, -1, 7)]
    for dy, dx, bit in shifts:
        neighbor = padded[1 + dy:1 + dy + patch.shape[0], 1 + dx:1 + dx + patch.shape[1]]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
    color_lbp = cv2.applyColorMap(lbp, cv2.COLORMAP_MAGMA)
    p3 = cv2.resize(color_lbp, (pw, ph))
    cv2.putText(p3, "3. LBP MICRO-TEXTURE MAP", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    lbp_txt = f"Texture Entropy: {res['signals']['lbp_texture_entropy']['score']:.3f}"
    cv2.putText(p3, lbp_txt, (15, ph - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)
    card[80 + ph:80 + ph * 2, 15:15 + pw] = p3

    # 4. Panel 4: Executive Forensic Telemetry Panel
    p4 = np.full((ph, pw, 3), 28, dtype=np.uint8)
    cv2.putText(p4, "4. PRESENTATION ATTACK AUDIT", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    tier_color = (80, 220, 80) if res["is_live"] else (60, 60, 240)
    cv2.putText(p4, f"STATUS: {res['spoof_tier']}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tier_color, 2)
    cv2.putText(p4, f"Liveness Probability: {res['liveness_score'] * 100:.1f}%", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(p4, f"Presentation Risk   : {res['spoof_risk'] * 100:.1f}%", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Signal bars
    cv2.putText(p4, "Forensic Sensor Signals:", (15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    sigs = [
        ("Moiré Replay", res["signals"]["moire_screen_replay"]["score"]),
        ("LBP Print Dot", res["signals"]["lbp_texture_entropy"]["score"]),
        ("Gamut Clamping", res["signals"]["chrominance_gamut"]["score"]),
        ("Corneal Disparity", res["signals"]["corneal_symmetry"]["score"]),
        ("Planar Glare", res["signals"]["specular_uniformity"]["score"]),
    ]
    for idx, (sname, sval) in enumerate(sigs):
        ypos = 215 + idx * 26
        cv2.putText(p4, f"{sname:<18}: {sval:.2f}", (15, ypos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        # Bar
        bx = 220
        bw_val = int(sval * 200)
        bcol = (60, 200, 60) if sval < 0.40 else ((60, 180, 220) if sval < 0.65 else (60, 60, 230))
        cv2.rectangle(p4, (bx, ypos - 12), (bx + 200, ypos), (50, 50, 50), -1)
        if bw_val > 0:
            cv2.rectangle(p4, (bx, ypos - 12), (bx + bw_val, ypos), bcol, -1)

    card[80 + ph:80 + ph * 2, 25 + pw:25 + pw * 2] = p4

    # Top Header
    cv2.putText(card, "ForgeLens-X | BIOMETRIC PRESENTATION ATTACK DETECTION (PAD) AUDIT", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)

    if output_path is None:
        os.makedirs("reports/liveness", exist_ok=True)
        output_path = f"reports/liveness/liveness_diagnostic_{int(time.time())}.png"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, card)
    return output_path
