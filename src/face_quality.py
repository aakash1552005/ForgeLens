"""
ForgeLens-X — Milestone 2: Biometric Face Quality Assessment Module
====================================================================
Implements ISO/IEC 19794-5 and ICAO 9303 biometric pre-screening metrics:
    - Sharpness / Blur estimation (Laplacian variance)
    - Geometric resolution & Inter-Ocular Distance (IOD)
    - Illumination uniformity & dynamic range clipping
    - Head pose estimation from 5 facial landmarks (yaw, pitch, roll)
    - Overall biometric quality scoring & tiered triage
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Biometric Standard Thresholds (ISO/IEC 19794-5)
# ---------------------------------------------------------------------------

MIN_RECOMMENDED_IOD = 40.0   # Pixels between eye centers for reliable recognition
MIN_ACCEPTABLE_IOD = 28.0    # Pixels below which features degrade significantly
SHARPNESS_BLUR_THRESHOLD = 50.0  # Laplacian variance below which image is blurry
SHARPNESS_CRISP_THRESHOLD = 150.0
MIN_MEAN_LUMINANCE = 60.0
MAX_MEAN_LUMINANCE = 200.0
MAX_CLIPPING_RATIO = 0.12    # Max fraction of over/underexposed pixels


# ---------------------------------------------------------------------------
# Metric Functions
# ---------------------------------------------------------------------------

def calculate_sharpness(
    image_bgr: np.ndarray,
    bbox: Optional[List[int]] = None,
) -> float:
    """
    Calculate focus / edge sharpness using variance of Laplacian.
    Higher values indicate sharp edges; values < 50 indicate blur.
    Supports optional bounding box crop [x, y, w, h].
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0

    if bbox is not None and len(bbox) >= 4:
        bx, by, bw, bh = bbox
        ih, iw = image_bgr.shape[:2]
        x1 = max(0, min(iw - 1, int(bx)))
        y1 = max(0, min(ih - 1, int(by)))
        x2 = max(x1 + 1, min(iw, int(bx + bw)))
        y2 = max(y1 + 1, min(ih, int(by + bh)))
        image_bgr = image_bgr[y1:y2, x1:x2]
        if image_bgr.size == 0:
            return 0.0

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return round(variance, 2)


def calculate_iod(
    landmarks: Optional[List[List[float]]] = None,
    bbox: Optional[List[int]] = None,
) -> float:
    """
    Calculate Inter-Ocular Distance (IOD) in pixels.
    If 5 landmarks are provided: Euclidean distance between right and left eyes.
    If only bbox is provided: standard biometric approximation (approx 0.40 * bbox_width).
    """
    if landmarks is not None and len(landmarks) >= 2:
        r_eye = landmarks[0]
        l_eye = landmarks[1]
        dx = float(l_eye[0] - r_eye[0])
        dy = float(l_eye[1] - r_eye[1])
        return round(float(math.sqrt(dx * dx + dy * dy)), 1)

    if bbox is not None and len(bbox) >= 4:
        bw = float(bbox[2])
        return round(bw * 0.40, 1)

    return 0.0


def calculate_illumination(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Calculate illumination quality, mean luminance, and dynamic range clipping.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {
            "mean_luminance": 0.0,
            "contrast_std": 0.0,
            "overexposed_ratio": 0.0,
            "underexposed_ratio": 0.0,
            "illumination_status": "UNUSABLE",
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))

    total_pixels = float(gray.size)
    overexposed = float(np.sum(gray >= 245)) / total_pixels
    underexposed = float(np.sum(gray <= 15)) / total_pixels

    if overexposed > MAX_CLIPPING_RATIO:
        status = "OVEREXPOSED"
    elif underexposed > MAX_CLIPPING_RATIO:
        status = "UNDEREXPOSED"
    elif mean_lum < MIN_MEAN_LUMINANCE:
        status = "LOW_LIGHT"
    elif mean_lum > MAX_MEAN_LUMINANCE:
        status = "HARSH_LIGHT"
    else:
        status = "BALANCED"

    return {
        "mean_luminance": round(mean_lum, 1),
        "contrast_std": round(std_lum, 1),
        "overexposed_ratio": round(overexposed, 4),
        "underexposed_ratio": round(underexposed, 4),
        "illumination_status": status,
    }


def estimate_pose(landmarks: Optional[List[List[float]]]) -> Dict[str, Any]:
    """
    Estimate head pose angles (roll, yaw symmetry ratio, pitch offset) from 5 landmarks.
    Landmarks format: [right_eye, left_eye, nose, right_mouth, left_mouth].
    """
    if landmarks is None or len(landmarks) < 5:
        return {
            "roll_degrees": 0.0,
            "yaw_ratio": 1.0,
            "pitch_ratio": 1.0,
            "pose_status": "UNKNOWN",
        }

    re = np.array(landmarks[0], dtype=np.float64)
    le = np.array(landmarks[1], dtype=np.float64)
    nose = np.array(landmarks[2], dtype=np.float64)
    rm = np.array(landmarks[3], dtype=np.float64)
    lm = np.array(landmarks[4], dtype=np.float64)

    # Roll: angle between eye line and horizontal
    dx = le[0] - re[0]
    dy = le[1] - re[1]
    roll_deg = math.degrees(math.atan2(dy, dx))

    # Yaw: ratio of horizontal distance from nose to left eye vs nose to right eye
    d_nose_r = float(np.linalg.norm(nose - re))
    d_nose_l = float(np.linalg.norm(nose - le))
    yaw_ratio = d_nose_l / max(d_nose_r, 1e-3)

    # Pitch: vertical distance between eye midpoint and nose vs nose and mouth midpoint
    eye_mid = (re + le) / 2.0
    mouth_mid = (rm + lm) / 2.0
    d_eye_nose = abs(nose[1] - eye_mid[1])
    d_nose_mouth = abs(mouth_mid[1] - nose[1])
    pitch_ratio = d_eye_nose / max(d_nose_mouth, 1e-3)

    # Determine pose status
    flags = []
    if abs(roll_deg) > 15.0:
        flags.append("HEAD_TILT")
    if yaw_ratio < 0.60 or yaw_ratio > 1.67:
        flags.append("PROFILE_YAW")
    if pitch_ratio < 0.40 or pitch_ratio > 2.20:
        flags.append("PITCH_DEVIATION")

    pose_status = "FRONTAL" if not flags else "+".join(flags)

    return {
        "roll_degrees": round(float(roll_deg), 1),
        "roll_deg": round(float(roll_deg), 1),
        "yaw_ratio": round(float(yaw_ratio), 2),
        "yaw_symmetry": round(float(yaw_ratio), 2),
        "pitch_ratio": round(float(pitch_ratio), 2),
        "pose_status": pose_status,
        "is_frontal": (pose_status == "FRONTAL"),
    }


# ---------------------------------------------------------------------------
# Master Biometric Quality Assessment
# ---------------------------------------------------------------------------

def assess_face_quality(
    image_bgr: np.ndarray,
    landmarks: Optional[List[List[float]]] = None,
    bbox: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Assess overall biometric facial presentation quality across sharpness,
    resolution (IOD), illumination, and pose.

    Returns:
        Structured dictionary with quality metrics, flags, overall score (0-100),
        and quality tier ('EXCELLENT', 'ACCEPTABLE', 'DEGRADED', 'UNUSABLE').
    """
    if image_bgr is None or image_bgr.size == 0:
        return {
            "quality_score": 0.0,
            "overall_score": 0.0,
            "quality_tier": "UNUSABLE",
            "is_acceptable": False,
            "flags": ["EMPTY_IMAGE"],
            "penalties": ["EMPTY_IMAGE"],
            "recommendations": ["Provide a valid non-empty facial image."],
            "sharpness": 0.0,
            "iod": 0.0,
            "illumination": {"illumination_status": "UNUSABLE"},
            "pose": {"pose_status": "UNKNOWN"},
        }

    sharpness = calculate_sharpness(image_bgr)
    iod = calculate_iod(landmarks=landmarks, bbox=bbox)
    illum = calculate_illumination(image_bgr)
    pose = estimate_pose(landmarks=landmarks)

    flags = []

    # 1. Sharpness Scoring (30 pts max)
    if sharpness < 20.0:
        sharpness_pts = 5.0
        flags.append("SEVERE_BLUR")
    elif sharpness < SHARPNESS_BLUR_THRESHOLD:
        sharpness_pts = 15.0
        flags.append("MILD_BLUR")
    elif sharpness >= SHARPNESS_CRISP_THRESHOLD:
        sharpness_pts = 30.0
    else:
        sharpness_pts = 20.0 + 10.0 * ((sharpness - SHARPNESS_BLUR_THRESHOLD) / (SHARPNESS_CRISP_THRESHOLD - SHARPNESS_BLUR_THRESHOLD))

    # 2. IOD Resolution Scoring (35 pts max)
    if iod < 15.0:
        iod_pts = 0.0
        flags.append("EXTREME_LOW_RESOLUTION")
    elif iod < MIN_ACCEPTABLE_IOD:
        iod_pts = 12.0
        flags.append("LOW_RESOLUTION")
    elif iod >= MIN_RECOMMENDED_IOD:
        iod_pts = 35.0
    else:
        iod_pts = 20.0 + 15.0 * ((iod - MIN_ACCEPTABLE_IOD) / (MIN_RECOMMENDED_IOD - MIN_ACCEPTABLE_IOD))

    # 3. Illumination Scoring (20 pts max)
    lum_status = illum["illumination_status"]
    if lum_status == "BALANCED":
        illum_pts = 20.0
    elif lum_status in ["LOW_LIGHT", "HARSH_LIGHT"]:
        illum_pts = 12.0
        flags.append(lum_status)
    else:
        illum_pts = 6.0
        flags.append(lum_status)

    # 4. Pose Scoring (15 pts max)
    pose_status = pose["pose_status"]
    if pose_status == "FRONTAL":
        pose_pts = 15.0
    else:
        pose_pts = 8.0
        flags.append(pose_status)

    quality_score = float(round(sharpness_pts + iod_pts + illum_pts + pose_pts, 1))

    # Determine Quality Tier
    if quality_score >= 75.0 and not any(f in flags for f in ["SEVERE_BLUR", "EXTREME_LOW_RESOLUTION"]):
        quality_tier = "EXCELLENT"
    elif quality_score >= 50.0:
        quality_tier = "ACCEPTABLE"
    elif quality_score >= 30.0:
        quality_tier = "DEGRADED"
    else:
        quality_tier = "UNUSABLE"

    recommendations = []
    if "SEVERE_BLUR" in flags or "MILD_BLUR" in flags:
        recommendations.append("Ensure camera focus is locked and eliminate motion blur.")
    if "EXTREME_LOW_RESOLUTION" in flags or "LOW_RESOLUTION" in flags:
        recommendations.append("Position face closer to the camera to increase pixel density.")
    if "OVEREXPOSED" in flags or "HARSH_LIGHT" in flags:
        recommendations.append("Reduce frontal glare and diffuse direct illumination.")
    if "UNDEREXPOSED" in flags or "LOW_LIGHT" in flags:
        recommendations.append("Increase ambient lighting on the subject's face.")
    if "PROFILE_YAW" in flags or "HEAD_TILT" in flags:
        recommendations.append("Instruct subject to look directly into the camera lens with head straight.")

    return {
        "quality_score": quality_score,
        "overall_score": quality_score,
        "quality_tier": quality_tier,
        "is_acceptable": quality_tier in ["EXCELLENT", "ACCEPTABLE"],
        "flags": flags if flags else ["OPTIMAL_PRESENTATION"],
        "penalties": flags,
        "recommendations": recommendations,
        "sharpness": sharpness,
        "iod": iod,
        "illumination": illum,
        "pose": pose,
    }
