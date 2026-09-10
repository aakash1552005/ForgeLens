"""
ForgeLens-X — Biometric Face Quality Module Test Suite
========================================================
Tests ISO/IEC 19794-5 compliant biometric face quality assessment:
- Sharpness calculation (Laplacian variance)
- Inter-ocular distance (IOD)
- Illumination analysis (mean luminance & dynamic range clipping)
- Pose estimation (yaw symmetry, pitch ratio, roll angle)
- Overall quality score synthesis and tiered triage
- Zero-crash error handling for invalid/degraded inputs
"""

import os
import cv2
import numpy as np
import pytest

from src.face_quality import (
    assess_face_quality,
    calculate_illumination,
    calculate_iod,
    calculate_sharpness,
    estimate_pose,
)


@pytest.fixture(scope="session")
def sample_face_img():
    """Load or generate a sample face image for testing."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_faces")
    d1 = os.path.join(base_dir, "david1.jpg")
    if os.path.exists(d1):
        img = cv2.imread(d1)
        if img is not None:
            return img

    # Fallback synthetic portrait
    syn = np.full((200, 200, 3), 140, dtype=np.uint8)
    cv2.circle(syn, (100, 100), 60, (180, 160, 140), -1)
    return syn


class TestFaceQualityMetrics:
    """Test individual quality metrics."""

    def test_sharpness_sharp_vs_blurred(self):
        # Create a high-frequency checkerboard pattern
        sharp = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp[::10, :] = 255
        sharp[:, ::10] = 255

        blurred = cv2.GaussianBlur(sharp, (15, 15), 5.0)

        s_score = calculate_sharpness(sharp)
        b_score = calculate_sharpness(blurred)

        assert s_score > b_score
        assert b_score >= 0.0

    def test_sharpness_with_bbox(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add high frequency only inside [50, 50, 100, 100]
        img[60:140, 60:140] = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)

        # Full image has low sharpness, crop has high sharpness
        full_score = calculate_sharpness(img)
        bbox_score = calculate_sharpness(img, bbox=[50, 50, 100, 100])

        assert bbox_score > full_score

    def test_iod_from_landmarks(self):
        # 5-point landmarks: left_eye=(40, 50), right_eye=(100, 50) -> dist = 60 px
        landmarks = [
            [40.0, 50.0],
            [100.0, 50.0],
            [70.0, 80.0],
            [50.0, 110.0],
            [90.0, 110.0],
        ]
        iod = calculate_iod(landmarks=landmarks)
        assert abs(iod - 60.0) < 1e-4

    def test_iod_from_bbox_fallback(self):
        # Bbox width = 100 px -> fallback IOD = 100 * 0.40 = 40.0 px
        bbox = [20, 20, 100, 120]
        iod = calculate_iod(bbox=bbox)
        assert abs(iod - 40.0) < 1e-4

    def test_iod_none_fallback(self):
        iod = calculate_iod(landmarks=None, bbox=None)
        assert iod == 0.0

    def test_illumination_normal(self):
        # Balanced uniform mid-grey
        balanced = np.full((100, 100, 3), 128, dtype=np.uint8)
        res = calculate_illumination(balanced)

        assert 120 <= res["mean_luminance"] <= 135
        assert res["underexposed_ratio"] == 0.0
        assert res["overexposed_ratio"] == 0.0

    def test_illumination_overexposed(self):
        bright = np.full((100, 100, 3), 253, dtype=np.uint8)
        res = calculate_illumination(bright)

        assert res["mean_luminance"] > 240
        assert res["overexposed_ratio"] > 0.90
        assert res["underexposed_ratio"] == 0.0

    def test_illumination_underexposed(self):
        dark = np.full((100, 100, 3), 10, dtype=np.uint8)
        res = calculate_illumination(dark)

        assert res["mean_luminance"] < 25
        assert res["underexposed_ratio"] > 0.90
        assert res["overexposed_ratio"] == 0.0

    def test_pose_estimation_frontal(self):
        # Perfectly frontal face landmarks
        landmarks = [
            [40.0, 50.0],   # left eye
            [80.0, 50.0],   # right eye
            [60.0, 70.0],   # nose (equidistant 20px from both eyes)
            [45.0, 95.0],   # left mouth
            [75.0, 95.0],   # right mouth
        ]
        pose = estimate_pose(landmarks)

        assert abs(pose["yaw_symmetry"] - 1.0) < 0.05
        assert abs(pose["roll_deg"]) < 1.0
        assert pose["is_frontal"] is True

    def test_pose_estimation_turned_head(self):
        # Turned head: nose much closer to right eye than left eye
        landmarks = [
            [30.0, 50.0],   # left eye (d=45 to nose)
            [85.0, 50.0],   # right eye (d=10 to nose)
            [75.0, 70.0],   # nose
            [40.0, 95.0],   # left mouth
            [80.0, 95.0],   # right mouth
        ]
        pose = estimate_pose(landmarks)

        assert pose["yaw_symmetry"] < 0.55
        assert pose["is_frontal"] is False


class TestFaceQualityAssessment:
    """Test combined quality scoring and triage."""

    def test_assess_real_fixture(self, sample_face_img):
        res = assess_face_quality(sample_face_img)

        assert "overall_score" in res
        assert "quality_tier" in res
        assert "is_acceptable" in res
        assert 0.0 <= res["overall_score"] <= 100.0
        assert res["quality_tier"] in ["EXCELLENT", "ACCEPTABLE", "DEGRADED", "UNUSABLE"]
        assert isinstance(res["penalties"], list)
        assert isinstance(res["recommendations"], list)

    def test_severely_degraded_quality_triage(self):
        # Very blurry and tiny and dark
        bad_img = np.zeros((30, 30, 3), dtype=np.uint8)
        res = assess_face_quality(bad_img)

        assert res["overall_score"] < 40.0
        assert res["quality_tier"] in ["DEGRADED", "UNUSABLE"]
        assert res["is_acceptable"] is False
        assert len(res["penalties"]) > 0

    def test_zero_crash_on_empty_and_none(self):
        # Empty array
        empty_res = assess_face_quality(np.empty((0, 0, 3), dtype=np.uint8))
        assert empty_res["quality_tier"] == "UNUSABLE"
        assert empty_res["overall_score"] == 0.0

        # None image handled gracefully
        none_res = assess_face_quality(None)
        assert none_res["quality_tier"] == "UNUSABLE"
        assert none_res["overall_score"] == 0.0
