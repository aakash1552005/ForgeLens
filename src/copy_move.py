"""
ForgeLens-X — Copy-Move Forgery Detection
============================================
Uses classical ORB keypoint detection + self-matching + RANSAC
geometric verification to detect duplicated regions within an image.

IMPORTANT:
    Copy-move matching identifies two matching regions but CANNOT
    inherently determine which is the original and which is the
    pasted forgery. Both candidates are reported.
"""

import cv2
import numpy as np

from src.utils import bbox_from_mask


def detect_copy_move(
    image_path: str,
    n_features: int = 5000,
    match_threshold: int = 30,
    min_spatial_distance: float = 50.0,
    ransac_threshold: float = 5.0,
    min_inliers: int = 35,
    min_confidence: float = 0.15,
) -> dict:
    """
    Detect copy-move forgery using ORB + RANSAC.

    Algorithm:
        1. Detect ORB keypoints and compute descriptors
        2. Self-match descriptors (image against itself)
        3. Filter: remove identity matches, enforce spatial distance
        4. RANSAC geometric verification
        5. Cluster inlier keypoints → bounding boxes

    Args:
        image_path: path to image
        n_features: max ORB features to detect
        match_threshold: max Hamming distance for a valid match
        min_spatial_distance: min pixel distance between matched points
        ransac_threshold: RANSAC reprojection threshold
        min_inliers: minimum inlier matches to declare detection
        min_confidence: minimum inlier ratio (inliers / good_matches)

    Returns:
        {
            "detected": bool,
            "candidate_bbox": [x1,y1,x2,y2] or None,
            "alt_bbox": [x1,y1,x2,y2] or None,
            "num_matches": int,
            "num_inliers": int,
            "displacement": [dx, dy] or None,
            "confidence": float or None
        }
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return _empty_result("image_load_failed")

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None or len(keypoints) < 10:
        return _empty_result("insufficient_keypoints")

    # Self-match using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # knnMatch with k=2 for ratio test
    try:
        matches = bf.knnMatch(descriptors, descriptors, k=3)
    except cv2.error:
        return _empty_result("matching_failed")

    # Filter matches
    good_matches = []
    for match_group in matches:
        for m in match_group:
            # Skip identity match
            if m.queryIdx == m.trainIdx:
                continue

            # Hamming distance threshold
            if m.distance > match_threshold:
                continue

            # Spatial distance check — avoid matching nearby similar features
            pt1 = np.array(keypoints[m.queryIdx].pt)
            pt2 = np.array(keypoints[m.trainIdx].pt)
            spatial_dist = np.linalg.norm(pt1 - pt2)

            if spatial_dist < min_spatial_distance:
                continue

            good_matches.append(m)
            break  # take best non-identity match per query

    if len(good_matches) < min_inliers:
        return _empty_result("insufficient_matches", num_matches=len(good_matches))

    # Prepare points for RANSAC
    src_pts = np.float32(
        [keypoints[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [keypoints[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # RANSAC geometric verification
    try:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    except cv2.error:
        return _empty_result("ransac_failed", num_matches=len(good_matches))

    if mask is None:
        return _empty_result("ransac_no_mask", num_matches=len(good_matches))

    inlier_mask = mask.ravel().astype(bool)
    num_inliers = int(inlier_mask.sum())

    # Confidence: ratio of inliers to total good matches
    confidence = num_inliers / max(len(good_matches), 1)

    if num_inliers < min_inliers or confidence < min_confidence:
        return _empty_result(
            "insufficient_inliers_or_confidence",
            num_matches=len(good_matches),
            num_inliers=num_inliers,
        )

    # Extract inlier points
    inlier_src = src_pts[inlier_mask].reshape(-1, 2)
    inlier_dst = dst_pts[inlier_mask].reshape(-1, 2)

    # Compute bounding boxes
    candidate_bbox = _points_to_bbox(inlier_src)
    alt_bbox = _points_to_bbox(inlier_dst)

    # Compute displacement
    displacement = np.mean(inlier_dst - inlier_src, axis=0).tolist()

    return {
        "detected": True,
        "candidate_bbox": candidate_bbox,
        "alt_bbox": alt_bbox,
        "num_matches": len(good_matches),
        "num_inliers": num_inliers,
        "displacement": [round(d, 1) for d in displacement],
        "confidence": round(confidence, 3),
    }


def _points_to_bbox(points: np.ndarray) -> list:
    """Convert array of points to bounding box [x1, y1, x2, y2]."""
    x1 = int(np.min(points[:, 0]))
    y1 = int(np.min(points[:, 1]))
    x2 = int(np.max(points[:, 0])) + 1
    y2 = int(np.max(points[:, 1])) + 1
    return [x1, y1, x2, y2]


def _empty_result(
    reason: str = "",
    num_matches: int = 0,
    num_inliers: int = 0,
) -> dict:
    """Return a no-detection result."""
    return {
        "detected": False,
        "candidate_bbox": None,
        "alt_bbox": None,
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "displacement": None,
        "confidence": None,
        "reason": reason,
    }
