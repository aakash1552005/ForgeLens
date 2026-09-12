"""
ForgeLens-X — Milestone 8: Perspective Homography & Document Rectification
==========================================================================
Detects document boundaries, extracts 4 corner coordinates, and computes
a 3x3 Homography perspective transformation matrix via RANSAC to normalize
tilted smartphone captures to a standardized flat rectangle.
"""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


def order_quad_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sorts 4 coordinate points into consistent order:
    [top-left, top-right, bottom-right, bottom-left].
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest x + y
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest x + y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest y - x
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest y - x
    return rect


def rectify_document_perspective(
    image: np.ndarray,
    target_width: int = 1200,
    target_height: int = 800,
    min_area_ratio: float = 0.25,
) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
    """
    Detects document boundary corners and applies perspective warping.

    Args:
        image: BGR document image.
        target_width: Normalized destination width in pixels.
        target_height: Normalized destination height in pixels.
        min_area_ratio: Minimum fraction of total image area to qualify as document.

    Returns:
        (rectified_bgr, was_rectified, corner_points)
    """
    if image is None or image.size == 0:
        return image, False, None

    h, w = image.shape[:2]
    total_area = h * w

    # Downscale for fast edge detection
    scale = 600.0 / max(h, w) if max(h, w) > 600 else 1.0
    small = cv2.resize(image, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Blur and morphological gradient edge enhancement
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 40, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_corners = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (total_area * (scale**2) * min_area_ratio):
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Found 4-point quadrilateral
            doc_corners = approx.reshape(4, 2) / scale
            break

    # If no distinct quad found (or already tightly cropped), return original
    if doc_corners is None:
        return image, False, None

    # Order corners
    ordered_pts = order_quad_corners(doc_corners)

    # Calculate natural width and height from corner distances
    tl, tr, br, bl = ordered_pts
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_w = max(int(width_a), int(width_b), target_width)

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_h = max(int(height_a), int(height_b), target_height)

    dst_pts = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1],
    ], dtype=np.float32)

    # Compute 3x3    # Apply perspective warp
    M = cv2.getPerspectiveTransform(ordered_pts.astype(np.float32), dst_pts)
    rectified = cv2.warpPerspective(image, M, (max_w, max_h), flags=cv2.INTER_LANCZOS4)
    return rectified, True, ordered_pts


def compute_quad_iou(
    quad_a: Union[np.ndarray, list],
    quad_b: Union[np.ndarray, list],
    image_shape: Tuple[int, int] = (800, 1200),
) -> float:
    """
    Computes polygon Intersection-over-Union (IoU) between two 4-point quadrilaterals.

    Args:
        quad_a: 4 corner points [[x0,y0], [x1,y1], [x2,y2], [x3,y3]].
        quad_b: 4 corner points [[x0,y0], [x1,y1], [x2,y2], [x3,y3]].
        image_shape: Height and width tuple of canvas for polygon rendering.

    Returns:
        IoU score in [0.0, 1.0].
    """
    if quad_a is None or quad_b is None:
        return 0.0
    h, w = image_shape[:2]
    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    try:
        pts_a = np.array(quad_a, dtype=np.int32).reshape((-1, 1, 2))
        pts_b = np.array(quad_b, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask_a, [pts_a], 255)
        cv2.fillPoly(mask_b, [pts_b], 255)
        inter = int(np.logical_and(mask_a > 0, mask_b > 0).sum())
        union = int(np.logical_or(mask_a > 0, mask_b > 0).sum())
        return round(float(inter / union), 4) if union > 0 else 0.0
    except Exception:
        return 0.0
