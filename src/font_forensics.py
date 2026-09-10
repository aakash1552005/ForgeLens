"""
ForgeLens-X — Milestone 4: Stroke Width Transform (SWT) & Typography Forensics
=============================================================================
Audits document typographic consistency across text fields using distance-transform
stroke width modeling and character bounding aspect ratios.

Detects spliced text, digital font insertion, or altered digits where an attacker
uses a font with different stroke weight or glyph aspect ratios than the genuine template.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml


def _load_m4_config() -> Dict[str, Any]:
    """Load default M4 configuration."""
    cfg_path = os.path.join(os.getcwd(), "configs", "m4_config.yaml")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def extract_field_stroke_metrics(
    crop_bgr: np.ndarray,
    min_area: int = 100,
) -> Dict[str, Any]:
    """
    Extract stroke width, glyph height, and aspect ratio metrics from a text field crop.

    Uses:
    1. Adaptive / Otsu binarization with polarity normalization (ensures text is foreground = 255).
    2. cv2.distanceTransform (L2 norm) to estimate half-stroke radius.
    3. Connected component analysis to calculate character height and width.
    """
    empty_result = {
        "stroke_mean": 0.0,
        "stroke_median": 0.0,
        "stroke_std": 0.0,
        "height_mean": 0.0,
        "height_std": 0.0,
        "aspect_ratio_mean": 0.0,
        "glyph_count": 0,
        "valid": False,
    }

    if crop_bgr is None or crop_bgr.size == 0:
        return empty_result

    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8 or (h * w) < min_area:
        return empty_result

    # Convert to grayscale
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr

    # Subtle gaussian blur to suppress high-frequency sensor noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Polarity check: text is typically darker than background on identity cards,
    # or lighter on dark themes. Determine background by inspecting image borders.
    border_pixels = np.concatenate([
        thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]
    ])
    bg_is_white = (np.mean(border_pixels) > 127)

    # We want text = 255 (foreground), background = 0
    binary = cv2.bitwise_not(thresh) if bg_is_white else thresh

    # Connected component analysis for glyph bounding boxes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    glyph_heights = []
    glyph_aspect_ratios = []
    glyph_masks = []

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        # Filter out noise specks or huge background blobs
        if 8 <= area <= (h * w * 0.7) and 4 <= ch <= (h * 0.95) and 2 <= cw <= (w * 0.9):
            glyph_heights.append(float(ch))
            glyph_aspect_ratios.append(float(cw / max(1.0, ch)))
            glyph_masks.append(labels == i)

    if not glyph_heights:
        return empty_result

    # Compute Euclidean distance transform on binary text
    # In distanceTransform, pixel value is distance to nearest zero (background)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Full stroke width is approximately 2 * distance along ridges / skeletons
    # We sample distance at pixels where distance > 0.5
    fg_dist = dist[binary > 0]
    if len(fg_dist) == 0:
        return empty_result

    # Peak distance in each column or local maxima approximates radius;
    # 2 * distance gives stroke width
    stroke_samples = fg_dist * 2.0

    return {
        "stroke_mean": float(np.mean(stroke_samples)),
        "stroke_median": float(np.median(stroke_samples)),
        "stroke_std": float(np.std(stroke_samples)),
        "height_mean": float(np.mean(glyph_heights)),
        "height_std": float(np.std(glyph_heights)),
        "aspect_ratio_mean": float(np.mean(glyph_aspect_ratios)),
        "glyph_count": len(glyph_heights),
        "valid": True,
    }


def audit_document_font_consistency(
    doc_bgr: np.ndarray,
    fields: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Audit cross-field typographic consistency across all available text fields.

    Calculates Z-scores for stroke width and aspect ratio across fields.
    Flags fields with |Z| > 2.5 as suspect spliced / inserted fonts.
    """
    if config is None:
        config = _load_m4_config()

    f_cfg = config.get("font_forensics", {})
    max_z = f_cfg.get("max_stroke_zscore", 2.5)
    max_aspect_z = f_cfg.get("max_aspect_ratio_zscore", 3.5)
    min_area = f_cfg.get("min_crop_area", 200)

    h_doc, w_doc = doc_bgr.shape[:2] if doc_bgr is not None else (0, 0)
    field_metrics: Dict[str, Any] = {}
    valid_strokes: List[float] = []
    valid_aspects: List[float] = []

    # Iterate over candidate text fields
    for field_name, f_data in fields.items():
        if field_name.startswith("_"):
            continue

        # Extract bbox if available
        bbox = None
        crop = None
        if isinstance(f_data, dict):
            bbox = f_data.get("bbox")
            crop = f_data.get("crop")

        # If crop not pre-extracted, crop from doc_bgr using bbox
        if crop is None and bbox and doc_bgr is not None and h_doc > 0 and w_doc > 0:
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_doc, x2), min(h_doc, y2)
                if x2 > x1 and y2 > y1:
                    crop = doc_bgr[y1:y2, x1:x2]

        if crop is not None and isinstance(crop, np.ndarray) and crop.size > 0:
            metrics = extract_field_stroke_metrics(crop, min_area=min_area)
            if metrics["valid"]:
                field_metrics[field_name] = metrics
                valid_strokes.append(metrics["stroke_mean"])
                valid_aspects.append(metrics["aspect_ratio_mean"])

    if len(valid_strokes) < 2:
        return {
            "typography_verdict": "INSUFFICIENT_TEXT_REGIONS",
            "is_consistent": True,
            "max_stroke_zscore": 0.0,
            "max_aspect_zscore": 0.0,
            "anomalous_fields": [],
            "field_metrics": field_metrics,
            "detail": f"Only {len(valid_strokes)} text regions available for cross-field font audit (minimum 2 required).",
        }

    # Document-level baseline statistics
    doc_stroke_mean = float(np.mean(valid_strokes))
    doc_stroke_std = float(np.std(valid_strokes))

    # Robust baseline estimation using median and MAD
    doc_stroke_median = float(np.median(valid_strokes))
    stroke_devs = [abs(s - doc_stroke_median) for s in valid_strokes]
    mad_stroke = float(np.median(stroke_devs)) * 1.4826
    eff_stroke_std = max(mad_stroke, 0.15)

    doc_aspect_median = float(np.median(valid_aspects))
    aspect_devs = [abs(a - doc_aspect_median) for a in valid_aspects]
    mad_aspect = float(np.median(aspect_devs)) * 1.4826
    eff_aspect_std = max(mad_aspect, 0.05)

    anomalous_fields = []
    max_observed_stroke_z = 0.0
    max_observed_aspect_z = 0.0

    for fname, m in field_metrics.items():
        stroke_z = abs(m["stroke_mean"] - doc_stroke_median) / eff_stroke_std
        aspect_z = abs(m["aspect_ratio_mean"] - doc_aspect_median) / eff_aspect_std
        m["stroke_zscore"] = round(float(stroke_z), 2)
        m["aspect_zscore"] = round(float(aspect_z), 2)

        max_observed_stroke_z = max(max_observed_stroke_z, stroke_z)
        max_observed_aspect_z = max(max_observed_aspect_z, aspect_z)

        if stroke_z > max_z:
            anomalous_fields.append({
                "field": fname,
                "stroke_zscore": round(float(stroke_z), 2),
                "aspect_zscore": round(float(aspect_z), 2),
                "stroke_mean": round(m["stroke_mean"], 2),
                "baseline_stroke": round(doc_stroke_mean, 2),
            })

    is_consistent = (len(anomalous_fields) == 0)
    verdict = "TYPOGRAPHY_CONSISTENT" if is_consistent else "SUSPECT_FONT_INCONSISTENCY"

    detail = (
        f"Typography across {len(valid_strokes)} fields is consistent (max Z={max_observed_stroke_z:.2f})."
        if is_consistent
        else f"Typographic outlier detected in {len(anomalous_fields)} field(s) with Z > {max_z}: {[a['field'] for a in anomalous_fields]}."
    )

    return {
        "typography_verdict": verdict,
        "is_consistent": is_consistent,
        "max_stroke_zscore": round(max_observed_stroke_z, 2),
        "max_aspect_zscore": round(max_observed_aspect_z, 2),
        "doc_stroke_mean": round(doc_stroke_mean, 2),
        "doc_stroke_std": round(doc_stroke_std, 2),
        "anomalous_fields": anomalous_fields,
        "field_metrics": field_metrics,
        "detail": detail,
    }
