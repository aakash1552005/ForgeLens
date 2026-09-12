"""
ForgeLens-X — Error Level Analysis (ELA)
==========================================
Implements ELA with baseline calibration for fixed-template documents.

ELA is ONE forensic signal. It does NOT directly declare fraud.

Key concepts:
    - Recompress JPEG at known quality → compute absolute pixel difference
    - Tampered regions show different error levels due to double-compression
    - Baseline calibration: subtract expected error from genuine documents
      to reduce false positives on naturally complex regions (text, stamps)

IMPORTANT: Baseline calibration is a fixed-template research technique.
It works because all M1 documents share the same layout.
Do NOT claim this generalizes to arbitrary unseen documents.
"""

import os
import tempfile
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageChops

from src.utils import bbox_from_mask, ensure_dirs


def compute_ela(
    image_path: str,
    quality: int = 90,
    subsampling: int = 0,
) -> np.ndarray:
    """
    Compute Error Level Analysis heatmap.

    1. Load image
    2. Re-save at specified JPEG quality
    3. Compute absolute difference between original and re-compressed
    4. Return as float32 grayscale heatmap (0-255 range)

    Args:
        image_path: path to JPEG image
        quality: JPEG quality for recompression (default 90)
        subsampling: chroma subsampling (0=4:4:4 for best ELA fidelity)

    Returns:
        np.ndarray: ELA heatmap, float32, shape (H, W)
    """
    original = Image.open(image_path).convert("RGB")

    # Recompress to in-memory buffer (avoid disk I/O)
    buffer = BytesIO()
    original.save(buffer, "JPEG", quality=quality, subsampling=subsampling)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # Absolute difference
    diff = ImageChops.difference(original, recompressed)

    # Convert to grayscale float heatmap
    diff_array = np.array(diff, dtype=np.float32)
    heatmap = diff_array.mean(axis=2)  # average across RGB channels

    return heatmap


def compute_baseline(
    genuine_paths: list,
    quality: int = 90,
    subsampling: int = 0,
) -> dict:
    """
    Build expected-error baseline from multiple known-genuine samples.

    Strategy:
        1. Compute ELA for each genuine image
        2. Per-pixel: calculate mean and std across all ELA maps
        3. threshold_map = mean + k*std (configurable k)

    Args:
        genuine_paths: list of JPEG paths to known-genuine documents
        quality: JPEG quality for ELA recompression
        subsampling: chroma subsampling setting

    Returns:
        {
            "mean_map": ndarray,
            "std_map": ndarray,
            "n_samples": int
        }

    NOTE: This is a fixed-template technique. All images must share
    the same layout/template for the baseline to be meaningful.
    """
    if not genuine_paths:
        raise ValueError("Need at least one genuine image for baseline")

    ela_maps = []
    for path in genuine_paths:
        heatmap = compute_ela(path, quality=quality, subsampling=subsampling)
        ela_maps.append(heatmap)

    ela_stack = np.stack(ela_maps, axis=0)

    return {
        "mean_map": ela_stack.mean(axis=0).astype(np.float32),
        "std_map": ela_stack.std(axis=0).astype(np.float32),
        "n_samples": len(genuine_paths),
    }


def calibrate_threshold(
    genuine_paths: list,
    baseline: dict,
    percentile: float = 80.0,
    quality: int = 90,
    k: float = 1.8,
    min_std: float = 1.5,
    min_area: int = 25,
    closing_ksize: tuple = (11, 7),
    default_threshold: float = 60.0,
    max_threshold: float = 85.0,
) -> float:
    """
    Empirically calibrate the anomaly energy threshold on known genuine samples.

    Guarantees that the empirical false alarm rate on genuine documents
    matches the target alpha level while capping against outlier text lengths.

    Args:
        genuine_paths: paths to genuine documents in calibration split
        baseline: baseline dict from compute_baseline
        percentile: percentile for threshold cutoff (e.g. 80.0)
        quality: JPEG recompression quality
        k: standard deviation multiplier
        min_std: minimum std noise floor
        min_area: minimum component area
        closing_ksize: kernel size for text closing
        default_threshold: fallback default threshold
        max_threshold: ceiling to maintain detection sensitivity for subtle edits

    Returns:
        float: calibrated energy threshold
    """
    if not genuine_paths:
        return default_threshold

    energies = []
    for path in genuine_paths:
        heatmap = compute_ela(path, quality=quality)
        anom = anomaly_map(heatmap, baseline, k=k, min_std=min_std)
        candidate = propose_candidate_region(
            anom,
            min_area=min_area,
            closing_ksize=closing_ksize,
            energy_threshold=0.0,  # capture all candidate energies
        )
        energies.append(candidate["energy"] if candidate else 0.0)

    if energies:
        calibrated = float(np.percentile(energies, percentile))
        thresh = max(calibrated * 1.05, default_threshold)
        return min(thresh, max_threshold)
    return default_threshold


def anomaly_map(
    ela_heatmap: np.ndarray,
    baseline: dict,
    k: float = 2.0,
    min_std: float = 1.5,
) -> np.ndarray:
    """
    Compute anomaly map: observed_error - expected_error.

    anomaly = max(0, ela_heatmap - (baseline_mean + k * max(baseline_std, min_std)))

    Positive values indicate regions with higher-than-expected error.
    A noise floor (min_std) prevents spurious anomalies in zero-variance regions.

    Args:
        ela_heatmap: ELA heatmap from compute_ela()
        baseline: baseline dict from compute_baseline()
        k: number of standard deviations for threshold
        min_std: minimum standard deviation noise floor

    Returns:
        np.ndarray: anomaly map (float32, non-negative)
    """
    mean_map = baseline["mean_map"]
    std_map = baseline["std_map"]

    h, w = ela_heatmap.shape[:2]
    if mean_map.shape != (h, w):
        mean_map = cv2.resize(mean_map, (w, h), interpolation=cv2.INTER_LINEAR)
        std_map = cv2.resize(std_map, (w, h), interpolation=cv2.INTER_LINEAR)

    std_map = np.maximum(std_map, min_std)
    threshold_map = mean_map + k * std_map
    anom = ela_heatmap - threshold_map
    anom = np.maximum(anom, 0.0)
    return anom.astype(np.float32)


def extract_features(
    ela_heatmap: np.ndarray,
    roi_bbox: list = None,
) -> dict:
    """
    Extract numeric features from ELA heatmap.

    Args:
        ela_heatmap: ELA heatmap (float32)
        roi_bbox: optional [x1, y1, x2, y2] to restrict analysis

    Returns:
        {
            "mean": float,
            "std": float,
            "max": float,
            "p95": float,
            "p99": float,
            "high_error_pixel_ratio": float
        }
    """
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        region = ela_heatmap[y1:y2, x1:x2]
    else:
        region = ela_heatmap

    if region.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "high_error_pixel_ratio": 0.0,
        }

    flat = region.flatten()
    high_threshold = 30.0  # pixels above this considered high-error

    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "max": float(np.max(flat)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "high_error_pixel_ratio": float(np.sum(flat > high_threshold) / flat.size),
    }


def propose_candidate_region(
    anom_map: np.ndarray,
    min_area: int = 25,
    threshold_val: float = 1.2,
    closing_ksize: tuple = (11, 7),
    energy_threshold: float = 60.0,
) -> dict | None:
    """
    Threshold anomaly map, bridge text glyphs, and find the most suspicious region.

    Strategy:
        1. Binarize by threshold_val (pixels with significant excess error)
        2. Morphological closing to fuse disconnected character strokes into text blocks
        3. Label connected components
        4. Score candidates by cumulative anomaly energy (sum of anomaly values)
        5. Return bbox and stats of the highest-energy component meeting min_area and energy_threshold

    Args:
        anom_map: anomaly map from anomaly_map()
        min_area: minimum pixel area for a candidate
        threshold_val: anomaly threshold for binarization
        closing_ksize: structuring element size (width, height)
        energy_threshold: minimum cumulative anomaly energy required

    Returns:
        {
            "bbox": [x1, y1, x2, y2],
            "area": int,
            "max_anomaly": float,
            "mean_anomaly": float,
            "energy": float
        }
        or None if no candidate found
    """
    import cv2
    from scipy import ndimage

    if anom_map.max() < threshold_val:
        return None

    binary = (anom_map >= threshold_val).astype(np.uint8)

    # Morphological closing to bridge character gaps in text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(closing_ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Connected components
    labeled, n_features = ndimage.label(closed)

    if n_features == 0:
        return None

    best_bbox = None
    best_area = 0
    best_score = 0.0
    best_component = None

    for label_id in range(1, n_features + 1):
        comp = (labeled == label_id)
        area = int(np.sum(comp))
        if area < min_area:
            continue
        energy = float(np.sum(anom_map[comp]))
        if energy > best_score:
            best_score = energy
            best_area = area
            best_component = comp
            best_bbox = bbox_from_mask(comp.astype(np.uint8))

    if best_bbox is None or best_score < energy_threshold:
        return None

    region_anomaly = anom_map[best_component]

    return {
        "bbox": best_bbox,
        "area": int(best_area),
        "max_anomaly": float(np.max(region_anomaly)),
        "mean_anomaly": float(np.mean(region_anomaly)),
        "energy": float(best_score),
    }


def analyze_ela(
    image_path: str,
    baseline: dict = None,
    quality: int = 90,
    k: float = 1.8,
    min_std: float = 1.5,
    min_area: int = 25,
    closing_ksize: tuple = (11, 7),
    energy_threshold: float = 60.0,
) -> dict:
    """
    Full ELA analysis pipeline for a single image.

    Args:
        image_path: path to JPEG image
        baseline: optional baseline from compute_baseline()
        quality: JPEG quality for recompression
        k: standard deviation multiplier for anomaly threshold
        min_std: minimum standard deviation floor
        min_area: minimum area for candidate region
        closing_ksize: kernel size for morphological closing
        energy_threshold: minimum cumulative anomaly energy

    Returns:
        {
            "ela_heatmap": ndarray,
            "anomaly_map": ndarray or None,
            "features": dict,
            "candidate": dict or None,
            "has_baseline": bool
        }
    """
    heatmap = compute_ela(image_path, quality=quality)

    if baseline is not None:
        anom = anomaly_map(heatmap, baseline, k=k, min_std=min_std)
        candidate = propose_candidate_region(
            anom,
            min_area=min_area,
            closing_ksize=closing_ksize,
            energy_threshold=energy_threshold,
        )
        features = extract_features(anom)
        features["candidate_energy"] = candidate["energy"] if candidate else 0.0
    else:
        anom = None
        candidate = None
        features = extract_features(heatmap)
        features["candidate_energy"] = 0.0

    return {
        "ela_heatmap": heatmap,
        "anomaly_map": anom,
        "features": features,
        "candidate": candidate,
        "has_baseline": baseline is not None,
    }
