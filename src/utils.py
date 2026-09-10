"""
ForgeLens-X — Shared Utilities
==============================
All modules import from here. No duplicate implementations.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set deterministic random seed for numpy and random."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Bounding-box / mask utilities
# ---------------------------------------------------------------------------

def compute_iou(box_a: list, box_b: list) -> float:
    """
    Compute Intersection-over-Union for two bounding boxes.
    
    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]
    
    Returns:
        IoU value in [0.0, 1.0]
    """
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute IoU between two binary masks.
    
    Args:
        mask_a: binary ndarray (0/1 or bool)
        mask_b: binary ndarray (0/1 or bool)
    
    Returns:
        IoU value in [0.0, 1.0]
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def bbox_from_mask(mask: np.ndarray) -> list | None:
    """
    Extract bounding box from a binary mask.
    
    Args:
        mask: binary ndarray
    
    Returns:
        [x1, y1, x2, y2] or None if mask is empty
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    return [int(x1), int(y1), int(x2 + 1), int(y2 + 1)]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def ensure_dirs(*paths: str) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_metadata(metadata: dict, path: str) -> None:
    """Save metadata dict as formatted JSON."""
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(path: str) -> dict:
    """Load metadata from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration.
    Defaults to configs/m1_config.yaml relative to project root.
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "configs" / "m1_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Return the data/ directory."""
    return get_project_root() / "data"


def get_generated_dir() -> Path:
    """Return data/generated/ directory."""
    return get_data_dir() / "generated"


def get_forensic_dir() -> Path:
    """Return data/forensic/ directory."""
    return get_data_dir() / "forensic"


def get_reports_dir() -> Path:
    """Return reports/ directory."""
    return get_project_root() / "reports"
