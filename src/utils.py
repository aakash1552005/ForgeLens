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


def get_splits_dir() -> Path:
    """Return data/splits/ directory."""
    return get_data_dir() / "splits"


def create_dataset_splits(
    master_index: dict,
    output_dir: str = None,
    train_ratio: float = 0.70,
    cal_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Partition master_index samples by source_id into train/cal/test splits.

    CRITICAL FOR FORENSIC VALIDITY:
        All attack variants of a given source_id must remain in the
        same split partition to prevent identity/template data leakage.

    Args:
        master_index: dict containing "samples" list
        output_dir: directory to save split JSON files (defaults to data/splits/)
        train_ratio: fraction for training (default 0.70)
        cal_ratio: fraction for calibration (default 0.15)
        test_ratio: fraction for final evaluation (default 0.15)
        seed: random seed for reproducible partition

    Returns:
        {
            "train": [samples...],
            "cal": [samples...],
            "test": [samples...],
            "split_summary": {...}
        }
    """
    samples = master_index.get("samples", [])
    if not samples:
        return {"train": [], "cal": [], "test": [], "split_summary": {}}

    # Group samples by source_id
    from collections import defaultdict
    source_groups = defaultdict(list)
    for s in samples:
        source_groups[s["source_id"]].append(s)

    unique_sources = sorted(list(source_groups.keys()))
    rng = random.Random(seed)
    shuffled_sources = list(unique_sources)
    rng.shuffle(shuffled_sources)

    n_sources = len(shuffled_sources)
    if n_sources == 1:
        train_sources = set(shuffled_sources)
        cal_sources = set()
        test_sources = set()
    elif n_sources == 2:
        train_sources = {shuffled_sources[0]}
        cal_sources = set()
        test_sources = {shuffled_sources[1]}
    else:
        n_train = max(1, int(n_sources * train_ratio))
        n_cal = max(1, int(n_sources * cal_ratio))
        # Ensure at least 1 in test if n_sources >= 3
        if n_train + n_cal >= n_sources:
            n_train = max(1, n_sources - 2)
            n_cal = 1
        train_sources = set(shuffled_sources[:n_train])
        cal_sources = set(shuffled_sources[n_train:n_train + n_cal])
        test_sources = set(shuffled_sources[n_train + n_cal:])

    train_samples = [s for src in train_sources for s in source_groups[src]]
    cal_samples = [s for src in cal_sources for s in source_groups[src]]
    test_samples = [s for src in test_sources for s in source_groups[src]]

    splits_data = {
        "train": train_samples,
        "cal": cal_samples,
        "test": test_samples,
        "split_summary": {
            "n_sources_total": n_sources,
            "n_sources_train": len(train_sources),
            "n_sources_cal": len(cal_sources),
            "n_sources_test": len(test_sources),
            "n_samples_train": len(train_samples),
            "n_samples_cal": len(cal_samples),
            "n_samples_test": len(test_samples),
        },
    }

    # Save to disk if requested or default
    target_dir = str(output_dir) if output_dir else str(get_splits_dir())
    ensure_dirs(target_dir)

    save_metadata({"samples": train_samples, "source_ids": sorted(list(train_sources))},
                  os.path.join(target_dir, "train.json"))
    save_metadata({"samples": cal_samples, "source_ids": sorted(list(cal_sources))},
                  os.path.join(target_dir, "cal.json"))
    save_metadata({"samples": test_samples, "source_ids": sorted(list(test_sources))},
                  os.path.join(target_dir, "test.json"))

    return splits_data
