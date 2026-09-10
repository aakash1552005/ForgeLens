"""
ForgeLens-X — Milestone 2: Face Dataset & Pair Generator
==========================================================
Manages standardized public face benchmark evaluation pairs
(same-person genuine pairs and different-person imposter pairs)
with simulated document-photo vs live-selfie cross-modality transformations.

Key Design Constraints:
    - Strictly public/licensed face fixtures (LFW, OpenCV Extra)
    - Zero real government identity documents
    - Deterministic, repeatable pair generation
    - Cross-demographic representation
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.utils import ensure_dirs, load_metadata, save_metadata, set_seed


# ---------------------------------------------------------------------------
# Cross-Modality Transformations: Document Photo vs Live Selfie
# ---------------------------------------------------------------------------

def simulate_document_photo(
    image: np.ndarray,
    target_size: Tuple[int, int] = (200, 240),
    quality: int = 82,
    seed: int = 42,
) -> np.ndarray:
    """
    Transform a portrait into a printed identity document photo crop:
    - Neutral passport-style framing
    - Subtle print/halftone texture
    - Document scanner contrast curve
    - Standard document JPEG compression history
    """
    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    # Resize to standard passport crop resolution
    doc_crop = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Convert to PIL for photometric adjustments
    pil_img = Image.fromarray(cv2.cvtColor(doc_crop, cv2.COLOR_BGR2RGB))

    # 1. Subtle contrast enhancement (typical of printed photo laminate)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.08)

    # 2. Subtle color saturation flattening (passport photo standardization)
    sat_enhancer = ImageEnhance.Color(pil_img)
    pil_img = sat_enhancer.enhance(0.92)

    # Convert back to numpy
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 3. Add subtle sensor/print micro-texture
    noise = rng.normal(0, 1.8, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 4. Simulate document compression history
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", arr, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def simulate_live_selfie(
    image: np.ndarray,
    target_size: Tuple[int, int] = (300, 360),
    quality: int = 92,
    seed: int = 1042,
) -> np.ndarray:
    """
    Transform a portrait into an unconstrained live selfie/webcam capture:
    - Subtle natural head tilt / rotation
    - Ambient lighting gradient (typical of indoor selfie)
    - Mobile/webcam camera sensor capture
    """
    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    # Resize to live selfie resolution
    selfie = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    sh, sw = selfie.shape[:2]

    # 1. Subtle camera tilt (-3 to +3 degrees)
    angle = rng.uniform(-3.0, 3.0)
    center = (sw // 2, sh // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    selfie = cv2.warpAffine(selfie, rot_mat, (sw, sh), borderMode=cv2.BORDER_REFLECT)

    # 2. Ambient lighting gradient (subtle top-left or top-right lighting)
    y_coords, x_coords = np.mgrid[0:sh, 0:sw]
    light_x = rng.choice([0, sw])
    gradient = 1.0 + 0.12 * (1.0 - np.sqrt((x_coords - light_x)**2 + y_coords**2) / np.sqrt(sw**2 + sh**2))
    gradient = np.dstack([gradient] * 3)

    arr = np.clip(selfie.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    # 3. Simulate high-res mobile JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", arr, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Benchmark Pair Generation & Management
# ---------------------------------------------------------------------------

def get_base_face_fixtures() -> List[Dict[str, Any]]:
    """
    Retrieve available public face fixtures in data/test_faces.
    """
    test_faces_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_faces")
    ensure_dirs(test_faces_dir)

    known_fixtures = [
        {"id": "david", "file1": "david1.jpg", "file2": "david2.jpg", "demographic": "male_caucasian"},
        {"id": "subject_lfw1", "file1": "100032540_1.jpg", "file2": None, "demographic": "male_adult"},
        {"id": "subject_lfw2", "file1": "100040721_1.jpg", "file2": None, "demographic": "female_adult"},
        {"id": "lena", "file1": "lena.jpg", "file2": None, "demographic": "female_adult"},
        {"id": "messi", "file1": "messi.jpg", "file2": None, "demographic": "male_adult"},
    ]

    valid = []
    for f in known_fixtures:
        p1 = os.path.join(test_faces_dir, f["file1"])
        if os.path.exists(p1):
            f["path1"] = p1
            f["path2"] = os.path.join(test_faces_dir, f["file2"]) if f.get("file2") else None
            valid.append(f)

    return valid


def create_benchmark_pairs(
    output_dir: str = "data/face_pairs",
    n_pairs: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a balanced public evaluation dataset with genuine and imposter pairs:
    - Exactly 50% Same-Person (Genuine) pairs
    - Exactly 50% Different-Person (Imposter) pairs
    - Real document vs live selfie cross-modality simulation

    Args:
        output_dir: directory to store images and pair metadata index
        n_pairs: total number of pairs to produce (half genuine, half imposter)
        seed: random seed for reproducibility

    Returns:
        Summary dict of generated benchmark dataset.
    """
    set_seed(seed)
    rng = random.Random(seed)

    images_dir = os.path.join(output_dir, "images")
    ensure_dirs(images_dir)

    fixtures = get_base_face_fixtures()
    if not fixtures:
        raise RuntimeError("No base face fixtures found in data/test_faces.")

    pairs = []
    n_genuine = n_pairs // 2
    n_imposter = n_pairs - n_genuine

    # 1. Generate Genuine Pairs (Same Person)
    for i in range(n_genuine):
        pair_id = f"pair_{i+1:04d}_genuine"
        fix = fixtures[i % len(fixtures)]
        subj_id = fix["id"]

        img_bgr = cv2.imread(fix["path1"])

        # If secondary natural capture exists, use it for live photo; else simulate live selfie
        doc_img = simulate_document_photo(img_bgr, seed=seed + i)
        if fix.get("path2") and os.path.exists(fix["path2"]):
            live_raw = cv2.imread(fix["path2"])
            live_img = simulate_live_selfie(live_raw, seed=seed + 1000 + i)
        else:
            live_img = simulate_live_selfie(img_bgr, seed=seed + 1000 + i)

        doc_path = os.path.join(images_dir, f"{pair_id}_doc.jpg")
        live_path = os.path.join(images_dir, f"{pair_id}_live.jpg")

        cv2.imwrite(doc_path, doc_img)
        cv2.imwrite(live_path, live_img)

        pairs.append({
            "pair_id": pair_id,
            "label": "genuine",
            "is_same_person": True,
            "subject_doc": subj_id,
            "subject_live": subj_id,
            "demographic": fix["demographic"],
            "doc_image_path": doc_path,
            "live_image_path": live_path,
        })

    # 2. Generate Imposter Pairs (Different Person)
    for j in range(n_imposter):
        pair_id = f"pair_{n_genuine + j + 1:04d}_imposter"

        # Pick two distinct subjects
        idx1 = j % len(fixtures)
        idx2 = (j + 1 + (j // len(fixtures))) % len(fixtures)
        if idx1 == idx2:
            idx2 = (idx1 + 1) % len(fixtures)

        fix1 = fixtures[idx1]
        fix2 = fixtures[idx2]

        doc_raw = cv2.imread(fix1["path1"])
        live_raw = cv2.imread(fix2["path1"])

        doc_img = simulate_document_photo(doc_raw, seed=seed + 2000 + j)
        live_img = simulate_live_selfie(live_raw, seed=seed + 3000 + j)

        doc_path = os.path.join(images_dir, f"{pair_id}_doc.jpg")
        live_path = os.path.join(images_dir, f"{pair_id}_live.jpg")

        cv2.imwrite(doc_path, doc_img)
        cv2.imwrite(live_path, live_img)

        pairs.append({
            "pair_id": pair_id,
            "label": "imposter",
            "is_same_person": False,
            "subject_doc": fix1["id"],
            "subject_live": fix2["id"],
            "demographic": f"{fix1['demographic']}_vs_{fix2['demographic']}",
            "doc_image_path": doc_path,
            "live_image_path": live_path,
        })

    index_data = {
        "dataset_name": "ForgeLens-X Face Verification Benchmark",
        "n_pairs": len(pairs),
        "n_genuine": n_genuine,
        "n_imposter": n_imposter,
        "seed": seed,
        "pairs": pairs,
    }

    index_path = os.path.join(output_dir, "pairs_index.json")
    save_metadata(index_data, index_path)

    return {
        "index_path": index_path,
        "n_pairs": len(pairs),
        "n_genuine": n_genuine,
        "n_imposter": n_imposter,
        "output_dir": output_dir,
    }


def load_benchmark_pairs(
    pairs_index_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load benchmark face pairs from pairs_index.json.
    If missing, automatically creates a default set of 20 benchmark pairs.
    """
    if pairs_index_path is None:
        pairs_index_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "face_pairs",
            "pairs_index.json",
        )
    elif os.path.isdir(pairs_index_path):
        pairs_index_path = os.path.join(pairs_index_path, "pairs_index.json")

    if not os.path.exists(pairs_index_path):
        out_dir = os.path.dirname(pairs_index_path)
        create_benchmark_pairs(output_dir=out_dir, n_pairs=20, seed=42)

    meta = load_metadata(pairs_index_path)
    return meta.get("pairs", [])


# Alias for consistency
generate_benchmark_pairs = create_benchmark_pairs
