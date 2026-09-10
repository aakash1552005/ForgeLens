"""
ForgeLens-X — Tamper Generator
================================
Generates four attack types with pixel-accurate ground-truth masks.

CRITICAL: The JPEG reload pipeline is mandatory for ELA to work.

    Generate in-memory → save JPEG #1 → RELOAD #1 → apply tamper → save JPEG #2

Without the reload step, both images share the same compression
history and ELA cannot distinguish them.

Attack types:
    1. date_edit   — modify DOB, issue date, or expiry date
    2. text_edit   — modify name or document number
    3. photo_swap  — replace placeholder photo
    4. copy_move   — duplicate a region to another location
"""

import os
import random
from io import BytesIO

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

from src.document_template import (
    BG_COLOR,
    DOC_HEIGHT,
    DOC_WIDTH,
    FIELD_VALUE_COLOR,
    _draw_placeholder_photo,
    _get_font,
)
from src.utils import ensure_dirs, save_metadata, set_seed


def _save_jpeg(image: Image.Image, path: str, quality: int = 85) -> None:
    """Save image as JPEG with controlled parameters."""
    ensure_dirs(os.path.dirname(path))
    image.save(path, "JPEG", quality=quality, subsampling=0)


def _reload_jpeg(path: str) -> Image.Image:
    """Reload a saved JPEG from disk (embeds compression artifacts)."""
    return Image.open(path).convert("RGB")


def _create_mask(width: int, height: int) -> np.ndarray:
    """Create an empty binary mask."""
    return np.zeros((height, width), dtype=np.uint8)


def _fill_mask_bbox(mask: np.ndarray, bbox: list) -> None:
    """Fill a bounding box region in the mask with 255."""
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 255


def _paint_over_region(
    draw: ImageDraw.Draw, bbox: list, bg_color: tuple = BG_COLOR
) -> None:
    """Paint over a region with background color to erase original content."""
    draw.rectangle(bbox, fill=bg_color)


def apply_date_edit(
    image: Image.Image,
    field_bboxes: dict,
    seed: int = 42,
) -> dict:
    """
    Attack type 1: Modify a date field.
    
    Selects DOB, issue_date, or expiry_date randomly.
    Paints over original date and renders a new one.
    """
    rng = random.Random(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)

    # Choose target field
    date_fields = ["dob", "issue_date", "expiry_date"]
    target_field = rng.choice(date_fields)
    target_info = field_bboxes[target_field]
    original_value = target_info["value"]
    orig_bbox = target_info["bbox"]
    render_pos = tuple(target_info.get("pos", (orig_bbox[0] + 2, orig_bbox[1] + 2)))

    # Generate a different date
    if target_field == "dob":
        new_date = fake.date_of_birth(minimum_age=18, maximum_age=65)
    else:
        new_date = fake.date_between(start_date="-10y", end_date="+10y")
    new_value = new_date.strftime("%d/%m/%Y")

    # Ensure the new value is actually different
    attempts = 0
    while new_value == original_value and attempts < 50:
        new_date = fake.date_between(start_date="-15y", end_date="+15y")
        new_value = new_date.strftime("%d/%m/%Y")
        attempts += 1

    # Apply tamper
    tampered = image.copy()
    draw = ImageDraw.Draw(tampered)
    value_font = _get_font(16)
    new_tb = draw.textbbox(render_pos, new_value, font=value_font)
    new_bbox = [
        max(0, new_tb[0] - 2),
        max(0, new_tb[1] - 2),
        min(image.width, new_tb[2] + 2),
        min(image.height, new_tb[3] + 2),
    ]

    # Tamper region encompasses both the erased original text area and new text area
    tamper_bbox = [
        min(orig_bbox[0], new_bbox[0]),
        min(orig_bbox[1], new_bbox[1]),
        max(orig_bbox[2], new_bbox[2]),
        max(orig_bbox[3], new_bbox[3]),
    ]
    _paint_over_region(draw, tamper_bbox)
    draw.text(render_pos, new_value, fill=FIELD_VALUE_COLOR, font=value_font)

    # Create mask
    mask = _create_mask(image.width, image.height)
    _fill_mask_bbox(mask, tamper_bbox)

    return {
        "tampered_image": tampered,
        "ground_truth_mask": mask,
        "ground_truth_bbox": tamper_bbox,
        "attack_type": "date_edit",
        "target_field": target_field,
        "original_value": original_value,
        "tampered_value": new_value,
    }


def apply_text_edit(
    image: Image.Image,
    field_bboxes: dict,
    seed: int = 42,
) -> dict:
    """
    Attack type 2: Modify name or document number.
    """
    rng = random.Random(seed)
    fake = Faker("en_IN")
    Faker.seed(seed + 1000)  # Different seed to avoid same name

    # Choose target field
    text_fields = ["name", "document_number"]
    target_field = rng.choice(text_fields)
    target_info = field_bboxes[target_field]
    original_value = target_info["value"]
    orig_bbox = target_info["bbox"]
    render_pos = tuple(target_info.get("pos", (orig_bbox[0] + 2, orig_bbox[1] + 2)))

    # Generate new value
    if target_field == "name":
        new_value = fake.name()
        attempts = 0
        while new_value == original_value and attempts < 50:
            new_value = fake.name()
            attempts += 1
    else:
        new_value = f"FGL-{rng.randint(100000, 999999):06d}-{rng.randint(10, 99):02d}"
        while new_value == original_value:
            new_value = f"FGL-{rng.randint(100000, 999999):06d}-{rng.randint(10, 99):02d}"

    # Apply tamper
    tampered = image.copy()
    draw = ImageDraw.Draw(tampered)
    value_font = _get_font(16)
    new_tb = draw.textbbox(render_pos, new_value, font=value_font)
    new_bbox = [
        max(0, new_tb[0] - 2),
        max(0, new_tb[1] - 2),
        min(image.width, new_tb[2] + 2),
        min(image.height, new_tb[3] + 2),
    ]

    # Tamper region encompasses both the erased original text area and new text area
    tamper_bbox = [
        min(orig_bbox[0], new_bbox[0]),
        min(orig_bbox[1], new_bbox[1]),
        max(orig_bbox[2], new_bbox[2]),
        max(orig_bbox[3], new_bbox[3]),
    ]
    _paint_over_region(draw, tamper_bbox)
    draw.text(render_pos, new_value, fill=FIELD_VALUE_COLOR, font=value_font)

    # Create mask
    mask = _create_mask(image.width, image.height)
    _fill_mask_bbox(mask, tamper_bbox)

    return {
        "tampered_image": tampered,
        "ground_truth_mask": mask,
        "ground_truth_bbox": tamper_bbox,
        "attack_type": "text_edit",
        "target_field": target_field,
        "original_value": original_value,
        "tampered_value": new_value,
    }


def apply_photo_swap(
    image: Image.Image,
    field_bboxes: dict,
    seed: int = 42,
) -> dict:
    """
    Attack type 3: Replace photo with an externally sourced portrait.

    Splicing realism:
        1. Render alternative portrait with distinct seed
        2. Inject photographic sensor micro-texture (subtle Gaussian noise)
        3. Pre-compress as external JPEG at quality=92
        4. Paste into host document canvas
    """
    photo_info = field_bboxes["photo"]
    bbox = photo_info["bbox"]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    # Create standalone cropped photo
    swapped_crop = Image.new("RGB", (w, h), (200, 200, 210))
    draw = ImageDraw.Draw(swapped_crop)
    _draw_placeholder_photo(draw, [0, 0, w, h], seed=seed + 9999)

    # Inject photographic sensor micro-texture
    arr = np.array(swapped_crop, dtype=np.int16)
    noise_rng = np.random.RandomState(seed + 9999)
    noise = noise_rng.normal(0, 6.0, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    textured_crop = Image.fromarray(arr)

    # Pre-compress as external JPEG at Q=92
    buf = BytesIO()
    textured_crop.save(buf, "JPEG", quality=92, subsampling=0)
    buf.seek(0)
    external_photo = Image.open(buf).convert("RGB")

    # Paste onto host image
    tampered = image.copy()
    tampered.paste(external_photo, (x1, y1))

    # Create mask
    mask = _create_mask(image.width, image.height)
    _fill_mask_bbox(mask, bbox)

    return {
        "tampered_image": tampered,
        "ground_truth_mask": mask,
        "ground_truth_bbox": bbox,
        "attack_type": "photo_swap",
        "target_field": "photo",
        "original_value": "placeholder_original",
        "tampered_value": "placeholder_swapped",
    }


def apply_copy_move(
    image: Image.Image,
    field_bboxes: dict,
    seed: int = 42,
) -> dict:
    """
    Attack type 4: Duplicate an existing region to another location.
    
    Copies the stamp region to an empty area of the document.
    """
    rng = random.Random(seed)

    # Source region: stamp
    stamp_info = field_bboxes["stamp"]
    src_bbox = stamp_info["bbox"]
    sx1, sy1, sx2, sy2 = src_bbox
    sw, sh = sx2 - sx1, sy2 - sy1

    # Crop source region
    source_patch = image.crop((sx1, sy1, sx2, sy2))

    # Destination: bottom-left area (avoid overlap with existing fields)
    dx1 = rng.randint(30, 180)
    dy1 = rng.randint(300, DOC_HEIGHT - sh - 20)
    dx2 = dx1 + sw
    dy2 = dy1 + sh

    # Clamp to image bounds
    dx2 = min(dx2, DOC_WIDTH - 10)
    dy2 = min(dy2, DOC_HEIGHT - 10)

    dst_bbox = [dx1, dy1, dx2, dy2]

    # Apply tamper
    tampered = image.copy()
    tampered.paste(source_patch, (dx1, dy1))

    # Create mask — the pasted (destination) region is the tampered area
    mask = _create_mask(image.width, image.height)
    _fill_mask_bbox(mask, dst_bbox)

    return {
        "tampered_image": tampered,
        "ground_truth_mask": mask,
        "ground_truth_bbox": dst_bbox,
        "attack_type": "copy_move",
        "target_field": "stamp_copy",
        "original_value": f"source:{src_bbox}",
        "tampered_value": f"destination:{dst_bbox}",
        "source_bbox": src_bbox,
        "destination_bbox": dst_bbox,
    }


# ---------------------------------------------------------------------------
# Main generation pipeline with JPEG reload
# ---------------------------------------------------------------------------

ATTACK_FUNCTIONS = {
    "date_edit": apply_date_edit,
    "text_edit": apply_text_edit,
    "photo_swap": apply_photo_swap,
    "copy_move": apply_copy_move,
}


def generate_tampered_dataset(
    document_result: dict,
    output_dir: str,
    jpeg_quality: int = 85,
    seed: int = 42,
) -> list:
    """
    Generate all attack variants for a single source document.

    CRITICAL JPEG PIPELINE:
        1. Save genuine document as JPEG #1
        2. RELOAD JPEG #1 from disk
        3. Apply each tamper on the RELOADED image
        4. Save tampered as JPEG #2

    Args:
        document_result: output from generate_document()
        output_dir: base directory for saving
        jpeg_quality: JPEG quality for saving
        seed: random seed

    Returns:
        List of metadata dicts for all generated samples (genuine + 4 attacks)
    """
    source_id = document_result["metadata"]["source_id"]
    image = document_result["image"]
    field_bboxes = document_result["field_bboxes"]

    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    metadata_dir = os.path.join(output_dir, "metadata")
    ensure_dirs(images_dir, masks_dir, metadata_dir)

    results = []

    # --- Step 1: Save genuine JPEG ---
    genuine_path = os.path.join(images_dir, f"{source_id}_genuine.jpg")
    _save_jpeg(image, genuine_path, quality=jpeg_quality)

    # Save genuine mask (all zeros)
    genuine_mask = _create_mask(image.width, image.height)
    genuine_mask_path = os.path.join(masks_dir, f"{source_id}_genuine.png")
    Image.fromarray(genuine_mask).save(genuine_mask_path)

    genuine_meta = {
        "source_id": source_id,
        "attack_type": "none",
        "label": "genuine",
        "image_path": genuine_path,
        "mask_path": genuine_mask_path,
        "bbox": None,
        "random_seed": seed,
        "target_field": None,
    }
    save_metadata(genuine_meta, os.path.join(metadata_dir, f"{source_id}_genuine.json"))
    results.append(genuine_meta)

    # --- Step 2: Reload genuine JPEG (critical for ELA) ---
    reloaded_image = _reload_jpeg(genuine_path)

    # --- Step 3: Apply each attack on the RELOADED image ---
    for attack_type, attack_fn in ATTACK_FUNCTIONS.items():
        attack_seed = seed + hash(attack_type) % 10000

        attack_result = attack_fn(
            image=reloaded_image,
            field_bboxes=field_bboxes,
            seed=attack_seed,
        )

        # Save tampered JPEG
        tampered_path = os.path.join(images_dir, f"{source_id}_{attack_type}.jpg")
        _save_jpeg(
            attack_result["tampered_image"], tampered_path, quality=jpeg_quality
        )

        # Save ground-truth mask
        mask_path = os.path.join(masks_dir, f"{source_id}_{attack_type}.png")
        Image.fromarray(attack_result["ground_truth_mask"]).save(mask_path)

        # Build metadata
        meta = {
            "source_id": source_id,
            "attack_type": attack_type,
            "label": "tampered",
            "image_path": tampered_path,
            "mask_path": mask_path,
            "bbox": attack_result["ground_truth_bbox"],
            "random_seed": attack_seed,
            "target_field": attack_result.get("target_field"),
            "original_value": attack_result.get("original_value"),
            "tampered_value": attack_result.get("tampered_value"),
        }

        # Copy-move extra metadata
        if "source_bbox" in attack_result:
            meta["source_bbox"] = attack_result["source_bbox"]
        if "destination_bbox" in attack_result:
            meta["destination_bbox"] = attack_result["destination_bbox"]

        save_metadata(meta, os.path.join(metadata_dir, f"{source_id}_{attack_type}.json"))
        results.append(meta)

    return results
