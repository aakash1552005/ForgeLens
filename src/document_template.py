"""
ForgeLens-X — Fictional Document Template Generator
=====================================================
Generates deterministic fictional identity-card-style documents
for the Republic of Forgelensia using Indian-style names/dates.

Safety:
    - Entirely fictional layout and country
    - Procedurally generated placeholder photo (NOT a real face)
    - No real government document layouts reproduced
"""

import math
import os
import random
from datetime import datetime, timedelta

from faker import Faker
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.utils import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOC_WIDTH = 800
DOC_HEIGHT = 500

# Color palette — muted official-looking scheme
BG_COLOR = (245, 240, 230)          # cream
HEADER_BG = (31, 61, 107)           # dark navy
HEADER_TEXT = (255, 255, 255)       # white
FIELD_LABEL_COLOR = (80, 80, 80)    # grey
FIELD_VALUE_COLOR = (10, 10, 10)    # near black
BORDER_COLOR = (31, 61, 107)        # navy
STAMP_COLOR = (180, 50, 50, 100)    # red semi-transparent
PATTERN_COLOR = (220, 215, 205)     # subtle pattern
PHOTO_BG = (200, 200, 210)          # light grey-blue


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Get a font, falling back gracefully.
    Tries system fonts, then Pillow default.
    """
    font_candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            continue
    # Ultimate fallback — Pillow's built-in bitmap font
    return ImageFont.load_default()


def _get_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a bold font, falling back to regular."""
    bold_candidates = [
        "arialbd.ttf",
        "Arial Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for font_name in bold_candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            continue
    return _get_font(size)


def _draw_guilloche_pattern(draw: ImageDraw.Draw, width: int, height: int) -> None:
    """Draw subtle security-style background pattern."""
    # Concentric ellipses
    cx, cy = width // 2, height // 2
    for r in range(30, max(width, height), 40):
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=PATTERN_COLOR,
            width=1,
        )
    # Diagonal lines
    for offset in range(-max(width, height), max(width, height), 25):
        draw.line(
            [(offset, 0), (offset + height, height)],
            fill=PATTERN_COLOR,
            width=1,
        )


def _draw_placeholder_photo(draw: ImageDraw.Draw, bbox: list, seed: int) -> None:
    """
    Draw a procedurally generated geometric placeholder.
    NOT a real or realistic face.
    """
    rng = random.Random(seed)
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    # Background
    draw.rectangle(bbox, fill=PHOTO_BG, outline=BORDER_COLOR, width=2)

    # Geometric shapes as placeholder
    colors = [
        (rng.randint(100, 200), rng.randint(100, 200), rng.randint(100, 200))
        for _ in range(4)
    ]

    # Head circle
    head_r = min(w, h) // 4
    cx, cy = x1 + w // 2, y1 + h // 3
    draw.ellipse(
        [cx - head_r, cy - head_r, cx + head_r, cy + head_r],
        fill=colors[0],
        outline=colors[1],
        width=2,
    )

    # Body trapezoid
    body_top = cy + head_r + 5
    draw.polygon(
        [
            (cx - head_r - 10, y2 - 5),
            (cx + head_r + 10, y2 - 5),
            (cx + head_r - 5, body_top),
            (cx - head_r + 5, body_top),
        ],
        fill=colors[2],
        outline=colors[3],
    )

    # "PHOTO" text
    font = _get_font(12)
    draw.text((x1 + 5, y2 - 20), "PLACEHOLDER", fill=(100, 100, 100), font=font)


def _draw_stamp(draw: ImageDraw.Draw, bbox: list) -> None:
    """Draw a fictional government seal/stamp."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r = min(x2 - x1, y2 - y1) // 2

    # Outer circle
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(180, 50, 50), width=3)
    # Inner circle
    draw.ellipse(
        [cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8],
        outline=(180, 50, 50),
        width=2,
    )

    # Star in center
    star_r = r // 3
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        rad = star_r if i % 2 == 0 else star_r // 2
        px = cx + int(rad * math.cos(angle))
        py = cy - int(rad * math.sin(angle))
        points.append((px, py))
    draw.polygon(points, fill=(180, 50, 50))

    # Text around stamp
    font = _get_font(10)
    draw.text((cx - r + 12, cy - r + 3), "FORGELENSIA", fill=(180, 50, 50), font=font)
    draw.text((cx - r + 18, cy + r - 18), "OFFICIAL SEAL", fill=(180, 50, 50), font=font)


def generate_document(source_id: str, seed: int = 42, face_photo: Optional[Any] = None) -> dict:
    """
    Generate a single fictional identity document.

    Args:
        source_id: unique identifier (e.g. "src_0001")
        seed: random seed for deterministic generation
        face_photo: optional image path, numpy array, or PIL Image of portrait

    Returns:
        {
            "image": PIL.Image.Image,
            "field_bboxes": { field_name: {"bbox": [...], "value": "..."} },
            "metadata": { "source_id", "seed", "country" }
        }
    """
    set_seed(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)

    # --- Generate field data ---
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=65)
    issue_date = fake.date_between(start_date="-5y", end_date="today")
    from datetime import timedelta
    import random
    expiry_date = issue_date + timedelta(days=random.randint(3650, 7300))

    dob_str = dob.strftime("%d/%m/%Y")
    issue_str = issue_date.strftime("%d/%m/%Y")
    expiry_str = expiry_date.strftime("%d/%m/%Y")
    doc_number = f"FGL-{random.randint(100000, 999999):06d}-{random.randint(10, 99):02d}"

    # --- Create canvas ---
    image = Image.new("RGB", (DOC_WIDTH, DOC_HEIGHT), BG_COLOR)
    img = image
    draw = ImageDraw.Draw(image)

    # Security guilloche background pattern
    _draw_guilloche_pattern(draw, DOC_WIDTH, DOC_HEIGHT)

    # Outer border
    draw.rectangle([5, 5, DOC_WIDTH - 6, DOC_HEIGHT - 6], outline=BORDER_COLOR, width=3)
    draw.rectangle([10, 10, DOC_WIDTH - 11, DOC_HEIGHT - 11], outline=BORDER_COLOR, width=1)

    # Header band
    draw.rectangle([12, 12, DOC_WIDTH - 13, 55], fill=HEADER_BG)
    header_font = _get_bold_font(18)
    draw.text(
        (20, 15),
        "REPUBLIC OF FORGELENSIA — NATIONAL IDENTITY CARD",
        fill=HEADER_TEXT,
        font=header_font,
    )

    # --- Field positions (fixed layout) ---
    label_font = _get_font(12)
    value_font = _get_font(16)

    fields = {}

    # Photo region (left side)
    photo_bbox = [30, 80, 200, 260]
    pw, ph = photo_bbox[2] - photo_bbox[0], photo_bbox[3] - photo_bbox[1]

    if face_photo is not None:
        try:
            if isinstance(face_photo, str) and os.path.exists(face_photo):
                p_img = Image.open(face_photo).convert("RGB")
            elif isinstance(face_photo, np.ndarray):
                p_img = Image.fromarray(cv2.cvtColor(face_photo, cv2.COLOR_BGR2RGB))
            elif isinstance(face_photo, Image.Image):
                p_img = face_photo.convert("RGB")
            else:
                p_img = None

            if p_img is not None:
                from PIL import ImageOps
                p_resized = ImageOps.fit(p_img, (pw, ph), method=Image.Resampling.LANCZOS)
                image.paste(p_resized, (photo_bbox[0], photo_bbox[1]))
                draw.rectangle(photo_bbox, outline=BORDER_COLOR, width=2)
                fields["photo"] = {"bbox": photo_bbox, "value": "portrait"}
            else:
                _draw_placeholder_photo(draw, photo_bbox, seed)
                fields["photo"] = {"bbox": photo_bbox, "value": "placeholder"}
        except Exception:
            _draw_placeholder_photo(draw, photo_bbox, seed)
            fields["photo"] = {"bbox": photo_bbox, "value": "placeholder"}
    else:
        _draw_placeholder_photo(draw, photo_bbox, seed)
        fields["photo"] = {"bbox": photo_bbox, "value": "placeholder"}

    # Name
    name_pos = (230, 85)
    draw.text(name_pos, "Full Name", fill=FIELD_LABEL_COLOR, font=label_font)
    name_val_pos = (230, 103)
    draw.text(name_val_pos, name, fill=FIELD_VALUE_COLOR, font=value_font)
    tb = draw.textbbox(name_val_pos, name, font=value_font)
    name_bbox = [max(0, tb[0] - 2), max(0, tb[1] - 2), min(DOC_WIDTH, tb[2] + 2), min(DOC_HEIGHT, tb[3] + 2)]
    fields["name"] = {"bbox": name_bbox, "value": name, "pos": name_val_pos}

    # DOB
    dob_pos = (230, 140)
    draw.text(dob_pos, "Date of Birth", fill=FIELD_LABEL_COLOR, font=label_font)
    dob_val_pos = (230, 158)
    draw.text(dob_val_pos, dob_str, fill=FIELD_VALUE_COLOR, font=value_font)
    tb = draw.textbbox(dob_val_pos, dob_str, font=value_font)
    dob_bbox = [max(0, tb[0] - 2), max(0, tb[1] - 2), min(DOC_WIDTH, tb[2] + 2), min(DOC_HEIGHT, tb[3] + 2)]
    fields["dob"] = {"bbox": dob_bbox, "value": dob_str, "pos": dob_val_pos}

    # Document Number
    doc_num_pos = (230, 195)
    draw.text(doc_num_pos, "Document Number", fill=FIELD_LABEL_COLOR, font=label_font)
    doc_num_val_pos = (230, 213)
    draw.text(doc_num_val_pos, doc_number, fill=FIELD_VALUE_COLOR, font=value_font)
    tb = draw.textbbox(doc_num_val_pos, doc_number, font=value_font)
    doc_num_bbox = [max(0, tb[0] - 2), max(0, tb[1] - 2), min(DOC_WIDTH, tb[2] + 2), min(DOC_HEIGHT, tb[3] + 2)]
    fields["document_number"] = {"bbox": doc_num_bbox, "value": doc_number, "pos": doc_num_val_pos}

    # Issue Date
    issue_pos = (230, 250)
    draw.text(issue_pos, "Issue Date", fill=FIELD_LABEL_COLOR, font=label_font)
    issue_val_pos = (230, 268)
    draw.text(issue_val_pos, issue_str, fill=FIELD_VALUE_COLOR, font=value_font)
    tb = draw.textbbox(issue_val_pos, issue_str, font=value_font)
    issue_bbox = [max(0, tb[0] - 2), max(0, tb[1] - 2), min(DOC_WIDTH, tb[2] + 2), min(DOC_HEIGHT, tb[3] + 2)]
    fields["issue_date"] = {"bbox": issue_bbox, "value": issue_str, "pos": issue_val_pos}

    # Expiry Date
    expiry_pos = (450, 250)
    draw.text(expiry_pos, "Expiry Date", fill=FIELD_LABEL_COLOR, font=label_font)
    expiry_val_pos = (450, 268)
    draw.text(expiry_val_pos, expiry_str, fill=FIELD_VALUE_COLOR, font=value_font)
    tb = draw.textbbox(expiry_val_pos, expiry_str, font=value_font)
    expiry_bbox = [max(0, tb[0] - 2), max(0, tb[1] - 2), min(DOC_WIDTH, tb[2] + 2), min(DOC_HEIGHT, tb[3] + 2)]
    fields["expiry_date"] = {"bbox": expiry_bbox, "value": expiry_str, "pos": expiry_val_pos}

    # Stamp (bottom right)
    stamp_bbox = [620, 320, 770, 470]
    _draw_stamp(draw, stamp_bbox)
    fields["stamp"] = {"bbox": stamp_bbox, "value": "seal"}

    # Footer
    footer_font = _get_font(10)
    draw.text(
        (20, DOC_HEIGHT - 30),
        f"Document ID: {doc_number}  |  This is a fictional research artifact.",
        fill=(120, 120, 120),
        font=footer_font,
    )

    return {
        "image": img,
        "field_bboxes": fields,
        "metadata": {
            "source_id": source_id,
            "seed": seed,
            "country": "FORGELENSIA",
            "generated_at": datetime.now().isoformat(),
        },
    }
