"""
ForgeLens-X — 5 Calibrated Passport Test Pairs Generator
=========================================================
Generates 5 realistic passport test pairs for manual dashboard evaluation:
- Folder 1: "original image" (Authentic documents)
- Folder 2: "tamper image"   (Manipulated documents with unique forensic vectors)

Real human portraits from the benchmark dataset are embedded into all 5 pairs.
Each pair exhibits one distinct, real-world forensic anomaly:
  Pair 1: Date Tampering (Expiry Date Extension from 2024 to 2034)
  Pair 2: Text Modification (Surname Alteration from Mehra to Singhania)
  Pair 3: Security Stamp Cloning (Copy-Move Forgery of Official Consular Seal)
  Pair 4: Photo Swap (Biometric Facial Substitution with Impostor Portrait)
  Pair 5: Document Number & MRZ Splicing (Altered Number + Checksum Mismatch)
"""

import os
import re
from io import BytesIO
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

# Root and Target Directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_DIR = os.path.join(ROOT_DIR, "original image")
TAMPER_DIR = os.path.join(ROOT_DIR, "tamper image")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(TAMPER_DIR, exist_ok=True)

# Standard Canvas & Security Palette
DOC_W, DOC_H = 800, 500
BG_COLOR = (245, 240, 230)          # Cream security parchment
NAVY_HEADER = (31, 61, 107)         # Diplomatic navy
GOLD_ACCENT = (197, 160, 89)        # Gold foil accent
TEXT_DARK = (10, 10, 10)            # High contrast dark text
TEXT_MUTED = (80, 80, 80)           # Field labels
BORDER_NAVY = (31, 61, 107)
GUILLOCHE_COLOR = (222, 216, 202)   # Guilloche background pattern


def get_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    """Load system font with reliable fallbacks."""
    if mono:
        candidates = ["C:/Windows/Fonts/consola.ttf", "consola.ttf", "DejaVuSansMono.ttf"]
    elif bold:
        candidates = ["C:/Windows/Fonts/arialbd.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
    else:
        candidates = ["C:/Windows/Fonts/arial.ttf", "arial.ttf", "DejaVuSans.ttf"]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_guilloche_security(draw: ImageDraw.Draw):
    """Draw subtle biometric passport background guilloche security lines."""
    cx, cy = DOC_W // 2, DOC_H // 2
    for r in range(35, max(DOC_W, DOC_H), 38):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=GUILLOCHE_COLOR, width=1)
    for offset in range(-DOC_W, DOC_W, 26):
        draw.line([(offset, 0), (offset + DOC_H, DOC_H)], fill=GUILLOCHE_COLOR, width=1)


def draw_official_seal(draw: ImageDraw.Draw, bbox: List[int]):
    """Draw realistic circular official border / consular stamp."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r_outer = min(x2 - x1, y2 - y1) // 2
    r_inner = r_outer - 7

    # Concentric rings
    draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline=(175, 45, 45), width=2)
    draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], outline=(175, 45, 45), width=1)

    # Center emblem
    font_star = get_font(16, bold=True)
    draw.text((cx - 7, cy - 11), "★", fill=(175, 45, 45), font=font_star)

    # Circular text
    font_text = get_font(8, bold=True)
    draw.text((cx - 38, cy - 24), "REPUBLIC OF", fill=(175, 45, 45), font=font_text)
    draw.text((cx - 42, cy + 12), "FORGELENSIA", fill=(175, 45, 45), font=font_text)


def compute_icao_check_digit(s: str) -> str:
    """ICAO Doc 9303 Modulo-10 checksum using cyclic weights [7, 3, 1]."""
    weights = [7, 3, 1]
    vals = {"<": 0}
    for i in range(10): vals[str(i)] = i
    for i in range(26): vals[chr(ord('A') + i)] = 10 + i
    total = sum(vals.get(c, 0) * weights[idx % 3] for idx, c in enumerate(s))
    return str(total % 10)


def build_td3_mrz(
    country: str,
    surname: str,
    given_names: str,
    doc_number: str,
    dob_yymmdd: str,
    sex: str,
    exp_yymmdd: str,
) -> Tuple[str, str]:
    """Generate canonical 2-line ICAO Doc 9303 TD3 MRZ."""
    # Line 1: P<COUNTRY<SURNAME<<GIVEN<NAMES
    clean_sur = re.sub(r"[^A-Z]", "", surname.upper())
    clean_giv = re.sub(r"[^A-Z ]", "", given_names.upper()).replace(" ", "<")
    name_field = f"{clean_sur}<<{clean_giv}"
    line1 = f"P<{country}{name_field}".ljust(44, "<")[:44]

    # Line 2: DOCNUM + CD + COUNTRY + DOB + CD + SEX + EXP + CD + OPT + COMP_CD
    clean_doc = re.sub(r"[^A-Z0-9]", "", doc_number.upper()).ljust(9, "<")[:9]
    doc_cd = compute_icao_check_digit(clean_doc)
    dob_cd = compute_icao_check_digit(dob_yymmdd)
    exp_cd = compute_icao_check_digit(exp_yymmdd)

    opt_data = "".ljust(14, "<")
    opt_cd = compute_icao_check_digit(opt_data)

    composite = clean_doc + doc_cd + dob_yymmdd + dob_cd + exp_yymmdd + exp_cd + opt_data + opt_cd
    comp_cd = compute_icao_check_digit(composite)

    line2 = f"{clean_doc}{doc_cd}{country}{dob_yymmdd}{dob_cd}{sex}{exp_yymmdd}{exp_cd}{opt_data}{opt_cd}{comp_cd}"
    line2 = line2[:44]

    return line1, line2


def render_passport(
    face_path: str,
    full_name: str,
    dob: str,
    doc_number: str,
    issue_date: str,
    expiry_date: str,
    nationality: str = "FORGELENSIAN",
    sex: str = "M",
    mrz_lines: Tuple[str, str] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Render canonical passport data page compliant with ForgeLens-X OCR & semantic specs."""
    img = Image.new("RGB", (DOC_W, DOC_H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 1. Security Guilloche Pattern
    draw_guilloche_security(draw)

    # 2. Outer Border Frame
    draw.rectangle([6, 6, DOC_W - 7, DOC_H - 7], outline=BORDER_NAVY, width=3)
    draw.rectangle([10, 10, DOC_W - 11, DOC_H - 11], outline=GOLD_ACCENT, width=1)

    # 3. Top Diplomatic Banner
    draw.rectangle([12, 12, DOC_W - 13, 55], fill=NAVY_HEADER)
    font_title = get_font(17, bold=True)
    draw.text((24, 17), "REPUBLIC OF FORGELENSIA — PASSPORT", fill=(255, 255, 255), font=font_title)
    font_sub = get_font(10)
    draw.text((24, 38), "OFFICIAL TRAVEL CREDENTIAL  ·  ICAO DOC 9303  ·  SPECIMEN", fill=GOLD_ACCENT, font=font_sub)

    # 4. Real Human Facial Photograph (Left Pane)
    photo_box = [30, 80, 200, 260]
    pw, ph = photo_box[2] - photo_box[0], photo_box[3] - photo_box[1]
    if os.path.exists(face_path):
        face_img = Image.open(face_path).convert("RGB")
        face_fit = ImageOps.fit(face_img, (pw, ph), method=Image.Resampling.LANCZOS)
        img.paste(face_fit, (photo_box[0], photo_box[1]))
    draw.rectangle(photo_box, outline=BORDER_NAVY, width=2)

    # 5. Field Positioning (Calibrated strictly to RapidOCR bounding targets)
    font_lbl = get_font(11)
    font_val = get_font(14, bold=True)
    coords = {}

    # Full Name
    draw.text((230, 80), "Full Name", fill=TEXT_MUTED, font=font_lbl)
    draw.text((230, 98), full_name, fill=TEXT_DARK, font=font_val)
    tb_name = draw.textbbox((230, 98), full_name, font=font_val)
    coords["name"] = {"bbox": list(tb_name), "pos": (230, 98), "val": full_name}

    # Date of Birth
    draw.text((230, 135), "Date of Birth", fill=TEXT_MUTED, font=font_lbl)
    draw.text((230, 153), dob, fill=TEXT_DARK, font=font_val)
    tb_dob = draw.textbbox((230, 153), dob, font=font_val)
    coords["dob"] = {"bbox": list(tb_dob), "pos": (230, 153), "val": dob}

    # Document Number
    draw.text((230, 190), "Document Number", fill=TEXT_MUTED, font=font_lbl)
    draw.text((230, 208), doc_number, fill=TEXT_DARK, font=font_val)
    tb_doc = draw.textbbox((230, 208), doc_number, font=font_val)
    coords["document_number"] = {"bbox": list(tb_doc), "pos": (230, 208), "val": doc_number}

    # Nationality
    draw.text((450, 190), "Nationality", fill=TEXT_MUTED, font=font_lbl)
    draw.text((450, 208), nationality, fill=TEXT_DARK, font=font_val)

    # Issue Date
    draw.text((230, 245), "Issue Date", fill=TEXT_MUTED, font=font_lbl)
    draw.text((230, 263), issue_date, fill=TEXT_DARK, font=font_val)
    tb_iss = draw.textbbox((230, 263), issue_date, font=font_val)
    coords["issue_date"] = {"bbox": list(tb_iss), "pos": (230, 263), "val": issue_date}

    # Expiry Date
    draw.text((450, 245), "Expiry Date", fill=TEXT_MUTED, font=font_lbl)
    draw.text((450, 263), expiry_date, fill=TEXT_DARK, font=font_val)
    tb_exp = draw.textbbox((450, 263), expiry_date, font=font_val)
    coords["expiry_date"] = {"bbox": list(tb_exp), "pos": (450, 263), "val": expiry_date}

    # 6. Official Consular Seal Stamp (Right Margin)
    seal_box = [615, 230, 755, 370]
    draw_official_seal(draw, seal_box)
    coords["stamp"] = {"bbox": seal_box}
    coords["photo"] = {"bbox": photo_box}

    # 7. Separator Line above MRZ
    draw.line([(15, 395), (DOC_W - 15, 395)], fill=BORDER_NAVY, width=1)

    # 8. Machine Readable Zone (MRZ) Optical TD3 Lines
    if mrz_lines:
        font_mrz = get_font(13, mono=True, bold=True)
        draw.text((32, 412), mrz_lines[0], fill=TEXT_DARK, font=font_mrz)
        draw.text((32, 436), mrz_lines[1], fill=TEXT_DARK, font=font_mrz)
        coords["mrz"] = {"line1": mrz_lines[0], "line2": mrz_lines[1]}

    return img, coords


def generate_all_5_passport_pairs():
    print("=" * 70)
    print("Generating 5 Calibrated Passport Test Pairs with Real Human Images")
    print("=" * 70)

    dataset_configs = [
        # --- PAIR 1: Date Tampering Issue ---
        {
            "id": "passport_01",
            "issue": "Date Tampering (Expiry Extended to 2034)",
            "face": "data/face_pairs/images/pair_0001_genuine_doc.jpg",
            "full_name": "Arun Kumar Verma",
            "surname": "Verma",
            "given": "Arun Kumar",
            "dob": "14/08/1988",
            "dob_yymmdd": "880814",
            "doc_num": "FGL-491820-05",
            "issue_date": "10/05/2014",
            "expiry_orig": "09/05/2024",
            "exp_orig_yymmdd": "240509",
            "tamper_action": "date_edit",
            "expiry_tampered": "09/05/2034",
            "primary_anomaly": "ELA Recompression Discontinuity on Expiry Date + Chronology Conflict",
        },
        # --- PAIR 2: Text Modification Issue ---
        {
            "id": "passport_02",
            "issue": "Text Modification (Surname Alteration)",
            "face": "data/face_pairs/images/pair_0002_genuine_doc.jpg",
            "full_name": "Rajesh Mehra",
            "surname": "Mehra",
            "given": "Rajesh",
            "dob": "22/11/1992",
            "dob_yymmdd": "921122",
            "doc_num": "FGL-810492-18",
            "issue_date": "18/02/2019",
            "expiry_orig": "17/02/2029",
            "exp_orig_yymmdd": "290217",
            "tamper_action": "text_edit",
            "name_tampered": "Rajesh Singhania",
            "primary_anomaly": "Typographic Stroke-Width Anomaly + OCR-to-MRZ Name Mismatch",
        },
        # --- PAIR 3: Stamp / Seal Issue (Copy-Move) ---
        {
            "id": "passport_03",
            "issue": "Security Stamp Cloning (Copy-Move Forgery)",
            "face": "data/face_pairs/images/pair_0003_genuine_doc.jpg",
            "full_name": "Siddharth Iyer",
            "surname": "Iyer",
            "given": "Siddharth",
            "dob": "05/03/1990",
            "dob_yymmdd": "900305",
            "doc_num": "FGL-391827-41",
            "issue_date": "12/07/2020",
            "expiry_orig": "11/07/2030",
            "exp_orig_yymmdd": "300711",
            "tamper_action": "stamp_clone",
            "primary_anomaly": "ORB Keypoint Copy-Move Clustered Correspondence Vectors",
        },
        # --- PAIR 4: Photo Swap Issue ---
        {
            "id": "passport_04",
            "issue": "Photo Swap (Facial Substitution with Impostor)",
            "face": "data/face_pairs/images/pair_0004_genuine_doc.jpg",
            "imposter_face": "data/face_pairs/images/pair_0011_imposter_doc.jpg",
            "full_name": "Priya Sharma",
            "surname": "Sharma",
            "given": "Priya",
            "dob": "19/09/1995",
            "dob_yymmdd": "950919",
            "doc_num": "FGL-720194-82",
            "issue_date": "01/10/2021",
            "expiry_orig": "30/09/2031",
            "exp_orig_yymmdd": "310930",
            "tamper_action": "photo_swap",
            "primary_anomaly": "Photo Splicing Boundary Artifact + ArcFace Facial Mismatch",
        },
        # --- PAIR 5: Document Number / Checksum Issue ---
        {
            "id": "passport_05",
            "issue": "Document Number & MRZ Tampering (Altered ID Splicing)",
            "face": "data/face_pairs/images/pair_0005_genuine_doc.jpg",
            "full_name": "Manoj Joshi",
            "surname": "Joshi",
            "given": "Manoj",
            "dob": "30/06/1985",
            "dob_yymmdd": "850630",
            "doc_num": "FGL-619284-07",
            "issue_date": "04/02/2021",
            "expiry_orig": "03/02/2031",
            "exp_orig_yymmdd": "310203",
            "tamper_action": "docnum_edit",
            "doc_num_tampered": "FGL-999999-99",
            "primary_anomaly": "ICAO Modulo-10 Checksum Failure + MRZ Concordance Mismatch",
        },
    ]

    manifest = []

    for idx, pinfo in enumerate(dataset_configs, start=1):
        pid = pinfo["id"]
        action = pinfo["tamper_action"]
        itype = pinfo["issue"]

        # 1. Generate ICAO TD3 MRZ for Original
        mrz1, mrz2 = build_td3_mrz(
            country="FGL",
            surname=pinfo["surname"],
            given_names=pinfo["given"],
            doc_number=pinfo["doc_num"],
            dob_yymmdd=pinfo["dob_yymmdd"],
            sex="M" if "Priya" not in pinfo["full_name"] else "F",
            exp_yymmdd=pinfo["exp_orig_yymmdd"],
        )

        # 2. Render Original Passport
        orig_canvas, field_coords = render_passport(
            face_path=pinfo["face"],
            full_name=pinfo["full_name"],
            dob=pinfo["dob"],
            doc_number=pinfo["doc_num"],
            issue_date=pinfo["issue_date"],
            expiry_date=pinfo["expiry_orig"],
            nationality="FORGELENSIAN",
            sex="M" if "Priya" not in pinfo["full_name"] else "F",
            mrz_lines=(mrz1, mrz2),
        )

        orig_filename = f"{pid}_original.jpg"
        orig_path = os.path.join(ORIGINAL_DIR, orig_filename)
        # Save Original JPEG #1 (Quality 90)
        orig_canvas.save(orig_path, "JPEG", quality=90, subsampling=0)

        # 3. RELOAD from disk (Mandatory for physical JPEG recompression forensics)
        tamper_canvas = Image.open(orig_path).convert("RGB")
        tdraw = ImageDraw.Draw(tamper_canvas)
        font_val = get_font(14, bold=True)

        # 4. Apply Unique Tampering Vector
        if action == "date_edit":
            # Alter expiry date visually to 2034 while leaving MRZ 2024
            exp_box = field_coords["expiry_date"]["bbox"]
            pos = tuple(field_coords["expiry_date"]["pos"])
            pw, ph = exp_box[2] - exp_box[0] + 8, exp_box[3] - exp_box[1] + 6
            patch = Image.new("RGB", (pw, ph), BG_COLOR)
            pdraw = ImageDraw.Draw(patch)
            pdraw.text((4, 4), pinfo["expiry_tampered"], fill=TEXT_DARK, font=font_val)
            pbuf = BytesIO()
            patch.save(pbuf, "JPEG", quality=72)
            pbuf.seek(0)
            patch_jpg = Image.open(pbuf)
            tamper_canvas.paste(patch_jpg, (exp_box[0] - 4, exp_box[1] - 3))
            tamper_filename = f"{pid}_tampered_date_edit.jpg"

        elif action == "text_edit":
            # Alter surname visually to Singhania
            name_box = field_coords["name"]["bbox"]
            pos = tuple(field_coords["name"]["pos"])
            pw, ph = name_box[2] - name_box[0] + 90, name_box[3] - name_box[1] + 6
            patch = Image.new("RGB", (pw, ph), BG_COLOR)
            pdraw = ImageDraw.Draw(patch)
            font_altered = get_font(15, bold=True)
            pdraw.text((4, 3), pinfo["name_tampered"], fill=(0, 0, 0), font=font_altered)
            pbuf = BytesIO()
            patch.save(pbuf, "JPEG", quality=72)
            pbuf.seek(0)
            patch_jpg = Image.open(pbuf)
            tamper_canvas.paste(patch_jpg, (name_box[0] - 4, name_box[1] - 3))
            tamper_filename = f"{pid}_tampered_text_edit.jpg"

        elif action == "stamp_clone":
            # Copy-Move: Clone official seal over photo/passport margin
            sbox = field_coords["stamp"]["bbox"]
            seal_crop = tamper_canvas.crop(sbox)
            dest_pos = (20, 260)
            tamper_canvas.paste(seal_crop, dest_pos)
            tamper_filename = f"{pid}_tampered_cloned_stamp.jpg"

        elif action == "photo_swap":
            # Photo Swap: Splice impostor face into document portrait window
            pbox = field_coords["photo"]["bbox"]
            pw, ph = pbox[2] - pbox[0], pbox[3] - pbox[1]
            imp_img = Image.open(pinfo["imposter_face"]).convert("RGB")
            imp_fit = ImageOps.fit(imp_img, (pw, ph), method=Image.Resampling.LANCZOS)
            tamper_canvas.paste(imp_fit, (pbox[0], pbox[1]))
            # Physical cut-and-paste boundary seam along portrait border
            tdraw.rectangle([pbox[0]-1, pbox[1]-1, pbox[2]+1, pbox[3]+1], outline=(168, 162, 155), width=1)
            tdraw.rectangle(pbox, outline=BORDER_NAVY, width=2)
            tamper_filename = f"{pid}_tampered_photo_swap.jpg"

        elif action == "docnum_edit":
            # Spliced Document Number + Corrupted MRZ checksum
            doc_box = field_coords["document_number"]["bbox"]
            pw, ph = doc_box[2] - doc_box[0] + 8, doc_box[3] - doc_box[1] + 6
            patch = Image.new("RGB", (pw, ph), BG_COLOR)
            pdraw = ImageDraw.Draw(patch)
            pdraw.text((4, 4), pinfo["doc_num_tampered"], fill=TEXT_DARK, font=font_val)
            pbuf = BytesIO()
            patch.save(pbuf, "JPEG", quality=72)
            pbuf.seek(0)
            patch_jpg = Image.open(pbuf)
            tamper_canvas.paste(patch_jpg, (doc_box[0] - 4, doc_box[1] - 3))

            # Corrupt MRZ line 2 checksum (replace check digit with 9)
            bad_line2 = mrz2[:9] + "9" + mrz2[10:]
            font_mrz = get_font(13, mono=True, bold=True)
            tdraw.rectangle([30, 434, DOC_W - 30, 455], fill=BG_COLOR)
            tdraw.text((32, 436), bad_line2, fill=TEXT_DARK, font=font_mrz)
            tamper_filename = f"{pid}_tampered_docnum_edit.jpg"

        # Save Tampered JPEG at Quality 90 (preserving pristine background/stamp fidelity)
        tamper_path = os.path.join(TAMPER_DIR, tamper_filename)
        tamper_canvas.save(tamper_path, "JPEG", quality=90, subsampling=0)

        # Save genuine selfie reference for biometric testing of Pair 4
        if pid == "passport_04":
            import shutil
            shutil.copy(pinfo["face"], os.path.join(ORIGINAL_DIR, "passport_04_selfie_genuine.jpg"))
            shutil.copy(pinfo["face"], os.path.join(TAMPER_DIR, "passport_04_selfie_genuine.jpg"))

        print(f"[{idx}/5] Pair {idx} Created:")
        print(f"      Original : {orig_path}")
        print(f"      Tampered : {tamper_path}")

        manifest.append({
            "pair": idx,
            "original_file": orig_filename,
            "tampered_file": tamper_filename,
            "citizen_name": pinfo["full_name"],
            "issue_type": itype,
            "primary_anomaly": pinfo["primary_anomaly"],
            "original_path": orig_path,
            "tampered_path": tamper_path,
        })

    # Write Complete Testing Guide Manifest
    guide_md = """# ForgeLens-X · Manual Testing Dataset Guide
## 5 Curated Passport Test Pairs with Real Human Portraits

Each pair contains one **Authentic Document** in `original image/` and one **Tampered Document** in `tamper image/`.

| Pair | Citizen Name | Original File (`original image/`) | Tampered File (`tamper image/`) | Specific Attack Vector | Forensic Evidence Triggered |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Pair 1** | Arun Kumar Verma | `passport_01_original.jpg` | `passport_01_tampered_date_edit.jpg` | **Date Tampering** (10-yr extension) | ELA compression residue discontinuity + ICAO MRZ Expiry mismatch |
| **Pair 2** | Rajesh Mehra | `passport_02_original.jpg` | `passport_02_tampered_text_edit.jpg` | **Text Alteration** (Surname changed) | Typographic stroke-width anomaly + MRZ Name concordance failure |
| **Pair 3** | Siddharth Iyer | `passport_03_original.jpg` | `passport_03_tampered_cloned_stamp.jpg` | **Security Stamp Cloning** | ORB keypoint Copy-Move clustered vector correspondences |
| **Pair 4** | Priya Sharma | `passport_04_original.jpg` | `passport_04_tampered_photo_swap.jpg` | **Photo Swap** (Impostor portrait) | Splicing edge discontinuity + ArcFace biometric verification failure |
| **Pair 5** | Manoj Joshi | `passport_05_original.jpg` | `passport_05_tampered_docnum_edit.jpg` | **Document Number Tampering** | ICAO Modulo-10 checksum failure + VIZ-to-MRZ Document No. mismatch |

---

### How to Test in the ForgeLens-X Console:
1. Open your browser at `http://localhost:8501`.
2. In the top screening banner, click on the **Upload Document** tab.
3. Drag and drop any image from `original image/` to see the **Authentic / Verified** verdict and clean forensic layers.
4. Drag and drop the corresponding image from `tamper image/` to see the **Tampering Detected / High Risk** verdict with highlighted bounding boxes, thermal heatmaps, and plain-language driver explanations!
"""

    with open(os.path.join(ORIGINAL_DIR, "MANUAL_TESTING_GUIDE.md"), "w", encoding="utf-8") as f:
        f.write(guide_md)
    with open(os.path.join(TAMPER_DIR, "MANUAL_TESTING_GUIDE.md"), "w", encoding="utf-8") as f:
        f.write(guide_md)
    with open(os.path.join(ROOT_DIR, "MANUAL_TESTING_GUIDE.md"), "w", encoding="utf-8") as f:
        f.write(guide_md)

    print("\n[+] Successfully created all 5 pairs and exported MANUAL_TESTING_GUIDE.md.")


if __name__ == "__main__":
    generate_all_5_passport_pairs()
