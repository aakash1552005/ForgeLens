import os
import sys
sys.path.insert(0, os.path.abspath("."))
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
from src.ocr import extract_structured_fields
from src.semantic_checks import run_semantic_rule_battery

DOC_W, DOC_H = 800, 500
BG_COLOR = (245, 240, 230)
NAVY_HEADER = (31, 61, 107)

img = Image.new("RGB", (DOC_W, DOC_H), BG_COLOR)
draw = ImageDraw.Draw(img)

# Border
draw.rectangle([6, 6, DOC_W - 7, DOC_H - 7], outline=NAVY_HEADER, width=3)
draw.rectangle([10, 10, DOC_W - 11, DOC_H - 11], outline=NAVY_HEADER, width=1)

# Header
draw.rectangle([12, 12, DOC_W - 13, 55], fill=NAVY_HEADER)
font_title = ImageFont.truetype("arialbd.ttf", 17)
draw.text((24, 18), "REPUBLIC OF FORGELENSIA — PASSPORT", fill=(255, 255, 255), font=font_title)
font_sub = ImageFont.truetype("arial.ttf", 10)
draw.text((24, 38), "TRAVEL CREDENTIAL  ·  ICAO DOC 9303  ·  SPECIMEN", fill=(200, 210, 230), font=font_sub)

# Photo
photo_box = [30, 80, 200, 260]
face_img = Image.open("data/face_pairs/images/pair_0001_genuine_doc.jpg").convert("RGB")
face_fit = ImageOps.fit(face_img, (170, 180), method=Image.Resampling.LANCZOS)
img.paste(face_fit, (30, 80))
draw.rectangle(photo_box, outline=NAVY_HEADER, width=2)

font_lbl = ImageFont.truetype("arial.ttf", 11)
font_val = ImageFont.truetype("arialbd.ttf", 14)

# Full Name
draw.text((230, 80), "Full Name", fill=(80, 80, 80), font=font_lbl)
draw.text((230, 98), "Arun Kumar Verma", fill=(10, 10, 10), font=font_val)

# Date of Birth
draw.text((230, 135), "Date of Birth", fill=(80, 80, 80), font=font_lbl)
draw.text((230, 153), "14/08/1988", fill=(10, 10, 10), font=font_val)

# Document Number
draw.text((230, 190), "Document Number", fill=(80, 80, 80), font=font_lbl)
draw.text((230, 208), "FGL-491820-05", fill=(10, 10, 10), font=font_val)

# Issue Date
draw.text((230, 245), "Issue Date", fill=(80, 80, 80), font=font_lbl)
draw.text((230, 263), "10/05/2014", fill=(10, 10, 10), font=font_val)

# Expiry Date
draw.text((450, 245), "Expiry Date", fill=(80, 80, 80), font=font_lbl)
draw.text((450, 263), "09/05/2024", fill=(10, 10, 10), font=font_val)

# Nationality
draw.text((450, 190), "Nationality", fill=(80, 80, 80), font=font_lbl)
draw.text((450, 208), "FORGELENSIAN", fill=(10, 10, 10), font=font_val)

# Save
img_path = "scripts/test_passport_layout.jpg"
img.save(img_path, "JPEG", quality=90, subsampling=0)

# Test OCR and Semantics
fields_res = extract_structured_fields(img_path)
print("=== EXTRACTED FIELDS ===")
for k, v in fields_res["fields"].items():
    print(f"  {k:15}: {v['value']} (status={v['status']}, conf={v['confidence']})")

sem_res = run_semantic_rule_battery(fields_res["fields"])
print("=== SEMANTIC RULES ===")
for c in sem_res:
    print(f"  {c['check']:25}: {c['status']} - {c['detail']}")
