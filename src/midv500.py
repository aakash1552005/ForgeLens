"""
ForgeLens-X — Real-World Identity Benchmark Loader (MIDV-500 & MIDV-2020)
========================================================================
Loads, parses, partitions, and synthesizes international real-world mobile
identity document benchmarks from MIDV-500 (ICPR) and MIDV-2020 (L3i/Smart Engines).

CRITICAL SPLIT RULE:
    Frames belonging to the same source document or video clip MUST remain
    in the same split partition.
    Random frame-level splitting is strictly prohibited to prevent data leakage.
"""

import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils import ensure_dirs, set_seed


# ---------------------------------------------------------------------------
# MIDV-500 & MIDV-2020 Annotation Parsers
# ---------------------------------------------------------------------------

def parse_midv500_annotation(json_path: str) -> Dict[str, Any]:
    """
    Parse MIDV-500 / MIDV-2020 ground-truth annotation file.
    MIDV annotations provide quadrilateral coordinates, field values, and attack tags.
    """
    if not os.path.exists(json_path):
        return {"fields": {}, "quad": None, "attack_type": "genuine", "is_presentation_attack": False}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"fields": {}, "quad": None, "attack_type": "genuine", "is_presentation_attack": False}

    fields = {}
    raw_fields = data.get("fields", {})

    field_mapping = {
        "sur_name": "name",
        "first_name": "name",
        "name": "name",
        "full_name": "name",
        "birth_date": "dob",
        "date_of_birth": "dob",
        "dob": "dob",
        "doc_number": "document_number",
        "document_number": "document_number",
        "doc_num": "document_number",
        "number": "document_number",
        "issue_date": "issue_date",
        "date_of_issue": "issue_date",
        "expiry_date": "expiry_date",
        "date_of_expiry": "expiry_date",
        "nationality": "nationality",
        "sex": "sex",
    }

    names = []
    for k, v in raw_fields.items():
        k_lower = k.lower()
        val_str = str(v.get("value", "") if isinstance(v, dict) else v).strip()
        quad = v.get("quad") if isinstance(v, dict) else None

        if k_lower in ["sur_name", "first_name"]:
            names.append(val_str)
        elif k_lower in field_mapping:
            canonical = field_mapping[k_lower]
            fields[canonical] = {
                "value": val_str,
                "quad": quad,
            }

    if names:
        fields["name"] = {
            "value": " ".join(names).strip(),
            "quad": None,
        }

    quad = data.get("quad")
    attack_type = data.get("attack_type", "genuine")
    is_pa = bool(data.get("is_presentation_attack", attack_type in ["screen_replay", "print_spoof", "presentation_attack"]))

    return {
        "doc_type": data.get("doc_type", "id_card"),
        "doc_category": data.get("doc_category", "passport" if "passport" in str(data.get("doc_type", "")).lower() else "id_card"),
        "country": data.get("country", "unknown"),
        "quad": quad,
        "fields": fields,
        "attack_type": attack_type,
        "is_presentation_attack": is_pa,
        "lighting_condition": data.get("lighting_condition", "normal"),
        "capture_device": data.get("capture_device", "smartphone"),
    }


# ---------------------------------------------------------------------------
# Zero-Leakage Source-Clip Splitter
# ---------------------------------------------------------------------------

def split_midv500_by_source_clip(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    cal_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Partition MIDV frames strictly at the source-document / video-clip level.
    Guarantees zero-leakage: all frames from a physical clip stay together.
    """
    rng = random.Random(seed)

    clip_groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        clip_id = s.get("source_clip_id") or s.get("doc_type") or Path(s["image_path"]).parent.name
        if clip_id not in clip_groups:
            clip_groups[clip_id] = []
        clip_groups[clip_id].append(s)

    unique_clips = sorted(list(clip_groups.keys()))
    rng.shuffle(unique_clips)

    n_total = len(unique_clips)
    n_train = int(n_total * train_ratio)
    n_cal = int(n_total * cal_ratio)
    n_test = n_total - n_train - n_cal

    if n_test <= 0 and n_total >= 3:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_cal > 0:
            n_cal -= 1

    if n_cal <= 0 and n_total >= 3:
        n_cal = 1
        if n_train > 1:
            n_train -= 1

    train_clips = set(unique_clips[:n_train])
    cal_clips = set(unique_clips[n_train:n_train + n_cal])
    test_clips = set(unique_clips[n_train + n_cal:])

    train_samples = []
    cal_samples = []
    test_samples = []

    for cid, items in clip_groups.items():
        if cid in train_clips:
            for it in items:
                it["split"] = "train"
                train_samples.append(it)
        elif cid in cal_clips:
            for it in items:
                it["split"] = "cal"
                cal_samples.append(it)
        else:
            for it in items:
                it["split"] = "test"
                test_samples.append(it)

    return {
        "train": train_samples,
        "cal": cal_samples,
        "test": test_samples,
        "clip_allocation": {
            "train_clips": sorted(list(train_clips)),
            "cal_clips": sorted(list(cal_clips)),
            "test_clips": sorted(list(test_clips)),
        },
        "summary": {
            "n_clips_total": n_total,
            "n_train_samples": len(train_samples),
            "n_cal_samples": len(cal_samples),
            "n_test_samples": len(test_samples),
        },
    }


# ---------------------------------------------------------------------------
# MIDV-500 Dataset Loader & Synthesizer
# ---------------------------------------------------------------------------

def load_midv500_dataset(
    data_dir: str = "data/midv500",
    sample_fallback_dir: str = "data/midv500_sample",
) -> List[Dict[str, Any]]:
    """
    Load real-world MIDV-500 dataset samples from local directory.
    Falls back to sample_fallback_dir if local dataset is not downloaded.
    """
    target_dir = data_dir if os.path.exists(data_dir) and os.listdir(data_dir) else sample_fallback_dir

    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        generate_sample_midv500_dataset(output_dir=target_dir, n_clips=6, frames_per_clip=3)

    samples = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                img_path = os.path.join(root, f)
                stem = Path(f).stem
                clip_dir = Path(root).name
                if clip_dir in ["images", "ground_truth"]:
                    clip_id = Path(root).parent.name
                else:
                    clip_id = clip_dir

                ann_candidates = [
                    os.path.join(root, f"{stem}.json"),
                    os.path.join(root, "..", "ground_truth", f"{stem}.json"),
                    os.path.join(root, f"{clip_id}.json"),
                    os.path.join(root, "..", f"{clip_id}.json"),
                ]
                annotation_data = {"fields": {}}
                for cand in ann_candidates:
                    if os.path.exists(cand):
                        annotation_data = parse_midv500_annotation(cand)
                        break

                samples.append({
                    "sample_id": f"{clip_id}_{stem}",
                    "image_path": img_path,
                    "source_clip_id": clip_id,
                    "doc_type": annotation_data.get("doc_type", "id_card"),
                    "doc_category": annotation_data.get("doc_category", "id_card"),
                    "country": annotation_data.get("country", "unknown"),
                    "quad": annotation_data.get("quad"),
                    "ground_truth_fields": annotation_data.get("fields", {}),
                    "fields": annotation_data.get("fields", {}),
                    "attack_type": annotation_data.get("attack_type", "genuine"),
                    "is_presentation_attack": annotation_data.get("is_presentation_attack", False),
                })

    return sorted(samples, key=lambda x: x["sample_id"])


def generate_sample_midv500_dataset(
    output_dir: str = "data/midv500_sample",
    n_clips: int = 6,
    frames_per_clip: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate realistic multi-country MIDV-500 benchmark sample fixtures.
    Includes Passports, ID Cards, and Driving Licenses across diverse jurisdictions.
    """
    ensure_dirs(output_dir)
    set_seed(seed)

    doc_specs = [
        {"type": "deu_passport", "category": "passport", "country": "DEU", "name": "ANNA SCHMIDT", "doc_num": "C01X00T47", "dob": "15/04/1986", "issue": "10/05/2018", "expiry": "09/05/2028", "mrz": "P<D<<SCHMIDT<<ANNA<<<<<<<<<<<<<<<<<<<<<<<\nC01X00T478DEU8604153F2805094<<<<<<<<<<<<<<08"},
        {"type": "usa_passport", "category": "passport", "country": "USA", "name": "JOHNATHAN DOE", "doc_num": "530291847", "dob": "22/08/1990", "issue": "14/01/2020", "expiry": "13/01/2030", "mrz": "P<USA<<DOE<<JOHNATHAN<<<<<<<<<<<<<<<<<<<<<<\n5302918474USA9008221M3001132<<<<<<<<<<<<<<04"},
        {"type": "fra_id", "category": "id_card", "country": "FRA", "name": "JEAN BOURGEOIS", "doc_num": "180275819201", "dob": "18/02/1975", "issue": "14/09/2016", "expiry": "14/09/2031"},
        {"type": "ind_id", "category": "id_card", "country": "IND", "name": "VIKRAM SHARMA", "doc_num": "FGL-849201-92", "dob": "12/04/1985", "issue": "15/08/2021", "expiry": "15/08/2031"},
        {"type": "esp_dl", "category": "driving_license", "country": "ESP", "name": "CARLOS GARCIA", "doc_num": "ES-94029184", "dob": "30/10/1983", "issue": "11/02/2019", "expiry": "11/02/2029"},
        {"type": "fin_id", "category": "id_card", "country": "FIN", "name": "MATTI VIRTANEN", "doc_num": "A93820192", "dob": "03/09/1990", "issue": "20/06/2020", "expiry": "20/06/2025"},
    ]

    total_samples = 0

    for i in range(min(n_clips, len(doc_specs))):
        spec = doc_specs[i]
        clip_id = f"clip_{i+1:02d}_{spec['type']}"
        clip_dir = os.path.join(output_dir, clip_id)
        img_dir = os.path.join(clip_dir, "images")
        gt_dir = os.path.join(clip_dir, "ground_truth")
        ensure_dirs(img_dir)
        ensure_dirs(gt_dir)

        for f_idx in range(frames_per_clip):
            f_seed = seed + i * 100 + f_idx
            frame_rng = random.Random(f_seed)

            canvas_w, canvas_h = 960, 640
            frame_img = Image.new("RGB", (canvas_w, canvas_h), (215, 218, 222))
            draw = ImageDraw.Draw(frame_img)

            # Natural tabletop background with slight tilt quad
            quad_corners = [
                [60 + frame_rng.randint(-5, 5), 50 + frame_rng.randint(-5, 5)],
                [900 + frame_rng.randint(-5, 5), 65 + frame_rng.randint(-5, 5)],
                [885 + frame_rng.randint(-5, 5), 585 + frame_rng.randint(-5, 5)],
                [75 + frame_rng.randint(-5, 5), 570 + frame_rng.randint(-5, 5)],
            ]

            # Draw card surface
            card_box = [70, 60, 890, 580]
            bg_lum = frame_rng.randint(220, 240)
            draw.rectangle(card_box, fill=(bg_lum, bg_lum - 4, bg_lum - 8), outline=(90, 90, 90), width=2)
            draw.rectangle([card_box[0] + 5, card_box[1] + 5, card_box[2] - 5, card_box[1] + 50], fill=(40, 60, 95))

            try:
                font_h = ImageFont.truetype("arialbd.ttf", 18)
                font_v = ImageFont.truetype("arial.ttf", 16)
                font_l = ImageFont.truetype("arial.ttf", 12)
                font_mrz = ImageFont.truetype("courbd.ttf", 14)
            except Exception:
                font_h = ImageFont.load_default()
                font_v = font_h
                font_l = font_h
                font_mrz = font_h

            title_text = f"{spec['country']} {spec['category'].upper().replace('_', ' ')}"
            draw.text((card_box[0] + 20, card_box[1] + 15), title_text, fill=(255, 255, 255), font=font_h)

            # Photo box
            photo_rect = [card_box[0] + 25, card_box[1] + 70, card_box[0] + 190, card_box[1] + 270]
            draw.rectangle(photo_rect, fill=(165, 175, 185), outline=(70, 70, 70), width=2)
            draw.text((photo_rect[0] + 35, photo_rect[1] + 85), "PHOTO", fill=(70, 70, 70), font=font_h)

            # Field text
            draw.text((card_box[0] + 220, card_box[1] + 75), "Full Name", fill=(90, 90, 90), font=font_l)
            draw.text((card_box[0] + 220, card_box[1] + 95), spec["name"], fill=(20, 20, 20), font=font_v)

            draw.text((card_box[0] + 220, card_box[1] + 135), "Date of Birth", fill=(90, 90, 90), font=font_l)
            draw.text((card_box[0] + 220, card_box[1] + 155), spec["dob"], fill=(20, 20, 20), font=font_v)

            draw.text((card_box[0] + 220, card_box[1] + 195), "Document Number", fill=(90, 90, 90), font=font_l)
            draw.text((card_box[0] + 220, card_box[1] + 215), spec["doc_num"], fill=(20, 20, 20), font=font_v)

            draw.text((card_box[0] + 220, card_box[1] + 255), "Issue Date", fill=(90, 90, 90), font=font_l)
            draw.text((card_box[0] + 220, card_box[1] + 275), spec["issue"], fill=(20, 20, 20), font=font_v)

            draw.text((card_box[0] + 450, card_box[1] + 255), "Expiry Date", fill=(90, 90, 90), font=font_l)
            draw.text((card_box[0] + 450, card_box[1] + 275), spec["expiry"], fill=(20, 20, 20), font=font_v)

            # MRZ line if passport
            if "mrz" in spec:
                draw.rectangle([card_box[0] + 10, card_box[3] - 110, card_box[2] - 10, card_box[3] - 10], fill=(245, 245, 248))
                draw.text((card_box[0] + 25, card_box[3] - 95), spec["mrz"], fill=(10, 10, 10), font=font_mrz)

            # Camera noise
            frame_arr = np.array(frame_img, dtype=np.int16)
            noise = np.random.RandomState(f_seed).normal(0, 2.5, frame_arr.shape).astype(np.int16)
            frame_arr = np.clip(frame_arr + noise, 0, 255).astype(np.uint8)
            final_frame = Image.fromarray(frame_arr)

            frame_filename = f"{clip_id}_frame_{f_idx+1:02d}.jpg"
            frame_path = os.path.join(img_dir, frame_filename)
            final_frame.save(frame_path, "JPEG", quality=92)

            gt_data = {
                "doc_type": spec["type"],
                "doc_category": spec["category"],
                "country": spec["country"],
                "source_clip_id": clip_id,
                "frame_id": f_idx + 1,
                "quad": quad_corners,
                "attack_type": "genuine",
                "is_presentation_attack": False,
                "fields": {
                    "name": {"value": spec["name"]},
                    "dob": {"value": spec["dob"]},
                    "document_number": {"value": spec["doc_num"]},
                    "issue_date": {"value": spec["issue"]},
                    "expiry_date": {"value": spec["expiry"]},
                },
            }
            gt_path = os.path.join(gt_dir, f"{clip_id}_frame_{f_idx+1:02d}.json")
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt_data, f, indent=2)

            total_samples += 1

    return {
        "status": "success",
        "output_dir": output_dir,
        "n_clips": n_clips,
        "total_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# MIDV-2020 Dataset Loader & Synthesizer (Real-World & Presentation Attacks)
# ---------------------------------------------------------------------------

def load_midv2020_dataset(
    data_dir: str = "data/midv2020",
    sample_fallback_dir: str = "data/midv2020_sample",
) -> List[Dict[str, Any]]:
    """
    Load real-world MIDV-2020 dataset samples (including Presentation Attacks).
    Falls back to sample_fallback_dir if local raw dataset is not downloaded.
    """
    target_dir = data_dir if os.path.exists(data_dir) and os.listdir(data_dir) else sample_fallback_dir

    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        generate_sample_midv2020_dataset(output_dir=target_dir, n_clips=6, frames_per_clip=3)

    return load_midv500_dataset(data_dir=target_dir, sample_fallback_dir=sample_fallback_dir)


def generate_sample_midv2020_dataset(
    output_dir: str = "data/midv2020_sample",
    n_clips: int = 6,
    frames_per_clip: int = 3,
    seed: int = 142,
) -> Dict[str, Any]:
    """
    Generate MIDV-2020 benchmark fixtures containing mobile camera artifacts:
    - Screen Replay Moiré presentation attack
    - Color Photo Print spoof
    - Harsh optical glare and low-angle skew
    - Genuine mobile captures
    """
    ensure_dirs(output_dir)
    set_seed(seed)

    scenarios = [
        {"type": "ita_id_genuine", "country": "ITA", "category": "id_card", "attack": "genuine", "name": "MARCO ROSSI", "doc_num": "CA1029384", "dob": "19/07/1987", "issue": "12/04/2019", "expiry": "12/04/2029"},
        {"type": "gbr_pass_screen_replay", "country": "GBR", "category": "passport", "attack": "screen_replay", "name": "OLIVER SMITH", "doc_num": "948201928", "dob": "08/11/1992", "issue": "01/09/2021", "expiry": "01/09/2031"},
        {"type": "esp_id_print_spoof", "country": "ESP", "category": "id_card", "attack": "print_spoof", "name": "SOFIA ALONSO", "doc_num": "84920192X", "dob": "24/03/1984", "issue": "18/06/2017", "expiry": "18/06/2027"},
        {"type": "deu_id_glare_mobile", "country": "DEU", "category": "id_card", "attack": "genuine", "name": "LUKAS WEBER", "doc_num": "L89201948", "dob": "14/02/1991", "issue": "20/10/2020", "expiry": "20/10/2030"},
        {"type": "usa_pass_photo_swap", "country": "USA", "category": "passport", "attack": "photo_swap", "name": "EMILY DAVIS", "doc_num": "492019482", "dob": "05/06/1993", "issue": "15/03/2021", "expiry": "14/03/2031"},
        {"type": "fra_pass_text_edit", "country": "FRA", "category": "passport", "attack": "text_edit", "name": "LUCAS MOREAU", "doc_num": "2001948201", "dob": "11/09/1980", "issue": "05/05/2017", "expiry": "05/05/2027"},
    ]

    total_samples = 0

    for i in range(min(n_clips, len(scenarios))):
        sc = scenarios[i]
        clip_id = f"clip_{i+1:02d}_{sc['type']}"
        clip_dir = os.path.join(output_dir, clip_id)
        img_dir = os.path.join(clip_dir, "images")
        gt_dir = os.path.join(clip_dir, "ground_truth")
        ensure_dirs(img_dir)
        ensure_dirs(gt_dir)

        is_pa = sc["attack"] in ["screen_replay", "print_spoof"]

        for f_idx in range(frames_per_clip):
            f_seed = seed + i * 100 + f_idx
            frame_rng = random.Random(f_seed)

            canvas_w, canvas_h = 960, 640
            frame_img = Image.new("RGB", (canvas_w, canvas_h), (210, 215, 220))
            draw = ImageDraw.Draw(frame_img)

            # Quad corner definition
            quad_corners = [
                [80 + frame_rng.randint(-8, 8), 60 + frame_rng.randint(-8, 8)],
                [880 + frame_rng.randint(-8, 8), 75 + frame_rng.randint(-8, 8)],
                [865 + frame_rng.randint(-8, 8), 570 + frame_rng.randint(-8, 8)],
                [95 + frame_rng.randint(-8, 8), 555 + frame_rng.randint(-8, 8)],
            ]

            card_box = [90, 70, 870, 560]
            bg_lum = frame_rng.randint(215, 238)
            draw.rectangle(card_box, fill=(bg_lum, bg_lum - 3, bg_lum - 6), outline=(85, 85, 85), width=2)
            draw.rectangle([card_box[0] + 5, card_box[1] + 5, card_box[2] - 5, card_box[1] + 50], fill=(35, 55, 90))

            try:
                font_h = ImageFont.truetype("arialbd.ttf", 18)
                font_v = ImageFont.truetype("arial.ttf", 16)
                font_l = ImageFont.truetype("arial.ttf", 12)
            except Exception:
                font_h = ImageFont.load_default()
                font_v = font_h
                font_l = font_h

            title = f"{sc['country']} {sc['category'].upper()}"
            draw.text((card_box[0] + 20, card_box[1] + 15), title, fill=(255, 255, 255), font=font_h)

            photo_rect = [card_box[0] + 25, card_box[1] + 70, card_box[0] + 190, card_box[1] + 270]
            draw.rectangle(photo_rect, fill=(160, 170, 180), outline=(70, 70, 70), width=2)
            draw.text((photo_rect[0] + 35, photo_rect[1] + 85), "PHOTO", fill=(70, 70, 70), font=font_h)

            draw.text((card_box[0] + 220, card_box[1] + 75), "Full Name", fill=(90, 90, 90), font=font_l)
            draw.text((card_box[0] + 220, card_box[1] + 95), sc["name"], fill=(20, 20, 20), font=font_v)

            draw.text((card_box[0] + 220, card_box[1] + 135), "Date of Birth", fill=(90, 90, 90), font=font_l)
            draw.text((card_box[0] + 220, card_box[1] + 155), sc["dob"], fill=(20, 20, 20), font=font_v)

            draw.text((card_box[0] + 220, card_box[1] + 195), "Document Number", fill=(90, 90, 90), font=font_l)
            draw.text((card_box[0] + 220, card_box[1] + 215), sc["doc_num"], fill=(20, 20, 20), font=font_v)

            draw.text((card_box[0] + 220, card_box[1] + 255), "Issue Date", fill=(90, 90, 90), font=font_l)
            draw.text((card_box[0] + 220, card_box[1] + 275), sc["issue"], fill=(20, 20, 20), font=font_v)

            draw.text((card_box[0] + 450, card_box[1] + 255), "Expiry Date", fill=(90, 90, 90), font=font_l)
            draw.text((card_box[0] + 450, card_box[1] + 275), sc["expiry"], fill=(20, 20, 20), font=font_v)

            frame_arr = np.array(frame_img, dtype=np.float32)

            # Apply specific mobile presentation attack / artifact simulation
            if sc["attack"] == "screen_replay":
                # Add high-frequency 2D periodic Moiré lattice pattern
                y_coords, x_coords = np.mgrid[0:canvas_h, 0:canvas_w]
                moire_pattern = 18.0 * np.sin(x_coords * 0.45 + y_coords * 0.35)
                frame_arr[:, :, 0] += moire_pattern
                frame_arr[:, :, 2] -= moire_pattern * 0.5
            elif sc["attack"] == "print_spoof":
                # Simulated halftone printing grain
                grain = np.random.RandomState(f_seed).normal(0, 10.0, frame_arr.shape)
                frame_arr += grain

            # Baseline sensor noise
            base_noise = np.random.RandomState(f_seed).normal(0, 2.5, frame_arr.shape)
            frame_arr = np.clip(frame_arr + base_noise, 0, 255).astype(np.uint8)

            if sc["attack"] == "photo_swap":
                # Splicing boundary artifact in photo area
                cv2.rectangle(frame_arr, (photo_rect[0], photo_rect[1]), (photo_rect[2], photo_rect[3]), (130, 140, 150), -1)
                cv2.putText(frame_arr, "SWAPPED", (photo_rect[0] + 15, photo_rect[1] + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
            elif sc["attack"] == "text_edit":
                # Splicing artifact in name area
                cv2.rectangle(frame_arr, (card_box[0] + 215, card_box[1] + 90), (card_box[0] + 450, card_box[1] + 120), (245, 245, 245), -1)
                cv2.putText(frame_arr, "ALTERED NAME", (card_box[0] + 220, card_box[1] + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)

            final_frame = Image.fromarray(frame_arr)

            frame_filename = f"{clip_id}_frame_{f_idx+1:02d}.jpg"
            frame_path = os.path.join(img_dir, frame_filename)
            final_frame.save(frame_path, "JPEG", quality=90)

            gt_data = {
                "doc_type": sc["type"],
                "doc_category": sc["category"],
                "country": sc["country"],
                "source_clip_id": clip_id,
                "frame_id": f_idx + 1,
                "quad": quad_corners,
                "attack_type": sc["attack"],
                "is_presentation_attack": is_pa,
                "fields": {
                    "name": {"value": sc["name"]},
                    "dob": {"value": sc["dob"]},
                    "document_number": {"value": sc["doc_num"]},
                    "issue_date": {"value": sc["issue"]},
                    "expiry_date": {"value": sc["expiry"]},
                },
            }
            gt_path = os.path.join(gt_dir, f"{clip_id}_frame_{f_idx+1:02d}.json")
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt_data, f, indent=2)

            total_samples += 1

    return {
        "status": "success",
        "output_dir": output_dir,
        "n_clips": n_clips,
        "total_samples": total_samples,
    }
