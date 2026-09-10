"""
ForgeLens-X — Milestone 4: JPEG EXIF & Image Provenance Forensics
================================================================
Extracts and audits digital image metadata to detect software manipulation
signatures (Photoshop, GIMP, Canva, etc.), chronological timestamp skews,
and camera hardware provenance.

Adheres to non-punitive conditional availability:
- Scanned, synthetic, or web-optimized images lacking EXIF markers return
  'STRIPPED_OR_ABSENT' with status 'NOT_APPLICABLE' rather than 'FAIL'.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from PIL import ExifTags, Image
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


def extract_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive metadata, EXIF tags, and filesystem timestamps from an image.
    """
    result: Dict[str, Any] = {
        "file_path": image_path,
        "file_name": os.path.basename(image_path) if image_path else "",
        "file_size_bytes": 0,
        "file_mtime": None,
        "file_ctime": None,
        "has_exif": False,
        "exif_tags": {},
        "software": None,
        "make": None,
        "model": None,
        "datetime_original": None,
        "datetime_digitized": None,
        "datetime_modified": None,
        "raw_strings": [],
    }

    if not image_path or not os.path.exists(image_path):
        return result

    try:
        stat = os.stat(image_path)
        result["file_size_bytes"] = stat.st_size
        result["file_mtime"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        result["file_ctime"] = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    try:
        with Image.open(image_path) as img:
            result["image_format"] = img.format
            result["image_mode"] = img.mode
            result["image_size"] = img.size

            raw_exif = img._getexif() if hasattr(img, "_getexif") else None
            if raw_exif:
                result["has_exif"] = True
                named_tags = {}
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Convert bytes to string representation if needed
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", errors="replace").strip("\x00")
                        except Exception:
                            value = str(value)
                    elif isinstance(value, tuple):
                        value = list(value)
                    named_tags[tag_name] = value

                result["exif_tags"] = named_tags
                result["software"] = named_tags.get("Software")
                result["make"] = named_tags.get("Make")
                result["model"] = named_tags.get("Model")
                result["datetime_original"] = named_tags.get("DateTimeOriginal")
                result["datetime_digitized"] = named_tags.get("DateTimeDigitized")
                result["datetime_modified"] = named_tags.get("DateTime")

            # Check info dict for XMP or other markers
            if hasattr(img, "info") and img.info:
                info_software = img.info.get("software") or img.info.get("Software")
                if info_software and not result["software"]:
                    result["software"] = str(info_software)
    except Exception as e:
        result["read_error"] = str(e)

    # Check raw file bytes for signature strings if EXIF wasn't cleanly parsed
    try:
        with open(image_path, "rb") as f:
            header_sample = f.read(65536)  # Inspect first 64KB for APP markers
            for sw_marker in [b"Photoshop", b"GIMP", b"Canva", b"Lightroom", b"Paint.NET", b"CorelDRAW"]:
                if sw_marker.lower() in header_sample.lower():
                    result["raw_strings"].append(sw_marker.decode("ascii"))
    except Exception:
        pass

    return result


def _parse_exif_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse standard EXIF timestamp format 'YYYY:MM:DD HH:MM:SS'."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        return datetime.strptime(ts_str.strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        try:
            return datetime.strptime(ts_str.strip()[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def audit_metadata_provenance(
    metadata: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Audit extracted metadata against forensic tampering indicators.

    Checks:
        1. Software Fingerprint Blacklist (Photoshop, GIMP, Canva, etc.)
        2. Timestamp Chronology & Discrepancy
        3. Camera Hardware Presence vs Stripped Metadata
    """
    if config is None:
        config = _load_m4_config()

    m_cfg = config.get("metadata_forensics", {})
    blacklist = m_cfg.get("software_blacklist", [
        "photoshop", "gimp", "canva", "lightroom", "paint.net",
        "coreldraw", "snapseed", "pixelmator", "affinity photo"
    ])
    max_skew_sec = m_cfg.get("max_timestamp_skew_seconds", 86400)

    # 1. Software Signature Check
    detected_sw = None
    sw_field = metadata.get("software")
    raw_sw = metadata.get("raw_strings", [])

    candidates_to_check = []
    if sw_field:
        candidates_to_check.append(str(sw_field))
    candidates_to_check.extend(raw_sw)

    for cand in candidates_to_check:
        cand_lower = cand.lower()
        for banned in blacklist:
            if banned in cand_lower:
                detected_sw = cand
                break
        if detected_sw:
            break

    if detected_sw:
        sw_status = "FAIL"
        sw_detail = f"Suspect photo-manipulation software detected in metadata: '{detected_sw}'"
    elif sw_field:
        sw_status = "PASS"
        sw_detail = f"Software tag present and clean: '{sw_field}'"
    else:
        sw_status = "NOT_APPLICABLE"
        sw_detail = "No editing software signature detected in headers."

    # 2. Timestamp Chronology Check
    dt_orig_raw = metadata.get("datetime_original")
    dt_digi_raw = metadata.get("datetime_digitized")
    dt_mod_raw = metadata.get("datetime_modified")

    dt_orig = _parse_exif_timestamp(dt_orig_raw)
    dt_digi = _parse_exif_timestamp(dt_digi_raw)
    dt_mod = _parse_exif_timestamp(dt_mod_raw)

    ts_status = "NOT_APPLICABLE"
    ts_detail = "Insufficient EXIF timestamps present to audit chronology."
    ts_evidence = {
        "datetime_original": dt_orig_raw,
        "datetime_digitized": dt_digi_raw,
        "datetime_modified": dt_mod_raw,
    }

    if dt_orig and dt_digi:
        skew = (dt_digi - dt_orig).total_seconds()
        ts_evidence["skew_seconds"] = skew
        if skew < 0:
            ts_status = "FAIL"
            ts_detail = f"Temporal inversion: DateTimeDigitized ({dt_digi_raw}) precedes DateTimeOriginal ({dt_orig_raw})."
        elif skew > max_skew_sec:
            ts_status = "FAIL"
            ts_detail = f"Excessive timestamp skew ({skew:.0f}s > {max_skew_sec}s) between original capture and digitization."
        else:
            ts_status = "PASS"
            ts_detail = f"Consistent timestamp chronology (skew: {skew:.0f}s within allowed tolerance)."
    elif dt_orig or dt_digi or dt_mod:
        ts_status = "PASS"
        ts_detail = f"Single EXIF timestamp available ({dt_orig_raw or dt_digi_raw or dt_mod_raw}), no temporal contradiction."

    # 3. EXIF Presence Check
    has_exif = metadata.get("has_exif", False)
    make = metadata.get("make")
    model = metadata.get("model")

    if has_exif and (make or model):
        exif_status = "PASS"
        exif_detail = f"Hardware capture provenance verified: Make='{make}', Model='{model}'"
    elif has_exif:
        exif_status = "PASS"
        exif_detail = "EXIF markers present without hardware make/model specification."
    else:
        exif_status = "NOT_APPLICABLE"
        exif_detail = "Image has stripped or absent EXIF markers (standard for scans, web assets, or synthetic IDs)."

    # Composite Verdict
    if sw_status == "FAIL":
        verdict = "SUSPECT_SOFTWARE_FINGERPRINT"
        is_tampered = True
    elif ts_status == "FAIL":
        verdict = "TIMESTAMP_ANOMALY"
        is_tampered = True
    elif exif_status == "PASS" and sw_status == "PASS":
        verdict = "AUTHENTIC_CAMERA_ORIGIN"
        is_tampered = False
    elif not has_exif and not detected_sw:
        verdict = "STRIPPED_OR_ABSENT"
        is_tampered = False
    else:
        verdict = "CLEAN_PROVENANCE"
        is_tampered = False

    return {
        "provenance_verdict": verdict,
        "is_tampered": is_tampered,
        "software_detected": detected_sw,
        "checks": {
            "software_signature": {
                "check": "software_signature",
                "status": sw_status,
                "detail": sw_detail,
                "evidence": {"software": detected_sw or sw_field},
            },
            "timestamp_chronology": {
                "check": "timestamp_chronology",
                "status": ts_status,
                "detail": ts_detail,
                "evidence": ts_evidence,
            },
            "exif_presence": {
                "check": "exif_presence",
                "status": exif_status,
                "detail": exif_detail,
                "evidence": {"has_exif": has_exif, "make": make, "model": model},
            },
        },
        "raw_metadata": metadata,
    }
