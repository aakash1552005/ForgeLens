"""
ForgeLens-X — Milestone 3: OCR & Structured Field Extraction Module
=====================================================================
Unified interface for optical character recognition on identity documents.
Extracts structured identity fields:
    - name
    - dob (Date of Birth)
    - document_number
    - issue_date
    - expiry_date

Supported Engines:
    - RapidOCR (PaddleOCR PP-OCRv4 ONNX, CPU-optimized, Primary)
    - Pytesseract (Tesseract OCR, Secondary Fallback)
    - Template/Geometry-Guided Fallback (Resilient Native Baseline)

Invariants:
    - Output schema: {"field": ..., "value": ..., "confidence": 0.0-1.0, "bbox": [x1, y1, x2, y2], "status": ...}
    - Low-confidence / unextractable fields return status="UNKNOWN" without crashing.
    - Missing information is not automatically treated as fraud.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Global Engine Cache
# ---------------------------------------------------------------------------

_RAPID_OCR_INSTANCE = None
_RAPID_OCR_INITIALIZED = False


def get_ocr_engine():
    """Lazy initialize RapidOCR (PaddleOCR ONNX engine)."""
    global _RAPID_OCR_INSTANCE, _RAPID_OCR_INITIALIZED
    if _RAPID_OCR_INITIALIZED:
        return _RAPID_OCR_INSTANCE

    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        _RAPID_OCR_INSTANCE = RapidOCR()
        _RAPID_OCR_INITIALIZED = True
    except Exception:
        _RAPID_OCR_INSTANCE = None
        _RAPID_OCR_INITIALIZED = True

    return _RAPID_OCR_INSTANCE


def load_image_for_ocr(image_input: Union[str, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
    """Load image into BGR numpy array for OCR processing."""
    if image_input is None:
        return None

    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            return None
        img = cv2.imread(image_input)
        return img

    if isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return image_input

    return None


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Adaptive preprocessing for OCR:
    Enhances contrast using CLAHE on luminance channel while preserving edge detail.
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception:
        return img_bgr


# ---------------------------------------------------------------------------
# Text Extraction Core
# ---------------------------------------------------------------------------

def extract_text_lines(
    image_input: Union[str, np.ndarray, Image.Image],
    engine: str = "auto",
    use_preprocessing: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract all raw text lines from an image with bounding boxes and confidences.

    Args:
        image_input: file path, numpy BGR array, or PIL Image
        engine: 'auto', 'rapidocr', 'tesseract'

    Returns:
        List of dicts:
        [
            {
                "text": "Aryan Maharaj",
                "confidence": 0.92,
                "bbox": [x1, y1, x2, y2],
                "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            },
            ...
        ]
    """
    img_bgr = load_image_for_ocr(image_input)
    if img_bgr is None or img_bgr.size == 0:
        return []

    if use_preprocessing:
        img_bgr = preprocess_for_ocr(img_bgr)

    lines = []

    # 1. Primary Engine: RapidOCR (PaddleOCR ONNX)
    if engine in ["auto", "rapidocr", "paddleocr"]:
        rapid_engine = get_ocr_engine()
        if rapid_engine is not None:
            try:
                res, _ = rapid_engine(img_bgr)
                if res:
                    for item in res:
                        # item format: [polygon_coords, text, confidence_str]
                        poly = item[0]
                        text = str(item[1]).strip()
                        try:
                            conf = float(item[2])
                        except (ValueError, TypeError):
                            conf = 0.50

                        pts = np.array(poly, dtype=np.float32)
                        x1 = float(np.min(pts[:, 0]))
                        y1 = float(np.min(pts[:, 1]))
                        x2 = float(np.max(pts[:, 0]))
                        y2 = float(np.max(pts[:, 1]))

                        lines.append({
                            "text": text,
                            "confidence": round(conf, 3),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "polygon": [[float(p[0]), float(p[1])] for p in poly],
                        })
                    return lines
            except Exception:
                pass

    # 2. Secondary Engine: Pytesseract Fallback
    if engine in ["auto", "tesseract"] and not lines:
        try:
            import pytesseract  # type: ignore
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf_val = float(data["conf"][i])
                if text and conf_val > 0:
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]
                    lines.append({
                        "text": text,
                        "confidence": round(conf_val / 100.0, 3),
                        "bbox": [x, y, x + w, y + h],
                        "polygon": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    })
            if lines:
                return lines
        except Exception:
            pass

    return lines


# ---------------------------------------------------------------------------
# Structured Identity Field Extraction
# ---------------------------------------------------------------------------

DATE_REGEX = re.compile(r"\b(\d{2}[/.-]\d{2}[/.-]\d{4})\b")
DOC_NUM_REGEX = re.compile(r"\b(FGL-\d{6}-\d{2}|[A-Z0-9]{8,15})\b", re.IGNORECASE)


def _clean_field_value(raw: str, field_type: str) -> str:
    """Standardize field values and strip OCR noise."""
    if not raw:
        return ""
    val = raw.strip()
    if field_type == "date":
        # Search for date pattern first if OCR included label text
        m = DATE_REGEX.search(val)
        if m:
            val = m.group(1)
        # Normalize delimiters (dots, dashes) to slash
        val = re.sub(r"[.-]", "/", val)
    elif field_type == "document_number":
        m = DOC_NUM_REGEX.search(val)
        if m:
            val = m.group(1)
        val = val.replace(" ", "").upper()
    elif field_type == "text":
        # Strip label prefixes if any remain
        val = re.sub(r"^(Full\s*Name|Name)\s*[:\s]*", "", val, flags=re.IGNORECASE).strip()
        # Insert space before capitalized word if glued (e.g. AryanMaharaj -> Aryan Maharaj)
        val = re.sub(r"([a-z])([A-Z])", r"\1 \2", val)
    return val


def _find_overlapping_ocr_text(
    ocr_lines: List[Dict[str, Any]],
    target_bbox: List[int],
    min_overlap: float = 0.20,
) -> Tuple[Optional[str], float, Optional[List[int]]]:
    """Find text lines that intersect with a designated target bounding box."""
    tx1, ty1, tx2, ty2 = target_bbox
    t_area = max((tx2 - tx1) * (ty2 - ty1), 1)

    matched_texts = []
    confidences = []
    matched_bboxes = []

    for item in ocr_lines:
        bx1, by1, bx2, by2 = item["bbox"]
        # Compute intersection
        ix1 = max(tx1, bx1)
        iy1 = max(ty1, by1)
        ix2 = min(tx2, bx2)
        iy2 = min(ty2, by2)

        if ix2 > ix1 and iy2 > iy1:
            i_area = (ix2 - ix1) * (iy2 - iy1)
            b_area = max((bx2 - bx1) * (by2 - by1), 1)
            overlap = i_area / min(t_area, b_area)
            if overlap >= min_overlap:
                matched_texts.append(item["text"])
                confidences.append(item["confidence"])
                matched_bboxes.append(item["bbox"])

    if not matched_texts:
        return None, 0.0, None

    combined_text = " ".join(matched_texts).strip()
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    # Union bounding box
    ux1 = min(b[0] for b in matched_bboxes)
    uy1 = min(b[1] for b in matched_bboxes)
    ux2 = max(b[2] for b in matched_bboxes)
    uy2 = max(b[3] for b in matched_bboxes)

    return combined_text, round(avg_conf, 3), [ux1, uy1, ux2, uy2]


def extract_structured_fields(
    image_input: Union[str, np.ndarray, Image.Image],
    template_type: str = "forgelensia",
    engine: str = "auto",
    field_bboxes: Optional[Dict[str, List[int]]] = None,
    use_preprocessing: bool = True,
    tamper_signals: Optional[Dict[str, Any]] = None,
    tamper_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Extract canonical identity document fields:
        name, dob, document_number, issue_date, expiry_date

    Args:
        image_input: document image path or array
        template_type: document template model ('forgelensia', 'generic')
        engine: OCR engine ('auto', 'rapidocr', 'tesseract')
        field_bboxes: optional bounding box overrides per field
        use_preprocessing: whether to apply adaptive contrast enhancement
        tamper_signals: optional M1 tamper forensic signals for cross-modal linking
        tamper_mask: optional binary tamper mask

    Returns:
        Structured dictionary matching canonical specification:
        {
            "fields": {
                "name": {"value": ..., "confidence": ..., "bbox": ..., "status": ...},
                "dob": {...},
                "document_number": {...},
                "issue_date": {...},
                "expiry_date": {...},
            },
            "ocr_lines": [...],
            "chronology_audit": {...},
            "mrz_data": {...} or None,
            "mrz_cross_validation": {...},
            "tamper_correlation": {...},
            "engine": str,
            "time_seconds": float,
        }
    """
    start_time = time.time()
    img_bgr = load_image_for_ocr(image_input)

    # Fallback template box layout for Forgelensia standard credentials
    default_boxes = {
        "name": [225, 95, 550, 135],
        "dob": [225, 150, 390, 185],
        "document_number": [225, 205, 460, 240],
        "issue_date": [225, 260, 390, 295],
        "expiry_date": [440, 260, 610, 295],
    }
    target_boxes = field_bboxes if field_bboxes else default_boxes

    # Initialize empty structured field schema
    fields: Dict[str, Dict[str, Any]] = {}
    for fname in ["name", "dob", "document_number", "issue_date", "expiry_date"]:
        fields[fname] = {
            "field": fname,
            "value": None,
            "confidence": 0.0,
            "bbox": None,
            "status": "UNKNOWN",
            "raw_text": "",
        }

    if img_bgr is None or img_bgr.size == 0:
        return {
            "fields": fields,
            "ocr_lines": [],
            "chronology_audit": {"chronology_valid": False, "status": "NO_IMAGE"},
            "mrz_data": None,
            "mrz_cross_validation": {"cross_validation_status": "NO_IMAGE"},
            "tamper_correlation": {"tampered_fields_count": 0, "integrity_verdict": "NO_IMAGE"},
            "engine": engine,
            "error": "invalid_input",
            "time_seconds": round(time.time() - start_time, 3),
        }

    # 1. Run OCR line extraction
    ocr_lines = extract_text_lines(img_bgr, engine=engine, use_preprocessing=use_preprocessing)

    # 2. Strategy A: Spatial Bounding Box Intersection
    for fname, t_box in target_boxes.items():
        text, conf, bbox = _find_overlapping_ocr_text(ocr_lines, t_box)
        if text:
            # Strip label artifacts if OCR merged label into value (e.g. "FullName Aryan Maharaj")
            cleaned_val = text
            if fname == "name":
                cleaned_val = re.sub(r"^(Full\s*Name\s*[:\s]*)", "", text, flags=re.IGNORECASE).strip()
            elif fname == "dob":
                cleaned_val = re.sub(r"^(Date\s*of\s*Birth\s*[:\s]*)", "", text, flags=re.IGNORECASE).strip()
            elif fname == "document_number":
                cleaned_val = re.sub(r"^(Document\s*Number\s*[:\s]*)", "", text, flags=re.IGNORECASE).strip()
            elif fname == "issue_date":
                cleaned_val = re.sub(r"^(Issue\s*Date\s*[:\s]*)", "", text, flags=re.IGNORECASE).strip()
            elif fname == "expiry_date":
                cleaned_val = re.sub(r"^(Expiry\s*Date\s*[:\s]*)", "", text, flags=re.IGNORECASE).strip()

            ftype = "date" if "date" in fname or fname == "dob" else "document_number" if fname == "document_number" else "text"
            final_val = _clean_field_value(cleaned_val, ftype)

            status = "EXTRACTED" if conf >= 0.50 else "LOW_CONFIDENCE"

            fields[fname] = {
                "field": fname,
                "value": final_val if final_val else None,
                "confidence": conf,
                "bbox": bbox,
                "status": status if final_val else "UNKNOWN",
                "raw_text": text,
            }

    # 3. Strategy B: Semantic Pattern Recovery for Any Missing Fields
    # Check for document number pattern across all OCR lines if still UNKNOWN
    if fields["document_number"]["status"] == "UNKNOWN":
        for item in ocr_lines:
            m = DOC_NUM_REGEX.search(item["text"])
            if m:
                fields["document_number"] = {
                    "field": "document_number",
                    "value": m.group(1),
                    "confidence": item["confidence"],
                    "bbox": item["bbox"],
                    "status": "EXTRACTED" if item["confidence"] >= 0.50 else "LOW_CONFIDENCE",
                    "raw_text": item["text"],
                }
                break

    # Check for date patterns across all OCR lines
    date_matches = []
    for item in ocr_lines:
        for m in DATE_REGEX.finditer(item["text"]):
            date_matches.append((m.group(1), item["confidence"], item["bbox"]))

    # If date fields are still UNKNOWN, assign chronological dates (DOB earliest, Issue next, Expiry latest)
    unfilled_date_fields = [f for f in ["dob", "issue_date", "expiry_date"] if fields[f]["status"] == "UNKNOWN"]
    if unfilled_date_fields and date_matches:
        # Sort dates by year
        parsed_dates = []
        for d_str, conf, bbox in date_matches:
            try:
                parts = re.split(r"[/.-]", d_str)
                yr = int(parts[2])
                parsed_dates.append((yr, d_str, conf, bbox))
            except Exception:
                pass
        parsed_dates.sort(key=lambda x: x[0])

        if len(parsed_dates) >= len(unfilled_date_fields):
            for i, fname in enumerate(unfilled_date_fields):
                _, d_val, conf, bbox = parsed_dates[i]
                fields[fname] = {
                    "field": fname,
                    "value": _clean_field_value(d_val, "date"),
                    "confidence": conf,
                    "bbox": bbox,
                    "status": "EXTRACTED" if conf >= 0.50 else "LOW_CONFIDENCE",
                    "raw_text": d_val,
                }

    # 4. Contextual Character Confusion Glyph Repair
    try:
        from src.ocr_postprocess import repair_glyph_confusions
        for fname, f_info in fields.items():
            if f_info.get("value"):
                ftype = "date" if "date" in fname or fname == "dob" else "document_number" if fname == "document_number" else "text"
                repaired_res = repair_glyph_confusions(f_info["value"], ftype)
                repaired = repaired_res[0] if isinstance(repaired_res, tuple) else repaired_res
                if repaired:
                    f_info["value"] = repaired
    except Exception:
        pass

    # 5. Date Chronology & Physical Sanity Audit
    chronology_res = {"chronology_valid": True, "status": "NOT_AUDITED", "anomalies": []}
    try:
        from src.ocr_postprocess import validate_date_chronology
        chronology_res = validate_date_chronology(fields)
    except Exception:
        pass

    # 6. ICAO Doc 9303 MRZ Parsing & Checksum Audit
    mrz_res = None
    mrz_cross_res = {"cross_validation_status": "NO_MRZ_DETECTED", "is_consistent": True}
    try:
        from src.ocr_postprocess import cross_validate_viz_and_mrz, parse_mrz_lines
        mrz_res = parse_mrz_lines(ocr_lines)
        if mrz_res:
            mrz_cross_res = cross_validate_viz_and_mrz(fields, mrz_res)
    except Exception:
        pass

    # 7. Forensic Cross-Modality Tamper Correlation (M1 x M3 Bridge)
    tamper_corr_res = {
        "tampered_fields_count": 0,
        "tampered_field_names": [],
        "integrity_verdict": "NOT_EVALUATED",
    }
    if tamper_signals is not None or tamper_mask is not None:
        try:
            from src.ocr_forensic_bridge import correlate_tamper_with_fields
            tamper_corr_res = correlate_tamper_with_fields(
                extracted_fields=fields,
                tamper_signals=tamper_signals,
                tamper_mask=tamper_mask,
            )
        except Exception:
            pass

    elapsed = round(time.time() - start_time, 3)

    return {
        "fields": fields,
        "ocr_lines": ocr_lines,
        "chronology_audit": chronology_res,
        "mrz_data": mrz_res,
        "mrz_cross_validation": mrz_cross_res,
        "tamper_correlation": tamper_corr_res,
        "engine": "rapidocr" if engine in ["auto", "rapidocr", "paddleocr"] else engine,
        "document_type": template_type,
        "time_seconds": elapsed,
    }
