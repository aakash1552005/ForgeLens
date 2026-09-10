"""
ForgeLens-X — Milestone 3: OCR Post-Processing & Forensic Validation Engine
=============================================================================
Provides production-grade post-processing for identity document OCR:
    1. Contextual character confusion disambiguation (glyph repair)
    2. Document date chronology & physical sanity audit
    3. ICAO Doc 9303 MRZ parser with cyclic [7, 3, 1] modulo-10 checksums
    4. Visual Inspection Zone (VIZ) vs Machine Readable Zone (MRZ) cross-validation
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# 1. Contextual Glyph Repair & Character Disambiguation
# ---------------------------------------------------------------------------

_DATE_CONFUSION_MAP = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1", "!": "1", "]": "1", "[": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8",
    "g": "9", "q": "9",
}

_NUMERIC_CONFUSION_MAP = {
    "O": "0", "o": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8",
}

_ALPHA_CONFUSION_MAP = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "2": "Z",
}


def repair_glyph_confusions(text: Optional[str], field_type: str) -> Tuple[Optional[str], int]:
    """
    Repair systematic optical character recognition glyph confusions
    based on the structural context of the identity field.
    Returns:
        Tuple of (repaired_text, repair_count)
    """
    if text is None:
        return (None, 0)
    if not text:
        return ("", 0)

    val = str(text).strip()
    repair_count = 0

    if field_type in ["date", "dob", "issue_date", "expiry_date"]:
        # Extract potential date portion if label leaked into text
        date_pattern = re.compile(r"([0-9OolI|!SsZzBb]{1,2}[/.-][0-9OolI|!SsZzBb]{1,2}[/.-][0-9OolI|!SsZzBb]{2,4})")
        m = date_pattern.search(val)
        if m:
            val = m.group(1)

        # Substitute numeric confusions
        repaired = []
        for ch in val:
            if ch in _DATE_CONFUSION_MAP:
                repaired.append(_DATE_CONFUSION_MAP[ch])
                repair_count += 1
            elif ch in [".", "-"]:
                repaired.append("/")
            else:
                repaired.append(ch)
        val = "".join(repaired)

        # Ensure valid 2-digit day and month formatting if 1-digit produced
        parts = val.split("/")
        if len(parts) == 3:
            d, m_part, y = parts
            d = d.zfill(2)
            m_part = m_part.zfill(2)
            if len(y) == 2:
                # Century assumption: > 40 -> 1900s, <= 40 -> 2000s
                y = ("19" if int(y) > 40 else "20") + y
            val = f"{d}/{m_part}/{y}"

    elif field_type in ["document_number", "doc_num"]:
        # Check for Forgelensia standard format: FGL-XXXXXX-XX
        fgl_m = re.search(r"FGL[-_\s]?([0-9A-Z]{6})[-_\s]?([0-9A-Z]{2})", val, re.IGNORECASE)
        if fgl_m:
            mid = fgl_m.group(1)
            tail = fgl_m.group(2)
            # Both mid and tail should be purely digits in Forgelensia schema
            mid_rep = []
            for c in mid:
                if c in _NUMERIC_CONFUSION_MAP:
                    mid_rep.append(_NUMERIC_CONFUSION_MAP[c])
                    repair_count += 1
                else:
                    mid_rep.append(c)
            tail_rep = []
            for c in tail:
                if c in _NUMERIC_CONFUSION_MAP:
                    tail_rep.append(_NUMERIC_CONFUSION_MAP[c])
                    repair_count += 1
                else:
                    tail_rep.append(c)
            val = f"FGL-{''.join(mid_rep)}-{''.join(tail_rep)}"
        else:
            # Generic alphanumeric document number
            val = val.replace(" ", "").upper()

    elif field_type in ["text", "name", "full_name"]:
        # Full Name cleanup
        val = re.sub(r"^(Full\s*Name|Name|Given\s*Name|Surname)\s*[:\s]*", "", val, flags=re.IGNORECASE).strip()
        # Repair accidental digits embedded inside words (e.g. "J0HN" or "Ary4n")
        repaired_words = []
        for word in val.split():
            clean_chars = []
            for c in word:
                if c in _ALPHA_CONFUSION_MAP:
                    clean_chars.append(_ALPHA_CONFUSION_MAP[c])
                    repair_count += 1
                else:
                    clean_chars.append(c)
            clean_word = "".join(clean_chars)
            clean_word = re.sub(r"([a-z])([A-Z])", r"\1 \2", clean_word)
            repaired_words.append(clean_word)
        val = " ".join(repaired_words).strip()

    return (val, repair_count)


# ---------------------------------------------------------------------------
# 2. Date Chronology & Physical Integrity Audit
# ---------------------------------------------------------------------------

def parse_date_flexible(d_str: Optional[str]) -> Optional[datetime]:
    """Parse date from multiple canonical formats."""
    if not d_str:
        return None
    cleaned = str(d_str).strip()
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%y"]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


# Backward-compatible internal alias
_parse_date_flexible = parse_date_flexible


def validate_date_chronology(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Forensically audit the logical chronology of document dates:
        1. Calendar validity
        2. DOB < Issue Date
        3. Issue Date < Expiry Date
        4. Minimum Age at Issue >= 0 (Adult ID guideline >= 18)
        5. Credential Validity Window <= 25 years
        6. Issue Date <= Current Date + 30 days tolerance
    """
    # Extract raw or structured date values
    def _get_val(f_name: str) -> Optional[str]:
        item = fields.get(f_name)
        if isinstance(item, dict):
            return item.get("value")
        return item

    dob_str = _get_val("dob")
    issue_str = _get_val("issue_date")
    expiry_str = _get_val("expiry_date")

    dob_dt = parse_date_flexible(dob_str)
    issue_dt = parse_date_flexible(issue_str)
    expiry_dt = parse_date_flexible(expiry_str)

    anomalies = []
    errors = []
    has_calendar_error = False

    # Check if all dates are missing
    if not dob_str and not issue_str and not expiry_str:
        return {
            "chronology_valid": True,
            "status": "DATES_UNAVAILABLE",
            "errors": [],
            "anomalies": [],
            "age_at_issue_years": None,
            "validity_years": None,
            "validity_period_years": None,
            "parsed_dates": {"dob": None, "issue_date": None, "expiry_date": None},
        }

    # Check calendar parse failures on non-empty inputs
    if dob_str and not dob_dt:
        anomalies.append(f"Invalid calendar date for Date of Birth: '{dob_str}'")
        errors.append("IMPOSSIBLE_CALENDAR_DATE")
        has_calendar_error = True
    if issue_str and not issue_dt:
        anomalies.append(f"Invalid calendar date for Issue Date: '{issue_str}'")
        errors.append("IMPOSSIBLE_CALENDAR_DATE")
        has_calendar_error = True
    if expiry_str and not expiry_dt:
        anomalies.append(f"Invalid calendar date for Expiry Date: '{expiry_str}'")
        errors.append("IMPOSSIBLE_CALENDAR_DATE")
        has_calendar_error = True

    age_at_issue = None
    validity_years = None

    if dob_dt and issue_dt:
        age_days = (issue_dt - dob_dt).days
        age_at_issue = round(age_days / 365.25, 1)
        if issue_dt < dob_dt:
            anomalies.append(f"Chronology inversion: Issue date ({issue_dt.strftime('%d/%m/%Y')}) precedes birth date ({dob_dt.strftime('%d/%m/%Y')})")
            errors.append("ISSUE_PRE_DOB")
        elif age_at_issue < 0:
            anomalies.append(f"Negative age at issuance: {age_at_issue} years")
            errors.append("NEGATIVE_AGE_AT_ISSUE")

    if issue_dt and expiry_dt:
        val_days = (expiry_dt - issue_dt).days
        validity_years = round(val_days / 365.25, 1)
        if expiry_dt <= issue_dt:
            anomalies.append(f"Chronology inversion: Expiry date ({expiry_dt.strftime('%d/%m/%Y')}) precedes or equals issue date ({issue_dt.strftime('%d/%m/%Y')})")
            errors.append("EXPIRY_PRE_ISSUE")
        elif validity_years > 25.0:
            anomalies.append(f"Suspiciously long validity window: {validity_years} years (expected <= 25 years)")
            errors.append("EXCESSIVE_VALIDITY")

    if dob_dt and expiry_dt and expiry_dt < dob_dt:
        anomalies.append("Chronology inversion: Expiry date precedes date of birth")
        errors.append("EXPIRY_PRE_DOB")

    # Check future issue date
    now = datetime.now()
    if issue_dt and (issue_dt - now).days > 30:
        anomalies.append(f"Anachronistic issue date: Document issued in the future ({issue_dt.strftime('%d/%m/%Y')})")
        errors.append("FUTURE_ISSUE_DATE")

    # Determine verdict status
    if has_calendar_error:
        status = "IMPOSSIBLE_CALENDAR_DATE"
        chronology_valid = False
    elif anomalies:
        status = "SUSPECT_CHRONOLOGY"
        chronology_valid = False
    elif dob_dt and issue_dt and expiry_dt:
        status = "VERIFIED_CHRONOLOGY"
        chronology_valid = True
    else:
        status = "PARTIAL_DATES_AVAILABLE"
        chronology_valid = True

    return {
        "chronology_valid": chronology_valid,
        "status": status,
        "errors": errors,
        "anomalies": anomalies,
        "age_at_issue_years": age_at_issue,
        "validity_years": validity_years,
        "validity_period_years": validity_years,
        "parsed_dates": {
            "dob": dob_dt.strftime("%d/%m/%Y") if dob_dt else None,
            "issue_date": issue_dt.strftime("%d/%m/%Y") if issue_dt else None,
            "expiry_date": expiry_dt.strftime("%d/%m/%Y") if expiry_dt else None,
        },
    }


# ---------------------------------------------------------------------------
# 3. ICAO Doc 9303 MRZ Parser & Modulo-10 Checksum Engine
# ---------------------------------------------------------------------------

_ICAO_WEIGHTS = [7, 3, 1]


def compute_icao_checksum(chars: Optional[str]) -> int:
    """
    Compute ICAO Doc 9303 weighted modulo-10 checksum over character string.
    Weights: 7, 3, 1 repeating.
    A-Z: 10-35, 0-9: 0-9, '<': 0.
    """
    if not chars:
        return 0
    total = 0
    for idx, ch in enumerate(chars):
        weight = _ICAO_WEIGHTS[idx % 3]
        if ch.isdigit():
            val = int(ch)
        elif ch.isalpha() and ch.isupper():
            val = ord(ch) - ord('A') + 10
        elif ch.isalpha() and ch.islower():
            val = ord(ch) - ord('a') + 10
        else:
            val = 0  # '<' or filler
        total += val * weight
    return total % 10


def parse_mrz_lines(ocr_lines: List[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Detect and parse Machine Readable Zone (MRZ) per ICAO Doc 9303.
    Supports TD1 (3 lines of 30 chars) and TD3 (2 lines of 44 chars).
    """
    # Extract raw text strings
    text_candidates = []
    for item in ocr_lines:
        t = item["text"] if isinstance(item, dict) else str(item)
        clean_t = t.replace(" ", "").upper()
        if len(clean_t) >= 28 and ("<" in clean_t or re.search(r"[A-Z0-9<]{28,44}", clean_t)):
            text_candidates.append(clean_t)

    # 1. Try TD1 Format (3 lines x 30 characters)
    td1_lines = [l for l in text_candidates if 28 <= len(l) <= 32]
    if len(td1_lines) >= 3:
        l1 = td1_lines[-3][:30].ljust(30, "<")
        l2 = td1_lines[-2][:30].ljust(30, "<")
        l3 = td1_lines[-1][:30].ljust(30, "<")

        doc_type = l1[0:2].replace("<", "")
        country = l1[2:5].replace("<", "")
        doc_num_raw = l1[5:14]
        doc_num_check = l1[14]

        dob_raw = l2[0:6]
        dob_check = l2[6]
        sex = l2[7]
        expiry_raw = l2[8:14]
        expiry_check = l2[14]
        nationality = l2[15:18].replace("<", "")

        name_raw = l3.strip("<")
        name_parts = name_raw.split("<<")
        surname = name_parts[0].replace("<", " ").strip() if name_parts else ""
        given = name_parts[1].replace("<", " ").strip() if len(name_parts) > 1 else ""
        full_name = f"{given} {surname}".strip() if given else surname

        c_doc = compute_icao_checksum(doc_num_raw)
        c_dob = compute_icao_checksum(dob_raw)
        c_exp = compute_icao_checksum(expiry_raw)

        v_doc = (str(c_doc) == doc_num_check)
        v_dob = (str(c_dob) == dob_check)
        v_exp = (str(c_exp) == expiry_check)
        all_valid = v_doc and v_dob and v_exp

        dob_formatted = _format_mrz_date(dob_raw)
        expiry_formatted = _format_mrz_date(expiry_raw)

        return {
            "format": "TD1",
            "document_type": doc_type,
            "country": country,
            "document_number": doc_num_raw.replace("<", ""),
            "dob": dob_raw,
            "dob_formatted": dob_formatted,
            "expiry_date": expiry_raw,
            "expiry_date_formatted": expiry_formatted,
            "sex": sex,
            "nationality": nationality,
            "surname": surname,
            "given_names": given,
            "name": full_name,
            "checksums": {
                "document_number_valid": v_doc,
                "doc_number_valid": v_doc,
                "dob_valid": v_dob,
                "expiry_valid": v_exp,
                "all_valid": all_valid,
            },
            "raw_mrz_lines": [l1, l2, l3],
        }

    # 2. Try TD3 Format (2 lines x 44 characters, Passport)
    td3_lines = [l for l in text_candidates if 42 <= len(l) <= 46]
    if len(td3_lines) >= 2:
        l1 = td3_lines[-2][:44].ljust(44, "<")
        l2 = td3_lines[-1][:44].ljust(44, "<")

        doc_type = l1[0:2].replace("<", "")
        country = l1[2:5].replace("<", "")
        name_raw = l1[5:].strip("<")
        name_parts = name_raw.split("<<")
        surname = name_parts[0].replace("<", " ").strip() if name_parts else ""
        given = name_parts[1].replace("<", " ").strip() if len(name_parts) > 1 else ""
        full_name = f"{given} {surname}".strip() if given else surname

        doc_num_raw = l2[0:9]
        doc_num_check = l2[9]
        nationality = l2[10:13].replace("<", "")
        dob_raw = l2[13:19]
        dob_check = l2[19]
        sex = l2[20]
        expiry_raw = l2[21:27]
        expiry_check = l2[27]

        c_doc = compute_icao_checksum(doc_num_raw)
        c_dob = compute_icao_checksum(dob_raw)
        c_exp = compute_icao_checksum(expiry_raw)

        v_doc = (str(c_doc) == doc_num_check)
        v_dob = (str(c_dob) == dob_check)
        v_exp = (str(c_exp) == expiry_check)

        return {
            "format": "TD3",
            "document_type": doc_type,
            "country": country,
            "document_number": doc_num_raw.replace("<", ""),
            "dob": dob_raw,
            "dob_formatted": _format_mrz_date(dob_raw),
            "expiry_date": expiry_raw,
            "expiry_date_formatted": _format_mrz_date(expiry_raw),
            "sex": sex,
            "nationality": nationality,
            "surname": surname,
            "given_names": given,
            "name": full_name,
            "checksums": {
                "document_number_valid": v_doc,
                "doc_number_valid": v_doc,
                "dob_valid": v_dob,
                "expiry_valid": v_exp,
                "all_valid": (v_doc and v_dob and v_exp),
            },
            "raw_mrz_lines": [l1, l2],
        }

    return None


def _format_mrz_date(yymmdd: str) -> str:
    """Format YYMMDD string to DD/MM/YYYY."""
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return yymmdd
    yy = int(yymmdd[0:2])
    mm = yymmdd[2:4]
    dd = yymmdd[4:6]
    century = "19" if yy > 40 else "20"
    return f"{dd}/{mm}/{century}{yy:02d}"


# ---------------------------------------------------------------------------
# 4. Visual Inspection Zone (VIZ) vs MRZ Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate_viz_and_mrz(
    viz_fields: Dict[str, Any],
    mrz_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Cross-validate structured visual fields (VIZ) against MRZ ground truth.
    Any divergence between human-readable text and cryptographically checksummed
    MRZ fields indicates document modification fraud.
    """
    if not mrz_data:
        return {
            "cross_validation_status": "NO_MRZ_PRESENT",
            "is_consistent": True,
            "document_number_match": True,
            "name_match": True,
            "dob_match": True,
            "expiry_date_match": True,
            "overall_viz_mrz_match": True,
            "discrepancies": [],
        }

    discrepancies = []

    def _get(fname):
        item = viz_fields.get(fname)
        if isinstance(item, dict):
            return str(item.get("value") or "").strip()
        return str(item or "").strip()

    v_name = _get("name").upper()
    v_doc = _get("document_number").upper().replace("-", "").replace(" ", "")
    v_dob = _get("dob")
    v_exp = _get("expiry_date")

    m_name = str(mrz_data.get("name") or "").upper()
    m_doc = str(mrz_data.get("document_number") or "").upper().replace("-", "").replace(" ", "")
    m_dob = str(mrz_data.get("dob") or "")
    m_dob_fmt = str(mrz_data.get("dob_formatted") or "")
    m_exp = str(mrz_data.get("expiry_date") or "")
    m_exp_fmt = str(mrz_data.get("expiry_date_formatted") or "")

    # Compare Document Number
    doc_match = True
    if v_doc and m_doc:
        if v_doc not in m_doc and m_doc not in v_doc:
            discrepancies.append(f"Document Number mismatch: VIZ='{v_doc}' vs MRZ='{m_doc}'")
            doc_match = False

    # Compare DOB
    dob_match = True
    if v_dob and (m_dob or m_dob_fmt):
        v_dt = parse_date_flexible(v_dob)
        v_yymmdd = v_dt.strftime("%y%m%d") if v_dt else v_dob.replace("/", "")
        if v_dob != m_dob and v_dob != m_dob_fmt and v_yymmdd != m_dob:
            discrepancies.append(f"DOB mismatch: VIZ='{v_dob}' vs MRZ='{m_dob}'")
            dob_match = False

    # Compare Expiry Date
    exp_match = True
    if v_exp and (m_exp or m_exp_fmt):
        v_dt = parse_date_flexible(v_exp)
        v_yymmdd = v_dt.strftime("%y%m%d") if v_dt else v_exp.replace("/", "")
        if v_exp != m_exp and v_exp != m_exp_fmt and v_yymmdd != m_exp:
            discrepancies.append(f"Expiry Date mismatch: VIZ='{v_exp}' vs MRZ='{m_exp}'")
            exp_match = False

    # Compare Name
    name_match = True
    if v_name and m_name:
        v_parts = set(v_name.split())
        m_parts = set(m_name.split())
        common = v_parts.intersection(m_parts)
        if not common:
            discrepancies.append(f"Name mismatch: VIZ='{v_name}' vs MRZ='{m_name}'")
            name_match = False

    status = "IDENTITY_TAMPERING_DISCREPANCY" if discrepancies else "VIZ_MRZ_CONSISTENT"

    return {
        "cross_validation_status": status,
        "is_consistent": (len(discrepancies) == 0),
        "document_number_match": doc_match,
        "name_match": name_match,
        "dob_match": dob_match,
        "expiry_date_match": exp_match,
        "overall_viz_mrz_match": (len(discrepancies) == 0),
        "discrepancies": discrepancies,
        "mrz_format": mrz_data.get("format"),
        "mrz_checksums_valid": mrz_data.get("checksums", {}).get("all_valid", False),
    }
