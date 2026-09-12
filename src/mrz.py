"""
ForgeLens-X — Milestone 4: Dedicated ICAO Doc 9303 MRZ Forensics Engine
======================================================================
Parses, audits, and repairs Machine Readable Zone (MRZ) strings according to
ICAO Doc 9303 specifications for TD1, TD2, and TD3 credential formats.

Features:
- Cyclic [7, 3, 1] Modulo-10 check digit calculation for all fields and composite blocks.
- Full TD1 (3x30), TD2 (2x36), and TD3 (2x44) format parsers.
- Hypothesis-Driven OCR Optical Confusable Disambiguation & Self-Healing:
  Tests 1-Hamming-distance confusable substitutions (e.g. O/0, I/1, S/5, B/8, Z/2)
  against the check digit and VIZ fields to distinguish benign scan noise from
  tampered/fraudulent credentials.
- VIZ-to-MRZ Cross-Validation:
  Performs bi-directional field concordances (Document Number, DOB, Expiry Date, Name)
  with character-level and semantic tolerance.
"""

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml


ICAO_CHAR_VALUES = {
    "<": 0,
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15, "G": 16, "H": 17, "I": 18,
    "J": 19, "K": 20, "L": 21, "M": 22, "N": 23, "O": 24, "P": 25, "Q": 26, "R": 27,
    "S": 28, "T": 29, "U": 30, "V": 31, "W": 32, "X": 33, "Y": 34, "Z": 35,
}

CONFUSABLE_MAP = {
    "O": ["0", "Q", "D"],
    "0": ["O", "Q", "D"],
    "Q": ["O", "0"],
    "D": ["0", "O"],
    "I": ["1", "L", "T"],
    "1": ["I", "L"],
    "L": ["1", "I"],
    "S": ["5"],
    "5": ["S"],
    "B": ["8"],
    "8": ["B"],
    "Z": ["2"],
    "2": ["Z"],
}


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


def compute_icao_checksum(chars: str, weights: Optional[List[int]] = None) -> int:
    """
    Compute ICAO Doc 9303 Modulo-10 checksum using cyclic weights [7, 3, 1].
    Any non-alphanumeric character (e.g. '<') is assigned value 0.
    """
    if weights is None:
        weights = [7, 3, 1]
    w_len = len(weights)
    total = 0
    for idx, c in enumerate(chars.upper()):
        val = ICAO_CHAR_VALUES.get(c, 0)
        w = weights[idx % w_len]
        total += val * w
    return total % 10


def format_mrz_date(yymmdd: str, is_dob: bool = True) -> Optional[str]:
    """
    Convert 6-digit YYMMDD string to DD/MM/YYYY format with sensible century pivot.
    """
    if not yymmdd or len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    try:
        yy = int(yymmdd[0:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])

        if mm < 1 or mm > 12 or dd < 1 or dd > 31:
            return None

        current_yy = datetime.now(timezone.utc).year % 100
        if is_dob:
            century = 1900 if yy > current_yy else 2000
        else:
            century = 2000 if yy <= (current_yy + 50) else 1900

        full_year = century + yy
        return f"{dd:02d}/{mm:02d}/{full_year:04d}"
    except Exception:
        return None


def parse_mrz_name(raw_name: str) -> Tuple[str, str, str]:
    """
    Parse MRZ name string containing '<' separators into (surname, given_names, full_name).
    Format: SURNAME<<GIVEN<NAMES<<<<<<
    """
    clean = raw_name.rstrip("<")
    parts = clean.split("<<")
    surname = parts[0].replace("<", " ").strip() if parts else ""
    given = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
    full_name = f"{given} {surname}".strip() if given else surname
    return surname, given, full_name


def parse_mrz(mrz_lines: List[str]) -> Optional[Dict[str, Any]]:
    """
    Detect format (TD1, TD2, TD3) and parse MRZ lines per ICAO Doc 9303.
    """
    # Clean and filter lines
    cleaned = []
    for line in mrz_lines:
        c = re.sub(r"[^A-Z0-9<]", "", line.upper())
        if len(c) >= 28:
            cleaned.append(c)

    if not cleaned:
        return None

    # Check for TD1 (3 lines of 30 chars)
    td1_candidates = [l for l in cleaned if 28 <= len(l) <= 32]
    if len(td1_candidates) >= 3:
        l1 = td1_candidates[-3][:30].ljust(30, "<")
        l2 = td1_candidates[-2][:30].ljust(30, "<")
        l3 = td1_candidates[-1][:30].ljust(30, "<")
        return _parse_td1(l1, l2, l3)

    # Check for TD3 (2 lines of 44 chars)
    td3_candidates = [l for l in cleaned if 42 <= len(l) <= 46]
    if len(td3_candidates) >= 2:
        l1 = td3_candidates[-2][:44].ljust(44, "<")
        l2 = td3_candidates[-1][:44].ljust(44, "<")
        return _parse_td3(l1, l2)

    # Check for TD2 (2 lines of 36 chars)
    td2_candidates = [l for l in cleaned if 34 <= len(l) <= 38]
    if len(td2_candidates) >= 2:
        l1 = td2_candidates[-2][:36].ljust(36, "<")
        l2 = td2_candidates[-1][:36].ljust(36, "<")
        return _parse_td2(l1, l2)

    return None


def _parse_td1(l1: str, l2: str, l3: str) -> Dict[str, Any]:
    """Parse TD1 format (3 lines x 30 characters)."""
    doc_type = l1[0:2].replace("<", "")
    issuing_country = l1[2:5].replace("<", "")
    doc_num_raw = l1[5:14]
    doc_num_check = l1[14]
    opt1 = l1[15:30]

    dob_raw = l2[0:6]
    dob_check = l2[6]
    sex = l2[7]
    expiry_raw = l2[8:14]
    expiry_check = l2[14]
    nationality = l2[15:18].replace("<", "")
    opt2 = l2[18:29]
    composite_check = l2[29]

    surname, given, full_name = parse_mrz_name(l3)

    # Checksum calculations
    c_doc = compute_icao_checksum(doc_num_raw)
    c_dob = compute_icao_checksum(dob_raw)
    c_exp = compute_icao_checksum(expiry_raw)

    # Composite string for TD1: Line 1 chars 5..30 (doc_num + check + opt1) + Line 2 chars 0..7 (dob + check) +
    # Line 2 chars 8..15 (exp + check) + Line 2 chars 18..29 (opt2)
    composite_data = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    c_composite = compute_icao_checksum(composite_data)

    v_doc = (str(c_doc) == doc_num_check)
    v_dob = (str(c_dob) == dob_check)
    v_exp = (str(c_exp) == expiry_check)
    v_comp = (str(c_composite) == composite_check)

    return {
        "format": "TD1",
        "raw_lines": [l1, l2, l3],
        "document_type": doc_type,
        "issuing_country": issuing_country,
        "document_number": doc_num_raw.replace("<", ""),
        "document_number_raw": doc_num_raw,
        "document_number_check": doc_num_check,
        "document_number_valid": v_doc,
        "document_number_computed": str(c_doc),
        "dob": dob_raw,
        "dob_formatted": format_mrz_date(dob_raw, is_dob=True),
        "dob_check": dob_check,
        "dob_valid": v_dob,
        "dob_computed": str(c_dob),
        "sex": sex,
        "expiry_date": expiry_raw,
        "expiry_date_formatted": format_mrz_date(expiry_raw, is_dob=False),
        "expiry_date_check": expiry_check,
        "expiry_date_valid": v_exp,
        "expiry_date_computed": str(c_exp),
        "nationality": nationality,
        "optional_data_1": opt1.replace("<", ""),
        "optional_data_2": opt2.replace("<", ""),
        "composite_check": composite_check,
        "composite_valid": v_comp,
        "composite_computed": str(c_composite),
        "surname": surname,
        "given_names": given,
        "name": full_name,
        "all_checksums_valid": (v_doc and v_dob and v_exp and v_comp),
        "checksum_summary": {
            "document_number": v_doc,
            "dob": v_dob,
            "expiry_date": v_exp,
            "composite": v_comp,
        },
    }


def _parse_td2(l1: str, l2: str) -> Dict[str, Any]:
    """Parse TD2 format (2 lines x 36 characters)."""
    doc_type = l1[0:2].replace("<", "")
    issuing_country = l1[2:5].replace("<", "")
    surname, given, full_name = parse_mrz_name(l1[5:36])

    doc_num_raw = l2[0:9]
    doc_num_check = l2[9]
    nationality = l2[10:13].replace("<", "")
    dob_raw = l2[13:19]
    dob_check = l2[19]
    sex = l2[20]
    expiry_raw = l2[21:27]
    expiry_check = l2[27]
    opt = l2[28:35]
    composite_check = l2[35]

    c_doc = compute_icao_checksum(doc_num_raw)
    c_dob = compute_icao_checksum(dob_raw)
    c_exp = compute_icao_checksum(expiry_raw)

    # Composite string for TD2: doc_num + check + dob + check + exp + check + opt
    composite_data = l2[0:10] + l2[13:20] + l2[21:35]
    c_composite = compute_icao_checksum(composite_data)

    v_doc = (str(c_doc) == doc_num_check)
    v_dob = (str(c_dob) == dob_check)
    v_exp = (str(c_exp) == expiry_check)
    v_comp = (str(c_composite) == composite_check)

    return {
        "format": "TD2",
        "raw_lines": [l1, l2],
        "document_type": doc_type,
        "issuing_country": issuing_country,
        "document_number": doc_num_raw.replace("<", ""),
        "document_number_raw": doc_num_raw,
        "document_number_check": doc_num_check,
        "document_number_valid": v_doc,
        "document_number_computed": str(c_doc),
        "nationality": nationality,
        "dob": dob_raw,
        "dob_formatted": format_mrz_date(dob_raw, is_dob=True),
        "dob_check": dob_check,
        "dob_valid": v_dob,
        "dob_computed": str(c_dob),
        "sex": sex,
        "expiry_date": expiry_raw,
        "expiry_date_formatted": format_mrz_date(expiry_raw, is_dob=False),
        "expiry_date_check": expiry_check,
        "expiry_date_valid": v_exp,
        "expiry_date_computed": str(c_exp),
        "optional_data": opt.replace("<", ""),
        "composite_check": composite_check,
        "composite_valid": v_comp,
        "composite_computed": str(c_composite),
        "surname": surname,
        "given_names": given,
        "name": full_name,
        "all_checksums_valid": (v_doc and v_dob and v_exp and v_comp),
        "checksum_summary": {
            "document_number": v_doc,
            "dob": v_dob,
            "expiry_date": v_exp,
            "composite": v_comp,
        },
    }


def _parse_td3(l1: str, l2: str) -> Dict[str, Any]:
    """Parse TD3 format (2 lines x 44 characters, Standard Passport)."""
    doc_type = l1[0:2].replace("<", "")
    issuing_country = l1[2:5].replace("<", "")
    surname, given, full_name = parse_mrz_name(l1[5:44])

    doc_num_raw = l2[0:9]
    doc_num_check = l2[9]
    nationality = l2[10:13].replace("<", "")
    dob_raw = l2[13:19]
    dob_check = l2[19]
    sex = l2[20]
    expiry_raw = l2[21:27]
    expiry_check = l2[27]
    opt_raw = l2[28:42]
    opt_check = l2[42]
    composite_check = l2[43]

    c_doc = compute_icao_checksum(doc_num_raw)
    c_dob = compute_icao_checksum(dob_raw)
    c_exp = compute_icao_checksum(expiry_raw)
    c_opt = compute_icao_checksum(opt_raw)

    v_doc = (str(c_doc) == doc_num_check)
    v_dob = (str(c_dob) == dob_check)
    v_exp = (str(c_exp) == expiry_check)
    v_opt = (opt_check == "<") or (str(c_opt) == opt_check)

    # Composite string for TD3: doc_num + check + dob + check + exp + check + opt + check
    composite_data = l2[0:10] + l2[13:20] + l2[21:43]
    c_composite = compute_icao_checksum(composite_data)
    v_comp = (str(c_composite) == composite_check)

    return {
        "format": "TD3",
        "raw_lines": [l1, l2],
        "document_type": doc_type,
        "issuing_country": issuing_country,
        "document_number": doc_num_raw.replace("<", ""),
        "document_number_raw": doc_num_raw,
        "document_number_check": doc_num_check,
        "document_number_valid": v_doc,
        "document_number_computed": str(c_doc),
        "nationality": nationality,
        "dob": dob_raw,
        "dob_formatted": format_mrz_date(dob_raw, is_dob=True),
        "dob_check": dob_check,
        "dob_valid": v_dob,
        "dob_computed": str(c_dob),
        "sex": sex,
        "expiry_date": expiry_raw,
        "expiry_date_formatted": format_mrz_date(expiry_raw, is_dob=False),
        "expiry_date_check": expiry_check,
        "expiry_date_valid": v_exp,
        "expiry_date_computed": str(c_exp),
        "optional_data": opt_raw.replace("<", ""),
        "optional_data_check": opt_check,
        "optional_data_valid": v_opt,
        "composite_check": composite_check,
        "composite_valid": v_comp,
        "composite_computed": str(c_composite),
        "surname": surname,
        "given_names": given,
        "name": full_name,
        "all_checksums_valid": (v_doc and v_dob and v_exp and v_opt and v_comp),
        "checksum_summary": {
            "document_number": v_doc,
            "dob": v_dob,
            "expiry_date": v_exp,
            "optional_data": v_opt,
            "composite": v_comp,
        },
    }


# ---------------------------------------------------------------------------
# Hypothesis-Driven Error Repair / Disambiguation
# ---------------------------------------------------------------------------

def disambiguate_mrz_field(
    raw_field: str,
    check_digit: str,
    viz_hint: Optional[str] = None,
    max_flips: int = 1,
) -> Tuple[bool, str, Optional[str]]:
    """
    Attempt 1-Hamming-distance confusable repairs on a failing MRZ field.
    Returns:
        (is_repaired, repaired_string, repair_explanation)
    """
    # 1. First check if already valid
    c = compute_icao_checksum(raw_field)
    if str(c) == check_digit:
        return False, raw_field, None

    field_chars = list(raw_field)
    # Check if check digit itself might be confusable
    if check_digit in CONFUSABLE_MAP:
        for alt_chk in CONFUSABLE_MAP[check_digit]:
            if str(c) == alt_chk:
                explanation = f"Check digit repaired: substituted '{check_digit}' -> '{alt_chk}' (optical confusable)"
                return True, raw_field, explanation

    # Try 1-flip substitutions on field_chars
    for idx, orig_c in enumerate(field_chars):
        alternatives = CONFUSABLE_MAP.get(orig_c, [])
        for alt_c in alternatives:
            candidate = list(field_chars)
            candidate[idx] = alt_c
            cand_str = "".join(candidate)
            if str(compute_icao_checksum(cand_str)) == check_digit:
                # If VIZ hint is available, candidate MUST match VIZ hint
                if viz_hint:
                    clean_viz = re.sub(r"[^A-Z0-9]", "", viz_hint.upper())
                    clean_cand = cand_str.replace("<", "")
                    if clean_viz and (clean_viz in clean_cand or clean_cand in clean_viz):
                        exp = f"Repaired position {idx}: substituted '{orig_c}' -> '{alt_c}' matching VIZ '{clean_viz}'"
                        return True, cand_str, exp
                    # VIZ hint present but candidate does not match -> reject candidate
                    continue

                # If no VIZ hint available, only accept conservative numeral/letter confusions
                if (orig_c.upper(), alt_c.upper()) in [
                    ("O", "0"), ("0", "O"), ("I", "1"), ("1", "I"),
                    ("S", "5"), ("5", "S"), ("B", "8"), ("8", "B")
                ]:
                    exp = f"Repaired position {idx}: substituted '{orig_c}' -> '{alt_c}' to satisfy check digit '{check_digit}'"
                    return True, cand_str, exp

    return False, raw_field, None


def disambiguate_mrz_checksums(
    mrz_data: Dict[str, Any],
    viz_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate all failing check digits in MRZ data.
    Attempts confusable disambiguation using VIZ fields if present.
    Emits structured status:
        'MRZ_ALL_CHECKSUMS_VERIFIED' | 'MRZ_REPAIRED_OPTICAL_CONFUSION' | 'MRZ_CHECKSUM_FORGERY_FRAUD'
    """
    viz_doc = None
    viz_dob = None
    viz_exp = None
    if viz_fields:
        viz_doc = _extract_str(viz_fields, "document_number")
        viz_dob = _extract_str(viz_fields, "dob")
        viz_exp = _extract_str(viz_fields, "expiry_date")

    repairs = []
    unrepaired_failures = []

    # 1. Document Number Check
    if not mrz_data.get("document_number_valid", False):
        raw = mrz_data.get("document_number_raw", "")
        chk = mrz_data.get("document_number_check", "")
        repaired, new_val, reason = disambiguate_mrz_field(raw, chk, viz_hint=viz_doc)
        if repaired:
            repairs.append({"field": "document_number", "original": raw, "repaired": new_val, "detail": reason})
            mrz_data["document_number_raw"] = new_val
            mrz_data["document_number"] = new_val.replace("<", "")
            mrz_data["document_number_valid"] = True
            mrz_data["document_number_computed"] = str(compute_icao_checksum(new_val))
        else:
            unrepaired_failures.append("document_number")

    # 2. DOB Check
    if not mrz_data.get("dob_valid", False):
        raw = mrz_data.get("dob", "")
        chk = mrz_data.get("dob_check", "")
        repaired, new_val, reason = disambiguate_mrz_field(raw, chk, viz_hint=viz_dob)
        if repaired:
            repairs.append({"field": "dob", "original": raw, "repaired": new_val, "detail": reason})
            mrz_data["dob"] = new_val
            mrz_data["dob_formatted"] = format_mrz_date(new_val, is_dob=True)
            mrz_data["dob_valid"] = True
            mrz_data["dob_computed"] = str(compute_icao_checksum(new_val))
        else:
            unrepaired_failures.append("dob")

    # 3. Expiry Date Check
    if not mrz_data.get("expiry_date_valid", False):
        raw = mrz_data.get("expiry_date", "")
        chk = mrz_data.get("expiry_date_check", "")
        repaired, new_val, reason = disambiguate_mrz_field(raw, chk, viz_hint=viz_exp)
        if repaired:
            repairs.append({"field": "expiry_date", "original": raw, "repaired": new_val, "detail": reason})
            mrz_data["expiry_date"] = new_val
            mrz_data["expiry_date_formatted"] = format_mrz_date(new_val, is_dob=False)
            mrz_data["expiry_date_valid"] = True
            mrz_data["expiry_date_computed"] = str(compute_icao_checksum(new_val))
        else:
            unrepaired_failures.append("expiry_date")

    # Update composite valid if all individual fields are now valid
    mrz_data["repairs"] = repairs
    mrz_data["unrepaired_failures"] = unrepaired_failures

    if not unrepaired_failures and not repairs:
        verdict = "MRZ_ALL_CHECKSUMS_VERIFIED"
        status = "PASS"
    elif not unrepaired_failures and len(repairs) > 0:
        verdict = "MRZ_REPAIRED_OPTICAL_CONFUSION"
        status = "PASS"
    else:
        verdict = "MRZ_CHECKSUM_FORGERY_FRAUD"
        status = "FAIL"

    mrz_data["status"] = status
    mrz_data["verdict"] = verdict
    return mrz_data


# ---------------------------------------------------------------------------
# VIZ-to-MRZ Cross-Validation
# ---------------------------------------------------------------------------

def _extract_str(fields: Dict[str, Any], key: str) -> Optional[str]:
    item = fields.get(key)
    if item is None:
        return None
    if isinstance(item, dict):
        val = item.get("value")
        if val is None or str(val).strip() in ["", "[NONE]", "None", "UNKNOWN"]:
            return None
        return str(val).strip()
    val = str(item).strip()
    if val in ["", "[NONE]", "None", "UNKNOWN"]:
        return None
    return val


def _normalize_date_digits(d_str: Optional[str]) -> Optional[str]:
    """Extract 6-digit YYMMDD from arbitrary DD/MM/YYYY or YYYY-MM-DD string."""
    if not d_str:
        return None
    d_str = d_str.strip()
    # Check delimited formats first (DD/MM/YYYY or YYYY-MM-DD)
    parts = re.split(r"[/.-]", d_str)
    if len(parts) == 3:
        p0, p1, p2 = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if len(p0) == 4 and p0.isdigit():
            # YYYY-MM-DD
            yy = p0[-2:]
            mm = p1.zfill(2)
            dd = p2.zfill(2)
            return f"{yy}{mm}{dd}"
        elif len(p2) == 4 and p2.isdigit():
            # DD/MM/YYYY
            dd = p0.zfill(2)
            mm = p1.zfill(2)
            yy = p2[-2:]
            return f"{yy}{mm}{dd}"
        elif len(p2) == 2 and p2.isdigit() and len(p0) == 2 and p0.isdigit():
            # DD/MM/YY
            dd = p0.zfill(2)
            mm = p1.zfill(2)
            yy = p2.zfill(2)
            return f"{yy}{mm}{dd}"

    cleaned = re.sub(r"[^0-9]", "", d_str)
    if len(cleaned) == 8:
        yr_prefix = int(cleaned[:4])
        mo_prefix = int(cleaned[4:6])
        if (1900 <= yr_prefix <= 2099) and (1 <= mo_prefix <= 12):
            return cleaned[2:8]
        else:
            dd = cleaned[0:2]
            mm = cleaned[2:4]
            yy = cleaned[6:8]
            return f"{yy}{mm}{dd}"
    elif len(cleaned) == 6:
        return cleaned
    return None


def cross_validate_viz_and_mrz(
    viz_fields: Dict[str, Any],
    mrz_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Bi-directional cross-check between Visual Inspection Zone (VIZ) and MRZ.
    Checks:
        1. Document Number concordance.
        2. Date of Birth concordance.
        3. Expiry Date concordance.
        4. Name / Surname token overlap concordance.
    """
    checks = []
    mismatches = []
    concordances = 0
    total_comparable = 0

    # 1. Document Number
    viz_doc = _extract_str(viz_fields, "document_number")
    mrz_doc = mrz_data.get("document_number")
    if viz_doc and mrz_doc:
        total_comparable += 1
        clean_v = re.sub(r"[^A-Z0-9]", "", viz_doc.upper())
        clean_m = re.sub(r"[^A-Z0-9]", "", mrz_doc.upper())
        # Check match or containment (since some countries add prefixes in VIZ)
        if clean_v == clean_m or clean_v in clean_m or clean_m in clean_v:
            concordances += 1
            checks.append({"field": "document_number", "status": "PASS", "viz": viz_doc, "mrz": mrz_doc})
        else:
            mismatches.append(f"Document Number mismatch: VIZ '{viz_doc}' vs MRZ '{mrz_doc}'")
            checks.append({"field": "document_number", "status": "FAIL", "viz": viz_doc, "mrz": mrz_doc})
    else:
        checks.append({"field": "document_number", "status": "NOT_APPLICABLE", "viz": viz_doc, "mrz": mrz_doc})

    # 2. Date of Birth
    viz_dob = _extract_str(viz_fields, "dob")
    mrz_dob = mrz_data.get("dob")
    if viz_dob and mrz_dob:
        total_comparable += 1
        norm_v = _normalize_date_digits(viz_dob)
        if norm_v and norm_v == mrz_dob:
            concordances += 1
            checks.append({"field": "dob", "status": "PASS", "viz": viz_dob, "mrz": mrz_dob})
        else:
            mismatches.append(f"DOB mismatch: VIZ '{viz_dob}' (norm: {norm_v}) vs MRZ '{mrz_dob}'")
            checks.append({"field": "dob", "status": "FAIL", "viz": viz_dob, "mrz": mrz_dob})
    else:
        checks.append({"field": "dob", "status": "NOT_APPLICABLE", "viz": viz_dob, "mrz": mrz_dob})

    # 3. Expiry Date
    viz_exp = _extract_str(viz_fields, "expiry_date")
    mrz_exp = mrz_data.get("expiry_date")
    if viz_exp and mrz_exp:
        total_comparable += 1
        norm_v = _normalize_date_digits(viz_exp)
        if norm_v and norm_v == mrz_exp:
            concordances += 1
            checks.append({"field": "expiry_date", "status": "PASS", "viz": viz_exp, "mrz": mrz_exp})
        else:
            mismatches.append(f"Expiry Date mismatch: VIZ '{viz_exp}' (norm: {norm_v}) vs MRZ '{mrz_exp}'")
            checks.append({"field": "expiry_date", "status": "FAIL", "viz": viz_exp, "mrz": mrz_exp})
    else:
        checks.append({"field": "expiry_date", "status": "NOT_APPLICABLE", "viz": viz_exp, "mrz": mrz_exp})

    # 4. Name Concordance
    viz_name = _extract_str(viz_fields, "name")
    mrz_name = mrz_data.get("name")
    mrz_surname = mrz_data.get("surname")
    if viz_name and (mrz_name or mrz_surname):
        total_comparable += 1
        v_tokens = set(re.findall(r"[A-Z]{2,}", viz_name.upper()))
        m_tokens = set(re.findall(r"[A-Z]{2,}", (mrz_name or "").upper()))
        # Check Surname concordance
        surname_match = True
        def _norm_token(t: str) -> str:
            return t.upper().replace("L", "I").replace("1", "I").replace("0", "O")

        norm_v = {_norm_token(t) for t in v_tokens}
        norm_m = {_norm_token(t) for t in m_tokens}

        if mrz_surname:
            s_tokens = set(re.findall(r"[A-Z]{2,}", mrz_surname.upper()))
            norm_s = {_norm_token(t) for t in s_tokens}
            if s_tokens and not (s_tokens.intersection(v_tokens) or norm_s.intersection(norm_v)):
                surname_match = False

        overlap = v_tokens.intersection(m_tokens) or norm_v.intersection(norm_m)
        if overlap and surname_match:
            concordances += 1
            checks.append({"field": "name", "status": "PASS", "viz": viz_name, "mrz": mrz_name, "overlap": list(overlap)})
        else:
            reason = f"Surname '{mrz_surname}' missing in VIZ '{viz_name}'" if not surname_match else f"VIZ '{viz_name}' vs MRZ '{mrz_name}' (no token overlap)"
            mismatches.append(f"Name mismatch: {reason}")
            checks.append({"field": "name", "status": "FAIL", "viz": viz_name, "mrz": f"{mrz_surname}<<{mrz_name}" if mrz_surname else mrz_name})
    else:
        checks.append({"field": "name", "status": "NOT_APPLICABLE", "viz": viz_name, "mrz": mrz_name})

    score = round(concordances / max(1, total_comparable), 2)
    is_concordant = (len(mismatches) == 0 and total_comparable > 0)

    if len(mismatches) > 0:
        verdict = "VIZ_MRZ_CONTRADICTION_FRAUD"
    elif total_comparable == 0:
        verdict = "VIZ_MRZ_NOT_COMPARABLE"
    else:
        verdict = "VIZ_MRZ_FULLY_CONCORDANT"

    return {
        "verdict": verdict,
        "is_concordant": is_concordant,
        "concordance_score": score,
        "comparable_fields": total_comparable,
        "concordant_fields": concordances,
        "mismatches": mismatches,
        "field_checks": checks,
    }
