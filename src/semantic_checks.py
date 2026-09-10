"""
ForgeLens-X — Milestone 4: Canonical Semantic Validation Engine
===============================================================
Audits logical, temporal, and structural consistency across extracted identity document fields.
Implements the canonical check schema:
    {
        "check": "rule_name",
        "status": "PASS" | "FAIL" | "NOT_APPLICABLE" | "UNKNOWN",
        "detail": "Explanatory human forensic review string",
        "evidence": {...}
    }

Non-punitive conditional availability protocol:
    If a field is missing, the corresponding check returns NOT_APPLICABLE or UNKNOWN.
    Missing information is never treated as equivalent to fraudulent information.
"""

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

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


def _get_val(fields: Dict[str, Any], fname: str) -> Optional[str]:
    """Extract raw string value from fields dict, supporting both flat and structured schemas."""
    item = fields.get(fname)
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


def parse_calendar_date(d_str: Optional[str]) -> Tuple[Optional[datetime], Optional[str]]:
    """
    Parse date from string with strict calendar validity checking.
    Returns:
        (datetime_obj, error_reason)
        If valid: (datetime, None)
        If calendar impossible: (None, "Day 31 invalid for April")
        If unparseable: (None, "Unparseable format")
    """
    if not d_str:
        return None, None

    cleaned = str(d_str).strip()
    # Normalize separators
    normalized = re.sub(r"[\s.-]", "/", cleaned)
    parts = normalized.split("/")

    if len(parts) == 3:
        p1, p2, p3 = parts
        # Determine day, month, year based on lengths and standard formats
        try:
            if len(p1) == 4:  # YYYY/MM/DD
                y, m, d = int(p1), int(p2), int(p3)
            elif len(p3) == 4:  # DD/MM/YYYY
                d, m, y = int(p1), int(p2), int(p3)
            elif len(p3) == 2:  # DD/MM/YY
                d, m = int(p1), int(p2)
                yy = int(p3)
                y = (1900 if yy > 40 else 2000) + yy
            else:
                d, m, y = int(p1), int(p2), int(p3)

            # Check month range
            if m < 1 or m > 12:
                return None, f"Month {m} out of calendar range [1..12]"

            # Days in month table
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            # Leap year rule
            is_leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
            if is_leap and m == 2:
                max_d = 29
            else:
                max_d = days_in_month[m - 1]

            if d < 1 or d > max_d:
                if m == 2 and d == 29 and not is_leap:
                    return None, f"February 29 invalid in non-leap year {y}"
                return None, f"Day {d} exceeds maximum {max_d} days for month {m}"

            return datetime(y, m, d), None
        except ValueError:
            return None, f"Non-numeric values in date string '{d_str}'"

    return None, f"Cannot identify Day/Month/Year components in '{d_str}'"


# ---------------------------------------------------------------------------
# Individual Semantic Rules
# ---------------------------------------------------------------------------

def check_impossible_dates(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 1: Detect impossible calendar dates (e.g., 30 Feb, 31 April)."""
    date_fields = ["dob", "issue_date", "expiry_date"]
    checked = 0
    failures = []
    evidence = {}

    for fname in date_fields:
        val = _get_val(fields, fname)
        if val:
            checked += 1
            dt, err = parse_calendar_date(val)
            evidence[fname] = {"raw": val, "parsed": dt.strftime("%d/%m/%Y") if dt else None, "error": err}
            if err:
                failures.append(f"Field '{fname}' ('{val}'): {err}")

    if checked == 0:
        return {
            "check": "impossible_dates",
            "status": "NOT_APPLICABLE",
            "detail": "No date fields present to audit for calendar bounds.",
            "evidence": evidence,
        }

    if failures:
        return {
            "check": "impossible_dates",
            "status": "FAIL",
            "detail": f"Impossible calendar date detected: {'; '.join(failures)}",
            "evidence": evidence,
        }

    return {
        "check": "impossible_dates",
        "status": "PASS",
        "detail": f"All {checked} provided date fields represent physically valid calendar days.",
        "evidence": evidence,
    }


def check_chronology_order(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 2: Enforce temporal sequence: DOB < Issue Date < Expiry Date."""
    dob_str = _get_val(fields, "dob")
    issue_str = _get_val(fields, "issue_date")
    expiry_str = _get_val(fields, "expiry_date")

    dob_dt, _ = parse_calendar_date(dob_str)
    issue_dt, _ = parse_calendar_date(issue_str)
    expiry_dt, _ = parse_calendar_date(expiry_str)

    evidence = {
        "dob": dob_dt.strftime("%d/%m/%Y") if dob_dt else None,
        "issue_date": issue_dt.strftime("%d/%m/%Y") if issue_dt else None,
        "expiry_date": expiry_dt.strftime("%d/%m/%Y") if expiry_dt else None,
    }

    pairs_checked = 0
    inversions = []

    if dob_dt and issue_dt:
        pairs_checked += 1
        if issue_dt <= dob_dt:
            inversions.append(f"Issue date ({evidence['issue_date']}) precedes or equals DOB ({evidence['dob']})")

    if issue_dt and expiry_dt:
        pairs_checked += 1
        if expiry_dt <= issue_dt:
            inversions.append(f"Expiry date ({evidence['expiry_date']}) precedes or equals Issue date ({evidence['issue_date']})")

    if dob_dt and expiry_dt and not issue_dt:
        pairs_checked += 1
        if expiry_dt <= dob_dt:
            inversions.append(f"Expiry date ({evidence['expiry_date']}) precedes or equals DOB ({evidence['dob']})")

    if pairs_checked == 0:
        return {
            "check": "chronology_order",
            "status": "NOT_APPLICABLE",
            "detail": "Insufficient date pairs available to verify chronology sequence.",
            "evidence": evidence,
        }

    if inversions:
        return {
            "check": "chronology_order",
            "status": "FAIL",
            "detail": f"Chronological inversion detected: {'; '.join(inversions)}",
            "evidence": evidence,
        }

    return {
        "check": "chronology_order",
        "status": "PASS",
        "detail": "Chronological progression (DOB < Issue Date < Expiry Date) verified.",
        "evidence": evidence,
    }


def check_age_at_issue_sanity(
    fields: Dict[str, Any],
    min_age_years: float = 0.0,
    adult_threshold: float = 18.0,
) -> Dict[str, Any]:
    """Rule 3: Check age at issuance >= 0 years (and adult ID threshold)."""
    dob_str = _get_val(fields, "dob")
    issue_str = _get_val(fields, "issue_date")

    dob_dt, _ = parse_calendar_date(dob_str)
    issue_dt, _ = parse_calendar_date(issue_str)

    if not dob_dt or not issue_dt:
        return {
            "check": "age_at_issue_sanity",
            "status": "NOT_APPLICABLE",
            "detail": "Requires both DOB and Issue Date to compute issuance age.",
            "evidence": {"dob": dob_str, "issue_date": issue_str},
        }

    age_days = (issue_dt - dob_dt).days
    age_years = round(age_days / 365.25, 2)

    evidence = {
        "dob": dob_dt.strftime("%d/%m/%Y"),
        "issue_date": issue_dt.strftime("%d/%m/%Y"),
        "age_at_issue_years": age_years,
    }

    if age_years < min_age_years:
        return {
            "check": "age_at_issue_sanity",
            "status": "FAIL",
            "detail": f"Negative age at issuance: {age_years:.1f} years. Document issued before birth.",
            "evidence": evidence,
        }

    detail = f"Age at issuance is {age_years:.1f} years (valid non-negative age)."
    if age_years < adult_threshold:
        detail += f" Note: Holder is a minor ({age_years:.1f} < {adult_threshold} yrs)."

    return {
        "check": "age_at_issue_sanity",
        "status": "PASS",
        "detail": detail,
        "evidence": evidence,
    }


def check_validity_window_sanity(
    fields: Dict[str, Any],
    max_validity_years: float = 25.0,
) -> Dict[str, Any]:
    """Rule 4: Validate duration between issue and expiry does not exceed 25 years."""
    issue_str = _get_val(fields, "issue_date")
    expiry_str = _get_val(fields, "expiry_date")

    issue_dt, _ = parse_calendar_date(issue_str)
    expiry_dt, _ = parse_calendar_date(expiry_str)

    if not issue_dt or not expiry_dt:
        return {
            "check": "validity_window_sanity",
            "status": "NOT_APPLICABLE",
            "detail": "Requires both Issue Date and Expiry Date to compute validity window.",
            "evidence": {"issue_date": issue_str, "expiry_date": expiry_str},
        }

    val_days = (expiry_dt - issue_dt).days
    val_years = round(val_days / 365.25, 2)

    evidence = {
        "issue_date": issue_dt.strftime("%d/%m/%Y"),
        "expiry_date": expiry_dt.strftime("%d/%m/%Y"),
        "validity_years": val_years,
        "max_allowed_years": max_validity_years,
    }

    if val_years <= 0.0:
        return {
            "check": "validity_window_sanity",
            "status": "FAIL",
            "detail": f"Non-positive validity window: {val_years:.1f} years.",
            "evidence": evidence,
        }

    if val_years > max_validity_years:
        return {
            "check": "validity_window_sanity",
            "status": "FAIL",
            "detail": f"Excessive validity period: {val_years:.1f} years exceeds legal ceiling of {max_validity_years} years.",
            "evidence": evidence,
        }

    return {
        "check": "validity_window_sanity",
        "status": "PASS",
        "detail": f"Validity duration of {val_years:.1f} years complies with standard credential windows.",
        "evidence": evidence,
    }


def check_anachronism(
    fields: Dict[str, Any],
    tolerance_days: int = 30,
) -> Dict[str, Any]:
    """Rule 5: Verify Issue Date is not in the future (permitting 30-day buffer)."""
    issue_str = _get_val(fields, "issue_date")
    issue_dt, _ = parse_calendar_date(issue_str)

    if not issue_dt:
        return {
            "check": "anachronism_check",
            "status": "NOT_APPLICABLE",
            "detail": "No valid Issue Date available to check temporal anachronism.",
            "evidence": {"issue_date": issue_str},
        }

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    diff_days = (issue_dt - now).days

    evidence = {
        "issue_date": issue_dt.strftime("%d/%m/%Y"),
        "current_date": now.strftime("%d/%m/%Y"),
        "days_difference": diff_days,
        "tolerance_days": tolerance_days,
    }

    if diff_days > tolerance_days:
        return {
            "check": "anachronism_check",
            "status": "FAIL",
            "detail": f"Anachronistic future issue date: {issue_dt.strftime('%d/%m/%Y')} is {diff_days} days in the future.",
            "evidence": evidence,
        }

    return {
        "check": "anachronism_check",
        "status": "PASS",
        "detail": f"Issue date {issue_dt.strftime('%d/%m/%Y')} is chronologically consistent with current timeline.",
        "evidence": evidence,
    }


def check_document_number_format(
    fields: Dict[str, Any],
    doc_type: str = "forgelensia",
    custom_regex: Optional[str] = None,
) -> Dict[str, Any]:
    """Rule 6: Verify document number matches expected alphanumeric pattern."""
    doc_num = _get_val(fields, "document_number")

    if not doc_num:
        return {
            "check": "document_number_format",
            "status": "NOT_APPLICABLE",
            "detail": "Document number not extracted or absent.",
            "evidence": {"document_number": None},
        }

    if custom_regex:
        regex = custom_regex
    elif doc_type == "forgelensia":
        regex = r"^FGL-\d{6}-\d{2}$"
    elif doc_type == "passport":
        regex = r"^[A-Z0-9<]{8,10}$"
    else:
        regex = r"^[A-Z0-9\-_]{5,20}$"

    clean_num = doc_num.strip()
    is_match = bool(re.match(regex, clean_num, re.IGNORECASE))

    evidence = {
        "document_number": clean_num,
        "document_type": doc_type,
        "expected_regex": regex,
    }

    if not is_match:
        return {
            "check": "document_number_format",
            "status": "FAIL",
            "detail": f"Document number '{clean_num}' violates expected schema format ('{regex}').",
            "evidence": evidence,
        }

    return {
        "check": "document_number_format",
        "status": "PASS",
        "detail": f"Document number '{clean_num}' strictly adheres to {doc_type} credential schema.",
        "evidence": evidence,
    }


def check_duplicate_field_contradiction(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 7: Detect internal contradictions where repeated fields diverge."""
    # Look for alternate extractions (e.g. mrz vs viz or secondary text crops)
    contradictions = []
    evidence = {}

    # Check if raw text vs cleaned value divergent in an incompatible way
    for fname in ["dob", "document_number", "issue_date", "expiry_date"]:
        f_info = fields.get(fname)
        if isinstance(f_info, dict):
            raw = str(f_info.get("raw_text") or "").strip()
            val = str(f_info.get("value") or "").strip()
            if raw and val and len(raw) > 5 and len(val) > 5:
                # Normalize digits
                d_raw = re.sub(r"[^0-9]", "", raw)
                d_val = re.sub(r"[^0-9]", "", val)
                if len(d_raw) >= 6 and len(d_val) >= 6 and d_raw != d_val:
                    contradictions.append(f"Field '{fname}': Raw OCR digits ({d_raw}) conflict with assigned value ({d_val})")
                    evidence[fname] = {"raw_digits": d_raw, "val_digits": d_val}

    if contradictions:
        return {
            "check": "duplicate_field_contradiction",
            "status": "FAIL",
            "detail": f"Internal field contradiction detected: {'; '.join(contradictions)}",
            "evidence": evidence,
        }

    return {
        "check": "duplicate_field_contradiction",
        "status": "PASS",
        "detail": "No internal field value contradictions or conflicting duplicate instances detected.",
        "evidence": evidence,
    }


def check_name_structure_sanity(
    fields: Dict[str, Any],
    blacklist_tokens: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Rule 8: Verify name contains alphabetic words and avoids placeholder strings."""
    name_str = _get_val(fields, "name")

    if not name_str:
        return {
            "check": "name_structure_sanity",
            "status": "NOT_APPLICABLE",
            "detail": "Name field not extracted or absent.",
            "evidence": {"name": None},
        }

    if blacklist_tokens is None:
        blacklist_tokens = [
            "TEST", "SAMPLE", "DEMO", "DUMMY", "UNKNOWN",
            "NOT APPLICABLE", "PLACEHOLDER", "JOHN DOE", "JANE DOE"
        ]

    upper_name = name_str.upper()
    found_placeholders = [tok for tok in blacklist_tokens if tok in upper_name]

    # Check for minimum alpha characters
    alpha_chars = sum(1 for c in name_str if c.isalpha())

    evidence = {
        "name": name_str,
        "alpha_char_count": alpha_chars,
        "found_placeholders": found_placeholders,
    }

    if found_placeholders:
        return {
            "check": "name_structure_sanity",
            "status": "FAIL",
            "detail": f"Name contains dummy/placeholder tokens: {found_placeholders}",
            "evidence": evidence,
        }

    if alpha_chars < 2:
        return {
            "check": "name_structure_sanity",
            "status": "FAIL",
            "detail": f"Name '{name_str}' lacks sufficient alphabetic characters ({alpha_chars} alpha chars).",
            "evidence": evidence,
        }

    return {
        "check": "name_structure_sanity",
        "status": "PASS",
        "detail": f"Name '{name_str}' is structurally valid with {alpha_chars} alphabetic characters.",
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# Master Semantic Rule Battery Execution
# ---------------------------------------------------------------------------

def run_semantic_rule_battery(
    fields: Dict[str, Any],
    doc_type: str = "forgelensia",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute all 8 canonical semantic rules across extracted fields.

    Returns:
        Structured audit dictionary with per-check statuses, overall verdict,
        and evidence breakdowns.
    """
    if config is None:
        config = _load_m4_config()

    s_cfg = config.get("semantic_rules", {})
    min_age = s_cfg.get("min_age_at_issue_years", 0.0)
    adult_age = s_cfg.get("adult_age_threshold_years", 18.0)
    max_val = s_cfg.get("max_validity_years", 25.0)
    tol_days = s_cfg.get("future_date_tolerance_days", 30)
    placeholders = s_cfg.get("placeholder_name_tokens")

    schema_cfg = s_cfg.get("schemas", {}).get(doc_type, {})
    doc_regex = schema_cfg.get("document_number_regex")

    # Run all 8 rules
    checks = [
        check_impossible_dates(fields),
        check_chronology_order(fields),
        check_age_at_issue_sanity(fields, min_age_years=min_age, adult_threshold=adult_age),
        check_validity_window_sanity(fields, max_validity_years=max_val),
        check_anachronism(fields, tolerance_days=tol_days),
        check_document_number_format(fields, doc_type=doc_type, custom_regex=doc_regex),
        check_duplicate_field_contradiction(fields),
        check_name_structure_sanity(fields, blacklist_tokens=placeholders),
    ]

    checks_by_name = {c["check"]: c for c in checks}

    passed = [c["check"] for c in checks if c["status"] == "PASS"]
    failed = [c["check"] for c in checks if c["status"] == "FAIL"]
    not_applicable = [c["check"] for c in checks if c["status"] == "NOT_APPLICABLE"]
    unknown = [c["check"] for c in checks if c["status"] == "UNKNOWN"]

    if len(failed) > 0:
        verdict = "SUSPECT_SEMANTIC_VIOLATION"
        is_valid = False
    elif len(passed) >= 3:
        verdict = "PASSED_SEMANTIC_AUDIT"
        is_valid = True
    else:
        verdict = "PARTIAL_SEMANTIC_DATA"
        is_valid = True

    return {
        "semantic_verdict": verdict,
        "is_valid": is_valid,
        "total_checks": len(checks),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "not_applicable_count": len(not_applicable),
        "unknown_count": len(unknown),
        "passed_checks": passed,
        "failed_checks": failed,
        "not_applicable_checks": not_applicable,
        "unknown_checks": unknown,
        "checks": checks_by_name,
        "checks_list": checks,
    }
