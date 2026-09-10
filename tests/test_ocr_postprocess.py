"""
ForgeLens-X — Tests for OCR Postprocessing & Forensic Validation (M3)
====================================================================
Tests for:
    - Contextual character glyph repair (dates, document numbers, names)
    - Date chronology audit (DOB, Issue Date, Expiry Date temporal consistency)
    - ICAO Doc 9303 MRZ parsing and modulo-10 cyclic [7, 3, 1] checksum verification
    - VIZ-to-MRZ cross-modality validation
"""

import pytest
from datetime import datetime, timezone

from src.ocr_postprocess import (
    compute_icao_checksum,
    parse_date_flexible,
    repair_glyph_confusions,
    validate_date_chronology,
    parse_mrz_lines,
    cross_validate_viz_and_mrz,
)


# ---------------------------------------------------------------------------
# Test 1: ICAO Doc 9303 Modulo-10 Checksum Tests
# ---------------------------------------------------------------------------

def test_compute_icao_checksum_known_vector():
    """Verify ICAO 7-3-1 modulo-10 checksum calculation with standard vector."""
    # "D23145890" -> 13*7 + 2*3 + 3*1 + 1*7 + 4*3 + 5*1 + 8*7 + 9*3 + 0*1 = 207 % 10 = 7
    cs = compute_icao_checksum("D23145890")
    assert cs == 7


def test_compute_icao_checksum_fillers():
    """Verify that filler character '<' is treated as numeric 0."""
    cs1 = compute_icao_checksum("AB<<<")
    cs2 = compute_icao_checksum("AB000")
    assert cs1 == cs2


def test_compute_icao_checksum_date_vector():
    """Verify checksum on a typical 6-digit YYMMDD date string."""
    # "740812" -> 7*7 + 4*3 + 0*1 + 8*7 + 1*3 + 2*1 = 49 + 12 + 0 + 56 + 3 + 2 = 122 % 10 = 2
    cs = compute_icao_checksum("740812")
    assert cs == 2


def test_compute_icao_checksum_empty():
    """Empty or None string should yield 0."""
    assert compute_icao_checksum("") == 0
    assert compute_icao_checksum(None) == 0


# ---------------------------------------------------------------------------
# Test 2: Glyph Confusion Repairs
# ---------------------------------------------------------------------------

def test_repair_glyph_confusions_date():
    """Verify OCR glyph repair for corrupted date strings."""
    raw = "1O/O5/2OOO"
    repaired, count = repair_glyph_confusions(raw, "dob")
    assert repaired == "10/05/2000"
    assert count == 5


def test_repair_glyph_confusions_doc_number():
    """Verify OCR glyph repair for alphanumeric document numbers."""
    raw = "FGL-2l6739-l3"
    repaired, count = repair_glyph_confusions(raw, "document_number")
    assert repaired == "FGL-216739-13"
    assert count == 2


def test_repair_glyph_confusions_name():
    """Verify OCR glyph repair for name fields where digits should be letters."""
    raw = "J0HN D0E"
    repaired, count = repair_glyph_confusions(raw, "name")
    assert repaired == "JOHN DOE"
    assert count == 2

    raw2 = "AL1CE SM1TH"
    repaired2, count2 = repair_glyph_confusions(raw2, "name")
    assert repaired2 == "ALICE SMITH"
    assert count2 == 2


def test_repair_glyph_confusions_no_change():
    """Clean strings should remain identical with count 0."""
    raw = "10/05/2020"
    repaired, count = repair_glyph_confusions(raw, "issue_date")
    assert repaired == raw
    assert count == 0


def test_repair_glyph_confusions_empty():
    """Empty strings should return unchanged with count 0."""
    assert repair_glyph_confusions("", "name") == ("", 0)
    assert repair_glyph_confusions(None, "dob") == (None, 0)


# ---------------------------------------------------------------------------
# Test 3: Date Parsing & Chronology Audits
# ---------------------------------------------------------------------------

def test_parse_date_flexible():
    """Verify parsing across multiple date formats."""
    d1 = parse_date_flexible("10/05/1990")
    assert d1 == datetime(1990, 5, 10)

    d2 = parse_date_flexible("1990-05-10")
    assert d2 == datetime(1990, 5, 10)

    d3 = parse_date_flexible("10-05-1990")
    assert d3 == datetime(1990, 5, 10)

    assert parse_date_flexible("INVALID_DATE") is None
    assert parse_date_flexible("") is None


def test_validate_date_chronology_valid():
    """Verify clean chronological document dates pass audit."""
    fields = {
        "dob": {"value": "15/04/1985"},
        "issue_date": {"value": "01/06/2018"},
        "expiry_date": {"value": "01/06/2028"},
    }
    audit = validate_date_chronology(fields)
    assert audit["chronology_valid"] is True
    assert audit["status"] == "VERIFIED_CHRONOLOGY"
    assert audit["age_at_issue_years"] == pytest.approx(33.1, abs=0.2)
    assert audit["validity_years"] == pytest.approx(10.0, abs=0.1)
    assert len(audit["errors"]) == 0


def test_validate_date_chronology_future_issue():
    """Verify future issue date is caught as temporal fraud."""
    fields = {
        "dob": {"value": "15/04/1985"},
        "issue_date": {"value": "01/06/2099"},
        "expiry_date": {"value": "01/06/2109"},
    }
    audit = validate_date_chronology(fields)
    assert audit["chronology_valid"] is False
    assert "FUTURE_ISSUE_DATE" in audit["errors"]


def test_validate_date_chronology_issue_before_dob():
    """Verify issue date preceding DOB is caught as temporal violation."""
    fields = {
        "dob": {"value": "15/04/2000"},
        "issue_date": {"value": "01/06/1995"},
        "expiry_date": {"value": "01/06/2005"},
    }
    audit = validate_date_chronology(fields)
    assert audit["chronology_valid"] is False
    assert "ISSUE_PRE_DOB" in audit["errors"]


def test_validate_date_chronology_expiry_before_issue():
    """Verify expiry date preceding issue date is caught."""
    fields = {
        "dob": {"value": "15/04/1985"},
        "issue_date": {"value": "01/06/2020"},
        "expiry_date": {"value": "01/06/2015"},
    }
    audit = validate_date_chronology(fields)
    assert audit["chronology_valid"] is False
    assert "EXPIRY_PRE_ISSUE" in audit["errors"]


def test_validate_date_chronology_excessive_validity():
    """Verify document validity exceeding 25 years is caught."""
    fields = {
        "dob": {"value": "15/04/1985"},
        "issue_date": {"value": "01/06/2010"},
        "expiry_date": {"value": "01/06/2045"},  # 35 years validity
    }
    audit = validate_date_chronology(fields)
    assert audit["chronology_valid"] is False
    assert "EXCESSIVE_VALIDITY" in audit["errors"]


def test_validate_date_chronology_missing_dates():
    """Verify missing dates do not crash the audit and report unverified."""
    fields = {
        "name": {"value": "Jane Doe"},
    }
    audit = validate_date_chronology(fields)
    assert audit["status"] == "DATES_UNAVAILABLE"
    assert audit["chronology_valid"] is True


# ---------------------------------------------------------------------------
# Test 4: MRZ Parsing & VIZ Cross-Validation
# ---------------------------------------------------------------------------

def test_parse_mrz_lines_td1():
    """Verify parsing of standard 3-line TD1 identity card MRZ."""
    line1 = "I<UTOD231458907<<<<<<<<<<<<<<<"
    line2 = "7408122F1204159UTO<<<<<<<<<<<6"
    line3 = "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"

    parsed = parse_mrz_lines([line1, line2, line3])
    assert parsed is not None
    assert parsed["format"] == "TD1"
    assert parsed["document_number"] == "D23145890"
    assert parsed["dob"] == "740812"
    assert parsed["expiry_date"] == "120415"
    assert parsed["surname"] == "ERIKSSON"
    assert parsed["given_names"] == "ANNA MARIA"
    assert parsed["checksums"]["doc_number_valid"] is True
    assert parsed["checksums"]["dob_valid"] is True
    assert parsed["checksums"]["expiry_valid"] is True
    assert parsed["checksums"]["all_valid"] is True


def test_parse_mrz_lines_td3():
    """Verify parsing of standard 2-line TD3 passport MRZ."""
    line1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    line2 = "D231458907UTO7408122F1204159<<<<<<<<<<<<<<<6"

    parsed = parse_mrz_lines([line1, line2])
    assert parsed is not None
    assert parsed["format"] == "TD3"
    assert parsed["document_number"] == "D23145890"
    assert parsed["dob"] == "740812"
    assert parsed["expiry_date"] == "120415"
    assert parsed["surname"] == "ERIKSSON"
    assert parsed["given_names"] == "ANNA MARIA"
    assert parsed["checksums"]["all_valid"] is True


def test_cross_validate_viz_and_mrz():
    """Verify cross-validation between visual inspection zone (VIZ) and MRZ."""
    viz_fields = {
        "document_number": {"value": "D23145890"},
        "dob": {"value": "12/08/1974"},
        "expiry_date": {"value": "15/04/2012"},
        "name": {"value": "Anna Maria Eriksson"},
    }

    mrz_data = {
        "document_number": "D23145890",
        "dob": "740812",
        "expiry_date": "120415",
        "surname": "ERIKSSON",
        "given_names": "ANNA MARIA",
        "checksums": {"all_valid": True},
    }

    xval = cross_validate_viz_and_mrz(viz_fields, mrz_data)
    assert xval["document_number_match"] is True
    assert xval["name_match"] is True
    assert xval["overall_viz_mrz_match"] is True
