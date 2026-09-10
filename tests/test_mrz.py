"""
Unit Tests for Milestone 4: ICAO Doc 9303 MRZ Engine & Hypothesis Repair
========================================================================
Tests checksum calculation, TD1/TD2/TD3 parsing, composite checksums,
confusable self-healing, and VIZ-to-MRZ cross-validation.
"""

import pytest
from src.mrz import (
    compute_icao_checksum,
    cross_validate_viz_and_mrz,
    disambiguate_mrz_checksums,
    disambiguate_mrz_field,
    format_mrz_date,
    parse_mrz,
    parse_mrz_name,
)


def test_compute_icao_checksum():
    """Verify cyclic [7, 3, 1] modulo-10 algorithm against official ICAO Doc 9303 examples."""
    # Standard ICAO test vectors
    # Document number: L898902C3 -> check digit is 6
    assert compute_icao_checksum("L898902C3") == 6

    # DOB: 740812 -> check digit is 2
    assert compute_icao_checksum("740812") == 2

    # Expiry: 120415 -> check digit is 9
    assert compute_icao_checksum("120415") == 9

    # Characters with filler '<'
    assert compute_icao_checksum("A12345<<<") == compute_icao_checksum("A12345000")


def test_format_mrz_date():
    """Verify conversion of YYMMDD to DD/MM/YYYY."""
    assert format_mrz_date("740812", is_dob=True) == "12/08/1974"
    assert format_mrz_date("120415", is_dob=False) == "15/04/2012"
    assert format_mrz_date("999999", is_dob=True) is None  # Invalid month/day


def test_parse_mrz_name():
    """Verify parsing of surname and given names separated by '<<'."""
    surname, given, full_name = parse_mrz_name("ERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<")
    assert surname == "ERIKSSON"
    assert given == "ANNA MARIA"
    assert full_name == "ANNA MARIA ERIKSSON"


def test_parse_td3_passport():
    """Verify TD3 format (2x44) passport parsing and checksum validation."""
    lines = [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    ]
    parsed = parse_mrz(lines)
    assert parsed is not None
    assert parsed["format"] == "TD3"
    assert parsed["document_type"] == "P"
    assert parsed["issuing_country"] == "UTO"
    assert parsed["document_number"] == "L898902C3"
    assert parsed["document_number_valid"] is True
    assert parsed["dob"] == "740812"
    assert parsed["dob_valid"] is True
    assert parsed["expiry_date"] == "120415"
    assert parsed["expiry_date_valid"] is True
    assert parsed["all_checksums_valid"] is True


def test_parse_td1_id_card():
    """Verify TD1 format (3x30) ID card parsing."""
    lines = [
        "I<UTOD231458907<<<<<<<<<<<<<<<",
        "7408122F1204159UTO<<<<<<<<<<<6",
        "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
    ]
    parsed = parse_mrz(lines)
    assert parsed is not None
    assert parsed["format"] == "TD1"
    assert parsed["document_type"] == "I"
    assert parsed["issuing_country"] == "UTO"
    assert parsed["document_number"] == "D23145890"
    assert parsed["document_number_valid"] is True
    assert parsed["dob"] == "740812"
    assert parsed["dob_valid"] is True
    assert parsed["expiry_date"] == "120415"
    assert parsed["expiry_date_valid"] is True
    assert parsed["surname"] == "ERIKSSON"
    assert parsed["given_names"] == "ANNA MARIA"


def test_hypothesis_disambiguation_optical_confusable():
    """Verify 1-flip hypothesis repair of 'O' vs '0' optical confusion."""
    # 'L8989O2C3' has letter 'O' instead of digit '0'
    repaired, cand, exp = disambiguate_mrz_field("L8989O2C3", "6")
    assert repaired is True
    assert cand == "L898902C3"
    assert "repaired" in exp.lower()

    # Unrepairable forgery
    repaired, cand, exp = disambiguate_mrz_field("L999999C9", "6")
    assert repaired is False


def test_disambiguate_mrz_checksums_engine():
    """Verify end-to-end disambiguation on a parsed MRZ object."""
    corrupted_td3 = [
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L8989O2C36UTO7408122F1204159ZE184226B<<<<<10",  # 'O' instead of '0'
    ]
    parsed = parse_mrz(corrupted_td3)
    assert parsed["document_number_valid"] is False

    disambiguated = disambiguate_mrz_checksums(parsed)
    assert disambiguated["status"] == "PASS"
    assert disambiguated["verdict"] == "MRZ_REPAIRED_OPTICAL_CONFUSION"
    assert disambiguated["document_number_valid"] is True
    assert disambiguated["document_number"] == "L898902C3"


def test_cross_validate_viz_and_mrz():
    """Verify VIZ-to-MRZ cross-validation concordance and discrepancy detection."""
    mrz_data = {
        "document_number": "L898902C3",
        "dob": "740812",
        "expiry_date": "120415",
        "name": "ANNA MARIA ERIKSSON",
        "surname": "ERIKSSON",
    }

    # Concordant VIZ
    viz_concordant = {
        "document_number": {"value": "L898902C3"},
        "dob": {"value": "12/08/1974"},
        "expiry_date": {"value": "15/04/2012"},
        "name": {"value": "Anna Maria Eriksson"},
    }
    res = cross_validate_viz_and_mrz(viz_concordant, mrz_data)
    assert res["is_concordant"] is True
    assert res["verdict"] == "VIZ_MRZ_FULLY_CONCORDANT"
    assert res["concordance_score"] == 1.0

    # Conflicting VIZ (Imposter)
    viz_imposter = {
        "document_number": {"value": "L898902C3"},
        "dob": {"value": "12/08/1974"},
        "expiry_date": {"value": "15/04/2012"},
        "name": {"value": "Robert Langdon"},  # Different name
    }
    res_imp = cross_validate_viz_and_mrz(viz_imposter, mrz_data)
    assert res_imp["is_concordant"] is False
    assert res_imp["verdict"] == "VIZ_MRZ_CONTRADICTION_FRAUD"
    assert len(res_imp["mismatches"]) >= 1
