"""
Unit Tests for Milestone 4: Canonical Semantic Validation Engine
================================================================
Tests all 8 canonical semantic rules, calendar parsing edge cases, leap years,
and non-punitive conditional availability protocol.
"""

from datetime import datetime
import pytest

from src.semantic_checks import (
    check_age_at_issue_sanity,
    check_anachronism,
    check_chronology_order,
    check_document_number_format,
    check_duplicate_field_contradiction,
    check_impossible_dates,
    check_name_structure_sanity,
    check_validity_window_sanity,
    parse_calendar_date,
    run_semantic_rule_battery,
)


def test_parse_calendar_date_valid():
    """Verify parsing valid dates in various standard formats."""
    dt, err = parse_calendar_date("15/05/1990")
    assert err is None
    assert dt == datetime(1990, 5, 15)

    dt, err = parse_calendar_date("1990-05-15")
    assert err is None
    assert dt == datetime(1990, 5, 15)

    dt, err = parse_calendar_date("29/02/2024")  # 2024 is a leap year
    assert err is None
    assert dt == datetime(2024, 2, 29)


def test_parse_calendar_date_leap_and_impossible():
    """Verify detection of impossible dates and non-leap year Feb 29."""
    # 2023 is not a leap year
    dt, err = parse_calendar_date("29/02/2023")
    assert dt is None
    assert "February 29 invalid in non-leap year 2023" in err

    # April has only 30 days
    dt, err = parse_calendar_date("31/04/2020")
    assert dt is None
    assert "exceeds maximum 30 days" in err

    # February 30 is always impossible
    dt, err = parse_calendar_date("30/02/2020")
    assert dt is None
    assert "exceeds maximum 29 days" in err

    # Month 13 out of range
    dt, err = parse_calendar_date("15/13/2020")
    assert dt is None
    assert "Month 13 out of calendar range" in err


def test_rule1_impossible_dates():
    """Test Rule 1 on valid, invalid, and empty fields."""
    valid_fields = {
        "dob": {"value": "15/05/1990"},
        "issue_date": {"value": "10/01/2020"},
        "expiry_date": {"value": "10/01/2030"},
    }
    res = check_impossible_dates(valid_fields)
    assert res["status"] == "PASS"

    invalid_fields = {
        "dob": {"value": "30/02/1990"},
        "issue_date": {"value": "10/01/2020"},
    }
    res = check_impossible_dates(invalid_fields)
    assert res["status"] == "FAIL"
    assert "30/02/1990" in res["detail"]

    empty_fields = {}
    res = check_impossible_dates(empty_fields)
    assert res["status"] == "NOT_APPLICABLE"


def test_rule2_chronology_order():
    """Test Rule 2 on valid chronology and temporal inversions."""
    valid_fields = {
        "dob": "15/05/1990",
        "issue_date": "10/01/2020",
        "expiry_date": "10/01/2030",
    }
    res = check_chronology_order(valid_fields)
    assert res["status"] == "PASS"

    # Inversion: Issue date before DOB
    inverted_fields = {
        "dob": "15/05/1990",
        "issue_date": "10/01/1985",
        "expiry_date": "10/01/2030",
    }
    res = check_chronology_order(inverted_fields)
    assert res["status"] == "FAIL"
    assert "precedes or equals DOB" in res["detail"]

    # Inversion: Expiry date before Issue date
    inverted_expiry = {
        "dob": "15/05/1990",
        "issue_date": "10/01/2020",
        "expiry_date": "10/01/2015",
    }
    res = check_chronology_order(inverted_expiry)
    assert res["status"] == "FAIL"
    assert "precedes or equals Issue date" in res["detail"]


def test_rule3_age_at_issue_sanity():
    """Test Rule 3 on normal adults, minors, and negative ages."""
    adult_fields = {"dob": "15/05/1990", "issue_date": "15/05/2020"}
    res = check_age_at_issue_sanity(adult_fields)
    assert res["status"] == "PASS"
    assert res["evidence"]["age_at_issue_years"] == pytest.approx(30.0, rel=0.05)

    minor_fields = {"dob": "15/05/2010", "issue_date": "15/05/2020"}
    res = check_age_at_issue_sanity(minor_fields)
    assert res["status"] == "PASS"
    assert "minor" in res["detail"].lower()

    negative_fields = {"dob": "15/05/2010", "issue_date": "15/05/2005"}
    res = check_age_at_issue_sanity(negative_fields)
    assert res["status"] == "FAIL"
    assert "Negative age" in res["detail"]


def test_rule4_validity_window_sanity():
    """Test Rule 4 on valid 10-year validity and excessive 30-year validity."""
    valid_fields = {"issue_date": "01/01/2020", "expiry_date": "01/01/2030"}
    res = check_validity_window_sanity(valid_fields, max_validity_years=25.0)
    assert res["status"] == "PASS"

    excessive_fields = {"issue_date": "01/01/2020", "expiry_date": "01/01/2055"}
    res = check_validity_window_sanity(excessive_fields, max_validity_years=25.0)
    assert res["status"] == "FAIL"
    assert "Excessive validity period" in res["detail"]


def test_rule5_anachronism():
    """Test Rule 5 on past, current, and future issue dates."""
    past_fields = {"issue_date": "01/01/2020"}
    res = check_anachronism(past_fields, tolerance_days=30)
    assert res["status"] == "PASS"

    future_fields = {"issue_date": "01/01/2045"}
    res = check_anachronism(future_fields, tolerance_days=30)
    assert res["status"] == "FAIL"
    assert "Anachronistic future issue date" in res["detail"]


def test_rule6_document_number_format():
    """Test Rule 6 schema regex enforcement."""
    valid_fields = {"document_number": "FGL-102948-19"}
    res = check_document_number_format(valid_fields, doc_type="forgelensia")
    assert res["status"] == "PASS"

    invalid_fields = {"document_number": "INVALID-123"}
    res = check_document_number_format(invalid_fields, doc_type="forgelensia")
    assert res["status"] == "FAIL"

    passport_fields = {"document_number": "L898902C3"}
    res = check_document_number_format(passport_fields, doc_type="passport")
    assert res["status"] == "PASS"


def test_rule7_duplicate_field_contradiction():
    """Test Rule 7 internal consistency."""
    clean_fields = {
        "dob": {"raw_text": "DOB: 15/05/1990", "value": "15/05/1990"},
        "document_number": {"raw_text": "FGL-102948-19", "value": "FGL-102948-19"},
    }
    res = check_duplicate_field_contradiction(clean_fields)
    assert res["status"] == "PASS"

    contradicting_fields = {
        "document_number": {"raw_text": "FGL-999999-99", "value": "FGL-102948-19"}
    }
    res = check_duplicate_field_contradiction(contradicting_fields)
    assert res["status"] == "FAIL"


def test_rule8_name_structure_sanity():
    """Test Rule 8 dummy name token detection and alphabetic character requirements."""
    valid_fields = {"name": "Maria Hernandez"}
    res = check_name_structure_sanity(valid_fields)
    assert res["status"] == "PASS"

    dummy_fields = {"name": "JOHN DOE TEST SAMPLE"}
    res = check_name_structure_sanity(dummy_fields)
    assert res["status"] == "FAIL"
    assert "placeholder tokens" in res["detail"]

    numeric_fields = {"name": "12345 6789"}
    res = check_name_structure_sanity(numeric_fields)
    assert res["status"] == "FAIL"


def test_master_semantic_battery_execution():
    """Test master semantic rule battery across clean and tampered credentials."""
    clean_fields = {
        "name": {"value": "Victoria Vance"},
        "document_number": {"value": "FGL-482019-21"},
        "dob": {"value": "12/08/1988"},
        "issue_date": {"value": "15/03/2018"},
        "expiry_date": {"value": "15/03/2028"},
    }
    battery = run_semantic_rule_battery(clean_fields, doc_type="forgelensia")
    assert battery["semantic_verdict"] == "PASSED_SEMANTIC_AUDIT"
    assert battery["is_valid"] is True
    assert battery["failed_count"] == 0
    assert battery["passed_count"] >= 5

    # Tampered credential with impossible date
    tampered_fields = dict(clean_fields)
    tampered_fields["dob"] = {"value": "30/02/1988"}
    battery_tampered = run_semantic_rule_battery(tampered_fields, doc_type="forgelensia")
    assert battery_tampered["semantic_verdict"] == "SUSPECT_SEMANTIC_VIOLATION"
    assert battery_tampered["is_valid"] is False
    assert "impossible_dates" in battery_tampered["failed_checks"]
