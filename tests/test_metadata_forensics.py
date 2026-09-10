"""
Unit Tests for Milestone 4: JPEG EXIF & Image Provenance Forensics
================================================================
Tests software signature auditing, timestamp chronology validation,
and non-punitive stripped metadata handling.
"""

import os
import pytest
from src.metadata_forensics import audit_metadata_provenance, extract_image_metadata


def test_audit_software_blacklist_detection():
    """Verify detection of blacklisted photo editing software signatures."""
    # Suspect Photoshop signature
    meta_ps = {
        "has_exif": True,
        "software": "Adobe Photoshop 2024 (Windows)",
        "make": "Canon",
        "model": "EOS R5",
    }
    res = audit_metadata_provenance(meta_ps)
    assert res["is_tampered"] is True
    assert res["provenance_verdict"] == "SUSPECT_SOFTWARE_FINGERPRINT"
    assert res["checks"]["software_signature"]["status"] == "FAIL"

    # Suspect GIMP signature
    meta_gimp = {
        "has_exif": True,
        "software": "GIMP 2.10.34",
    }
    res_g = audit_metadata_provenance(meta_gimp)
    assert res_g["is_tampered"] is True
    assert res_g["provenance_verdict"] == "SUSPECT_SOFTWARE_FINGERPRINT"

    # Clean camera firmware signature
    meta_clean = {
        "has_exif": True,
        "software": "Ver.1.02",
        "make": "Nikon",
        "model": "D850",
    }
    res_c = audit_metadata_provenance(meta_clean)
    assert res_c["is_tampered"] is False
    assert res_c["checks"]["software_signature"]["status"] == "PASS"


def test_audit_timestamp_chronology():
    """Verify timestamp sequence checks and anomaly detection."""
    # Consistent timestamps
    meta_valid_ts = {
        "has_exif": True,
        "datetime_original": "2023:05:15 10:30:00",
        "datetime_digitized": "2023:05:15 10:30:02",
        "datetime_modified": "2023:05:15 10:30:05",
    }
    res = audit_metadata_provenance(meta_valid_ts)
    assert res["checks"]["timestamp_chronology"]["status"] == "PASS"

    # Inverted timestamps (digitized before original capture)
    meta_inverted_ts = {
        "has_exif": True,
        "datetime_original": "2023:05:15 12:00:00",
        "datetime_digitized": "2023:05:15 10:00:00",  # 2 hours earlier
    }
    res_inv = audit_metadata_provenance(meta_inverted_ts)
    assert res_inv["checks"]["timestamp_chronology"]["status"] == "FAIL"
    assert "inversion" in res_inv["checks"]["timestamp_chronology"]["detail"].lower()
    assert res_inv["is_tampered"] is True


def test_stripped_metadata_non_punitive():
    """Verify that images lacking EXIF return STRIPPED_OR_ABSENT rather than FAIL."""
    meta_stripped = {
        "has_exif": False,
        "software": None,
        "make": None,
        "model": None,
        "datetime_original": None,
        "raw_strings": [],
    }
    res = audit_metadata_provenance(meta_stripped)
    assert res["is_tampered"] is False
    assert res["provenance_verdict"] == "STRIPPED_OR_ABSENT"
    assert res["checks"]["exif_presence"]["status"] == "NOT_APPLICABLE"
    assert res["checks"]["software_signature"]["status"] == "NOT_APPLICABLE"


def test_extract_metadata_real_file():
    """Verify metadata extraction on an existing workspace image."""
    img_path = "data/generated/images/src_0000_genuine.jpg"
    if os.path.exists(img_path):
        meta = extract_image_metadata(img_path)
        assert meta["file_path"] == img_path
        assert meta["file_size_bytes"] > 0
        assert "image_size" in meta


def test_xmp_packet_provenance_audit(tmp_path):
    """Verify detection of Photoshop signatures embedded in raw XMP byte packets."""
    # Create synthetic test file with embedded XMP packet
    dummy_img = tmp_path / "xmp_tampered.jpg"
    xmp_payload = (
        b"\xff\xd8\xff\xe1\x00\x80Exif\x00\x00"
        b"<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
        b"<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
        b"<xmp:CreatorTool>Adobe Photoshop CC 2024</xmp:CreatorTool>"
        b"<photoshop:History>tampered</photoshop:History>"
        b"</rdf:RDF></x:xmpmeta>"
        b"\xff\xd9"
    )
    dummy_img.write_bytes(xmp_payload)

    meta = extract_image_metadata(str(dummy_img))
    assert meta["has_xmp"] is True
    assert meta["xmp_creator_tool"] == "Adobe Photoshop CC 2024"
    assert meta["has_photoshop_history"] is True

    audit = audit_metadata_provenance(meta)
    assert audit["is_tampered"] is True
    assert audit["provenance_verdict"] == "SUSPECT_SOFTWARE_FINGERPRINT"

