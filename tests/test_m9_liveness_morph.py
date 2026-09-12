"""
Unit and Integration Tests for Milestone 9:
- Presentation Attack Detection (Liveness / Anti-Spoofing PAD)
- Facial Morphing Attack Forensics (S-MAD & D-MAD)
- Mass-Scale High-Volume Batch Streaming (100+ Samples with Fault Isolation & Ledgers)
- Milestone 9 REST Microservice Endpoints
"""

import os
import shutil
import tempfile
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from src.liveness_pad import evaluate_face_liveness
from src.morph_forensics import evaluate_photo_morphing
from src.batch_streaming import process_bulk_batch, stream_document_directory
from api.app import app


@pytest.fixture(scope="module")
def sample_face_images(tmp_path_factory):
    """Generate synthetic genuine face, screen replay (Moiré), and morphed test faces."""
    tmp_dir = tmp_path_factory.mktemp("m9_faces")
    
    # 1. Genuine clean face-like pattern (smooth skin tones, realistic illumination)
    genuine_face = np.full((256, 256, 3), (180, 200, 230), dtype=np.uint8)
    # Add gentle gradients and simulated facial eyes/features
    cv2.circle(genuine_face, (85, 100), 20, (140, 160, 190), -1)
    cv2.circle(genuine_face, (170, 100), 20, (140, 160, 190), -1)
    # Corneal reflection catchlights (subtle bright specular spots)
    cv2.circle(genuine_face, (88, 98), 3, (255, 255, 255), -1)
    cv2.circle(genuine_face, (173, 98), 3, (255, 255, 255), -1)
    # Soft mouth curve
    cv2.ellipse(genuine_face, (128, 175), (45, 18), 0, 0, 180, (120, 130, 180), 3)
    genuine_path = str(tmp_dir / "genuine_selfie.png")
    cv2.imwrite(genuine_path, genuine_face)

    # 2. Screen replay attack (intense periodic Moiré high-frequency grid pattern)
    screen_spoof = genuine_face.copy().astype(np.float32)
    y_coords, x_coords = np.mgrid[0:256, 0:256]
    # Physical LCD/OLED raster scanlines create strong Fourier peaks along coordinate axes
    moire_grid = 35.0 * np.sin(x_coords * 0.6) + 35.0 * np.sin(y_coords * 0.6)
    screen_spoof = np.clip(screen_spoof + moire_grid[:, :, None], 0, 255).astype(np.uint8)
    screen_path = str(tmp_dir / "screen_replay_spoof.png")
    cv2.imwrite(screen_path, screen_spoof)

    # 3. Print attack (sharp LBP micro-texture with clipped matte gamut)
    print_spoof = genuine_face.copy()
    # Add halftoning / speckle noise
    noise = np.random.RandomState(42).randint(-35, 35, (256, 256, 3))
    print_spoof = np.clip(print_spoof.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    print_path = str(tmp_dir / "print_matte_spoof.png")
    cv2.imwrite(print_path, print_spoof)

    # 4. Morphed Portrait (double-exposure edge ghosting with blending artifacts)
    morphed_portrait = genuine_face.copy()
    # Introduce secondary ghosted contour around facial boundary
    cv2.circle(morphed_portrait, (92, 102), 22, (100, 120, 160), 2)
    cv2.circle(morphed_portrait, (177, 102), 22, (100, 120, 160), 2)
    cv2.ellipse(morphed_portrait, (132, 178), (48, 20), 0, 0, 180, (90, 100, 150), 3)
    morph_path = str(tmp_dir / "morphed_portrait.png")
    cv2.imwrite(morph_path, morphed_portrait)

    return {
        "genuine": genuine_path,
        "screen_spoof": screen_path,
        "print_spoof": print_path,
        "morphed": morph_path,
    }


def test_liveness_pad_signals(sample_face_images):
    """Verify presentation attack detection on clean vs replay vs print attacks."""
    # 1. Test clean genuine selfie
    gen_res = evaluate_face_liveness(sample_face_images["genuine"])
    assert "is_live" in gen_res
    assert "liveness_score" in gen_res
    assert "spoof_risk" in gen_res
    assert "signals" in gen_res
    assert gen_res["signals"]["moire_screen_replay"]["score"] < 0.85

    # 2. Test screen replay Moiré spoof
    screen_res = evaluate_face_liveness(sample_face_images["screen_spoof"])
    assert screen_res["spoof_risk"] > gen_res["spoof_risk"]
    assert screen_res["signals"]["moire_screen_replay"]["score"] > gen_res["signals"]["moire_screen_replay"]["score"]

    # 3. Test print matte attack
    print_res = evaluate_face_liveness(sample_face_images["print_spoof"])
    assert print_res["signals"]["lbp_texture_entropy"]["score"] > gen_res["signals"]["lbp_texture_entropy"]["score"]


def test_morphing_forensics_smad_and_dmad(sample_face_images):
    """Verify single-image (S-MAD) and differential (D-MAD) morphing forensics."""
    # 1. Test clean portrait S-MAD
    clean_morph = evaluate_photo_morphing(sample_face_images["genuine"])
    assert "morphing_detected" in clean_morph
    assert "morphing_score" in clean_morph
    assert "signals" in clean_morph
    assert "smad_single_image" in clean_morph["signals"]

    # 2. Test ghosted morphed portrait S-MAD
    morphed_res = evaluate_photo_morphing(sample_face_images["morphed"])
    assert morphed_res["morphing_score"] >= clean_morph["morphing_score"]

    # 3. Test differential D-MAD with reference selfie
    dmad_res = evaluate_photo_morphing(
        sample_face_images["morphed"],
        selfie_input=sample_face_images["genuine"]
    )
    assert dmad_res["signals"]["dmad_differential"]["evaluated"] is True
    assert "dmad_score" in dmad_res["signals"]["dmad_differential"]


def test_batch_streaming_50_samples_with_ledgers(tmp_path):
    """Simulate 50 document processing with streaming chunks, fault isolation, and ledgers."""
    docs_dir = tmp_path / "bulk_docs"
    docs_dir.mkdir()

    # Generate 50 test document samples (45 valid synthetic docs + 5 intentionally corrupted/zero-byte files)
    file_paths = []
    base_img = np.full((300, 450, 3), 245, dtype=np.uint8)
    cv2.putText(base_img, "PASSPORT FORGELENS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(base_img, "NAME: SPECIMEN TEST", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

    for i in range(1, 46):
        doc_img = base_img.copy()
        cv2.putText(doc_img, f"DOC-NUM: A{i:06d}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
        p = str(docs_dir / f"doc_{i:03d}.jpg")
        cv2.imwrite(p, doc_img)
        file_paths.append(p)

    # 5 intentionally corrupted/broken files to prove fault tolerance
    for i in range(46, 51):
        corrupt_p = str(docs_dir / f"corrupt_{i:03d}.jpg")
        with open(corrupt_p, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_STREAM")
        file_paths.append(corrupt_p)

    assert len(file_paths) == 50

    # Stream directory test
    streamed_chunks = list(stream_document_directory(str(docs_dir), chunk_size=10))
    total_streamed = sum(len(c) for c in streamed_chunks)
    assert total_streamed == 50

    csv_ledger = str(tmp_path / "compliance_ledger.csv")
    jsonl_ledger = str(tmp_path / "compliance_ledger.jsonl")

    progress_events = []
    def on_progress(ev):
        progress_events.append(ev)

    # Run batch processing engine
    summary = process_bulk_batch(
        file_paths=file_paths,
        max_workers=4,
        ledger_csv=csv_ledger,
        ledger_jsonl=jsonl_ledger,
        progress_callback=on_progress,
    )

    # Assertions
    assert summary["total_screened"] == 50
    assert summary["total_verified"] + summary["total_flagged"] == 50
    assert summary["throughput_docs_per_sec"] > 0
    assert summary["mean_latency_ms"] > 0
    assert len(progress_events) == 50

    # Verify CSV ledger exists and has 51 lines (header + 50 docs)
    assert os.path.exists(csv_ledger)
    with open(csv_ledger, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 51
        assert "document_id" in lines[0]
        assert "risk_score" in lines[0]
        assert "decision" in lines[0]

    # Verify JSONL ledger exists and has 50 lines
    assert os.path.exists(jsonl_ledger)
    with open(jsonl_ledger, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 50


def test_api_m9_liveness_and_morph_endpoints(sample_face_images):
    """Test FastAPI REST endpoints for Liveness and Morphing."""
    client = TestClient(app)

    # 1. POST /api/v1/liveness/detect
    with open(sample_face_images["genuine"], "rb") as f:
        resp = client.post(
            "/api/v1/liveness/detect",
            files={"selfie": ("selfie.png", f, "image/png")}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "is_live" in data
    assert "liveness_score" in data
    assert "spoof_risk" in data
    assert "signals" in data

    # 2. POST /api/v1/morph/detect (Single portrait S-MAD)
    with open(sample_face_images["morphed"], "rb") as f:
        resp = client.post(
            "/api/v1/morph/detect",
            files={"portrait": ("portrait.png", f, "image/png")}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "morphing_detected" in data
    assert "morphing_score" in data
    assert "morph_tier" in data

    # 3. POST /api/v1/morph/detect (Differential D-MAD with reference selfie)
    with open(sample_face_images["morphed"], "rb") as f1, open(sample_face_images["genuine"], "rb") as f2:
        resp = client.post(
            "/api/v1/morph/detect",
            files={
                "portrait": ("portrait.png", f1, "image/png"),
                "selfie": ("selfie.png", f2, "image/png"),
            }
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["signals"]["dmad_differential"]["evaluated"] is True


def test_diagnostic_cards_and_batch_analytics(sample_face_images, tmp_path):
    """Verify generation of visual PAD & MAD diagnostic cards and executive analytics."""
    from src.liveness_pad import generate_liveness_diagnostic_card
    from src.morph_forensics import generate_morph_diagnostic_card
    from src.batch_streaming import generate_batch_analytics_report

    # 1. PAD visual card
    pad_card = str(tmp_path / "pad_card.png")
    out_pad = generate_liveness_diagnostic_card(sample_face_images["genuine"], output_path=pad_card)
    assert os.path.exists(out_pad)
    assert os.path.getsize(out_pad) > 1000

    # 2. MAD visual card
    mad_card = str(tmp_path / "mad_card.png")
    out_mad = generate_morph_diagnostic_card(sample_face_images["morphed"], selfie_input=sample_face_images["genuine"], output_path=mad_card)
    assert os.path.exists(out_mad)
    assert os.path.getsize(out_mad) > 1000

    # 3. Batch Analytics
    mock_batch = {
        "total_screened": 10,
        "total_verified": 8,
        "total_flagged": 2,
        "throughput_docs_per_sec": 4.5,
        "results": [
            {"filename": f"doc_{i}.jpg", "status": "PASS", "decision": "VERIFIED", "risk_score": 5.0, "attack_type_guess": "none", "pipeline_latency_ms": 220.0}
            for i in range(8)
        ] + [
            {"filename": "doc_8.jpg", "status": "FAIL", "decision": "HIGH_RISK", "risk_score": 95.0, "attack_type_guess": "date_edit", "pipeline_latency_ms": 310.0},
            {"filename": "doc_9.jpg", "status": "FAIL", "decision": "HIGH_RISK", "risk_score": 88.0, "attack_type_guess": "photo_swap", "pipeline_latency_ms": 290.0},
        ]
    }
    rep_md = str(tmp_path / "report.md")
    rep_json = str(tmp_path / "report.json")
    analytics = generate_batch_analytics_report(mock_batch, output_md=rep_md, output_json=rep_json)
    assert analytics["total_screened"] == 10
    assert analytics["pass_rate_pct"] == 80.0
    assert os.path.exists(rep_md)
    assert os.path.exists(rep_json)
