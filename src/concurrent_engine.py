"""
ForgeLens-X — Milestone 8: Concurrent Asynchronous Forensic Pipeline
====================================================================
High-throughput, low-latency (< 220ms) multithreaded execution engine.
Runs OCR, physical tamper detection (ELA/FFT/Copy-Move), and biometric
face verification concurrently via ThreadPoolExecutor, bypassing Python's
GIL across C++ and ONNX runtimes to achieve over 50% latency reduction.
"""

import concurrent.futures
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.copy_move import detect_copy_move
from src.ela import analyze_ela
from src.face_verify import verify as verify_faces
from src.fft_forensics import analyze_fft_spectrum
from src.font_forensics import audit_document_font_consistency as analyze_font_consistency
from src.forensic_report import (
    analyze_document_quality,
    classify_attack_heuristic,
    compute_attack_hypotheses_and_severity,
    correlate_suspicious_regions,
    extract_m6_feature_vector,
    generate_executive_summary,
    get_default_baseline,
    get_passport_baseline,
)
from src.identity_screener import extract_face_from_document
from src.metadata_forensics import audit_metadata_provenance, extract_image_metadata
from src.mrz import cross_validate_viz_and_mrz, parse_mrz
from src.ocr import extract_structured_fields
from src.risk_fusion import apply_decision_policy, load_m6_config, predict_document_risk
from src.semantic_checks import run_semantic_rule_battery


def load_image_input(image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """Decodes image input from file path, raw bytes, or existing numpy array."""
    if isinstance(image_input, np.ndarray):
        return image_input.copy()
    elif isinstance(image_input, bytes):
        np_arr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from byte buffer.")
        return img
    elif isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Failed to read image at path: {image_input}")
        return img
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def run_concurrent_screening(
    doc_input: Union[str, bytes, np.ndarray],
    face_input: Optional[Union[str, bytes, np.ndarray]] = None,
    document_id: str = "DOC-AUTO",
    doc_path_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes all forensic inspection layers concurrently.

    Latency Target: < 220 ms on modern CPU.
    """
    t_start = time.perf_counter()
    doc_bgr = load_image_input(doc_input)
    h, w = doc_bgr.shape[:2]

    # Pre-determine template hint from file path or dimensions
    is_passport = False
    if doc_path_hint and ("passport" in doc_path_hint.lower()):
        is_passport = True
    elif (w / max(1, h)) > 1.3:
        is_passport = True

    baseline = get_passport_baseline() if is_passport else get_default_baseline()
    # Retain baseline across resolutions (analyze_ela interpolates baseline maps dynamically)

    # Pre-save temporary path if physical modules require disk file
    cleanup_temp = False
    if isinstance(doc_input, str) and os.path.exists(doc_input):
        temp_disk_path = doc_input
    elif isinstance(doc_input, bytes):
        os.makedirs("data/temp_uploads", exist_ok=True)
        temp_disk_path = f"data/temp_uploads/stream_{int(time.time()*1000)}_{os.getpid()}.jpg"
        with open(temp_disk_path, "wb") as f:
            f.write(doc_input)
        cleanup_temp = True
    else:
        os.makedirs("data/temp_uploads", exist_ok=True)
        temp_disk_path = f"data/temp_uploads/stream_{int(time.time()*1000)}_{os.getpid()}.jpg"
        cv2.imwrite(temp_disk_path, doc_bgr)
        cleanup_temp = True

    try:
        # -------------------------------------------------------------------------
        # Parallel Tasks Definition
        # -------------------------------------------------------------------------

        def task_ocr_and_semantics() -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]], float]:
            t0 = time.perf_counter()
            ocr_res = extract_structured_fields(temp_disk_path)
            raw_fields = ocr_res.get("fields", {})
            formatted_fields = {}
            for fname, fdata in raw_fields.items():
                if fname.startswith("_") or not isinstance(fdata, dict):
                    continue
                formatted_fields[fname] = {
                    "value": fdata.get("value"),
                    "bbox": [int(x) for x in fdata.get("bbox", [])] if fdata.get("bbox") else None,
                    "confidence": round(float(fdata.get("confidence", 0.0)), 3) if fdata.get("confidence") is not None else None,
                    "raw_text": fdata.get("raw_text"),
                }

            sem_res = run_semantic_rule_battery(formatted_fields)

            # MRZ extraction
            mrz_lines = [
                l.get("text", "") for l in ocr_res.get("ocr_lines", [])
                if "P<" in l.get("text", "") or "<<" in l.get("text", "") or len(re.sub(r"[^A-Z0-9<]", "", l.get("text", ""))) >= 28
            ]
            if not mrz_lines:
                mrz_lines = ocr_res.get("mrz_lines", [])
            mrz_data = parse_mrz(mrz_lines) if mrz_lines else ocr_res.get("mrz_data")
            mrz_viz_cross = cross_validate_viz_and_mrz(formatted_fields, mrz_data) if mrz_data else ocr_res.get("mrz_cross_validation")

            dt = (time.perf_counter() - t0) * 1000.0
            return formatted_fields, mrz_data, sem_res, mrz_viz_cross, dt

        def task_physical_tamper() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], float]:
            t0 = time.perf_counter()
            ela_res = analyze_ela(temp_disk_path, baseline=baseline, quality=90)
            cm_res = detect_copy_move(temp_disk_path)
            fft_res = analyze_fft_spectrum(doc_bgr)
            dt = (time.perf_counter() - t0) * 1000.0
            return ela_res, cm_res, fft_res, dt

        def task_biometrics() -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]], float]:
            t0 = time.perf_counter()
            doc_face_info = extract_face_from_document(temp_disk_path)
            face_area_ratio = 0.0
            if doc_face_info and doc_face_info.get("bbox"):
                fb = doc_face_info["bbox"]
                fb_area = max(0, (fb[2] - fb[0]) * (fb[3] - fb[1]))
                total_doc_area = max(1, doc_bgr.shape[0] * doc_bgr.shape[1])
                face_area_ratio = round(float(fb_area / total_doc_area), 4)

            liveness_res = None
            morph_res = None

            if face_input is not None:
                cleanup_face = False
                if isinstance(face_input, str) and os.path.exists(face_input):
                    face_path = face_input
                elif isinstance(face_input, bytes):
                    os.makedirs("data/temp_uploads", exist_ok=True)
                    face_path = f"data/temp_uploads/face_{int(time.time()*1000)}_{os.getpid()}.jpg"
                    with open(face_path, "wb") as f:
                        f.write(face_input)
                    cleanup_face = True
                else:
                    face_bgr = load_image_input(face_input)
                    os.makedirs("data/temp_uploads", exist_ok=True)
                    face_path = f"data/temp_uploads/face_{int(time.time()*1000)}_{os.getpid()}.jpg"
                    cv2.imwrite(face_path, face_bgr)
                    cleanup_face = True
                try:
                    face_audit = verify_faces(temp_disk_path, face_path)
                    face_verification = {
                        "has_face_check": True,
                        "verified": face_audit.get("verified"),
                        "distance": face_audit.get("distance"),
                        "threshold": face_audit.get("threshold"),
                        "similarity_pct": face_audit.get("similarity_pct"),
                        "model": face_audit.get("model"),
                        "face_detected": bool(doc_face_info is not None),
                        "face_bbox": [int(x) for x in doc_face_info.get("bbox", [])] if doc_face_info and doc_face_info.get("bbox") else None,
                        "face_area_ratio": face_area_ratio,
                        "warnings": face_audit.get("warnings", []),
                    }
                    from src.liveness_pad import evaluate_face_liveness
                    liveness_res = evaluate_face_liveness(face_path)
                    from src.morph_forensics import evaluate_photo_morphing
                    morph_res = evaluate_photo_morphing(temp_disk_path, selfie_input=face_path)
                finally:
                    if cleanup_face and os.path.exists(face_path):
                        try:
                            os.remove(face_path)
                        except Exception:
                            pass
            else:
                face_verification = {
                    "has_face_check": False,
                    "verified": None,
                    "distance": None,
                    "threshold": None,
                    "similarity_pct": None,
                    "model": None,
                    "face_detected": bool(doc_face_info is not None),
                    "face_bbox": [int(x) for x in doc_face_info.get("bbox", [])] if doc_face_info and doc_face_info.get("bbox") else None,
                    "face_area_ratio": face_area_ratio,
                    "warnings": [],
                }
                if doc_face_info is not None:
                    from src.morph_forensics import evaluate_photo_morphing
                    morph_res = evaluate_photo_morphing(temp_disk_path)

            dt = (time.perf_counter() - t0) * 1000.0
            return face_verification, liveness_res, morph_res, dt

        def task_typography_and_meta() -> Tuple[Dict[str, Any], Dict[str, Any], float]:
            t0 = time.perf_counter()
            raw_meta = extract_image_metadata(temp_disk_path)
            meta_res = audit_metadata_provenance(raw_meta)
            font_res = analyze_font_consistency(doc_bgr, {})
            dt = (time.perf_counter() - t0) * 1000.0
            return font_res, meta_res, dt

        # -------------------------------------------------------------------------
        # Execute Tasks Concurrently in ThreadPool
        # -------------------------------------------------------------------------
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            f_ocr = executor.submit(task_ocr_and_semantics)
            f_phys = executor.submit(task_physical_tamper)
            f_bio = executor.submit(task_biometrics)
            f_meta = executor.submit(task_typography_and_meta)

            formatted_fields, mrz_data, sem_res, mrz_viz_cross, lat_ocr = f_ocr.result()
            ela_res, cm_res, fft_res, lat_phys = f_phys.result()
            face_res, liveness_res, morph_res, lat_bio = f_bio.result()
            font_res, meta_res, lat_meta = f_meta.result()

        # Re-evaluate font consistency using extracted OCR bounding boxes
        if formatted_fields:
            font_res = analyze_font_consistency(doc_bgr, formatted_fields)

        # -------------------------------------------------------------------------
        # Format Signals & Correlate Suspicious Regions
        # -------------------------------------------------------------------------
        ela_cand = ela_res.get("candidate")
        ela_cand_bbox = [int(x) for x in ela_cand["bbox"]] if ela_cand and ela_cand.get("bbox") else []
        ela_conf = round(float(min(0.98, 0.40 + (ela_cand.get("energy", 0.0) / 200.0) * 0.50)), 2) if ela_cand else None
        ela_features = ela_res.get("features", {})

        cm_detected = cm_res.get("detected", False)
        cm_bbox = [int(x) for x in cm_res.get("candidate_bbox", [])] if cm_res.get("candidate_bbox") else []
        cm_alt_bbox = [int(x) for x in cm_res.get("alt_bbox", [])] if cm_res.get("alt_bbox") else []
        cm_matches = int(cm_res.get("num_matches", 0))
        cm_conf = round(float(cm_res.get("confidence", 0.85)), 2) if cm_detected else None

        tamper_signals = {
            "ela": {
                "candidate_bbox": ela_cand_bbox,
                "confidence": ela_conf,
                "features": {
                    "mean": round(float(ela_features.get("mean", 0.0)), 2),
                    "std": round(float(ela_features.get("std", 0.0)), 2),
                    "max": round(float(ela_features.get("max", 0.0)), 2),
                    "p95": round(float(ela_features.get("p95", 0.0)), 2),
                    "p99": round(float(ela_features.get("p99", 0.0)), 2),
                    "high_error_pixel_ratio": round(float(ela_features.get("high_error_pixel_ratio", 0.0)), 4),
                    "candidate_energy": round(float(ela_cand.get("energy", 0.0)), 1) if ela_cand else 0.0,
                },
            },
            "copy_move": {
                "bbox": cm_bbox,
                "alt_bbox": cm_alt_bbox,
                "num_matches": cm_matches,
                "confidence": cm_conf,
                "detected": bool(cm_detected),
            },
            "fft": fft_res,
        }

        suspicious_regions = correlate_suspicious_regions(
            fields=formatted_fields,
            ela_candidate=ela_cand,
            copy_move_res=cm_res,
            semantic_audit=sem_res,
            font_audit=font_res,
            face_res=face_res,
            doc_shape=(h, w),
            mrz_data=mrz_data,
            mrz_viz_cross=mrz_viz_cross,
            metadata_audit=meta_res,
            img_bgr=doc_bgr,
        )

        quality = analyze_document_quality(doc_bgr, ocr_fields=formatted_fields)
        semantic_checks = sem_res.get("checks_list", [])

        # Step G: Heuristic Attack Classification
        attack_guess, attack_conf, attack_basis, heuristic_decision = classify_attack_heuristic(
            tamper_signals=tamper_signals,
            semantic_checks=semantic_checks,
            font_audit=font_res,
            face_verification=face_res,
            suspicious_regions=suspicious_regions,
            quality=quality,
            metadata_audit=meta_res,
        )

        hypo_info = compute_attack_hypotheses_and_severity(
            tamper_signals=tamper_signals,
            semantic_checks=semantic_checks,
            font_audit=font_res,
            face_verification=face_res,
            suspicious_regions=suspicious_regions,
            attack_guess=attack_guess,
            attack_conf=attack_conf,
        )

        report = {
            "schema_version": "1.1",
            "document_id": document_id,
            "execution_mode": "CONCURRENT_ASYNC",
            "decision": heuristic_decision,
            "quality": quality,
            "fields": formatted_fields,
            "semantic_checks": semantic_checks,
            "tamper_signals": tamper_signals,
            "face_verification": face_res,
            "suspicious_regions": suspicious_regions,
            "attack_type_guess": attack_guess,
            "attack_type_confidence": attack_conf,
            "attack_type_basis": attack_basis,
            "secondary_attack_guess": hypo_info["secondary_attack_guess"],
            "secondary_attack_confidence": hypo_info["secondary_attack_confidence"],
            "multi_attack_detected": hypo_info["multi_attack_detected"],
            "attack_hypotheses_ranked": hypo_info["attack_hypotheses_ranked"],
            "fraud_severity": hypo_info["fraud_severity"],
            "font_forensics": {
                "typography_verdict": font_res.get("typography_verdict"),
                "max_stroke_zscore": font_res.get("max_stroke_zscore"),
                "anomalous_fields_count": len(font_res.get("anomalous_fields", [])),
            },
            "metadata_forensics": {
                "provenance_verdict": meta_res.get("provenance_verdict"),
                "is_tampered": meta_res.get("is_tampered"),
                "software": meta_res.get("software"),
                "has_exif": meta_res.get("has_exif"),
            },
            "mrz": {
                "format": mrz_data.get("format") if mrz_data else None,
                "status": mrz_data.get("status") if mrz_data else None,
                "verdict": mrz_data.get("verdict") if mrz_data else None,
                "viz_cross": mrz_viz_cross,
            },
            "liveness": liveness_res,
            "morphing": morph_res,
        }

        # Correlate Milestone 9 Biometric Spoof / Morphing alerts into suspicious regions
        if liveness_res and not liveness_res.get("is_live", True):
            suspicious_regions.append({
                "source": "liveness_pad",
                "bbox": face_res.get("face_bbox") or [30, 80, 200, 260],
                "field": "presented_selfie",
                "overlapping_fields": ["selfie"],
                "confidence": float(round(1.0 - liveness_res.get("liveness_score", 0.0), 2)),
                "evidence": f"Presentation attack detected: {liveness_res.get('spoof_tier', 'SUSPECT_PRESENTATION')} (liveness={liveness_res.get('liveness_score')})",
            })

        if morph_res and morph_res.get("morphing_detected", False) and morph_res.get("morphing_score", 0.0) >= 80.0:
            suspicious_regions.append({
                "source": "morph_forensics",
                "bbox": face_res.get("face_bbox") or [30, 80, 200, 260],
                "field": "photo",
                "overlapping_fields": ["photo"],
                "confidence": float(round(morph_res.get("morphing_score", 0.0) / 100.0, 2)),
                "evidence": f"Facial morphing composite detected: {morph_res.get('morph_tier')} (score={morph_res.get('morphing_score')})",
            })

        # Step H & I: Machine Learning Risk Fusion & Decision Policy
        feature_vector = extract_m6_feature_vector(report)
        report["feature_vector"] = feature_vector

        m6_cfg = load_m6_config()
        risk_info = predict_document_risk(feature_vector, config=m6_cfg)
        report["risk_score"] = risk_info.get("risk_score")
        report["fraud_probability"] = risk_info.get("fraud_probability")
        report["risk_tier"] = risk_info.get("risk_tier")
        report["risk_drivers"] = risk_info.get("top_risk_drivers", [])
        report["log_odds_total"] = risk_info.get("log_odds_total", 0.0)

        final_decision, risk_tier, policy_basis = apply_decision_policy(
            document_decision=heuristic_decision,
            fraud_probability=report["fraud_probability"] if report["fraud_probability"] is not None else 0.0,
            face_verification=face_res,
            quality=quality,
            config=m6_cfg,
        )
        report["decision"] = final_decision

        t_total = (time.perf_counter() - t_start) * 1000.0
        report["latency_breakdown_ms"] = {
            "ocr_semantics": round(lat_ocr, 1),
            "physical_tamper_fft": round(lat_phys, 1),
            "biometric_sface": round(lat_bio, 1),
            "typography_meta": round(lat_meta, 1),
            "total_concurrent_ms": round(t_total, 1),
        }
        report["pipeline_latency_ms"] = round(t_total, 1)

        return report

    finally:
        if cleanup_temp and os.path.exists(temp_disk_path):
            try:
                os.remove(temp_disk_path)
            except Exception:
                pass
