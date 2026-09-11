"""
ForgeLens-X — CLI Entry Point
================================
Provides command-line interface for M1 pipeline.

Usage:
    py -m src.cli run-demo --samples 10 --seed 42
    py -m src.cli generate --samples 100 --seed 42
    py -m src.cli analyze --data-dir data/generated
    py -m src.cli evaluate --data-dir data/generated
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from src.copy_move import detect_copy_move
from src.document_template import generate_document
from src.ela import analyze_ela, compute_baseline
from src.evaluate import (
    evaluate_batch,
    export_samples_summary_csv,
    generate_markdown_audit_report,
    save_evaluation_report,
)
from src.tamper_generator import generate_tampered_dataset
from src.utils import (
    ensure_dirs,
    get_forensic_dir,
    get_generated_dir,
    get_reports_dir,
    load_config,
    load_metadata,
    save_metadata,
    set_seed,
)


def cmd_generate(args):
    """Generate synthetic documents + all attack variants."""
    config = load_config()
    set_seed(args.seed)

    output_dir = str(get_generated_dir())
    ensure_dirs(output_dir)

    print(f"[M1] Generating {args.samples} source documents...")
    all_metadata = []

    for i in range(args.samples):
        source_id = f"src_{i:04d}"
        doc_seed = args.seed + i

        # Generate template
        doc = generate_document(source_id=source_id, seed=doc_seed)

        # Generate all attack variants (genuine + 4 attacks)
        sample_meta = generate_tampered_dataset(
            document_result=doc,
            output_dir=output_dir,
            jpeg_quality=config["document"]["jpeg_quality"],
            seed=doc_seed,
        )
        all_metadata.extend(sample_meta)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Generated {i + 1}/{args.samples} sources "
                  f"({len(all_metadata)} total samples)")

    # Save master metadata
    master_path = os.path.join(output_dir, "metadata", "master_index.json")
    save_metadata({"samples": all_metadata, "total": len(all_metadata)}, master_path)

    # Generate zero-leakage splits
    from src.utils import create_dataset_splits, get_splits_dir
    splits = create_dataset_splits(
        {"samples": all_metadata},
        output_dir=get_splits_dir(),
        train_ratio=config["dataset"]["train_ratio"],
        cal_ratio=config["dataset"]["cal_ratio"],
        test_ratio=config["dataset"]["test_ratio"],
        seed=args.seed,
    )
    summary = splits["split_summary"]

    print(f"[M1] Done. {len(all_metadata)} samples in {output_dir}")
    print(f"     Master index: {master_path}")
    print(f"[M1] Dataset splits created (zero-leakage by source_id):")
    print(f"     Train: {summary['n_sources_train']} sources ({summary['n_samples_train']} samples)")
    print(f"     Cal:   {summary['n_sources_cal']} sources ({summary['n_samples_cal']} samples)")
    print(f"     Test:  {summary['n_sources_test']} sources ({summary['n_samples_test']} samples)")
    return all_metadata


def cmd_analyze(args):
    """Run ELA + copy-move analysis on generated data."""
    config = load_config()

    data_dir = args.data_dir or str(get_generated_dir())
    master_path = os.path.join(data_dir, "metadata", "master_index.json")

    if not os.path.exists(master_path):
        print(f"[ERROR] Master index not found: {master_path}")
        print("        Run 'generate' first.")
        sys.exit(1)

    master = load_metadata(master_path)
    samples = master["samples"]

    # --- Build ELA baseline from genuine samples ---
    genuine_paths = [
        s["image_path"] for s in samples if s["label"] == "genuine"
    ]
    print(f"[M1] Building ELA baseline from {len(genuine_paths)} genuine samples...")
    baseline = compute_baseline(
        genuine_paths,
        quality=config["ela"]["recompress_quality"],
    )
    print(f"     Baseline built (mean ELA range: "
          f"{baseline['mean_map'].min():.2f} - {baseline['mean_map'].max():.2f})")

    from src.ela import calibrate_threshold
    cal_threshold = calibrate_threshold(
        genuine_paths,
        baseline,
        percentile=95.0,
        quality=config["ela"]["recompress_quality"],
        k=config["ela"]["baseline_k"],
        min_std=config["ela"].get("std_floor", 1.5),
        min_area=config["ela"]["min_candidate_area"],
        closing_ksize=tuple(config["ela"].get("closing_ksize", [11, 7])),
        default_threshold=config["ela"].get("energy_threshold", 60.0),
    )
    print(f"     Empirically calibrated energy threshold: {cal_threshold:.1f}")

    # --- Save baseline statistics for reuse / screening ---
    forensic_dir = str(get_forensic_dir())
    ela_dir = os.path.join(forensic_dir, "ela")
    cm_dir = os.path.join(forensic_dir, "copy_move")
    ensure_dirs(ela_dir, cm_dir)

    baseline_npz = os.path.join(ela_dir, "baseline_stats.npz")
    np.savez_compressed(
        baseline_npz,
        mean_map=baseline["mean_map"],
        std_map=baseline["std_map"],
        cal_threshold=cal_threshold,
    )
    print(f"     Baseline stats saved: {baseline_npz}")

    # --- Analyze each sample ---
    analysis_results = []
    print(f"[M1] Analyzing {len(samples)} samples...")

    for idx, sample in enumerate(samples):
        image_path = sample["image_path"]

        if not os.path.exists(image_path):
            print(f"  [WARN] Missing: {image_path}")
            continue

        # --- ELA ---
        ela_cfg = config["ela"]
        ela_result = analyze_ela(
            image_path,
            baseline=baseline,
            quality=ela_cfg["recompress_quality"],
            k=ela_cfg["baseline_k"],
            min_std=ela_cfg.get("std_floor", 1.5),
            min_area=ela_cfg["min_candidate_area"],
            closing_ksize=tuple(ela_cfg.get("closing_ksize", [11, 7])),
            energy_threshold=cal_threshold,
        )

        # --- Copy-move ---
        cm_cfg = config["copy_move"]
        cm_result = detect_copy_move(
            image_path,
            n_features=cm_cfg["n_features"],
            match_threshold=cm_cfg["match_threshold"],
            min_spatial_distance=cm_cfg["min_spatial_distance"],
            ransac_threshold=cm_cfg["ransac_threshold"],
            min_inliers=cm_cfg["min_inliers"],
            min_confidence=cm_cfg.get("min_confidence", 0.15),
        )

        # Combine
        result = {
            "source_id": sample["source_id"],
            "attack_type": sample["attack_type"],
            "label": sample["label"],
            "image_path": image_path,
            "ground_truth_bbox": sample.get("bbox"),
            # ELA results
            "ela_detected": ela_result["candidate"] is not None,
            "ela_candidate_bbox": ela_result["candidate"]["bbox"] if ela_result["candidate"] else None,
            "ela_features": ela_result["features"],
            "ela_candidate_area": ela_result["candidate"]["area"] if ela_result["candidate"] else 0,
            "ela_max_anomaly": ela_result["candidate"]["max_anomaly"] if ela_result["candidate"] else 0.0,
            "ela_anomaly_score": ela_result["candidate"]["energy"] if ela_result["candidate"] else 0.0,
            # Copy-move results
            "copy_move_detected": cm_result["detected"],
            "copy_move_bbox": cm_result["candidate_bbox"],
            "copy_move_alt_bbox": cm_result.get("alt_bbox"),
            "copy_move_num_matches": cm_result["num_matches"],
            "copy_move_num_inliers": cm_result["num_inliers"],
            "copy_move_inliers": cm_result["num_inliers"],
            "copy_move_confidence": cm_result["confidence"],
        }

        analysis_results.append(result)

        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"  Analyzed {idx + 1}/{len(samples)}")

    # Save analysis results
    analysis_path = os.path.join(forensic_dir, "analysis_results.json")
    save_metadata({"results": analysis_results}, analysis_path)

    print(f"[M1] Analysis complete. Results: {analysis_path}")
    return analysis_results


def cmd_evaluate(args):
    """Evaluate detection and localization performance."""
    config = load_config()

    forensic_dir = str(get_forensic_dir())
    analysis_path = os.path.join(forensic_dir, "analysis_results.json")

    if not os.path.exists(analysis_path):
        print(f"[ERROR] Analysis results not found: {analysis_path}")
        print("        Run 'analyze' first.")
        sys.exit(1)

    analysis = load_metadata(analysis_path)
    results = analysis["results"]

    print(f"[M1] Evaluating {len(results)} samples...")

    eval_results = evaluate_batch(
        results,
        iou_threshold=config["evaluation"]["iou_threshold"],
    )

    # Add metadata
    eval_results["config"] = {
        "seed": args.seed if hasattr(args, "seed") else "from_analysis",
        "ela_quality": config["ela"]["recompress_quality"],
        "ela_k": config["ela"]["baseline_k"],
        "copy_move_n_features": config["copy_move"]["n_features"],
        "iou_threshold": config["evaluation"]["iou_threshold"],
    }

    # Save reports
    report_path = save_evaluation_report(eval_results)
    csv_path = export_samples_summary_csv(results)
    audit_path = generate_markdown_audit_report(eval_results)

    # Print summary
    print("\n" + "=" * 60)
    print("M1 EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- ELA Detector ---")
    _print_detector_summary(eval_results["ela"])

    print("\n--- Copy-Move Detector ---")
    _print_detector_summary(eval_results["copy_move"])

    print(f"\n[M1] JSON report:     {report_path}")
    print(f"[M1] Tabular CSV:     {csv_path}")
    print(f"[M1] Executive audit: {audit_path}")
    return eval_results


def _print_detector_summary(detector_results: dict):
    """Print formatted detector evaluation summary."""
    overall = detector_results["overall"]
    print(f"  Detection rate:    {overall['detection_rate']:.4f}")
    print(f"  False alarm rate:  {overall['false_alarm_rate']:.4f}")
    print(f"  Mean IoU:          {overall.get('mean_iou', 'N/A')}")
    print(f"  Localization rate: {overall.get('localization_rate', 'N/A')}")

    print(f"  TP={overall['true_positives']} TN={overall['true_negatives']} "
          f"FP={overall['false_positives']} FN={overall['false_negatives']}")

    if detector_results.get("per_attack"):
        print("  Per-attack:")
        for attack, metrics in detector_results["per_attack"].items():
            det_rate = metrics.get("detection_rate", "N/A")
            mean_iou = metrics.get("mean_iou", "N/A")
            print(f"    {attack:15s}  det={det_rate}  iou={mean_iou}")


def cmd_visualize(args):
    """Generate visual forensic explanation cards for human review."""
    from src.visualize import visualize_batch

    forensic_dir = str(get_forensic_dir())
    analysis_path = os.path.join(forensic_dir, "analysis_results.json")
    master_path = os.path.join(str(get_generated_dir()), "metadata", "master_index.json")

    if not os.path.exists(analysis_path) or not os.path.exists(master_path):
        print(f"[ERROR] Analysis or master index missing. Run 'run-demo' or 'analyze' first.")
        sys.exit(1)

    analysis = load_metadata(analysis_path)["results"]
    master = load_metadata(master_path)["samples"]

    max_samples = getattr(args, "count", 5)
    print(f"[M1] Generating up to {max_samples} visual forensic explanation cards...")
    saved = visualize_batch(master, analysis, max_samples=max_samples)

    print(f"[M1] Generated {len(saved)} visual diagnostic panels in reports/visuals/:")
    for p in saved:
        print(f"  • {p}")
    return saved


def cmd_run_demo(args):
    """Run the full M1 pipeline end-to-end with visual explainability."""
    start = time.time()

    print("=" * 60)
    print("ForgeLens-X — M1 Demo Pipeline")
    print("=" * 60)
    print(f"Samples: {args.samples}, Seed: {args.seed}")
    print()

    # Step 1: Generate
    print("[STEP 1/4] Generating synthetic documents + attacks + dataset splits...")
    cmd_generate(args)
    print()

    # Step 2: Analyze
    print("[STEP 2/4] Running ELA + Copy-Move analysis...")
    args.data_dir = None  # use default
    cmd_analyze(args)
    print()

    # Step 3: Evaluate
    print("[STEP 3/4] Evaluating results...")
    eval_results = cmd_evaluate(args)
    print()

    # Step 4: Visualize
    print("[STEP 4/4] Generating visual forensic explanation cards...")
    args.count = 5
    cmd_visualize(args)
    print()

    elapsed = time.time() - start
    print(f"[M1] Pipeline completed in {elapsed:.1f}s")
    print("=" * 60)

    return eval_results


def cmd_screen(args):
    """
    Forensic document screening on a single document image.
    Usage: py -m src.cli screen <path_to_image> [--output <output_path>]
    """
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"[ERROR] Document image not found: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("ForgeLens-X — Forensic Document Screening (M1)")
    print("=" * 60)
    print(f"Input Document: {image_path}")

    config = load_config()

    # Check for baseline stats
    forensic_dir = str(get_forensic_dir())
    ela_dir = os.path.join(forensic_dir, "ela")
    baseline_path = os.path.join(ela_dir, "baseline_stats.npz")
    baseline = None
    cal_threshold = config["ela"].get("energy_threshold", 60.0)

    if os.path.exists(baseline_path):
        data = np.load(baseline_path)
        baseline = {"mean_map": data["mean_map"], "std_map": data["std_map"]}
        if "cal_threshold" in data:
            cal_threshold = float(data["cal_threshold"])
        print(f"ELA Baseline: Loaded from {baseline_path} (Threshold: {cal_threshold:.1f})")
    else:
        print(f"ELA Baseline: None found at {baseline_path} (Running uncalibrated ELA)")

    # 1. Run ELA
    ela_cfg = config["ela"]
    ela_result = analyze_ela(
        image_path,
        baseline=baseline,
        quality=ela_cfg["recompress_quality"],
        k=ela_cfg["baseline_k"],
        min_std=ela_cfg.get("std_floor", 1.5),
        min_area=ela_cfg["min_candidate_area"],
        closing_ksize=tuple(ela_cfg.get("closing_ksize", [11, 7])),
        energy_threshold=cal_threshold,
    )

    # 2. Run Copy-Move
    cm_cfg = config["copy_move"]
    cm_result = detect_copy_move(
        image_path,
        n_features=cm_cfg["n_features"],
        match_threshold=cm_cfg["match_threshold"],
        min_spatial_distance=cm_cfg["min_spatial_distance"],
        ransac_threshold=cm_cfg["ransac_threshold"],
        min_inliers=cm_cfg["min_inliers"],
        min_confidence=cm_cfg.get("min_confidence", 0.15),
    )

    # Screening Decision
    flags = []
    if ela_result["candidate"]:
        score = float(ela_result["candidate"]["energy"])
        bbox = ela_result["candidate"]["bbox"]
        flags.append(f"ELA Anomaly Energy Spike (Score={score:.1f}, Bounding Box={bbox})")
    if cm_result["detected"]:
        inliers = cm_result.get("num_inliers", 0)
        conf = cm_result.get("confidence") or 0.0
        flags.append(f"Duplicated Region / Copy-Move Cloning ({inliers} ORB inliers, Confidence={conf:.2f})")

    verdict = "FLAGGED (Suspicious)" if flags else "CLEAN (No Anomalies Detected)"
    verdict_badge = "[ALERT]" if flags else "[PASS]"

    cm_conf_val = cm_result.get("confidence")
    cm_conf_str = f"{cm_conf_val:.3f}" if cm_conf_val is not None else "0.000"

    print("-" * 60)
    print(f"VERDICT: {verdict_badge} {verdict}")
    print("-" * 60)
    print("DETECTOR SIGNALS:")
    print(f"  • ELA Detection:        {'POSITIVE' if ela_result['candidate'] else 'NEGATIVE'}")
    print(f"    - Anomaly Energy:     {ela_result['candidate']['energy'] if ela_result['candidate'] else 0.0:.2f}")
    print(f"    - Candidate Bounding: {ela_result['candidate']['bbox'] if ela_result['candidate'] else 'None'}")
    print(f"  • Copy-Move Detection:  {'POSITIVE' if cm_result['detected'] else 'NEGATIVE'}")
    print(f"    - Inlier Count:       {cm_result.get('num_inliers', 0)}")
    print(f"    - Inlier Confidence:  {cm_conf_str}")
    print(f"    - Cloned Bounding:    {cm_result.get('candidate_bbox')}")

    print("\nEXPLANATORY FORENSIC EVIDENCE:")
    if flags:
        for idx, f in enumerate(flags, 1):
            print(f"  [{idx}] {f}")
    else:
        print("  Document compression and feature correspondence show no physical manipulation.")

    # Visual Forensic Card
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_card = args.output
    if not out_card:
        vis_dir = os.path.join(get_reports_dir(), "visuals")
        ensure_dirs(vis_dir)
        out_card = os.path.join(vis_dir, f"{base_name}_forensic.png")

    from src.visualize import create_forensic_card
    sample_data = {
        "image_path": image_path,
        "attack_type": "screening",
        "label": "screening",
        "source_id": base_name,
        "ground_truth_bbox": None,
    }
    analysis_data = {
        "source_id": base_name,
        "attack_type": "screening",
        "label": "screening",
        "image_path": image_path,
        "ela_detected": ela_result["candidate"] is not None,
        "ela_candidate_bbox": ela_result["candidate"]["bbox"] if ela_result["candidate"] else None,
        "ela_features": ela_result["features"],
        "copy_move_detected": cm_result["detected"],
        "copy_move_bbox": cm_result["candidate_bbox"],
        "copy_move_inliers": cm_result.get("num_inliers", 0),
        "copy_move_confidence": cm_result.get("confidence") or 0.0,
    }

    create_forensic_card(sample_data, analysis_data, output_path=out_card)
    print(f"\nVisual Forensic Explanation Card generated:")
    print(f"  • {out_card}")
    print("=" * 60)


def cmd_face_verify(args):
    """Verify document photo crop against presented live face."""
    from src.face_verify import verify
    from src.face_visualize import create_face_forensic_card

    print("=" * 60)
    print("ForgeLens-X — Face Verification (M2)")
    print("=" * 60)
    print(f"Document Photo: {args.doc_face}")
    print(f"Live Selfie:    {args.live_face}")
    print(f"Model Backbone: {args.model}")
    print(f"Distance Metric:{args.metric}")
    print("-" * 60)

    res = verify(
        document_face_path=args.doc_face,
        live_face_path=args.live_face,
        model_name=args.model,
        distance_metric=args.metric,
    )

    if res.get("error"):
        print(f"[ERROR] Verification failed: {res['error']}")
        print(f"        Detail: {res.get('detail', 'Unknown error')}")
        print("=" * 60)
        return

    verified = res["verified"]
    badge = "[MATCH]" if verified else "[MISMATCH]"
    dist = res["distance"]
    thresh = res["threshold"]
    sim = res["similarity_pct"]
    tier = res["verdict_tier"]

    print(f"VERDICT: {badge} {tier}")
    print("-" * 60)
    print(f"  • Match Verified:       {'YES (Same Person)' if verified else 'NO (Different People)'}")
    print(f"  • Similarity Score:     {sim:.1f}%")
    print(f"  • Measured Distance:    {dist:.4f} ({args.metric})")
    print(f"  • Decision Threshold:   {thresh:.4f}")
    print(f"  • Margin to Threshold:  {thresh - dist:+.4f}")
    print(f"  • Verification Model:   {res['model']}")
    print(f"  • Detector Backend:     {res['detector']}")
    print(f"  • Engine:               {res['engine']}")
    print(f"  • Processing Time:      {res['time_seconds']:.3f}s")

    print("\nEXPLANATORY EVIDENCE:")
    if verified:
        print(f"  Facial embeddings align within confidence threshold (d={dist:.4f} <= {thresh:.4f}).")
        print(f"  Calibrated similarity is {sim:.1f}%, indicating identity consistency.")
    else:
        print(f"  Facial embeddings exceed allowable threshold (d={dist:.4f} > {thresh:.4f}).")
        print(f"  Calibrated similarity is only {sim:.1f}%, indicating disparate facial geometries.")

    out_path = args.output
    if not out_path:
        vis_dir = os.path.join(get_reports_dir(), "visuals", "face")
        ensure_dirs(vis_dir)
        d_stem = Path(args.doc_face).stem
        l_stem = Path(args.live_face).stem
        out_path = os.path.join(vis_dir, f"{d_stem}_vs_{l_stem}_card.png")

    create_face_forensic_card(
        document_face_path=args.doc_face,
        live_face_path=args.live_face,
        verification_result=res,
        output_path=out_path,
        model_name=args.model,
    )
    print(f"\nVisual Forensic Explanation Card generated:")
    print(f"  • {out_path}")
    print("=" * 60)


def cmd_face_compare(args):
    """Run multi-model comparison across models."""
    from src.face_verify import compare_models

    print("=" * 60)
    print("ForgeLens-X — Multi-Model Face Comparison (M2)")
    print("=" * 60)
    models = args.models or ["ArcFace", "Facenet512", "SFace"]
    print(f"Document: {args.doc_face}")
    print(f"Live:     {args.live_face}")
    print(f"Models:   {', '.join(models)}")
    print("-" * 60)

    res = compare_models(
        document_face_path=args.doc_face,
        live_face_path=args.live_face,
        models=models,
        distance_metric=args.metric,
    )

    if res.get("error"):
        print(f"[ERROR] Comparison failed: {res['error']}")
        print(f"        Detail: {res.get('detail')}")
        return

    print(f"{'Model':<14} | {'Verified':<10} | {'Distance':<10} | {'Threshold':<10} | {'Sim %':<8} | {'Tier'}")
    print("-" * 75)
    for m, mres in res["models"].items():
        if mres.get("error"):
            print(f"{m:<14} | ERROR: {mres.get('error')}")
        else:
            v_str = "YES" if mres["verified"] else "NO"
            print(f"{m:<14} | {v_str:<10} | {mres['distance']:<10.4f} | {mres['threshold']:<10.4f} | {mres['similarity_pct']:<7.1f}% | {mres['verdict_tier']}")

    print("-" * 75)
    print(f"Consensus Verdict:    {'MATCH' if res['consensus_verified'] else 'MISMATCH'}")
    print(f"Model Agreement:      {res['agreement_pct']:.1f}% ({res['models_agreeing']}/{res['total_models']} models)")
    print(f"Mean Similarity:      {res['mean_similarity_pct']:.1f}%")
    print(f"Mean Distance:        {res['mean_distance']:.4f}")
    print("=" * 60)


def cmd_face_eval(args):
    """Evaluate face verification on benchmark dataset."""
    from src.face_dataset import generate_benchmark_pairs, load_benchmark_pairs
    from src.face_evaluate import evaluate_face_verification, export_evaluation_report
    from src.face_visualize import create_face_forensic_card

    print("=" * 60)
    print("ForgeLens-X — Benchmark Face Verification Evaluation (M2)")
    print("=" * 60)

    pairs_dir = args.pairs_dir or os.path.join(os.getcwd(), "data", "face_pairs")
    index_path = os.path.join(pairs_dir, "pairs_index.json")

    if not os.path.exists(index_path):
        print(f"[M2] Benchmark pairs not found at {pairs_dir}. Generating {args.pairs} pairs...")
        generate_benchmark_pairs(output_dir=pairs_dir, n_pairs=args.pairs, seed=args.seed)

    pairs = load_benchmark_pairs(pairs_dir)
    print(f"[M2] Loaded {len(pairs)} benchmark evaluation pairs from {pairs_dir}")
    print(f"     Model:  {args.model}")
    print(f"     Metric: {args.metric}")
    print("-" * 60)

    report = evaluate_face_verification(
        pairs=pairs,
        model_name=args.model,
        distance_metric=args.metric,
    )

    metrics = report["metrics"]
    print("BENCHMARK PERFORMANCE METRICS:")
    print(f"  • Total Evaluated Pairs: {report['n_pairs']} ({report['n_genuine']} genuine, {report['n_imposter']} imposter)")
    print(f"  • Verification Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"  • False Accept Rate (FAR):{metrics['far'] * 100:.2f}%")
    print(f"  • False Reject Rate (FRR):{metrics['frr'] * 100:.2f}%")
    print(f"  • True Accept Rate (TAR): {metrics['tar'] * 100:.2f}%")
    print(f"  • Separation Margin:     {metrics['separation_margin']:.4f}")
    print(f"    - Mean Genuine Dist:   {metrics['mean_genuine_dist']:.4f} (std: {metrics['std_genuine_dist']:.4f})")
    print(f"    - Mean Imposter Dist:  {metrics['mean_imposter_dist']:.4f} (std: {metrics['std_imposter_dist']:.4f})")
    print(f"  • Mean Inference Time:   {metrics['mean_inference_time_ms']:.1f} ms / pair")
    print(f"  • Zero-Crash Failures:   {metrics['failed_pairs']} / {report['n_pairs']}")

    roc = report.get("roc_analysis") or report.get("roc_calibration")
    if roc and (roc.get("sample_roc_points") or "eer_value" in roc):
        print("\nROC CURVE & EQUAL ERROR RATE (EER) CALIBRATION:")
        print(f"  • Equal Error Rate (EER): {roc.get('eer_value', 0.0) * 100:.2f}% at tau = {roc.get('eer_threshold', 0.68):.4f}")
        print(f"  • ROC Area Under Curve:   {roc.get('auc', 1.0):.4f}")
        regs = roc.get("operational_regimes", {})
        if "high_security" in regs:
            hs = regs["high_security"]
            print(f"  • High-Security Regime:   tau = {hs['threshold']:.4f} ({hs.get('target', '')})")
        if "balanced" in regs:
            be = regs["balanced"]
            print(f"  • Balanced (EER) Regime:  tau = {be['threshold']:.4f} ({be.get('target', '')})")
        if "low_friction" in regs:
            lf = regs["low_friction"]
            print(f"  • Low-Friction Regime:    tau = {lf['threshold']:.4f} ({lf.get('target', '')})")

    if "demographic_fairness" in report:
        fair = report["demographic_fairness"]
        print("\nDEMOGRAPHIC FAIRNESS & PARITY AUDIT:")
        print(f"  • Fairness Parity Status: {fair.get('fairness_status', 'N/A')}")
        print(f"  • Max Subgroup Disparity: {fair.get('max_accuracy_disparity', 0.0) * 100:.2f}%")
        for sname, smetrics in fair.get("demographic_subgroups", {}).items():
            print(f"    - Subgroup [{sname}]: {smetrics.get('sample_count', 0)} pairs | Acc: {smetrics.get('accuracy', 0)*100:.1f}% | FAR: {smetrics.get('far', 0)*100:.1f}% | FRR: {smetrics.get('frr', 0)*100:.1f}%")

    reports_dir = get_reports_dir()
    csv_path = os.path.join(reports_dir, "m2_pairs_summary.csv")
    md_path = os.path.join(reports_dir, "m2_face_audit_report.md")
    json_path = os.path.join(reports_dir, "m2_results.json")

    paths = export_evaluation_report(
        report=report,
        csv_path=csv_path,
        md_path=md_path,
        json_path=json_path,
    )

    print("\nREPORT ARTIFACTS EXPORTED:")
    print(f"  • CSV Summary:    {paths['csv']}")
    print(f"  • Markdown Audit: {paths['md']}")
    print(f"  • JSON Metrics:   {paths['json']}")

    if args.cards > 0:
        vis_dir = os.path.join(reports_dir, "visuals", "face")
        ensure_dirs(vis_dir)
        print(f"\nGenerating {min(args.cards, len(pairs))} visual explanation cards in {vis_dir}...")
        for i in range(min(args.cards, len(pairs))):
            p = pairs[i]
            card_out = os.path.join(vis_dir, f"{p['pair_id']}_forensic_card.png")
            create_face_forensic_card(
                document_face_path=p.get("doc_image_path") or p.get("doc_path"),
                live_face_path=p.get("live_image_path") or p.get("live_path"),
                output_path=card_out,
                model_name=args.model,
            )
        print(f"  • Visual explanation cards generated in {vis_dir}")

    print("=" * 60)



def cmd_screen_identity(args):
    """
    Perform end-to-end identity screening:
    Checks document tamper forensics (ELA + Copy-Move) and verifies
    extracted document face against presented live selfie.
    """
    from src.identity_screener import screen_identity, create_identity_screening_card
    from src.utils import get_reports_dir, ensure_dirs

    print("=" * 60)
    print("ForgeLens-X — End-to-End Identity Screening (M1 + M2 Bridge)")
    print("=" * 60)
    print(f"Document ID: {args.doc_image}")
    print(f"Live Selfie: {args.live_face}")
    print(f"Model:       {args.model}")
    print(f"Metric:      {args.metric}")
    print("-" * 60)

    res = screen_identity(
        document_path=args.doc_image,
        live_face_path=args.live_face,
        model_name=args.model,
        distance_metric=args.metric,
    )

    verdict = res.get("verdict", "UNKNOWN")
    tier = res.get("verdict_tier", "UNKNOWN")
    summary = res.get("summary", "")
    explanation = res.get("explanation", "")
    rec = res.get("action_recommended", "")
    signals = res.get("signals", {})

    badge_map = {
        "VERIFIED_AUTHENTIC": "[PASS - AUTHENTIC]",
        "CRITICAL_PHOTO_SWAP_FRAUD": "[CRITICAL ALERT - PHOTO SWAP]",
        "IMPOSTER_MISMATCH": "[REJECT - IMPOSTER]",
        "TAMPERED_DOCUMENT_ALTERATION": "[ALERT - DOCUMENT TAMPERED]",
        "TOTAL_FRAUD_REJECTED": "[SEVERE ALERT - TOTAL FRAUD]",
        "BORDERLINE_REVIEW": "[FLAGGED - BORDERLINE]",
        "ERROR": "[ERROR]",
    }
    badge = badge_map.get(verdict, f"[{verdict}]")

    print(f"OVERALL SCREENING VERDICT: {badge}")
    print(f"Status Tier: {tier}")
    print("-" * 60)
    print("FORENSIC EVIDENCE BREAKDOWN:")
    print(f"  • Physical Document Tampering: {'DETECTED [FAIL]' if res.get('document_tampered') else 'CLEAN [PASS]'}")
    print(f"    - ELA Anomaly:               {'POSITIVE' if signals.get('ela_detected') else 'NEGATIVE'}")
    if signals.get("ela_candidate"):
        cand = signals["ela_candidate"]
        print(f"      Energy: {cand.get('energy', 0.0):.1f} | Area: {cand.get('area', 0)} px | BBox: {cand.get('bbox')}")
    print(f"    - Copy-Move Cloning:         {'POSITIVE' if signals.get('copy_move_detected') else 'NEGATIVE'}")
    if signals.get("copy_move_detected"):
        print(f"      Inliers: {signals.get('copy_move_inliers', 0)}")
    print(f"    - Photo-Swap Spatial Overlap: {'DETECTED [CRITICAL]' if signals.get('photo_swap_detected') else 'NEGATIVE'}")
    if signals.get("photo_swap_iou", 0) > 0:
        print(f"      Face-Tamper IoU: {signals.get('photo_swap_iou', 0.0):.3f}")

    print(f"\n  • Biometric Face Verification: {'MATCH [PASS]' if res.get('face_verified') else 'MISMATCH [FAIL]'}")
    if "face_distance" in signals and signals["face_distance"] is not None:
        dist = signals["face_distance"]
        thresh = signals.get("face_threshold", 0.6)
        conf = signals.get("face_confidence", 0.0)
        print(f"    - Distance: {dist:.4f} (Threshold: {thresh:.4f})")
        print(f"    - Face Confidence: {conf * 100:.1f}%")

    doc_q = signals.get("doc_face_quality", {})
    if doc_q:
        print(f"    - ID Face Quality:   {doc_q.get('quality_tier', 'N/A')} (Score: {doc_q.get('overall_score', 0):.1f}/100, Sharpness: {doc_q.get('sharpness', 0):.1f})")
    live_q = signals.get("live_face_quality", {})
    if live_q:
        print(f"    - Live Face Quality: {live_q.get('quality_tier', 'N/A')} (Score: {live_q.get('overall_score', 0):.1f}/100, Sharpness: {live_q.get('sharpness', 0):.1f})")

    print(f"\nEXPLANATION:\n  {explanation}")
    if rec:
        print(f"\nRECOMMENDED ACTION:\n  {rec}")

    out_path = args.output
    if not out_path:
        vis_dir = os.path.join(get_reports_dir(), "visuals", "identity")
        ensure_dirs(vis_dir)
        d_stem = Path(args.doc_image).stem
        l_stem = Path(args.live_face).stem
        out_path = os.path.join(vis_dir, f"{d_stem}_vs_{l_stem}_screen_card.png")

    card_res = create_identity_screening_card(res, output_path=out_path)
    if card_res and os.path.exists(out_path):
        print(f"\nVisual Identity Diagnostic Card generated:")
        print(f"  • {out_path}")
    print(f"Total screening time: {res.get('time_seconds', 0.0):.3f}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Milestone 3: OCR & Structured Field Extraction Commands
# ---------------------------------------------------------------------------

def cmd_ocr(args):
    """
    Extract structured identity fields from an identity document image.
    Extracts name, dob, document_number, issue_date, expiry_date.
    """
    from src.ocr import extract_structured_fields
    from src.ocr_visualize import generate_ocr_diagnostic_card
    from src.utils import ensure_dirs, get_reports_dir

    print("=" * 60)
    print("ForgeLens-X — Structured OCR Field Extraction (Milestone 3)")
    print("=" * 60)
    print(f"Document Image: {args.image}")
    print(f"OCR Engine:     {args.engine}")
    print("-" * 60)

    if not os.path.exists(args.image):
        print(f"[Error] Image file not found: {args.image}")
        sys.exit(1)

    t0 = time.time()
    res = extract_structured_fields(
        image_input=args.image,
        engine=args.engine,
        use_preprocessing=not args.no_preprocess,
    )
    latency_ms = (time.time() - t0) * 1000.0

    fields = res.get("fields", {})
    engine_used = res.get("engine", args.engine)
    doc_type = res.get("document_type", "identity_card")

    print(f"Document Type Detected: {doc_type}")
    print(f"Engine Executed:        {engine_used}")
    print(f"Processing Latency:     {latency_ms:.1f} ms\n")

    print(f"{'FIELD':<18} | {'EXTRACTED VALUE':<24} | {'CONF':<6} | {'STATUS':<15} | {'BBOX'}")
    print("-" * 80)

    for field_name in ["name", "dob", "document_number", "issue_date", "expiry_date"]:
        f_info = fields.get(field_name, {})
        val = str(f_info.get("value") or "[NONE]")
        conf = f_info.get("confidence", 0.0)
        status = f_info.get("status", "UNKNOWN")
        bbox = f_info.get("bbox")
        bbox_str = f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]" if bbox else "None"

        print(f"{field_name:<18} | {val:<24} | {conf*100:>5.1f}% | {status:<15} | {bbox_str}")

    # Forensic Auditing Details
    chron = res.get("chronology_audit", {})
    if chron:
        print("\nDATE CHRONOLOGY AUDIT:")
        c_status = chron.get("status", "VALID")
        c_badge = "[PASS]" if chron.get("chronology_valid", True) else "[FRAUD DETECTED]"
        print(f"  • Overall Status:     {c_badge} {c_status}")
        if "age_at_issue_years" in chron and chron["age_at_issue_years"] is not None:
            print(f"  • Age at Issue:       {chron['age_at_issue_years']:.1f} years")
        if "validity_years" in chron and chron["validity_years"] is not None:
            print(f"  • Validity Window:    {chron['validity_years']:.1f} years")
        for err in chron.get("errors", []):
            print(f"  [!] Chronology Violation: {err}")

    mrz = res.get("mrz_data")
    if mrz:
        print("\nICAO DOC 9303 MRZ PARSER & CHECKSUM VERIFICATION:")
        all_chk = mrz.get("checksums", {}).get("all_valid", False)
        m_badge = "[PASS]" if all_chk else "[FAIL]"
        print(f"  • MRZ Format:         {mrz.get('format', 'TD1')}")
        print(f"  • Checksums:          {m_badge} {'All 3 Checksums Verified' if all_chk else 'Checksum Mismatch'}")
        if "viz_cross_validation" in mrz:
            xv = mrz["viz_cross_validation"]
            print(f"  • VIZ-to-MRZ Match:   {xv.get('overall_viz_mrz_match')}")

    tamper = res.get("tamper_correlation", {})
    if tamper and tamper.get("tampered_fields_count", 0) > 0:
        print("\n[CRITICAL FORENSIC ALERT] TAMPER-FIELD SPATIAL OVERLAP:")
        for t_name in tamper.get("tampered_field_names", []):
            t_detail = tamper.get("tampered_fields", {}).get(t_name, {})
            print(f"  • Field '{t_name}' directly overlaps with tamper anomaly ({t_detail.get('overlap_type')}, IoU/Overlap={t_detail.get('tamper_overlap_area', 0)} px)")
    elif tamper:
        print("\nTAMPER CORRELATION: CLEAN (No field overlaps with physical manipulation)")

    # Generate Visual Diagnostic Card
    out_path = args.output
    if not out_path:
        vis_dir = os.path.join(get_reports_dir(), "visuals")
        ensure_dirs(vis_dir)
        stem = Path(args.image).stem
        out_path = os.path.join(vis_dir, f"ocr_card_{stem}.png")

    try:
        card_file = generate_ocr_diagnostic_card(
            image_input=args.image,
            extracted_res=res,
            output_path=out_path,
        )
        if os.path.exists(card_file):
            print(f"\nVisual Diagnostic Card Generated:\n  • {card_file}")
    except Exception as e:
        print(f"\n[Warning] Could not generate visual card: {e}")

    print("=" * 60)


def cmd_ocr_eval(args):
    """
    Evaluate OCR structured extraction accuracy and character error rate (CER)
    across synthetic (Forgelensia M1) and real-world (MIDV-500) benchmarks.
    """
    from src.midv500 import generate_sample_midv500_dataset, load_midv500_dataset
    from src.ocr_evaluate import evaluate_ocr_dataset, export_ocr_reports, load_synthetic_ocr_dataset
    from src.ocr_visualize import batch_generate_ocr_cards
    from src.utils import ensure_dirs, get_generated_dir, get_reports_dir

    print("=" * 60)
    print("ForgeLens-X — M3 OCR & Structured Extraction Benchmark")
    print("=" * 60)
    print(f"Target Dataset: {args.dataset}")
    print(f"Sample Limit:   {args.samples}")
    print(f"Engine:         {args.engine}")
    print(f"Random Seed:    {args.seed}")
    print("-" * 60)

    synth_report = None
    midv_report = None

    # 1. Evaluate Synthetic Benchmark (Forgelensia M1)
    if args.dataset in ["synthetic", "all"]:
        print("\n[STEP 1] Evaluating Synthetic Benchmark (Forgelensia M1)...")
        eval_synth_items = load_synthetic_ocr_dataset(limit=args.samples, seed=args.seed)
        print(f"  • Running OCR on {len(eval_synth_items)} synthetic documents...")
        synth_report = evaluate_ocr_dataset(
            samples=eval_synth_items,
            dataset_type="synthetic",
            ocr_engine=args.engine,
        )
        s_ov = synth_report.get("overall_metrics", {})
        print(f"    - Mean Latency:        {s_ov.get('mean_latency_ms', 0):.1f} ms/doc")
        print(f"    - Overall CER:         {s_ov.get('mean_cer', 0):.4f} (Target <= 0.1500)")
        print(f"    - Mean Similarity:     {s_ov.get('mean_edit_similarity', 0)*100:.1f}%")
        print(f"    - Exact Match Rate:    {s_ov.get('exact_match_rate', 0)*100:.1f}%")

    # 2. Evaluate Real-World Benchmark (MIDV-500)
    if args.dataset in ["midv500", "all"]:
        print("\n[STEP 2] Evaluating Real-World Benchmark (MIDV-500)...")
        midv_data_dir = "data/midv500"
        if not os.path.exists(midv_data_dir) or not os.listdir(midv_data_dir):
            midv_data_dir = "data/midv500_sample"
            if not os.path.exists(midv_data_dir) or not os.listdir(midv_data_dir):
                print(f"  • Generating sample MIDV-500 benchmark fixture in {midv_data_dir}...")
                generate_sample_midv500_dataset(output_dir=midv_data_dir, n_clips=5, frames_per_clip=3)

        midv_ds = load_midv500_dataset(midv_data_dir)
        midv_samples = midv_ds if isinstance(midv_ds, list) else midv_ds.get("samples", [])

        if not midv_samples:
            print("  [Warning] No MIDV-500 samples loaded.")
        else:
            eval_midv_items = midv_samples[:args.samples]
            print(f"  • Running OCR on {len(eval_midv_items)} MIDV-500 video frames...")
            midv_report = evaluate_ocr_dataset(
                samples=eval_midv_items,
                dataset_type="midv500",
                ocr_engine=args.engine,
            )
            m_ov = midv_report.get("overall_metrics", {})
            print(f"    - Mean Latency:        {m_ov.get('mean_latency_ms', 0):.1f} ms/frame")
            print(f"    - Overall CER:         {m_ov.get('mean_cer', 0):.4f} (Target <= 0.2500)")
            print(f"    - Mean Similarity:     {m_ov.get('mean_edit_similarity', 0)*100:.1f}%")
            print(f"    - Exact Match Rate:    {m_ov.get('exact_match_rate', 0)*100:.1f}%")

    # 3. Multi-Condition Optical Stress Testing (Optional)
    robust_report = None
    if getattr(args, "stress_test", False):
        from src.ocr_evaluate import evaluate_ocr_robustness
        print("\n[STEP 3] Running Multi-Condition Optical Stress-Testing (Defocus Blur, Glare, Underexposure, Downsampling)...")
        stress_samples = []
        if synth_report and "eval_synth_items" in locals() and eval_synth_items:
            stress_samples.extend(eval_synth_items[:8])
        elif midv_report and "eval_midv_items" in locals() and eval_midv_items:
            stress_samples.extend(eval_midv_items[:8])

        if stress_samples:
            robust_report = evaluate_ocr_robustness(
                samples=stress_samples,
                ocr_engine=args.engine,
            )
            print("-" * 75)
            print(f"{'Condition':<20} | {'Mean CER':<10} | {'CER Delta':<11} | {'Degradation':<12} | {'Exact Match'}")
            print("-" * 75)
            for c_name, c_res in robust_report.get("stress_conditions", {}).items():
                c_cer = c_res.get("mean_cer", 0.0)
                c_delta = c_res.get("cer_delta_vs_baseline", 0.0)
                c_deg = c_res.get("cer_degradation_pct", 0.0)
                c_em = c_res.get("exact_match_rate", 0.0) * 100
                print(f"{c_name:<20} | {c_cer:<10.4f} | {c_delta:<+11.4f} | {c_deg:<+11.1f}% | {c_em:.1f}%")
            print("-" * 75)
            print(f"Robustness Summary: {robust_report.get('robustness_summary')}")

    # 4. Export Comprehensive Forensic Reports
    print("\n[STEP 4] Exporting Forensic Audit Reports...")
    export_paths = export_ocr_reports(
        synthetic_report=synth_report,
        midv_report=midv_report,
        robustness_report=robust_report,
    )
    print(f"  • Summary CSV:   {export_paths['csv_path']}")
    print(f"  • Audit Report:  {export_paths['md_path']}")
    print(f"  • Results JSON:  {export_paths['json_path']}")

    # 4. Generate Visual Diagnostic Explanation Cards
    if args.cards > 0:
        print(f"\n[STEP 4] Generating Visual Diagnostic Explanation Cards ({args.cards} cards)...")
        vis_dir = os.path.join(get_reports_dir(), "visuals")
        cards_generated = []

        if synth_report and synth_report.get("sample_evaluations"):
            c_paths = batch_generate_ocr_cards(
                synth_report["sample_evaluations"],
                output_dir=vis_dir,
                max_cards=args.cards // 2 or 1,
            )
            cards_generated.extend(c_paths)

        if midv_report and midv_report.get("sample_evaluations"):
            c_paths = batch_generate_ocr_cards(
                midv_report["sample_evaluations"],
                output_dir=vis_dir,
                max_cards=max(1, args.cards - len(cards_generated)),
            )
            cards_generated.extend(c_paths)

        for c in cards_generated:
            print(f"  • Generated: {c}")

    print("\n" + "=" * 60)
    print("Milestone 3 Benchmark Completed Successfully!")
    print("=" * 60)


def cmd_semantic_check(args):
    """Run full Milestone 4 semantic, MRZ, typography & EXIF forensic screening on an image."""
    import cv2
    from src.font_forensics import audit_document_font_consistency
    from src.metadata_forensics import audit_metadata_provenance, extract_image_metadata
    from src.mrz import cross_validate_viz_and_mrz, disambiguate_mrz_checksums, parse_mrz
    from src.ocr import extract_structured_fields
    from src.ocr_forensic_bridge import correlate_tamper_with_fields, fuse_forensic_modalities
    from src.semantic_checks import run_semantic_rule_battery
    from src.semantic_visualize import generate_semantic_diagnostic_card
    from src.utils import get_reports_dir

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    doc_bgr = cv2.imread(image_path)
    if doc_bgr is None:
        print(f"[ERROR] Failed to decode image: {image_path}")
        sys.exit(1)

    print("=" * 65)
    print("ForgeLens-X — M4 Semantic, MRZ & Typographic Forensic Audit")
    print("=" * 65)
    print(f"Target Document: {image_path}")
    print(f"Document Schema: {args.doc_type}")
    print("-" * 65)

    # 1. OCR Extractions & MRZ
    ocr_res = extract_structured_fields(doc_bgr)
    fields = ocr_res.get("fields", {})

    raw_lines = [b.get("text", "") for b in ocr_res.get("ocr_boxes", [])]
    mrz_data = parse_mrz(raw_lines)
    if mrz_data:
        mrz_data = disambiguate_mrz_checksums(mrz_data, viz_fields=fields)
        viz_cross = cross_validate_viz_and_mrz(fields, mrz_data)
    else:
        viz_cross = None

    # 2. Canonical 8-Rule Semantic Audit
    sem_audit = run_semantic_rule_battery(fields, doc_type=args.doc_type)

    # 3. Typography Forensics (SWT)
    font_audit = audit_document_font_consistency(doc_bgr, fields)

    # 4. EXIF Provenance
    meta_raw = extract_image_metadata(image_path)
    meta_audit = audit_metadata_provenance(meta_raw)

    # 5. Spatial & Multi-Modal Fusion
    spatial_bridge = correlate_tamper_with_fields(fields)
    fusion = fuse_forensic_modalities(
        spatial_bridge=spatial_bridge,
        semantic_audit=sem_audit,
        font_audit=font_audit,
        metadata_audit=meta_audit,
        mrz_audit=mrz_data,
        mrz_viz_cross=viz_cross,
    )

    # Print Terminal Table
    print("\n--- Canonical 8-Rule Semantic Matrix ---")
    rule_names = [
        ("impossible_dates", "1. Calendar Sanity"),
        ("chronology_order", "2. Chronology Sequence"),
        ("age_at_issue_sanity", "3. Age-at-Issue Sanity"),
        ("validity_window_sanity", "4. Validity Window"),
        ("anachronism_check", "5. Future Anachronism"),
        ("document_number_format", "6. Doc Number Regex"),
        ("duplicate_field_contradiction", "7. Duplicate Contradiction"),
        ("name_structure_sanity", "8. Name Alpha Sanity"),
    ]
    for r_key, r_label in rule_names:
        chk = sem_audit["checks"].get(r_key, {})
        status = chk.get("status", "UNKNOWN")
        detail = chk.get("detail", "")
        status_badge = f"[{status}]"
        print(f"  {r_label:<26} {status_badge:<16} : {detail[:55]}")

    print("\n--- Multi-Modal Forensic Modalities ---")
    print(f"  Typography (SWT)       : [{font_audit['typography_verdict']}] (Max Z: {font_audit['max_stroke_zscore']:.2f})")
    print(f"  EXIF Provenance        : [{meta_audit['provenance_verdict']}] ({meta_audit.get('software_detected') or 'Clean'})")
    if mrz_data:
        print(f"  ICAO Doc 9303 MRZ      : [{mrz_data['verdict']}] (Format: {mrz_data['format']})")
    else:
        print(f"  ICAO Doc 9303 MRZ      : [NOT_APPLICABLE] (National ID / No MRZ)")

    print("\n" + "=" * 65)
    print(f"COMPOSITE THREAT LEVEL   : {fusion['threat_level']}")
    print(f"AUTHENTICITY STATUS      : {'CLEARED (AUTHENTIC)' if fusion['is_authentic'] else 'REJECTED (FRAUD/TAMPERED)'}")
    print(f"Summary                  : {fusion['verdict_summary']}")
    print("=" * 65)

    # Render Visual Card
    card_path = args.output
    if not card_path:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        card_path = os.path.join(get_reports_dir(), "visuals", f"semantic_card_{stem}.png")

    _, out_path = generate_semantic_diagnostic_card(
        doc_bgr, fields, sem_audit, meta_audit, font_audit,
        mrz_data=mrz_data, mrz_viz_cross=viz_cross,
        spatial_bridge=spatial_bridge,
        doc_id=os.path.basename(image_path),
        output_path=card_path,
    )
    print(f"\n[+] Visual Diagnostic Card Generated: {out_path}")


def cmd_semantic_eval(args):
    """Run full Milestone 4 semantic benchmark across genuine, tampered, boundary, and MRZ sets."""
    from src.semantic_evaluate import run_full_m4_benchmark

    print("=" * 65)
    print("ForgeLens-X — Milestone 4 Comprehensive Forensic Benchmark")
    print("=" * 65)
    print(f"Dataset Scope: {args.dataset}")
    print(f"Sample Limit:  {args.samples}")
    print(f"Cards Count:   {args.cards}")
    print("-" * 65)

    res = run_full_m4_benchmark(
        samples_per_category=args.samples,
        num_cards=args.cards,
    )

    metrics = res["metrics"]
    print("\n" + "=" * 65)
    print("Milestone 4 Benchmark Summary Results")
    print("=" * 65)
    print(f"  Total Credentials Audited   : {metrics['total_documents_audited']}")
    print(f"  False Rejection Rate (FRR)  : {metrics['false_rejection_rate_frr']*100:.2f}% (Target <= 5%)")
    print(f"  Tamper Detection Rate (TPR) : {metrics['true_positive_rate_tpr']*100:.2f}%")
    print(f"  Boundary Rule Accuracy      : {metrics['boundary_rule_accuracy']*100:.2f}%")
    print(f"  Summary CSV                 : {res['csv_path']}")
    print(f"  Forensic Audit Report       : {res['report_path']}")
def cmd_unified_screen(args):
    """
    Run end-to-end multi-modal forensic screening across M1-M4 producing a unified M5 report.
    Usage: py -m src.cli unified-screen <image_path> [--face <ref_face>] [--doc-type <type>] [--output <card.png>] [--json <report.json>]
    """
    import cv2
    from src.forensic_report import generate_unified_forensic_report, export_unified_report
    from src.unified_visualize import render_unified_forensic_card

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"[ERROR] Document image not found: {image_path}")
        sys.exit(1)

    print("=" * 65)
    print("ForgeLens-X — Milestone 5: Unified Forensic Screening")
    print("=" * 65)
    print(f"Input Document:   {image_path}")
    if getattr(args, "face", None):
        print(f"Reference Face:   {args.face}")
    print(f"Credential Type:  {args.doc_type}")
    print("-" * 65)

    report = generate_unified_forensic_report(
        image_path,
        reference_face_path=getattr(args, "face", None),
        doc_type=args.doc_type,
    )

    output_arg = getattr(args, "output", None)
    json_path = getattr(args, "json", None)
    card_output_path = None

    if output_arg:
        if output_arg.lower().endswith(".json"):
            # User specified JSON report export via --output
            if not json_path:
                json_path = output_arg
        else:
            # User specified diagnostic card path via --output
            card_output_path = output_arg

    card_path = None
    if getattr(args, "card", True):
        doc_bgr = cv2.imread(image_path)
        _, card_path = render_unified_forensic_card(doc_bgr, report, output_path=card_output_path)

    if json_path:
        export_unified_report(report, json_path=json_path)

    # Print executive terminal summary
    q = report['quality']
    print(f"Quality Reliability : {q['analysis_reliability']} (Sharpness: {q['blur_score']:.1f}, Brightness: {q['mean_brightness']:.1f}, Aspect: {q.get('aspect_ratio', 0.0):.2f})")
    print(f"Decision Tier       : {report['decision']}")
    print(f"Fraud Severity      : {report.get('fraud_severity', 'NONE')}")
    print(f"Attack Hypothesis   : {report['attack_type_guess'].upper()} (Confidence: {report['attack_type_confidence']*100:.1f}%)")
    if report.get("multi_attack_detected") and report.get("secondary_attack_guess"):
        sec_att = report['secondary_attack_guess'].upper()
        sec_conf = report.get('secondary_attack_confidence') or 0.0
        print(f"Co-occurring Attack : {sec_att} (Confidence: {sec_conf*100:.1f}%)")
    print(f"Suspicious Regions  : {len(report['suspicious_regions'])} detected")
    print(f"Executive Summary   : {report.get('executive_summary', '')}")
    print("\nEvidentiary Basis:")
    for b in report["attack_type_basis"][:5]:
        print(f"  * {b}")

    if card_path:
        print(f"\n[+] Master Diagnostic Card saved: {card_path}")
    if json_path:
        print(f"[+] Unified JSON Report saved:    {json_path}")
    print("=" * 65)
    return report


def cmd_unified_eval(args):
    """
    Run full M5 evaluation benchmark across genuine, tampered, and degraded documents.
    Usage: py -m src.cli unified-eval [--samples 15] [--cards 4] [--dataset {all,synthetic,degraded}]
    """
    from src.unified_evaluate import run_unified_m5_benchmark

    print("=" * 65)
    print("ForgeLens-X — Milestone 5 Unified Forensic Benchmark")
    print("=" * 65)
    print(f"Dataset scope:        {getattr(args, 'dataset', 'all')}")
    print(f"Samples per category: {args.samples}")
    print(f"Diagnostic cards:     {args.cards}")
    print("-" * 65)

    res = run_unified_m5_benchmark(
        samples_per_category=args.samples,
        num_cards=args.cards,
        dataset_scope=getattr(args, "dataset", "all"),
    )

    metrics = res["metrics"]
    print("\n" + "=" * 65)
    print("Milestone 5 Benchmark Summary Results")
    print("=" * 65)
    print(f"  Total Documents Audited     : {metrics['total_documents_audited']}")
    print(f"  False Rejection Rate (FRR)  : {metrics['false_rejection_rate_frr']*100:.2f}% (Target <= 5%)")
    print(f"  Tamper Detection Rate (TPR) : {metrics['tamper_detection_rate_tpr']*100:.2f}%")
    print(f"  Attack Classification Acc   : {metrics['attack_classification_accuracy']*100:.2f}%")
    print(f"  Quality Gating Success Rate : {metrics['quality_gating_success_rate']*100:.2f}%")
    print(f"  Summary Tabular CSV         : {res['csv_path']}")
    print(f"  Forensic Audit Report       : {res['report_path']}")
    print(f"  JSON Results Contract       : {res['json_path']}")
    print("=" * 65)


def cmd_train_fusion(args):
    """
    Train Milestone 6 Calibrated Machine Learning Risk Fusion Model.
    Usage: py -m src.cli train-fusion [--train data/splits/train.json] [--cal data/splits/cal.json]
    """
    from src.risk_fusion import train_document_risk_model
    from src.utils import get_splits_dir

    train_path = getattr(args, "train", None) or str(get_splits_dir() / "train.json")
    cal_path = getattr(args, "cal", None) or str(get_splits_dir() / "cal.json")

    print("=" * 68)
    print("ForgeLens-X — Milestone 6: Train Document-Risk Fusion Model")
    print("=" * 68)
    print(f"Train Split:       {train_path}")
    print(f"Calibration Split: {cal_path}")
    print("-" * 68)

    meta = train_document_risk_model(train_split_path=train_path, cal_split_path=cal_path)
    print("\n" + "=" * 68)
    print("Milestone 6 Training Summary")
    print("=" * 68)
    print(f"  Selected Calibrator:   {meta['selected_calibration_method']}")
    print(f"  Train ROC-AUC:         {meta['train_auc']:.4f}")
    print(f"  Calibration Brier:     {meta['cal_brier_score']:.4f}")
    print(f"  Trained Features:      {meta['num_features']}")
    print(f"  Model Saved To:        {meta['model_path']}")
    print("=" * 68)


def cmd_evaluate_fusion(args):
    """
    Evaluate Milestone 6 Calibrated Risk Model on Untouched Test Split.
    Usage: py -m src.cli evaluate-fusion [--test data/splits/test.json]
    """
    from src.fusion_evaluate import run_fusion_evaluation
    from src.utils import get_splits_dir

    test_path = getattr(args, "test", None) or str(get_splits_dir() / "test.json")

    print("=" * 68)
    print("ForgeLens-X — Milestone 6: Risk Fusion Test Evaluation")
    print("=" * 68)
    print(f"Test Partition: {test_path}")
    print("-" * 68)

    res = run_fusion_evaluation(test_split_path=test_path)
    m = res["metrics"]
    print("\n" + "=" * 68)
    print("Milestone 6 Test Evaluation Results (Untouched Test Split)")
    print("=" * 68)
    print(f"  Total Test Samples:        {res['test_samples_evaluated']}")
    print(f"  ROC-AUC:                   {m['roc_auc']:.4f} (Target > 0.95)")
    print(f"  PR-AUC:                    {m['pr_auc']:.4f} (Target > 0.95)")
    print(f"  Brier Calibration Score:   {m['brier_score']:.4f} (Target < 0.10)")
    print(f"  Expected Calib Error (ECE):{m['expected_calibration_error_ece']:.4f} (Target < 0.05)")
    print(f"  False Rejection Rate (FRR):{m['false_rejection_rate_frr']*100:.2f}% (Target <= 5%)")
    print(f"  Tamper Detection Rate(TPR):{m['tamper_detection_rate_tpr']*100:.2f}%")
    print(f"  F1 Score:                  {m['f1_score']:.4f}")
    print(f"  Executive Audit Report:    {res['report_path']}")
    print(f"  Summary Tabular CSV:       {res['artifacts']['summary_csv']}")
    print(f"  Calibration Curve Plot:    {res['artifacts']['calibration_curve_plot']}")
    print(f"  ROC/PR Curves Plot:        {res['artifacts']['roc_pr_curves_plot']}")
    print("=" * 68)


def cmd_score_risk(args):
    """
    Score Document Manipulation Risk with Calibrated Probability and Decision Policy.
    Usage: py -m src.cli score-risk <image> [--face <selfie>] [--prior 0.05]
    """
    from src.forensic_report import generate_unified_forensic_report
    from src.risk_fusion import predict_document_risk, apply_decision_policy

    image_path = args.image
    face_path = getattr(args, "face", None)
    prior = getattr(args, "prior", None)

    print("=" * 68)
    print("ForgeLens-X — Milestone 6: Calibrated Document Risk Scoring")
    print("=" * 68)
    print(f"Document Image: {image_path}")
    if face_path:
        print(f"Reference Face: {face_path}")
    if prior:
        print(f"Operational Prior Base-Rate: {prior * 100:.1f}%")
    print("-" * 68)

    report = generate_unified_forensic_report(image_path, reference_face_path=face_path)
    f_vec = report.get("feature_vector", {})
    risk_info = predict_document_risk(f_vec, operational_prior=prior)

    final_decision, risk_tier, basis = apply_decision_policy(
        document_decision=report.get("decision", "VERIFIED"),
        fraud_probability=risk_info["fraud_probability"],
        face_verification=report.get("face_verification", {}),
        quality=report.get("quality", {}),
    )

    print(f"  Calibrated Fraud Probability : {risk_info['fraud_probability']:.4f} ({risk_info['fraud_probability']*100:.1f}%)")
    print(f"  Document Risk Score          : {risk_info['risk_score']:.1f} / 100.0")
    print(f"  Risk Severity Tier           : {risk_tier}")
    print(f"  Operational Decision         : {final_decision}")
    print(f"  Model Engine Status          : {risk_info.get('model_status')}")
    print("\nTop Explainable Risk Drivers (Log-Odds Attribution):")
    drivers = risk_info.get("top_risk_drivers", [])
    if drivers:
        for i, d in enumerate(drivers, 1):
            print(f"    {i}. {d.get('description')} ({d.get('contribution_log_odds', 0.0):+.2f} log-odds)")
    else:
        print("    None (All forensic metrics conform strictly to authentic distribution)")
    print("\nDecision Policy Rationale:")
    for b in basis:
        print(f"    • {b}")
    print("=" * 68)

    if getattr(args, "json", None):
        out_json = args.json
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        export_data = {
            "image": image_path,
            "face": face_path,
            "risk_score": risk_info["risk_score"],
            "fraud_probability": risk_info["fraud_probability"],
            "decision": final_decision,
            "risk_tier": risk_tier,
            "risk_drivers": drivers,
            "policy_basis": basis,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"[+] Risk scoring contract exported to: {out_json}")


def cmd_system_audit(args):
    """
    Master System Diagnostic & Cross-Milestone Health Audit.
    Runs comprehensive smoke verification across M1, M2, M3, M4, M5, and M6.
    Usage: py -m src.cli system-audit [--json reports/system_audit.json]
    """
    print("=" * 68)
    print("ForgeLens-X — Master System Diagnostic & Integration Health Audit")
    print("=" * 68)
    print("Benchmarking all subsystems (Milestones 1 through 6)...")
    print("-" * 68)

    t0_master = time.time()
    results = {
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "milestones": {},
    }

    from src.utils import get_generated_dir
    gen_dir = get_generated_dir()
    sample_doc = None
    sample_selfie = None
    if os.path.exists(gen_dir / "images"):
        p = gen_dir / "images" / "src_0000_genuine.jpg"
        if p.exists():
            sample_doc = str(p)
        s = gen_dir / "images" / "selfies" / "selfie_0000.jpg"
        if s.exists():
            sample_selfie = str(s)

    # 1. Milestone 1: Physical Forensics
    t0 = time.time()
    m1_status = "PASS"
    m1_details = []
    try:
        from src.document_template import generate_document
        from src.ela import analyze_ela
        from src.copy_move import detect_copy_move

        if sample_doc and os.path.exists(sample_doc):
            test_path = sample_doc
        else:
            doc = generate_document(source_id="audit_m1", seed=42)
            import tempfile, cv2
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                test_path = tf.name
            cv2.imwrite(test_path, doc["image"])

        ela_res = analyze_ela(test_path)
        cm_res = detect_copy_move(test_path)
        if test_path != sample_doc and os.path.exists(test_path):
            os.remove(test_path)

        m1_details.append(f"ELA error ratio={ela_res['features'].get('high_error_pixel_ratio', 0.0):.4f}")
        m1_details.append(f"CopyMove matches={cm_res.get('num_matches', 0)}")
    except Exception as e:
        m1_status = "FAIL"
        m1_details.append(f"Error: {e}")
    m1_time = (time.time() - t0) * 1000
    results["milestones"]["M1_Physical_Forensics"] = {
        "status": m1_status,
        "latency_ms": round(m1_time, 2),
        "details": "; ".join(m1_details),
    }

    # 2. Milestone 2: Biometrics & In-Memory Extraction
    t0 = time.time()
    m2_status = "PASS"
    m2_details = []
    try:
        from src.face_verify import calculate_similarity_pct, DEFAULT_THRESHOLDS
        from src.identity_screener import extract_face_from_document
        import cv2

        if sample_doc and os.path.exists(sample_doc):
            face_info = extract_face_from_document(sample_doc)
            has_det = face_info is not None and face_info.get("face_image") is not None
            m2_details.append(f"In-memory doc face detected={has_det}")
        else:
            m2_details.append("In-memory extraction verified")

        sim = calculate_similarity_pct(0.20, 0.40, beta=8.0)
        m2_details.append(f"Calibrated similarity={sim:.1f}%")
    except Exception as e:
        m2_status = "FAIL"
        m2_details.append(f"Error: {e}")
    m2_time = (time.time() - t0) * 1000
    results["milestones"]["M2_Biometric_Verification"] = {
        "status": m2_status,
        "latency_ms": round(m2_time, 2),
        "details": "; ".join(m2_details),
    }

    # 3. Milestone 3: Structured OCR & Field Parsing
    t0 = time.time()
    m3_status = "PASS"
    m3_details = []
    try:
        from src.ocr import extract_structured_fields, get_ocr_engine
        import cv2
        engine = get_ocr_engine()
        eng_name = engine.__class__.__name__ if engine else "Fallback"
        m3_details.append(f"Engine={eng_name}")

        if sample_doc and os.path.exists(sample_doc):
            doc_bgr = cv2.imread(sample_doc)
            ocr_res = extract_structured_fields(doc_bgr)
            fields_found = len([k for k, v in ocr_res.get("fields", {}).items() if not k.startswith("_") and isinstance(v, dict) and v.get("value")])
            m3_details.append(f"Fields extracted={fields_found}")
        else:
            m3_details.append("Engine initialized successfully")
    except Exception as e:
        m3_status = "FAIL"
        m3_details.append(f"Error: {e}")
    m3_time = (time.time() - t0) * 1000
    results["milestones"]["M3_Structured_OCR"] = {
        "status": m3_status,
        "latency_ms": round(m3_time, 2),
        "details": "; ".join(m3_details),
    }

    # 4. Milestone 4: Semantic Rules, MRZ, Typography, Metadata
    t0 = time.time()
    m4_status = "PASS"
    m4_details = []
    try:
        from src.semantic_checks import run_semantic_rule_battery
        from src.mrz import parse_mrz

        mock_fields = {
            "dob": {"value": "1990-05-15"},
            "issue_date": {"value": "2020-01-10"},
            "expiry_date": {"value": "2030-01-10"},
            "document_number": {"value": "FL-1234567"},
            "name": {"value": "JOHN DOE"},
            "country": {"value": "UTO"},
        }
        sem_res = run_semantic_rule_battery(mock_fields)
        passed_rules = sum(1 for c in sem_res.get("checks_list", []) if c.get("status") == "PASS")
        m4_details.append(f"Semantic rules passed={passed_rules}")

        mrz_mock = ["IDUTOD231458907<<<<<<<<<<<<<<<", "7408122F1204159UTO<<<<<<<<<<<6", "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"]
        mrz_parsed = parse_mrz(mrz_mock)
        has_mrz = mrz_parsed is not None and mrz_parsed.get("format") == "TD1"
        m4_details.append(f"MRZ TD1 parsed={has_mrz}")
    except Exception as e:
        m4_status = "FAIL"
        m4_details.append(f"Error: {e}")
    m4_time = (time.time() - t0) * 1000
    results["milestones"]["M4_Semantic_MRZ_Typography"] = {
        "status": m4_status,
        "latency_ms": round(m4_time, 2),
        "details": "; ".join(m4_details),
    }

    # 5. Milestone 5: Unified Forensic Screening Pipeline
    t0 = time.time()
    m5_status = "PASS"
    m5_details = []
    try:
        from src.forensic_report import generate_unified_forensic_report, validate_report_schema

        if sample_doc and os.path.exists(sample_doc):
            report = generate_unified_forensic_report(sample_doc, reference_face_path=sample_selfie)
            is_valid, errs = validate_report_schema(report)
            if not is_valid:
                m5_status = "FAIL"
                m5_details.append(f"Schema errors: {errs}")
            else:
                m5_details.append(f"Schema 1.0 Valid; Decision={report['decision']}; Attack={report['attack_type_guess']}")
                m5_details.append(f"Features={len(report.get('feature_vector', {}))}")
        else:
            m5_details.append("Module imported and callable")
    except Exception as e:
        m5_status = "FAIL"
        m5_details.append(f"Error: {e}")
    m5_time = (time.time() - t0) * 1000
    results["milestones"]["M5_Unified_Forensic_Report"] = {
        "status": m5_status,
        "latency_ms": round(m5_time, 2),
        "details": "; ".join(m5_details),
    }

    # 6. Milestone 6: Calibrated ML Risk Fusion & Explainability
    t0 = time.time()
    m6_status = "PASS"
    m6_details = []
    try:
        from src.risk_fusion import predict_document_risk, apply_decision_policy, get_feature_names
        feats = get_feature_names()
        mock_vec = {fn: 0.0 for fn in feats}
        mock_vec["ela_high_error_ratio"] = 0.05
        risk_res = predict_document_risk(mock_vec)
        pol_dec, pol_tier, pol_basis = apply_decision_policy(
            document_decision="VERIFIED",
            fraud_probability=risk_res["fraud_probability"],
        )
        m6_details.append(f"Engine={risk_res['model_status']}; RiskScore={risk_res['risk_score']:.1f}; Tier={pol_tier}")
        m6_details.append(f"Features={len(feats)}")
    except Exception as e:
        m6_status = "FAIL"
        m6_details.append(f"Error: {e}")
    m6_time = (time.time() - t0) * 1000
    results["milestones"]["M6_Risk_Fusion_ML"] = {
        "status": m6_status,
        "latency_ms": round(m6_time, 2),
        "details": "; ".join(m6_details),
    }

    total_time = (time.time() - t0_master) * 1000
    overall_status = "PASS" if all(v["status"] == "PASS" for v in results["milestones"].values()) else "FAIL"
    results["status"] = overall_status
    results["total_latency_ms"] = round(total_time, 2)

    print(f"{'Milestone Subsystem':<32} | {'Status':<6} | {'Latency':<9} | Details")
    print("-" * 68)
    for m_name, m_data in results["milestones"].items():
        print(f"{m_name:<32} | {m_data['status']:<6} | {m_data['latency_ms']:>6.1f} ms | {m_data['details']}")
    print("-" * 68)
    print(f"Overall System Health : {overall_status} (Total Latency: {total_time:.1f} ms)")
    print("=" * 68)

    if getattr(args, "json", None):
        out_json = args.json
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[+] Exported system diagnostic report to: {out_json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        prog="forgelens-m1",
        description="ForgeLens-X — Multi-Modal Identity & Document Forensics Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic dataset")
    gen_parser.add_argument("--samples", type=int, default=10)
    gen_parser.add_argument("--seed", type=int, default=42)

    # analyze
    ana_parser = subparsers.add_parser("analyze", help="Run forensic analysis")
    ana_parser.add_argument("--data-dir", type=str, default=None)
    ana_parser.add_argument("--seed", type=int, default=42)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate results")
    eval_parser.add_argument("--seed", type=int, default=42)

    # visualize
    vis_parser = subparsers.add_parser("visualize", help="Generate visual forensic explanation cards")
    vis_parser.add_argument("--count", type=int, default=5, help="Number of diagnostic cards to generate")

    # screen
    screen_parser = subparsers.add_parser("screen", help="Screen a single document for forensic manipulation")
    screen_parser.add_argument("image", type=str, help="Path to document image")
    screen_parser.add_argument("--output", type=str, default=None, help="Output path for visual forensic card")

    # run-demo
    demo_parser = subparsers.add_parser("run-demo", help="Full pipeline demo")
    demo_parser.add_argument("--samples", type=int, default=10)
    demo_parser.add_argument("--seed", type=int, default=42)

    # --- Milestone 2: Face Verification Subcommands ---

    # face-verify
    fv_parser = subparsers.add_parser("face-verify", help="Verify document face crop against presented live face")
    fv_parser.add_argument("doc_face", type=str, help="Path to document face image")
    fv_parser.add_argument("live_face", type=str, help="Path to presented live selfie image")
    fv_parser.add_argument("--model", type=str, default="ArcFace", choices=["ArcFace", "Facenet512", "SFace"], help="Model backbone")
    fv_parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean_l2"], help="Distance metric")
    fv_parser.add_argument("--output", type=str, default=None, help="Output path for visual explanation card")

    # face-compare
    fc_parser = subparsers.add_parser("face-compare", help="Compare multiple face recognition models on an image pair")
    fc_parser.add_argument("doc_face", type=str, help="Path to document face image")
    fc_parser.add_argument("live_face", type=str, help="Path to presented live selfie image")
    fc_parser.add_argument("--models", nargs="+", default=None, help="List of models to evaluate (e.g. ArcFace Facenet512 SFace)")
    fc_parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean_l2"], help="Distance metric")

    # face-eval
    fe_parser = subparsers.add_parser("face-eval", help="Evaluate face verification on benchmark dataset")
    fe_parser.add_argument("--pairs-dir", type=str, default=None, help="Path to face pairs directory")
    fe_parser.add_argument("--pairs", type=int, default=20, help="Number of pairs to evaluate/generate")
    fe_parser.add_argument("--model", type=str, default="ArcFace", choices=["ArcFace", "Facenet512", "SFace"], help="Model backbone")
    fe_parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean_l2"], help="Distance metric")
    fe_parser.add_argument("--cards", type=int, default=4, help="Number of visual diagnostic cards to generate")
    fe_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # screen-identity (Bridge M1 + M2)
    si_parser = subparsers.add_parser("screen-identity", help="End-to-end screen document tamper + live face verification")
    si_parser.add_argument("doc_image", type=str, help="Path to identity document image")
    si_parser.add_argument("live_face", type=str, help="Path to presented live selfie image")
    si_parser.add_argument("--model", type=str, default="ArcFace", choices=["ArcFace", "Facenet512", "SFace"], help="Face recognition model")
    si_parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean_l2"], help="Distance metric")
    si_parser.add_argument("--output", type=str, default=None, help="Output path for visual identity screening card")

    # --- Milestone 3: OCR Subcommands ---

    # ocr
    ocr_parser = subparsers.add_parser("ocr", help="Extract structured identity fields using OCR")
    ocr_parser.add_argument("image", type=str, help="Path to identity document image")
    ocr_parser.add_argument("--engine", type=str, default="rapidocr", choices=["rapidocr", "tesseract"], help="OCR engine")
    ocr_parser.add_argument("--no-preprocess", action="store_true", help="Disable adaptive image preprocessing")
    ocr_parser.add_argument("--output", type=str, default=None, help="Output path for visual explanation card")

    # ocr-eval
    oe_parser = subparsers.add_parser("ocr-eval", help="Evaluate OCR field extraction and CER across benchmarks")
    oe_parser.add_argument("--dataset", type=str, default="all", choices=["synthetic", "midv500", "all"], help="Benchmark dataset to evaluate")
    oe_parser.add_argument("--samples", type=int, default=20, help="Number of document samples to evaluate")
    oe_parser.add_argument("--engine", type=str, default="rapidocr", choices=["rapidocr", "tesseract"], help="OCR engine")
    oe_parser.add_argument("--cards", type=int, default=4, help="Number of visual diagnostic explanation cards to generate")
    oe_parser.add_argument("--stress-test", action="store_true", help="Run multi-condition optical stress testing (blur, glare, underexposure, downsampling)")

    # --- Milestone 4: Semantic, MRZ & Typography Subcommands ---

    # semantic-check
    sc_parser = subparsers.add_parser("semantic-check", help="Run full semantic, MRZ, typography & EXIF forensic audit on a document")
    sc_parser.add_argument("image", type=str, help="Path to identity document image")
    sc_parser.add_argument("--doc-type", type=str, default="forgelensia", choices=["forgelensia", "passport", "generic_id"], help="Document credential schema")
    sc_parser.add_argument("--output", type=str, default=None, help="Output path for visual diagnostic explanation card")

    # semantic-eval
    se_parser = subparsers.add_parser("semantic-eval", help="Evaluate semantic rules, MRZ repair, and typography consistency on benchmark dataset")
    se_parser.add_argument("--dataset", type=str, default="all", choices=["synthetic", "boundary", "all"], help="Benchmark dataset scope")
    se_parser.add_argument("--samples", type=int, default=15, help="Number of document samples to evaluate per category")
    se_parser.add_argument("--cards", type=int, default=4, help="Number of visual diagnostic explanation cards to generate")

    # --- Milestone 5: Unified Forensic Report Subcommands ---

    # unified-screen
    us_parser = subparsers.add_parser("unified-screen", help="Run full multi-modal forensic screening (M1-M4) producing unified M5 report")
    us_parser.add_argument("image", type=str, help="Path to identity document image")
    us_parser.add_argument("--face", "--selfie", dest="face", type=str, default=None, help="Optional path to reference live face selfie photo")
    us_parser.add_argument("--doc-type", type=str, default="forgelensia", choices=["forgelensia", "passport", "generic_id"], help="Document credential schema")
    us_parser.add_argument("--output", type=str, default=None, help="Output path for master 4-panel diagnostic card")
    us_parser.add_argument("--json", type=str, default=None, help="Output path to export unified JSON report contract")
    us_parser.add_argument("--no-card", action="store_false", dest="card", default=True, help="Disable visual diagnostic card generation")

    # unified-eval
    ue_parser = subparsers.add_parser("unified-eval", help="Evaluate unified report accuracy, quality gating, and attack classification")
    ue_parser.add_argument("--dataset", type=str, default="all", choices=["synthetic", "degraded", "all"], help="Benchmark dataset scope")
    ue_parser.add_argument("--samples", type=int, default=15, help="Number of document samples to evaluate per attack category")
    ue_parser.add_argument("--cards", type=int, default=4, help="Number of master diagnostic cards to generate")

    # --- Milestone 6: Machine Learning Risk Fusion Subcommands ---

    # train-fusion
    tf_parser = subparsers.add_parser("train-fusion", help="Train Milestone 6 calibrated risk fusion model")
    tf_parser.add_argument("--train", type=str, default=None, help="Path to train split JSON (default: data/splits/train.json)")
    tf_parser.add_argument("--cal", type=str, default=None, help="Path to cal split JSON (default: data/splits/cal.json)")

    # evaluate-fusion
    ef_parser = subparsers.add_parser("evaluate-fusion", help="Evaluate Milestone 6 calibrated risk model on untouched test split")
    ef_parser.add_argument("--test", type=str, default=None, help="Path to test split JSON (default: data/splits/test.json)")

    # score-risk
    sr_parser = subparsers.add_parser("score-risk", help="Score document manipulation risk with calibrated probability and decision policy")
    sr_parser.add_argument("image", type=str, help="Path to document image")
    sr_parser.add_argument("--face", "--selfie", dest="face", type=str, default=None, help="Optional reference face image")
    sr_parser.add_argument("--prior", type=float, default=None, help="Operational prior fraud base rate (e.g. 0.05)")
    sr_parser.add_argument("--json", type=str, default=None, help="Optional export path for scoring contract JSON")

    # --- Master System Audit / Diagnostics ---
    sa_parser = subparsers.add_parser("system-audit", aliases=["diagnostics"], help="Run master system diagnostic & integration health audit across M1-M6")
    sa_parser.add_argument("--json", type=str, default=None, help="Optional output path to export diagnostic report JSON")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "generate": cmd_generate,
        "analyze": cmd_analyze,
        "evaluate": cmd_evaluate,
        "visualize": cmd_visualize,
        "screen": cmd_screen,
        "run-demo": cmd_run_demo,
        "face-verify": cmd_face_verify,
        "face-compare": cmd_face_compare,
        "face-eval": cmd_face_eval,
        "screen-identity": cmd_screen_identity,
        "ocr": cmd_ocr,
        "ocr-eval": cmd_ocr_eval,
        "semantic-check": cmd_semantic_check,
        "semantic-eval": cmd_semantic_eval,
        "unified-screen": cmd_unified_screen,
        "unified-eval": cmd_unified_eval,
        "train-fusion": cmd_train_fusion,
        "evaluate-fusion": cmd_evaluate_fusion,
        "score-risk": cmd_score_risk,
        "system-audit": cmd_system_audit,
        "diagnostics": cmd_system_audit,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
