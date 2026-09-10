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
import sys
import time

import numpy as np

from src.copy_move import detect_copy_move
from src.document_template import generate_document
from src.ela import analyze_ela, compute_baseline
from src.evaluate import evaluate_batch, save_evaluation_report
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

    # --- Analyze each sample ---
    forensic_dir = str(get_forensic_dir())
    ela_dir = os.path.join(forensic_dir, "ela")
    cm_dir = os.path.join(forensic_dir, "copy_move")
    ensure_dirs(ela_dir, cm_dir)

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
            # Copy-move results
            "copy_move_detected": cm_result["detected"],
            "copy_move_bbox": cm_result["candidate_bbox"],
            "copy_move_alt_bbox": cm_result.get("alt_bbox"),
            "copy_move_num_matches": cm_result["num_matches"],
            "copy_move_num_inliers": cm_result["num_inliers"],
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

    # Save report
    report_path = save_evaluation_report(eval_results)

    # Print summary
    print("\n" + "=" * 60)
    print("M1 EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- ELA Detector ---")
    _print_detector_summary(eval_results["ela"])

    print("\n--- Copy-Move Detector ---")
    _print_detector_summary(eval_results["copy_move"])

    print(f"\n[M1] Full report saved: {report_path}")
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


def main():
    parser = argparse.ArgumentParser(
        prog="forgelens-m1",
        description="ForgeLens-X M1 — Synthetic Tamper + ELA + Copy-Move Pipeline",
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

    # run-demo
    demo_parser = subparsers.add_parser("run-demo", help="Full pipeline demo")
    demo_parser.add_argument("--samples", type=int, default=10)
    demo_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "generate": cmd_generate,
        "analyze": cmd_analyze,
        "evaluate": cmd_evaluate,
        "visualize": cmd_visualize,
        "run-demo": cmd_run_demo,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
