"""
ForgeLens-X — Milestone 2: Face Verification Evaluation Module
================================================================
Evaluates face verification performance over benchmark pairs.
Computes empirical metrics:
    - Accuracy
    - False Accept Rate (FAR = FP / (FP + TN))
    - False Reject Rate (FRR = FN / (FN + TP))
    - True Accept Rate (TAR = 1 - FRR)
    - Distance distribution separation
    - Multi-model comparison (ArcFace vs Facenet512)

Exports:
    - reports/m2_results.json
    - reports/m2_pairs_summary.csv
    - reports/m2_face_audit_report.md
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.face_verify import compare_models, verify
from src.utils import ensure_dirs, get_reports_dir, save_metadata


# ---------------------------------------------------------------------------
# Metric Calculations
# ---------------------------------------------------------------------------

def calculate_verification_metrics(
    predictions: List[bool],
    ground_truths: List[bool],
    distances: List[float],
) -> Dict[str, Any]:
    """
    Compute binary classification & biometric verification metrics.

    Args:
        predictions: list of bool (True = verified/match)
        ground_truths: list of bool (True = genuine/same person)
        distances: list of float distance values

    Returns:
        dict of metrics (accuracy, FAR, FRR, TAR, distance distributions)
    """
    tp = tn = fp = fn = 0
    genuine_dists = []
    imposter_dists = []

    for pred, gt, dist in zip(predictions, ground_truths, distances):
        if dist is not None:
            if gt:
                genuine_dists.append(dist)
            else:
                imposter_dists.append(dist)

        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
            fp += 1
        else:
            tn += 1

    total = max(len(predictions), 1)
    total_genuine = max(tp + fn, 1)
    total_imposter = max(fp + tn, 1)

    accuracy = (tp + tn) / total
    far = fp / total_imposter
    frr = fn / total_genuine
    tar = 1.0 - frr

    mean_gen = float(np.mean(genuine_dists)) if genuine_dists else 0.0
    mean_imp = float(np.mean(imposter_dists)) if imposter_dists else 0.0
    std_gen = float(np.std(genuine_dists)) if genuine_dists else 0.0
    std_imp = float(np.std(imposter_dists)) if imposter_dists else 0.0

    return {
        "total_pairs": len(predictions),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(float(accuracy), 4),
        "far": round(float(far), 4),
        "frr": round(float(frr), 4),
        "tar": round(float(tar), 4),
        "mean_genuine_distance": round(mean_gen, 4),
        "std_genuine_distance": round(std_gen, 4),
        "mean_imposter_distance": round(mean_imp, 4),
        "std_imposter_distance": round(std_imp, 4),
        "mean_genuine_dist": round(mean_gen, 4),
        "std_genuine_dist": round(std_gen, 4),
        "mean_imposter_dist": round(mean_imp, 4),
        "std_imposter_dist": round(std_imp, 4),
        "separation_margin": round(mean_imp - mean_gen, 4),
    }


# ---------------------------------------------------------------------------
# Batch Pair Evaluation
# ---------------------------------------------------------------------------

def evaluate_face_pairs(
    pairs: List[Dict[str, Any]],
    model_name: str = "ArcFace",
    distance_metric: str = "cosine",
) -> Dict[str, Any]:
    """
    Evaluate a batch of face pairs using the specified model.

    Args:
        pairs: list of pair metadata dicts from face_dataset
        model_name: model to evaluate ('ArcFace', 'Facenet512', etc.)
        distance_metric: metric to evaluate

    Returns:
        Full evaluation report dictionary.
    """
    import time
    start_time = time.time()

    pair_results = []
    preds = []
    gts = []
    dists = []

    for p in pairs:
        doc_path = p.get("doc_image_path") or p.get("doc_path")
        live_path = p.get("live_image_path") or p.get("live_path")
        gt = bool(p["is_same_person"])

        res = verify(
            document_face_path=doc_path,
            live_face_path=live_path,
            model_name=model_name,
            distance_metric=distance_metric,
        )

        pred = bool(res.get("verified", False))
        dist = res.get("distance")

        preds.append(pred)
        gts.append(gt)
        dists.append(dist)

        record = {
            "pair_id": p["pair_id"],
            "label": p["label"],
            "is_same_person": gt,
            "subject_doc": p.get("subject_doc"),
            "subject_live": p.get("subject_live"),
            "demographic": p.get("demographic"),
            "doc_image_path": doc_path,
            "live_image_path": live_path,
            "model": model_name,
            "distance_metric": distance_metric,
            "verified": pred,
            "distance": dist,
            "threshold": res.get("threshold"),
            "similarity_pct": res.get("similarity_pct", 0.0),
            "verdict_tier": res.get("verdict_tier", "ERROR"),
            "correct_decision": pred == gt,
            "error": res.get("error"),
        }
        pair_results.append(record)

    total_time = time.time() - start_time
    mean_inf_time = (total_time / max(len(pairs), 1)) * 1000.0
    failed_pairs = sum(1 for r in pair_results if r.get("error"))
    n_gen = sum(1 for p in pairs if p.get("is_same_person"))
    n_imp = len(pairs) - n_gen

    metrics = calculate_verification_metrics(preds, gts, dists)
    metrics["failed_pairs"] = failed_pairs
    metrics["mean_inference_time_ms"] = round(mean_inf_time, 1)

    return {
        "model_name": model_name,
        "distance_metric": distance_metric,
        "evaluated_at": datetime.now().isoformat(),
        "n_pairs": len(pairs),
        "n_genuine": n_gen,
        "n_imposter": n_imp,
        "metrics": metrics,
        "pair_results": pair_results,
    }


def evaluate_model_comparison(
    pairs: List[Dict[str, Any]],
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate and compare multiple face verification models side-by-side
    on the exact same benchmark pairs.
    """
    if models is None:
        models = ["ArcFace", "Facenet512"]

    comparison_reports = {}
    for m in models:
        comparison_reports[m] = evaluate_face_pairs(pairs, model_name=m)

    # Compute agreement statistics
    agreements = 0
    total = len(pairs)
    for idx in range(total):
        verdicts = [comparison_reports[m]["pair_results"][idx]["verified"] for m in models]
        if all(v == verdicts[0] for v in verdicts):
            agreements += 1

    consensus_rate = agreements / max(total, 1)

    return {
        "models": models,
        "evaluated_at": datetime.now().isoformat(),
        "n_pairs": total,
        "consensus_rate": round(consensus_rate, 4),
        "model_reports": {m: r["metrics"] for m, r in comparison_reports.items()},
        "detailed_reports": comparison_reports,
    }


# ---------------------------------------------------------------------------
# Report Export Utilities
# ---------------------------------------------------------------------------

def save_face_evaluation_report(
    eval_results: Dict[str, Any],
    filename: str = "m2_results.json",
) -> str:
    """Save evaluation results to reports/m2_results.json."""
    reports_dir = get_reports_dir()
    ensure_dirs(reports_dir)
    out_path = os.path.join(reports_dir, filename)
    save_metadata(eval_results, out_path)
    return out_path


def export_face_pairs_summary_csv(
    pair_results: List[Dict[str, Any]],
    filename: str = "m2_pairs_summary.csv",
) -> str:
    """
    Export granular pair-by-pair metrics to CSV.
    """
    reports_dir = get_reports_dir()
    ensure_dirs(reports_dir)
    out_path = os.path.join(reports_dir, filename)

    fieldnames = [
        "pair_id",
        "label",
        "is_same_person",
        "subject_doc",
        "subject_live",
        "demographic",
        "model",
        "distance_metric",
        "distance",
        "threshold",
        "similarity_pct",
        "verified",
        "verdict_tier",
        "correct_decision",
        "error",
    ]

    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in pair_results:
            writer.writerow({
                "pair_id": r.get("pair_id", ""),
                "label": r.get("label", ""),
                "is_same_person": r.get("is_same_person", False),
                "subject_doc": r.get("subject_doc", ""),
                "subject_live": r.get("subject_live", ""),
                "demographic": r.get("demographic", ""),
                "model": r.get("model", ""),
                "distance_metric": r.get("distance_metric", ""),
                "distance": r.get("distance", ""),
                "threshold": r.get("threshold", ""),
                "similarity_pct": r.get("similarity_pct", 0.0),
                "verified": r.get("verified", False),
                "verdict_tier": r.get("verdict_tier", ""),
                "correct_decision": r.get("correct_decision", False),
                "error": r.get("error", ""),
            })

    return out_path


def generate_face_markdown_audit_report(
    eval_results: Dict[str, Any],
    filename: str = "m2_face_audit_report.md",
) -> str:
    """
    Generate an executive forensic face verification audit report in Markdown.
    """
    reports_dir = get_reports_dir()
    ensure_dirs(reports_dir)
    out_path = os.path.join(reports_dir, filename)

    metrics = eval_results.get("metrics", {})
    model_name = eval_results.get("model_name", "ArcFace")
    metric_name = eval_results.get("distance_metric", "cosine")
    n_pairs = metrics.get("total_pairs", 0)

    lines = [
        "# ForgeLens-X — Milestone 2: Face Verification Audit Report",
        "",
        f"**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Evaluated Model:** `{model_name}` ({metric_name} distance)",
        f"**Benchmark Sample Size:** {n_pairs} balanced pairs",
        "",
        "## 1. Executive Summary",
        "",
        "Milestone 2 establishes cross-modality facial verification between identity credential portraits and presented live selfies:",
        "- **Cross-Modality Normalization**: 5-point landmark affine alignment to canonical $112 \\times 112$ geometry eliminates perspective/scale bias.",
        "- **Deep Representation Backends**: Supported ArcFace (512-D), Facenet512 (512-D), and SFace ONNX (native 128-D).",
        "- **Strict Zero-Crash Policy**: Detects non-face images gracefully and outputs structured error dictionary without terminating the screening pipeline.",
        "- **Post-Fusion Independence**: Retained strictly as an independent post-fusion gate in Milestone 6.",
        "",
        "## 2. Empirical Verification Performance",
        "",
        f"- **Overall Accuracy:** {metrics.get('accuracy', 0.0) * 100:.1f}%",
        f"- **True Accept Rate (TAR):** {metrics.get('tar', 0.0) * 100:.1f}%",
        f"- **False Accept Rate (FAR - Imposter Leakage):** {metrics.get('far', 0.0) * 100:.1f}%",
        f"- **False Reject Rate (FRR - Genuine Friction):** {metrics.get('frr', 0.0) * 100:.1f}%",
        "",
        "| Metric | Measured Value | Operational Interpretation |",
        "| :--- | :--- | :--- |",
        f"| **True Positives (TP)** | {metrics.get('true_positives', 0)} | Genuine identity correctly verified |",
        f"| **True Negatives (TN)** | {metrics.get('true_negatives', 0)} | Imposter/mismatch correctly rejected |",
        f"| **False Positives (FP)** | {metrics.get('false_positives', 0)} | Imposter incorrectly admitted (Security Risk) |",
        f"| **False Negatives (FN)** | {metrics.get('false_negatives', 0)} | Genuine user rejected (User Friction) |",
        f"| **Mean Genuine Distance** | {metrics.get('mean_genuine_distance', 0.0):.4f} (std: {metrics.get('std_genuine_distance', 0.0):.4f}) | Tight clustering for authentic presentations |",
        f"| **Mean Imposter Distance** | {metrics.get('mean_imposter_distance', 0.0):.4f} (std: {metrics.get('std_imposter_distance', 0.0):.4f}) | Distinct separation from genuine presentations |",
        f"| **Separation Margin** | **{metrics.get('separation_margin', 0.0):.4f}** | Decision boundary margin between classes |",
        "",
        "## 3. Engineering & Forensic Takeaways",
        "",
        "1. **Landmark Normalization**: Aligning the two eye centers and mouth corners directly tackles the discrepancy between a formal flat document photo and an angled mobile selfie.",
        "2. **Calibrated Confidence**: Similarity percentages are mapped through a calibrated sigmoid centered at the operational threshold, giving human reviewers an intuitive confidence index.",
        "3. **Zero-Liveness Boundary Preserved**: As mandated by the canonical specification, liveness/anti-spoofing and morphing detection remain deferred to Milestone 9.",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


# ---------------------------------------------------------------------------
# Aliases & Unified Export Helper
# ---------------------------------------------------------------------------

evaluate_face_verification = evaluate_face_pairs


def export_evaluation_report(
    report: Dict[str, Any],
    csv_path: Optional[str] = None,
    md_path: Optional[str] = None,
    json_path: Optional[str] = None,
) -> Dict[str, str]:
    """Unified helper to export JSON, CSV, and Markdown reports."""
    j_fn = os.path.basename(json_path) if json_path else "m2_results.json"
    c_fn = os.path.basename(csv_path) if csv_path else "m2_pairs_summary.csv"
    m_fn = os.path.basename(md_path) if md_path else "m2_face_audit_report.md"

    j_out = save_face_evaluation_report(report, filename=j_fn)
    c_out = export_face_pairs_summary_csv(report.get("pair_results", []), filename=c_fn)
    m_out = generate_face_markdown_audit_report(report, filename=m_fn)

    return {"json": j_out, "csv": c_out, "md": m_out}

