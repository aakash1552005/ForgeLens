"""
ForgeLens-X — Evaluation Module
==================================
Evaluates detection and localization performance.

Key distinction:
    - Detection rate: did the detector find SOMETHING?
    - Localization accuracy: did it find the RIGHT region?

A detector can detect something while localizing it incorrectly.
These are separate metrics. A high detection rate with low IoU means
the detector finds tampering but can't pinpoint where.
"""

import csv
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

from src.utils import compute_iou, get_reports_dir, save_metadata


def evaluate_detection(
    predictions: list,
    ground_truths: list,
) -> dict:
    """
    Evaluate binary detection performance.

    Args:
        predictions: list of {"detected": bool, ...}
        ground_truths: list of {"label": "genuine"|"tampered", ...}

    Returns:
        {
            "total": int,
            "true_positives": int,    # tampered correctly detected
            "true_negatives": int,    # genuine correctly not detected
            "false_positives": int,   # genuine incorrectly detected
            "false_negatives": int,   # tampered missed
            "detection_rate": float,  # TP / (TP + FN)
            "false_alarm_rate": float # FP / (FP + TN)
        }
    """
    tp = tn = fp = fn = 0

    for pred, gt in zip(predictions, ground_truths):
        is_tampered = gt.get("label") == "tampered"
        is_detected = pred.get("detected", False)

        if is_tampered and is_detected:
            tp += 1
        elif is_tampered and not is_detected:
            fn += 1
        elif not is_tampered and is_detected:
            fp += 1
        else:
            tn += 1

    detection_rate = tp / max(tp + fn, 1)
    false_alarm_rate = fp / max(fp + tn, 1)

    return {
        "total": len(predictions),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "detection_rate": round(detection_rate, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
    }


def evaluate_localization(
    pred_bbox: list | None,
    gt_bbox: list | None,
) -> float:
    """
    Evaluate localization using IoU.

    Args:
        pred_bbox: predicted [x1,y1,x2,y2] or None
        gt_bbox: ground-truth [x1,y1,x2,y2] or None

    Returns:
        IoU value (0.0 if either bbox is None)
    """
    if pred_bbox is None or gt_bbox is None:
        return 0.0
    return compute_iou(pred_bbox, gt_bbox)


def evaluate_batch(
    results: list,
    iou_threshold: float = 0.3,
) -> dict:
    """
    Full evaluation over a batch of samples.

    Each result should contain:
        {
            "source_id": str,
            "attack_type": str,        # "none", "date_edit", etc.
            "label": str,              # "genuine" or "tampered"
            "ground_truth_bbox": list or None,
            "ela_detected": bool,
            "ela_candidate_bbox": list or None,
            "copy_move_detected": bool,
            "copy_move_bbox": list or None,
        }

    Returns:
        {
            "ela": {
                "overall": {...detection metrics...},
                "per_attack": {
                    "date_edit": {detection_rate, mean_iou, ...},
                    ...
                }
            },
            "copy_move": {
                "overall": {...},
                "per_attack": {...}
            }
        }
    """
    # Separate by detector
    ela_eval = _evaluate_detector(results, "ela", iou_threshold)
    cm_eval = _evaluate_detector(results, "copy_move", iou_threshold)

    return {
        "ela": ela_eval,
        "copy_move": cm_eval,
        "iou_threshold": iou_threshold,
        "n_samples": len(results),
    }


def _evaluate_detector(
    results: list,
    detector_name: str,
    iou_threshold: float,
) -> dict:
    """Evaluate a specific detector across all samples."""
    detected_key = f"{detector_name}_detected"
    bbox_key = f"{detector_name}_candidate_bbox" if detector_name == "ela" else f"{detector_name}_bbox"

    # Overall detection
    predictions = []
    ground_truths = []
    iou_values = defaultdict(list)
    per_attack_data = defaultdict(lambda: {"preds": [], "gts": [], "ious": []})

    for r in results:
        attack_type = r.get("attack_type", "none")
        label = r.get("label", "genuine")
        detected = r.get(detected_key, False)
        pred_bbox = r.get(bbox_key)
        gt_bbox = r.get("ground_truth_bbox")

        predictions.append({"detected": detected})
        ground_truths.append({"label": label})

        if detector_name == "copy_move":
            alt_bbox = r.get("copy_move_alt_bbox")
            src_bbox = r.get("source_bbox")
            candidates = [c for c in [pred_bbox, alt_bbox] if c is not None]
            targets = [t for t in [gt_bbox, src_bbox] if t is not None]
            best_iou = 0.0
            for c in candidates:
                for t in targets:
                    best_iou = max(best_iou, evaluate_localization(c, t))
            iou = best_iou
        else:
            iou = evaluate_localization(pred_bbox, gt_bbox)

        per_attack_data[attack_type]["preds"].append({"detected": detected})
        per_attack_data[attack_type]["gts"].append({"label": label})
        per_attack_data[attack_type]["ious"].append(iou)

        if label == "tampered":
            iou_values[attack_type].append(iou)

    overall_detection = evaluate_detection(predictions, ground_truths)

    # Per-attack metrics
    per_attack = {}
    for attack_type, data in per_attack_data.items():
        det_metrics = evaluate_detection(data["preds"], data["gts"])
        ious = data["ious"]
        tampered_ious = [
            iou
            for iou, gt in zip(ious, data["gts"])
            if gt["label"] == "tampered"
        ]

        per_attack[attack_type] = {
            **det_metrics,
            "mean_iou": round(float(np.mean(tampered_ious)), 4) if tampered_ious else None,
            "median_iou": round(float(np.median(tampered_ious)), 4) if tampered_ious else None,
            "localization_rate": round(
                sum(1 for i in tampered_ious if i >= iou_threshold) / max(len(tampered_ious), 1),
                4,
            ) if tampered_ious else None,
        }

    # Overall IoU for tampered
    all_tampered_ious = [
        iou for attack_ious in iou_values.values() for iou in attack_ious
    ]

    return {
        "overall": {
            **overall_detection,
            "mean_iou": round(float(np.mean(all_tampered_ious)), 4) if all_tampered_ious else None,
            "localization_rate": round(
                sum(1 for i in all_tampered_ious if i >= iou_threshold) / max(len(all_tampered_ious), 1),
                4,
            ) if all_tampered_ious else None,
        },
        "per_attack": per_attack,
    }


def save_evaluation_report(eval_results: dict, filename: str = "m1_results.json") -> str:
    """Save evaluation results to reports/ directory."""
    reports_dir = get_reports_dir()
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, filename)
    save_metadata(eval_results, path)
    return path


def export_samples_summary_csv(
    results: list,
    filename: str = "m1_samples_summary.csv",
) -> str:
    """
    Export per-sample forensic metrics to CSV for granular auditing.
    """
    reports_dir = get_reports_dir()
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, filename)

    fieldnames = [
        "source_id",
        "attack_type",
        "label",
        "ground_truth_bbox",
        "ela_detected",
        "ela_anomaly_score",
        "ela_candidate_bbox",
        "ela_iou",
        "copy_move_detected",
        "copy_move_inliers",
        "copy_move_confidence",
        "copy_move_bbox",
        "copy_move_iou",
        "fused_flagged",
        "verdict_correct",
    ]

    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            label = r.get("label", "genuine")
            gt_bbox = r.get("ground_truth_bbox")
            ela_detected = bool(r.get("ela_detected", False))
            ela_score = round(float(r.get("ela_anomaly_score") or 0.0), 2)
            ela_bbox = r.get("ela_candidate_bbox")
            ela_iou = round(float(evaluate_localization(ela_bbox, gt_bbox)), 4) if gt_bbox else 0.0

            cm_detected = bool(r.get("copy_move_detected", False))
            cm_inliers = int(r.get("copy_move_inliers") or 0)
            cm_conf = round(float(r.get("copy_move_confidence") or 0.0), 4)
            cm_bbox = r.get("copy_move_bbox")

            # Copy-move localization checks dual bbox
            alt_bbox = r.get("copy_move_alt_bbox")
            src_bbox = r.get("source_bbox")
            candidates = [c for c in [cm_bbox, alt_bbox] if c is not None]
            targets = [t for t in [gt_bbox, src_bbox] if t is not None]
            best_cm_iou = 0.0
            for c in candidates:
                for t in targets:
                    best_cm_iou = max(best_cm_iou, evaluate_localization(c, t))
            cm_iou = round(float(best_cm_iou), 4) if gt_bbox else 0.0

            fused_flagged = ela_detected or cm_detected
            verdict_correct = fused_flagged == (label == "tampered")

            writer.writerow({
                "source_id": r.get("source_id", ""),
                "attack_type": r.get("attack_type", "none"),
                "label": label,
                "ground_truth_bbox": str(gt_bbox) if gt_bbox else "",
                "ela_detected": ela_detected,
                "ela_anomaly_score": ela_score,
                "ela_candidate_bbox": str(ela_bbox) if ela_bbox else "",
                "ela_iou": ela_iou,
                "copy_move_detected": cm_detected,
                "copy_move_inliers": cm_inliers,
                "copy_move_confidence": cm_conf,
                "copy_move_bbox": str(cm_bbox) if cm_bbox else "",
                "copy_move_iou": cm_iou,
                "fused_flagged": fused_flagged,
                "verdict_correct": verdict_correct,
            })

    return out_path


def generate_markdown_audit_report(
    eval_results: dict,
    filename: str = "m1_forensic_audit_report.md",
) -> str:
    """
    Generate an executive forensic audit report in Markdown format.
    """
    reports_dir = get_reports_dir()
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, filename)

    ela_overall = eval_results.get("ela", {}).get("overall", {})
    ela_per_attack = eval_results.get("ela", {}).get("per_attack", {})
    cm_overall = eval_results.get("copy_move", {}).get("overall", {})
    cm_per_attack = eval_results.get("copy_move", {}).get("per_attack", {})
    n_samples = eval_results.get("n_samples", 0)

    lines = [
        "# ForgeLens-X — Milestone 1 Forensic Audit Report",
        "",
        f"**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Samples Evaluated:** {n_samples}",
        "",
        "## 1. Executive Summary",
        "",
        "Milestone 1 establishes the baseline forensic generation and detection framework:",
        "- **Synthetic Generation**: Procedural identity documents with Indian demographic distribution (`Faker en_IN`), guilloche anti-counterfeiting patterns, and official seal stamps.",
        "- **JPEG History Simulation**: Rigorous two-stage JPEG compression lifecycle (`save #1 -> reload -> tamper -> save #2`) to realistically simulate compression artifact differentials.",
        "- **Physical Tamper Types**: Date modification, alphanumeric text editing, photographic portrait splicing (with sensor micro-noise injection), and copy-move region duplication.",
        "- **Forensic Detectors**: Calibrated Error Level Analysis (ELA) with statistical baseline z-score filtering, and ORB + RANSAC homography copy-move detection with spatial distance constraints.",
        "",
        "## 2. ELA Detection & Localization Performance",
        "",
        f"- **Overall Detection Rate (TP Rate):** {ela_overall.get('detection_rate', 0.0) * 100:.1f}%",
        f"- **False Alarm Rate (FP Rate on Genuine):** {ela_overall.get('false_alarm_rate', 0.0) * 100:.1f}%",
        f"- **Mean Localization IoU (Tampered):** {ela_overall.get('mean_iou', 0.0) if ela_overall.get('mean_iou') is not None else 'N/A'}",
        "",
        "| Attack Type | Total | Detected (TP/FP) | Detection Rate | Mean IoU |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for attack, stats in ela_per_attack.items():
        det_rate = f"{stats.get('detection_rate', 0.0) * 100:.1f}%"
        mean_iou = f"{stats.get('mean_iou'):.4f}" if stats.get('mean_iou') is not None else "—"
        total = stats.get("total", 0)
        detected = stats.get("true_positives", 0) if attack != "none" else stats.get("false_positives", 0)
        lines.append(f"| `{attack}` | {total} | {detected} | {det_rate} | {mean_iou} |")

    lines.extend([
        "",
        "## 3. Copy-Move Detection & Localization Performance",
        "",
        f"- **Overall Detection Rate:** {cm_overall.get('detection_rate', 0.0) * 100:.1f}%",
        f"- **False Alarm Rate (Genuine):** {cm_overall.get('false_alarm_rate', 0.0) * 100:.1f}%",
        f"- **Mean Localization IoU (on Copy-Move):** {cm_per_attack.get('copy_move', {}).get('mean_iou', 'N/A')}",
        "",
        "| Attack Type | Total | Inliers Detected | Detection Rate | Mean IoU |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for attack, stats in cm_per_attack.items():
        det_rate = f"{stats.get('detection_rate', 0.0) * 100:.1f}%"
        mean_iou = f"{stats.get('mean_iou'):.4f}" if stats.get('mean_iou') is not None else "—"
        total = stats.get("total", 0)
        detected = stats.get("true_positives", 0) if attack != "none" else stats.get("false_positives", 0)
        lines.append(f"| `{attack}` | {total} | {detected} | {det_rate} | {mean_iou} |")

    lines.extend([
        "",
        "## 4. Key Engineering & Forensic Findings",
        "",
        "1. **Zero-Leakage Splitting**: Splits strictly partitioned by `source_id` guarantee that the document layout and font geometry from a given template cannot leak into calibration or test evaluations.",
        "2. **Noise Floor Calibration**: Setting a baseline std floor (`sigma_floor = 1.5`) prevents untextured regions (such as solid headers) from generating divide-by-zero division spikes on 1-pixel rounding variations.",
        "3. **Dual Bounding Box Matching for Copy-Move**: Copy-move operations involve both a source cloning region and a destination pasted region. Checking predictions against both source and destination candidates accurately reflects forensic detection success.",
        "4. **Need for Multimodal Signal Fusion (M6)**: While ELA excels on photo splicing and text alterations and Copy-Move excels on region cloning, individual forensic detectors have blind spots. Fusion with OCR/MRZ semantic consistency (M3/M4) and Face Verification (M2) in M6 provides complete coverage targeting >95% system-level screening accuracy.",
        "",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path
