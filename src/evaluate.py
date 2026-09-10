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

import json
import os
from collections import defaultdict

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
