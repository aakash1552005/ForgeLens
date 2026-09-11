"""
ForgeLens-X — Milestone 6: Machine Learning Risk Fusion & Decision Policy
==========================================================================
Trains a leakage-free, calibrated Logistic Regression model to fuse multi-modal
forensic (M1 ELA + Copy-Move), semantic (M4 rules + MRZ), typography, and quality
signals into an explainable, continuous fraud probability [0.0, 1.0] and risk score [0, 100].

Scientific Invariant:
    Face verification (M2) is strictly EXCLUDED as a learned feature in the ML model
    because the project lacks a joint distribution with true document-tamper and face-mismatch
    labels. M2 face verification is applied post-fusion as an independent evidence gate
    governed by an ordered severity hierarchy:
        VERIFIED < MANUAL_REVIEW < HIGH_RISK
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------

_M6_CONFIG_CACHE = None
_MODEL_BUNDLE_CACHE = None


def load_m6_config() -> Dict[str, Any]:
    """Load configuration from configs/m6_config.yaml."""
    global _M6_CONFIG_CACHE
    if _M6_CONFIG_CACHE is not None:
        return _M6_CONFIG_CACHE

    cfg_path = os.path.join(os.getcwd(), "configs", "m6_config.yaml")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                _M6_CONFIG_CACHE = yaml.safe_load(f) or {}
                return _M6_CONFIG_CACHE
        except Exception:
            pass

    # Default fallback config
    _M6_CONFIG_CACHE = {
        "model": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "random_state": 42},
        "calibration": {"enabled": True, "method": "auto", "num_bins": 10, "target_brier_score": 0.10},
        "decision_policy": {
            "risk_threshold_low": 0.30,
            "risk_threshold_high": 0.70,
            "severity_order": ["VERIFIED", "MANUAL_REVIEW", "HIGH_RISK"],
            "face_mismatch_floor": "MANUAL_REVIEW",
            "quality_gating_enabled": True,
        },
        "features": {
            "include_interactions": True,
            "base_features": [
                "ela_mean", "ela_std", "ela_max", "ela_p95", "ela_p99", "ela_high_error_ratio", "ela_candidate_energy",
                "copy_move_num_matches", "copy_move_confidence", "copy_move_detected",
                "semantic_failed_count", "semantic_date_order_flag", "semantic_impossible_date_flag",
                "semantic_age_sanity_flag", "semantic_doc_number_flag", "semantic_contradiction_flag", "semantic_country_code_flag",
                "mrz_checksum_pass", "mrz_checksum_fail", "mrz_has_data", "mrz_viz_contradiction_flag",
                "font_max_stroke_zscore", "font_inconsistency_flag", "metadata_is_tampered", "metadata_has_exif",
                "quality_blur_score", "quality_resolution_ok", "quality_ocr_confidence", "quality_is_low_reliability",
                "quality_mean_brightness", "quality_contrast_std", "quality_aspect_ratio", "quality_field_completeness"
            ],
            "interaction_features": [
                "interaction_ela_copymove", "interaction_semantic_mrz", "interaction_font_docnumber"
            ],
        },
        "paths": {
            "model_dir": "models/fusion",
            "model_file": "logistic_regression_m6.joblib",
            "scaler_file": "scaler_m6.joblib",
            "metadata_file": "m6_model_metadata.json",
        },
    }
    return _M6_CONFIG_CACHE


def get_feature_names(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Get ordered list of learned forensic features for M6 model.
    Guaranteed to exclude any face verification signals.
    """
    if config is None or "base_features" not in config.get("features", {}):
        config = load_m6_config()
    feat_cfg = config.get("features", {})
    base = list(feat_cfg.get("base_features", []))
    if not base:
        base = list(load_m6_config().get("features", {}).get("base_features", []))
    if feat_cfg.get("include_interactions", True):
        interactions = feat_cfg.get("interaction_features", [
            "interaction_ela_copymove", "interaction_semantic_mrz", "interaction_font_docnumber"
        ])
        base.extend(interactions)

    # Strictly purge any accidental face features to prevent data leakage
    forbidden = {"has_face_check", "face_distance", "face_similarity", "face_verified", "face_mismatch", "face_area_ratio"}
    return [f for f in base if f not in forbidden]


# ---------------------------------------------------------------------------
# Feature Extraction Layer
# ---------------------------------------------------------------------------

def extract_learned_features(
    feature_vector: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Extract a numeric numpy feature vector from an M5 feature vector dict.
    Computes domain-specific interaction features and enforces face isolation.

    Args:
        feature_vector: dictionary of extracted features from Milestone 5 report.
        config: optional M6 configuration dict.
        feature_names: optional pre-specified list of feature names.

    Returns:
        1D float64 numpy array of ordered feature values.
    """
    if feature_names is not None:
        feat_names = feature_names
    else:
        if config is None or "base_features" not in config.get("features", {}):
            config = load_m6_config()
        feat_names = get_feature_names(config)

    values = []

    # Read base features with zero default and alias support
    for name in feat_names:
        if name == "interaction_ela_copymove":
            # Physical compound: ELA high error pixel ratio * Copy-Move keypoint matches
            v1 = float(feature_vector.get("ela_high_error_ratio") or 0.0)
            v2 = float(feature_vector.get("copy_move_num_matches") or feature_vector.get("copy_move_matches") or 0.0)
            val = round(v1 * v2, 5)
        elif name == "interaction_semantic_mrz":
            # Logical compound: Failed semantic checks * MRZ checksum failure flag
            v1 = float(feature_vector.get("semantic_failed_count") or 0.0)
            v2 = float(feature_vector.get("mrz_checksum_fail") or 0.0)
            val = round(v1 * v2, 3)
        elif name == "interaction_font_docnumber":
            # Targeted text forgery compound: Font inconsistency * Document number format failure
            v1 = float(feature_vector.get("font_inconsistency_flag") or feature_vector.get("font_inconsistent_flag") or 0.0)
            v2 = float(feature_vector.get("semantic_doc_number_flag") or feature_vector.get("docnumber_format_flag") or 0.0)
            val = round(v1 * v2, 3)
        else:
            raw_val = feature_vector.get(name)
            if raw_val is None:
                # Check known aliases across pipeline milestones
                if name == "copy_move_num_matches":
                    raw_val = feature_vector.get("copy_move_matches", 0.0)
                elif name == "font_inconsistency_flag":
                    raw_val = feature_vector.get("font_inconsistent_flag", 0.0)
                elif name == "semantic_doc_number_flag":
                    raw_val = feature_vector.get("docnumber_format_flag", 0.0)
                elif name == "ela_candidate_energy":
                    raw_val = feature_vector.get("candidate_energy", 0.0)
                else:
                    raw_val = 0.0

            if raw_val is None or (isinstance(raw_val, float) and (np.isnan(raw_val) or np.isinf(raw_val))):
                val = 0.0
            else:
                val = float(raw_val)
        values.append(val)

    return np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Training & Calibration Pipeline
# ---------------------------------------------------------------------------

def train_document_risk_model(
    train_split_path: str,
    cal_split_path: str,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train a leakage-free calibrated Logistic Regression model.

    Procedure:
    1. Load source-grouped train.json (~70%) and cal.json (~15%).
    2. Extract multi-modal forensic feature vectors for all samples.
    3. Fit StandardScaler on train split with outlier clipping.
    4. Fit LogisticRegression(penalty='l2', C=1.0) on train split.
    5. Evaluate probability calibrator (Sigmoid vs Isotonic) on cal.json.
    6. Select the calibrator achieving the lowest Brier score.
    7. Save trained artifacts to output_dir.

    Returns:
        Dictionary of training metadata, calibration metrics, and coefficients.
    """
    if config is None:
        config = load_m6_config()

    from src.forensic_report import generate_unified_forensic_report

    m_cfg = config.get("model", {})
    c_cfg = config.get("calibration", {})
    p_cfg = config.get("paths", {})
    feature_names = get_feature_names(config)

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), p_cfg.get("model_dir", "models/fusion"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[M6] Training learned document-risk model with {len(feature_names)} forensic features...")
    print(f"     Train split: {train_split_path}")
    print(f"     Cal split:   {cal_split_path}")

    # 1. Load splits
    with open(train_split_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)["samples"]
    with open(cal_split_path, "r", encoding="utf-8") as f:
        cal_data = json.load(f)["samples"]

    # 2. Extract features and labels
    def _extract_dataset_matrix(samples: List[Dict[str, Any]], desc: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X_list = []
        y_list = []
        src_ids = []
        print(f"[M6] Extracting forensic features from {len(samples)} {desc} samples...")
        for i, s in enumerate(samples):
            img_path = s["image_path"]
            label = 1 if s.get("label") == "tampered" else 0
            # Generate unified report to obtain standardized feature vector
            report = generate_unified_forensic_report(img_path, reference_face_path=None)
            f_vec = report.get("feature_vector", {})
            feat_arr = extract_learned_features(f_vec, config=config)
            X_list.append(feat_arr)
            y_list.append(label)
            src_ids.append(s.get("source_id", f"src_{i}"))
        return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.int32), src_ids

    X_train, y_train, train_sources = _extract_dataset_matrix(train_data, "train")
    X_cal, y_cal, cal_sources = _extract_dataset_matrix(cal_data, "cal")

    # Assert zero-leakage by source identity
    overlap = set(train_sources).intersection(set(cal_sources))
    if overlap:
        raise ValueError(f"Data leakage detected! Shared source identities between train and cal: {overlap}")

    # 3. Fit StandardScaler on train split with clipping
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clip_val = float(m_cfg.get("clip_scaled_features", 5.0))
    X_train_scaled = np.clip(X_train_scaled, -clip_val, clip_val)

    X_cal_scaled = scaler.transform(X_cal)
    X_cal_scaled = np.clip(X_cal_scaled, -clip_val, clip_val)

    # 4. Train base Logistic Regression model
    base_lr = LogisticRegression(
        C=float(m_cfg.get("C", 1.0)),
        class_weight=m_cfg.get("class_weight", "balanced"),
        solver=str(m_cfg.get("solver", "lbfgs")),
        max_iter=int(m_cfg.get("max_iter", 1000)),
        random_state=int(m_cfg.get("random_state", 42)),
    )
    base_lr.fit(X_train_scaled, y_train)

    train_probs_base = base_lr.predict_proba(X_train_scaled)[:, 1]
    train_auc = float(roc_auc_score(y_train, train_probs_base)) if len(np.unique(y_train)) > 1 else 1.0

    # 5. Probability Calibration Selection on cal.json
    selected_calibrator = None
    best_method = "sigmoid"
    best_brier = 1.0

    cal_methods = ["sigmoid", "isotonic"] if c_cfg.get("method", "auto") == "auto" else [c_cfg.get("method", "sigmoid")]
    cv_cal = [(list(range(len(X_cal))), list(range(len(X_cal))))]

    calibrator_evals = {}
    for method in cal_methods:
        try:
            try:
                from sklearn.frozen import FrozenEstimator
                est = FrozenEstimator(base_lr)
                calibrator = CalibratedClassifierCV(estimator=est, method=method, cv=cv_cal)
            except (ImportError, TypeError):
                calibrator = CalibratedClassifierCV(estimator=base_lr, method=method, cv="prefit")

            calibrator.fit(X_cal_scaled, y_cal)
            cal_probs = calibrator.predict_proba(X_cal_scaled)[:, 1]
            brier = float(brier_score_loss(y_cal, cal_probs))
            auc = float(roc_auc_score(y_cal, cal_probs)) if len(np.unique(y_cal)) > 1 else 1.0
            calibrator_evals[method] = {"brier_score": round(brier, 4), "roc_auc": round(auc, 4)}

            if brier < best_brier:
                best_brier = brier
                best_method = method
                selected_calibrator = calibrator
        except Exception as e:
            calibrator_evals[method] = {"error": str(e)}

    if selected_calibrator is None:
        # Fallback to sigmoid on train if cal fit fails
        try:
            from sklearn.frozen import FrozenEstimator
            est = FrozenEstimator(base_lr)
            cv_train = [(list(range(len(X_train))), list(range(len(X_train))))]
            selected_calibrator = CalibratedClassifierCV(estimator=est, method="sigmoid", cv=cv_train)
        except (ImportError, TypeError):
            selected_calibrator = CalibratedClassifierCV(estimator=base_lr, method="sigmoid", cv="prefit")
        selected_calibrator.fit(X_train_scaled, y_train)
        best_method = "sigmoid_train_fallback"
        best_brier = float(brier_score_loss(y_train, selected_calibrator.predict_proba(X_train_scaled)[:, 1]))

    # 6. Extract Feature Importance & Log-Odds Weights
    coefs = base_lr.coef_[0]
    intercept = float(base_lr.intercept_[0])
    feature_importance = [
        {"feature": name, "weight": round(float(w), 4), "abs_weight": round(abs(float(w)), 4)}
        for name, w in zip(feature_names, coefs)
    ]
    feature_importance.sort(key=lambda x: x["abs_weight"], reverse=True)

    # 7. Save Model Artifacts
    model_path = os.path.join(output_dir, p_cfg.get("model_file", "logistic_regression_m6.joblib"))
    scaler_path = os.path.join(output_dir, p_cfg.get("scaler_file", "scaler_m6.joblib"))
    meta_path = os.path.join(output_dir, p_cfg.get("metadata_file", "m6_model_metadata.json"))

    model_bundle = {
        "calibrator": selected_calibrator,
        "base_model": base_lr,
        "scaler": scaler,
        "feature_names": feature_names,
        "clip_val": clip_val,
        "calibration_method": best_method,
    }
    joblib.dump(model_bundle, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_version": "1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "train_samples": len(X_train),
        "cal_samples": len(X_cal),
        "train_auc": round(train_auc, 4),
        "calibration_evaluations": calibrator_evals,
        "selected_calibration_method": best_method,
        "cal_brier_score": round(best_brier, 4),
        "intercept": round(intercept, 4),
        "top_features": feature_importance[:8],
        "model_path": model_path,
        "scaler_path": scaler_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Invalidate memory cache so newly trained model loads
    global _MODEL_BUNDLE_CACHE
    _MODEL_BUNDLE_CACHE = model_bundle

    print(f"[+] M6 Model Training Complete:")
    print(f"    - Model Path:          {model_path}")
    print(f"    - Calibrator Method:   {best_method}")
    print(f"    - Train ROC-AUC:       {train_auc:.4f}")
    print(f"    - Calibration Brier:   {best_brier:.4f}")
    print(f"    - Metadata Index:      {meta_path}")

    return metadata


# ---------------------------------------------------------------------------
# Model Loading & Cache
# ---------------------------------------------------------------------------

def load_fusion_model(model_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load trained M6 model bundle (calibrator, scaler, feature names).
    Returns None if model has not yet been trained.
    """
    global _MODEL_BUNDLE_CACHE
    if _MODEL_BUNDLE_CACHE is not None:
        return _MODEL_BUNDLE_CACHE

    config = load_m6_config()
    p_cfg = config.get("paths", {})
    if model_dir is None:
        model_dir = os.path.join(os.getcwd(), p_cfg.get("model_dir", "models/fusion"))

    model_path = os.path.join(model_dir, p_cfg.get("model_file", "logistic_regression_m6.joblib"))
    if not os.path.exists(model_path):
        return None

    try:
        _MODEL_BUNDLE_CACHE = joblib.load(model_path)
        return _MODEL_BUNDLE_CACHE
    except Exception as e:
        print(f"[WARN] Failed to load M6 fusion model from {model_path}: {e}")
        return None


def clear_model_cache():
    """Clear memory cached model bundle."""
    global _MODEL_BUNDLE_CACHE
    _MODEL_BUNDLE_CACHE = None


# ---------------------------------------------------------------------------
# Log-Odds Attribution & Explainability
# ---------------------------------------------------------------------------

def compute_log_odds_attribution(
    features_scaled: np.ndarray,
    base_model: Any,
    feature_names: List[str],
    top_k: int = 3,
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    """
    Decompose raw prediction log-odds into exact linear feature contributions:
        logit(p) = beta_0 + sum(beta_i * x_i)

    Returns:
        (total_log_odds, feature_contributions_dict, top_risk_drivers_list)
    """
    intercept = float(base_model.intercept_[0])
    coefs = base_model.coef_[0]

    contributions = {}
    drivers = []
    total_log_odds = intercept

    for name, x_val, beta in zip(feature_names, features_scaled, coefs):
        contrib = float(x_val * beta)
        contributions[name] = round(contrib, 3)
        total_log_odds += contrib
        if contrib > 0.02:  # Positive evidence contributing to manipulation probability
            # Generate human-readable explanation for compliance review
            desc = _format_driver_description(name, x_val, contrib)
            drivers.append({
                "feature": name,
                "contribution_log_odds": round(contrib, 3),
                "description": desc,
            })

    # Sort drivers descending by fraud risk contribution
    drivers.sort(key=lambda d: d["contribution_log_odds"], reverse=True)

    return round(total_log_odds, 3), contributions, drivers[:top_k]


def _format_driver_description(feature: str, val: float, contrib: float) -> str:
    """Human-forensic translator for logistic regression risk drivers."""
    # 1. Compound cross-modal interaction terms
    if feature == "interaction_ela_copymove":
        return f"Dual physical tampering: Coincident high ELA compression residue and cloned ORB keypoint clusters (+{contrib:.2f} log-odds)"
    elif feature == "interaction_semantic_mrz":
        return f"Dual logical contradiction: Identity chronology violation corroborated by ICAO MRZ checksum mismatch (+{contrib:.2f} log-odds)"
    elif feature == "interaction_font_docnumber":
        return f"Targeted credential forgery: Document number format discrepancy coupled with typographic stroke anomaly (+{contrib:.2f} log-odds)"

    # 2. Cryptographic and Machine-Readable Zone (MRZ)
    elif feature == "mrz_checksum_fail":
        return f"Cryptographic ICAO Doc 9303 checksum failure on machine-readable zone (+{contrib:.2f} log-odds)"
    elif feature == "mrz_viz_contradiction_flag":
        return f"Discrepancy detected between machine-readable zone (MRZ) and visual zone (VIZ) (+{contrib:.2f} log-odds)"
    elif "mrz" in feature:
        return f"Cryptographic ICAO Doc 9303 check digit anomaly (+{contrib:.2f} log-odds)"

    # 3. Typography & Font Forensics
    elif feature == "font_max_stroke_zscore":
        return f"Abnormal typographic stroke-width variance exceeding character baseline (+{contrib:.2f} log-odds)"
    elif feature == "font_inconsistency_flag":
        return f"Typographic font variation detected across field character groups (+{contrib:.2f} log-odds)"
    elif "font" in feature:
        return f"Stroke-width typography outlier indicating character insertion (+{contrib:.2f} log-odds)"

    # 4. Copy-Move Forensics
    elif feature == "copy_move_detected" or feature == "copy_move_num_matches":
        return f"Cloned motif detected via verified ORB keypoint correspondences (+{contrib:.2f} log-odds)"
    elif "copy_move" in feature:
        return f"Duplicated image patch detected with matched feature points (+{contrib:.2f} log-odds)"

    # 5. Error Level Analysis (ELA)
    elif feature == "ela_high_error_ratio":
        return f"High fraction of abnormal pixel compression error in document canvas (+{contrib:.2f} log-odds)"
    elif feature == "ela_candidate_energy":
        return f"Localized compression artifact anomaly energy in high-error candidate bbox (+{contrib:.2f} log-odds)"
    elif "ela" in feature:
        return f"Error Level Analysis compression residue mismatch (+{contrib:.2f} log-odds)"

    # 6. Metadata & Provenance
    elif feature == "metadata_is_tampered":
        return f"Digital forensics EXIF audit confirms editing tool provenance or stripped metadata (+{contrib:.2f} log-odds)"
    elif "metadata" in feature:
        return f"Image metadata indicates editing software alteration (+{contrib:.2f} log-odds)"

    # 7. Semantic Identity Rules
    elif feature == "semantic_date_order_flag":
        return f"Impossible temporal sequence between birth, issue, and expiry dates (+{contrib:.2f} log-odds)"
    elif feature == "semantic_impossible_date_flag":
        return f"Calendar validation failure (e.g. invalid leap day or month > 12) (+{contrib:.2f} log-odds)"
    elif feature == "semantic_contradiction_flag":
        return f"Contradictory demographic or territorial indicators across credential fields (+{contrib:.2f} log-odds)"
    elif feature == "semantic_doc_number_flag":
        return f"Document number syntax/format violation for credential schema (+{contrib:.2f} log-odds)"
    elif "semantic" in feature:
        return f"Logical identity chronology or document rule violation (+{contrib:.2f} log-odds)"

    return f"Forensic risk indicator '{feature}' elevated (+{contrib:.2f} log-odds)"


# ---------------------------------------------------------------------------
# Risk Inference Layer
# ---------------------------------------------------------------------------

def predict_document_risk(
    feature_vector: Dict[str, Any],
    model_bundle: Optional[Dict[str, Any]] = None,
    operational_prior: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Predict calibrated document manipulation probability and risk score.

    Args:
        feature_vector: extracted Milestone 5 feature dictionary.
        model_bundle: preloaded model bundle or None to load from disk.
        operational_prior: optional real-world base-rate fraud probability (e.g. 0.05).
        config: optional M6 config dict.

    Returns:
        Structured risk output dict with probability, score, log-odds, and drivers.
    """
    if config is None:
        config = load_m6_config()
    if model_bundle is None:
        model_bundle = load_fusion_model()

    # Fallback to heuristic risk score if model is not yet trained
    if model_bundle is None:
        return _heuristic_risk_fallback(feature_vector, config)

    calibrator = model_bundle["calibrator"]
    base_model = model_bundle["base_model"]
    scaler = model_bundle["scaler"]
    feature_names = model_bundle["feature_names"]
    clip_val = model_bundle.get("clip_val", 5.0)

    # 1. Extract feature array & scale
    feat_arr = extract_learned_features(feature_vector, config=config, feature_names=feature_names).reshape(1, -1)
    feat_scaled = scaler.transform(feat_arr)[0]
    feat_scaled = np.clip(feat_scaled, -clip_val, clip_val)

    # 2. Predict calibrated probability
    probs = calibrator.predict_proba(feat_scaled.reshape(1, -1))[0]
    raw_prob = float(probs[1])

    # 3. Optional Bayesian Base-Rate Prior Adjustment (Logit Shift)
    if operational_prior is not None and 0.0001 <= operational_prior <= 0.9999:
        train_prior = 0.50  # Balanced training distribution
        logit_raw = math.log(max(1e-6, min(1.0 - 1e-6, raw_prob))) - math.log(1.0 - max(1e-6, min(1.0 - 1e-6, raw_prob)))
        logit_shift = math.log(operational_prior / (1.0 - operational_prior)) - math.log(train_prior / (1.0 - train_prior))
        adj_logit = logit_raw + logit_shift
        calibrated_prob = 1.0 / (1.0 + math.exp(-adj_logit))
    else:
        calibrated_prob = raw_prob

    calibrated_prob = round(float(np.clip(calibrated_prob, 0.0, 1.0)), 4)
    risk_score = round(calibrated_prob * 100.0, 1)
    risk_tier = "LOW" if calibrated_prob < 0.30 else ("MODERATE" if calibrated_prob < 0.70 else "CRITICAL")

    # 4. Compute explainable log-odds attribution & top-3 drivers
    total_log_odds, contributions, top_drivers = compute_log_odds_attribution(
        feat_scaled, base_model, feature_names, top_k=3
    )

    # For authentic low-risk documents (fraud_prob < 0.30), suppress spurious baseline noise drivers
    if calibrated_prob < 0.30:
        top_drivers = []

    return {
        "fraud_probability": calibrated_prob,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "log_odds_total": total_log_odds,
        "feature_contributions": contributions,
        "top_risk_drivers": top_drivers,
        "calibration_method": model_bundle.get("calibration_method", "sigmoid"),
        "model_status": "CALIBRATED_ML_MODEL",
    }


def _heuristic_risk_fallback(feature_vector: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Graceful zero-crash heuristic estimator when M6 ML model is not yet trained."""
    ela_en = float(feature_vector.get("ela_candidate_energy", 0.0) or 0.0)
    cm_matches = float(feature_vector.get("copy_move_num_matches", 0.0) or 0.0)
    sem_fail = float(feature_vector.get("semantic_failed_count", 0.0) or 0.0)
    mrz_fail = float(feature_vector.get("mrz_checksum_fail", 0.0) or 0.0)

    score = 5.0
    drivers = []
    if ela_en >= 50.0:
        score += min(35.0, (ela_en / 150.0) * 35.0)
        drivers.append({"feature": "ela_candidate_energy", "contribution_log_odds": 1.5, "description": "Elevated ELA compression anomaly"})
    if cm_matches >= 3.0:
        score += min(30.0, cm_matches * 3.0)
        drivers.append({"feature": "copy_move_num_matches", "contribution_log_odds": 1.8, "description": "Keypoint correspondence cluster"})
    if sem_fail >= 1.0:
        score += min(20.0, sem_fail * 10.0)
        drivers.append({"feature": "semantic_failed_count", "contribution_log_odds": 1.2, "description": "Semantic chronology contradiction"})
    if mrz_fail >= 1.0:
        score += 15.0
        drivers.append({"feature": "mrz_checksum_fail", "contribution_log_odds": 1.4, "description": "ICAO MRZ checksum mismatch"})

    score = round(min(100.0, max(0.0, score)), 1)
    prob = round(score / 100.0, 4)
    risk_tier = "LOW" if prob < 0.30 else ("MODERATE" if prob < 0.70 else "CRITICAL")

    return {
        "fraud_probability": prob,
        "risk_score": score,
        "risk_tier": risk_tier,
        "log_odds_total": 0.0,
        "feature_contributions": {},
        "top_risk_drivers": drivers[:3],
        "calibration_method": "heuristic_fallback",
        "model_status": "HEURISTIC_PRIOR_MODEL",
    }


# ---------------------------------------------------------------------------
# Ordered Decision Policy Layer
# ---------------------------------------------------------------------------

def apply_decision_policy(
    document_decision: str,
    fraud_probability: float,
    face_verification: Optional[Dict[str, Any]] = None,
    quality: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, List[str]]:
    """
    Apply canonical post-fusion decision policy enforcing the ordered severity hierarchy:
        VERIFIED < MANUAL_REVIEW < HIGH_RISK

    Rules:
    1. Low scan fidelity without decisive tamper signals -> MANUAL_REVIEW.
    2. Calibrated document risk establishes baseline operational disposition:
       - fraud_probability < threshold_low (0.30)  -> VERIFIED
       - threshold_low <= fraud_prob < threshold_high (0.70) -> MANUAL_REVIEW
       - fraud_probability >= threshold_high (0.70) -> HIGH_RISK
    3. Independent Face Verification Gate (M2):
       - If face is supplied and does not verify (mismatch):
         decision = max_floor(document_decision, "MANUAL_REVIEW")
         (Raises a VERIFIED document to MANUAL_REVIEW; HIGH_RISK document stays HIGH_RISK;
          extreme face distance >= 0.70 elevates to HIGH_RISK).
       - If face verified: supports genuine disposition.
       - If face missing/not supplied: retains document-risk decision.

    Returns:
        (final_decision, risk_tier, decision_basis_list)
    """
    if face_verification is None:
        face_verification = {}
    if quality is None:
        quality = {}
    if config is None:
        config = load_m6_config()
    pol_cfg = config.get("decision_policy", {})
    th_low = float(pol_cfg.get("risk_threshold_low", 0.30))
    th_high = float(pol_cfg.get("risk_threshold_high", 0.70))

    basis: List[str] = []

    # 1. Quality Gating Check
    if pol_cfg.get("quality_gating_enabled", True):
        if quality.get("analysis_reliability") == "LOW" or quality.get("quality_gating") == "FAIL":
            if fraud_probability < th_high:
                q_flags = quality.get("quality_flags") or quality.get("issues") or ["DEGRADED_SCAN"]
                basis.append(
                    f"Image scan quality is degraded ({', '.join(q_flags)}). "
                    "Evidence is insufficient to reliably confirm intentional tampering."
                )
                return "MANUAL_REVIEW", "UNVERIFIED", basis

    # 2. Evaluate Baseline Document-Risk Disposition
    if fraud_probability < th_low:
        doc_risk_decision = "VERIFIED"
        risk_tier = "LOW"
        basis.append(f"Calibrated document manipulation risk is low ({fraud_probability * 100:.1f}% < {th_low * 100:.0f}%).")
    elif fraud_probability < th_high:
        doc_risk_decision = "MANUAL_REVIEW"
        risk_tier = "MODERATE"
        basis.append(f"Borderline document risk ({fraud_probability * 100:.1f}%) warrants secondary examiner inspection.")
    else:
        doc_risk_decision = "HIGH_RISK"
        risk_tier = "CRITICAL"
        basis.append(f"High calibrated manipulation probability ({fraud_probability * 100:.1f}% >= {th_high * 100:.0f}%).")

    # 3. Independent Biometric Face Gate (M2)
    has_face = bool(face_verification.get("has_face_check")) or (face_verification.get("verified") is not None)
    face_verified = face_verification.get("verified")

    final_decision = doc_risk_decision

    if has_face:
        if face_verified is False:
            raw_dist = face_verification.get("distance")
            face_dist = float(raw_dist) if raw_dist is not None else None
            dist_desc = f"distance={face_dist:.2f}" if face_dist is not None else "face unverified/undetected"

            if face_dist is not None and face_dist >= 0.70:
                final_decision = "HIGH_RISK"
                basis.append(
                    f"Severe biometric face mismatch ({dist_desc} >= 0.70) "
                    "elevates disposition to HIGH_RISK."
                )
            elif doc_risk_decision == "VERIFIED":
                final_decision = "MANUAL_REVIEW"
                basis.append(
                    f"Biometric face mismatch ({dist_desc}) "
                    "overrides low document risk, elevating disposition to MANUAL_REVIEW."
                )
            else:
                basis.append(
                    f"Biometric face mismatch ({dist_desc}) "
                    f"confirms identity dispute alongside {doc_risk_decision} document status."
                )
        elif face_verified is True:
            basis.append(
                f"Biometric face match verified (distance={face_verification.get('distance')}, "
                f"similarity={face_verification.get('similarity_pct', 0.0)}%)."
            )
    else:
        basis.append("No live reference face supplied; decision reflects independent document risk.")

    return final_decision, risk_tier, basis
