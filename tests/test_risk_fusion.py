"""
Unit and Integration Tests for Milestone 6:
Calibrated Machine Learning Risk Fusion, Explainability, and Decision Policy
"""

import json
import os
import tempfile
import numpy as np
import pytest

from src.risk_fusion import (
    extract_learned_features,
    get_feature_names,
    train_document_risk_model,
    predict_document_risk,
    compute_log_odds_attribution,
    apply_decision_policy,
    load_fusion_model,
    clear_model_cache,
)
from src.fusion_evaluate import (
    run_fusion_evaluation,
    compute_expected_calibration_error,
)
from src.utils import get_splits_dir


class TestRiskFusionFeatureEngineering:
    """Test feature extraction and zero-face-signal isolation."""

    def test_face_signals_purged_from_learned_features(self):
        """Verify M2 face signals are never extracted into the ML feature vector."""
        raw_dict = {
            "ela_high_error_ratio": 0.12,
            "copy_move_matches": 15,
            # Injected face signals that must be strictly purged
            "face_distance": 0.42,
            "face_verified": True,
            "face_cosine_distance": 0.38,
            "face_crop_quality": 0.95,
        }
        feat_vec = extract_learned_features(raw_dict)
        feat_names = get_feature_names()

        assert len(feat_vec) == len(feat_names)
        for fn in feat_names:
            assert "face" not in fn.lower(), f"Face signal leaked into learned feature names: {fn}"

    def test_domain_interaction_terms_computed(self):
        """Verify the 3 forensic domain interaction terms compute correctly."""
        raw_dict = {
            "ela_high_error_ratio": 0.20,
            "copy_move_matches": 10.0,
            "semantic_failed_count": 2.0,
            "mrz_checksum_fail": 1.0,
            "font_inconsistent_flag": 1.0,
            "docnumber_format_flag": 1.0,
        }
        feat_vec = extract_learned_features(raw_dict)
        feat_names = get_feature_names()
        idx_ela_cm = feat_names.index("interaction_ela_copymove")
        idx_sem_mrz = feat_names.index("interaction_semantic_mrz")
        idx_font_doc = feat_names.index("interaction_font_docnumber")
        assert pytest.approx(feat_vec[idx_ela_cm], 1e-4) == 2.0
        assert pytest.approx(feat_vec[idx_sem_mrz], 1e-4) == 2.0
        assert pytest.approx(feat_vec[idx_font_doc], 1e-4) == 1.0

    def test_missing_features_default_to_zero(self):
        """Missing input keys should safely default to 0.0."""
        feat_vec = extract_learned_features({})
        for val in feat_vec:
            assert val == 0.0


class TestRiskModelTrainingAndInference:
    """Test model training, calibration selection, and probabilistic inference."""

    @pytest.fixture(autouse=True)
    def clean_cache(self):
        clear_model_cache()
        yield
        clear_model_cache()

    def test_zero_leakage_source_splits_disjoint(self):
        """Verify source_id zero-leakage across train, cal, and test splits."""
        splits_dir = get_splits_dir()
        train_path = splits_dir / "train.json"
        cal_path = splits_dir / "cal.json"
        test_path = splits_dir / "test.json"

        if not (train_path.exists() and cal_path.exists() and test_path.exists()):
            pytest.skip("Dataset split files not present in data/splits")

        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)["samples"]
        with open(cal_path, "r", encoding="utf-8") as f:
            cal_data = json.load(f)["samples"]
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)["samples"]

        train_sources = {item["source_id"] for item in train_data}
        cal_sources = {item["source_id"] for item in cal_data}
        test_sources = {item["source_id"] for item in test_data}

        assert len(train_sources & cal_sources) == 0, "Leakage detected: Source ID shared between train and cal!"
        assert len(train_sources & test_sources) == 0, "Leakage detected: Source ID shared between train and test!"
        assert len(cal_sources & test_sources) == 0, "Leakage detected: Source ID shared between cal and test!"

    def test_train_document_risk_model_end_to_end(self):
        """Train the model and verify output artifacts and calibration."""
        splits_dir = get_splits_dir()
        train_path = splits_dir / "train.json"
        cal_path = splits_dir / "cal.json"

        if not (train_path.exists() and cal_path.exists()):
            pytest.skip("Dataset split files not present")

        meta = train_document_risk_model(train_split_path=str(train_path), cal_split_path=str(cal_path))
        assert meta["selected_calibration_method"] in ["sigmoid", "isotonic", "sigmoid_train_fallback"]
        assert meta["train_auc"] >= 0.90
        assert meta["cal_brier_score"] <= 0.25
        assert os.path.exists(meta["model_path"])
        assert os.path.exists(meta["scaler_path"])

    def test_predict_document_risk_bounds_and_structure(self):
        """Verify output types, ranges, and explanations for genuine and tampered vectors."""
        # Low risk vector (genuine)
        low_feat = {fn: 0.0 for fn in get_feature_names()}
        low_res = predict_document_risk(low_feat)

        assert 0.0 <= low_res["fraud_probability"] <= 1.0
        assert 0.0 <= low_res["risk_score"] <= 100.0
        assert low_res["risk_tier"] in ["LOW", "MODERATE", "CRITICAL", "MEDIUM", "HIGH"]
        assert isinstance(low_res["top_risk_drivers"], list)

        # High risk vector (tampered)
        high_feat = {fn: 0.0 for fn in get_feature_names()}
        high_feat["ela_max"] = 220.0
        high_feat["ela_std"] = 55.0
        high_feat["ela_mean"] = 35.0
        high_feat["ela_high_error_ratio"] = 0.45
        high_feat["copy_move_num_matches"] = 25.0
        high_feat["semantic_failed_count"] = 3.0
        high_feat["font_inconsistency_flag"] = 1.0
        high_res = predict_document_risk(high_feat)

        assert high_res["fraud_probability"] > low_res["fraud_probability"]
        assert high_res["risk_score"] > low_res["risk_score"]
        assert len(high_res["top_risk_drivers"]) > 0

    def test_log_odds_decomposition_exact(self):
        """Verify linear log-odds decomposition matches logistic model raw output."""
        bundle = load_fusion_model()
        if bundle is None:
            pytest.skip("Trained fusion model not available")

        base_estimator = bundle["base_model"]
        scaler = bundle["scaler"]
        feature_names = bundle["feature_names"]

        raw_feat = {fn: np.random.uniform(0, 1) for fn in feature_names}
        vec = extract_learned_features(raw_feat)
        X = vec.reshape(1, -1)
        X_scaled = scaler.transform(X)[0]

        total_logit, contributions, drivers = compute_log_odds_attribution(
            X_scaled, base_estimator, feature_names
        )

        expected_logit = float(np.dot(base_estimator.coef_[0], X_scaled) + base_estimator.intercept_[0])
        assert pytest.approx(total_logit, 1e-3) == expected_logit

    def test_operational_prior_adjustment(self):
        """Verify Bayesian logit shift properly adjusts fraud probability."""
        feat = {fn: 0.1 for fn in get_feature_names()}
        # Base run
        res_default = predict_document_risk(feat)
        # Shift down to 5% fraud base rate
        res_low_prior = predict_document_risk(feat, operational_prior=0.05)
        # Shift up to 80% fraud base rate
        res_high_prior = predict_document_risk(feat, operational_prior=0.80)

        assert res_low_prior["fraud_probability"] < res_default["fraud_probability"]
        assert res_high_prior["fraud_probability"] > res_default["fraud_probability"]


class TestDecisionPolicy:
    """Test ordered severity hierarchy and decoupled M2 face verification policy gating."""

    def test_low_risk_authentic_document(self):
        decision, tier, basis = apply_decision_policy(
            document_decision="VERIFIED",
            fraud_probability=0.05,
            face_verification={"verified": True, "distance": 0.25},
            quality={"quality_gating": "PASS"},
        )
        assert decision == "VERIFIED"
        assert tier == "LOW"

    def test_intermediate_risk_triggers_manual_review(self):
        decision, tier, basis = apply_decision_policy(
            document_decision="VERIFIED",
            fraud_probability=0.35,  # between 0.20 and 0.65
            face_verification={"verified": True, "distance": 0.25},
            quality={"quality_gating": "PASS"},
        )
        assert decision == "MANUAL_REVIEW"
        assert tier in ["MODERATE", "MEDIUM"]

    def test_high_risk_triggers_high_risk_decision(self):
        decision, tier, basis = apply_decision_policy(
            document_decision="HIGH_RISK",
            fraud_probability=0.85,
            face_verification={"verified": True, "distance": 0.25},
            quality={"quality_gating": "PASS"},
        )
        assert decision == "HIGH_RISK"
        assert tier in ["CRITICAL", "HIGH"]

    def test_face_mismatch_acts_as_independent_floor(self):
        """Even with 0.00 document fraud probability, face mismatch cannot be VERIFIED."""
        decision, tier, basis = apply_decision_policy(
            document_decision="VERIFIED",
            fraud_probability=0.02,
            face_verification={"verified": False, "distance": 0.55},
            quality={"quality_gating": "PASS"},
        )
        assert decision in ["MANUAL_REVIEW", "HIGH_RISK"]
        assert any("face" in b.lower() for b in basis)

    def test_extreme_face_mismatch_elevates_to_high_risk(self):
        """Distance exceeding 0.70 should elevate to HIGH_RISK."""
        decision, tier, basis = apply_decision_policy(
            document_decision="VERIFIED",
            fraud_probability=0.05,
            face_verification={"verified": False, "distance": 0.85},
            quality={"quality_gating": "PASS"},
        )
        assert decision == "HIGH_RISK"

    def test_never_downgrades_higher_severity(self):
        """Document HIGH_RISK cannot be downgraded by a matching face."""
        decision, tier, basis = apply_decision_policy(
            document_decision="HIGH_RISK",
            fraud_probability=0.90,
            face_verification={"verified": True, "distance": 0.10},
            quality={"quality_gating": "PASS"},
        )
        assert decision == "HIGH_RISK"

    def test_poor_quality_elevates_to_manual_review(self):
        """Quality gating failure elevates clean doc to MANUAL_REVIEW."""
        decision, tier, basis = apply_decision_policy(
            document_decision="VERIFIED",
            fraud_probability=0.04,
            face_verification={"verified": True, "distance": 0.20},
            quality={"quality_gating": "FAIL", "issues": ["Severe motion blur"]},
        )
        assert decision == "MANUAL_REVIEW"
        assert any("quality" in b.lower() for b in basis)


class TestCalibrationMetricsAndEvaluation:
    """Test Brier score, ECE computation, and test split evaluation."""

    def test_calibration_metrics_calculation(self):
        from sklearn.metrics import brier_score_loss
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        brier = brier_score_loss(y_true, y_prob)
        ece, bin_details = compute_expected_calibration_error(y_true, y_prob, num_bins=5)
        assert brier < 0.05
        assert ece <= 0.20
        assert len(bin_details) == 5

    def test_run_fusion_evaluation_on_test_split(self):
        splits_dir = get_splits_dir()
        test_path = splits_dir / "test.json"
        if not test_path.exists():
            pytest.skip("test.json split not available")

        res = run_fusion_evaluation(test_split_path=str(test_path))
        m = res["metrics"]
        assert res["test_samples_evaluated"] == 10
        assert m["roc_auc"] >= 0.90
        assert m["brier_score"] <= 0.20
        assert m["false_rejection_rate_frr"] <= 0.05
        assert os.path.exists(res["report_path"])
        assert os.path.exists(res["artifacts"]["calibration_curve_plot"])

    def test_interaction_driver_descriptions(self):
        """Verify compound interaction terms return tailored explanations."""
        from src.risk_fusion import _format_driver_description
        d1 = _format_driver_description("interaction_ela_copymove", 1.0, 1.25)
        d2 = _format_driver_description("interaction_semantic_mrz", 1.0, 1.35)
        d3 = _format_driver_description("interaction_font_docnumber", 1.0, 0.95)
        assert "Dual physical tampering" in d1
        assert "Dual logical contradiction" in d2
        assert "Targeted credential forgery" in d3

    def test_extract_m6_feature_vector_includes_energy(self):
        """Verify extract_m6_feature_vector populates ela_candidate_energy."""
        from src.forensic_report import extract_m6_feature_vector
        mock_rep = {
            "tamper_signals": {
                "ela": {"features": {"candidate_energy": 84.5}},
            }
        }
        f_vec = extract_m6_feature_vector(mock_rep)
        assert f_vec.get("ela_candidate_energy") == 84.5

    def test_config_robustness_with_non_m6_config(self):
        """Verify predict_document_risk works when passed a generic config dictionary without base_features."""
        from src.risk_fusion import predict_document_risk
        generic_config = {"features": {"include_interactions": True}}
        res = predict_document_risk({"ela_mean": 10.0}, config=generic_config)
        assert "fraud_probability" in res
        assert "risk_score" in res
        assert 0.0 <= res["fraud_probability"] <= 1.0

    def test_threshold_sensitivity_generated_in_evaluation(self):
        """Verify sensitivity table is present and covers expected thresholds."""
        splits_dir = get_splits_dir()
        test_path = splits_dir / "test.json"
        if not test_path.exists():
            pytest.skip("test.json split not available")

        res = run_fusion_evaluation(test_split_path=str(test_path))
        st = res.get("threshold_sensitivity", [])
        assert len(st) == 9
        assert st[0]["threshold"] == 0.10
        assert st[-1]["threshold"] == 0.90
        assert st[4]["f1_score"] >= 0.90
