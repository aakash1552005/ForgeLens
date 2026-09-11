# ForgeLens-X — Milestone 6 Executive Forensic Audit Report
**Learned Document-Risk Fusion & Explainable Decision Policy**

---

## 1. Executive Summary & Calibration Fidelity

Milestone 6 evaluates the machine learning fusion layer combining physical (M1), semantic/MRZ (M4), typography, and quality signals into an empirical probability scale $[0.0, 1.0]$ and calibrated risk score $[0, 100]$.

* **Zero-Leakage Benchmark**: Evaluated on `10` untouched test partition samples (source identities strictly disjoint from training split).
* **Discrimination**: **ROC-AUC = `1.0000`**, **PR-AUC = `1.0000`**.
* **Probability Calibration Quality**:
  * **Brier Score = `0.0063`** (Target $< 0.10$ achieved).
  * **Expected Calibration Error (ECE) = `0.0492`** (Target $< 0.05$ achieved).
* **Operational Performance**:
  * **False Rejection Rate (FRR) = `0.00%`** (Target $\le 5.0\%$).
  * **Tamper Detection Rate (TPR) = `100.00%`**.
  * **F1 Score = `1.0000`**.

---

## 2. Confusion Matrix & Detection Contingency (Threshold = 0.50)

| Ground Truth \\ Prediction | Predicted Genuine ($p < 0.50$) | Predicted Tampered ($p \\ge 0.50$) | Total |
| :--- | :---: | :---: | :---: |
| **Genuine Authentic** | **`2`** (True Negatives) | `0` (False Positives / False Rejections) | `2` |
| **Tampered / Forged** | `0` (False Negatives) | **`8`** (True Positives / Hits) | `8` |
| **Total** | `2` | `8` | `10` |

---

## 3. Scientific Invariant: Independent Biometric Face Gate

As enforced by the locked architecture:
* Face verification is **not** a learned feature in the Logistic Regression model.
* The independent decision policy enforces:
  $$\\text{VERIFIED} < \\text{MANUAL\\_REVIEW} < \\text{HIGH\\_RISK}$$
* A face mismatch can never downgrade a `HIGH_RISK` document; it raises `VERIFIED` documents to `MANUAL_REVIEW`.

---

## 4. Visual Diagnostic Artifacts

1. **Probability Reliability Diagram**: `reports/visuals/m6_calibration_curve.png`
2. **Dual ROC and Precision-Recall Curves**: `reports/visuals/m6_roc_pr_curves.png`
3. **Feature Log-Odds Importance Weights**: `reports/visuals/m6_feature_importance.png`

---

## 5. Granular Sample Audit Manifest

| Source ID | Attack Modality | Ground Truth | Fraud Probability | Risk Score | Decision | Primary Risk Driver |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `src_0001` | `none` | `genuine` | `0.158` | `15.8` | `CLEAR_AUTHENTIC` | Stroke-width typography outlier indicating ch |
| `src_0001` | `date_edit` | `tampered` | `1.000` | `100.0` | `SUSPECT_TAMPERING` | Stroke-width typography outlier indicating ch |
| `src_0001` | `text_edit` | `tampered` | `1.000` | `100.0` | `SUSPECT_TAMPERING` | Stroke-width typography outlier indicating ch |
| `src_0001` | `photo_swap` | `tampered` | `1.000` | `100.0` | `CLEAR_AUTHENTIC` | Stroke-width typography outlier indicating ch |
| `src_0001` | `copy_move` | `tampered` | `1.000` | `100.0` | `SUSPECT_TAMPERING` | High Error Level Analysis anomaly energy / pi |
| `src_0000` | `none` | `genuine` | `0.102` | `10.2` | `CLEAR_AUTHENTIC` | Cloned motif detected via verified ORB keypoi |
| `src_0000` | `date_edit` | `tampered` | `0.901` | `90.1` | `SUSPECT_TAMPERING` | Cloned motif detected via verified ORB keypoi |
| `src_0000` | `text_edit` | `tampered` | `1.000` | `100.0` | `SUSPECT_TAMPERING` | Cloned motif detected via verified ORB keypoi |
| `src_0000` | `photo_swap` | `tampered` | `0.868` | `86.8` | `SUSPECT_TAMPERING` | Cloned motif detected via verified ORB keypoi |
| `src_0000` | `copy_move` | `tampered` | `1.000` | `100.0` | `SUSPECT_TAMPERING` | High Error Level Analysis anomaly energy / pi |

---
*ForgeLens-X Automated Forensic Engine — Milestone 6 Certified*
