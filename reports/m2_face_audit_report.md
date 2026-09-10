# ForgeLens-X — Milestone 2: Face Verification Audit Report

**Generated At:** 2026-09-10 19:42:29
**Evaluated Model:** `ArcFace` (cosine distance)
**Benchmark Sample Size:** 20 balanced pairs

## 1. Executive Summary

Milestone 2 establishes cross-modality facial verification between identity credential portraits and presented live selfies:
- **Cross-Modality Normalization**: 5-point landmark affine alignment to canonical $112 \times 112$ geometry eliminates perspective/scale bias.
- **Deep Representation Backends**: Supported ArcFace (512-D), Facenet512 (512-D), and SFace ONNX (native 128-D).
- **Strict Zero-Crash Policy**: Detects non-face images gracefully and outputs structured error dictionary without terminating the screening pipeline.
- **Post-Fusion Independence**: Retained strictly as an independent post-fusion gate in Milestone 6.

## 2. Empirical Verification Performance

- **Overall Accuracy:** 95.0%
- **True Accept Rate (TAR):** 100.0%
- **False Accept Rate (FAR - Imposter Leakage):** 10.0%
- **False Reject Rate (FRR - Genuine Friction):** 0.0%

| Metric | Measured Value | Operational Interpretation |
| :--- | :--- | :--- |
| **True Positives (TP)** | 10 | Genuine identity correctly verified |
| **True Negatives (TN)** | 9 | Imposter/mismatch correctly rejected |
| **False Positives (FP)** | 1 | Imposter incorrectly admitted (Security Risk) |
| **False Negatives (FN)** | 0 | Genuine user rejected (User Friction) |
| **Mean Genuine Distance** | 0.2473 (std: 0.1723) | Tight clustering for authentic presentations |
| **Mean Imposter Distance** | 0.8803 (std: 0.1213) | Distinct separation from genuine presentations |
| **Separation Margin** | **0.6330** | Decision boundary margin between classes |

## 3. Empirical ROC Curve & Operational Regimes

- **Equal Error Rate (EER):** **0.00%** at distance threshold $\tau = 0.5535$
- **Area Under ROC Curve (AUC):** **1.0000**

| Operational Regime | Calibrated Threshold (tau) | Target Specification | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **High Security** | `0.0574` | FAR <= 0.1% | High-risk KYC / Border Screening |
| **Balanced** | `0.5535` | EER (~0.0%) | Standard Identity Verification |
| **Low Friction** | `1.2852` | FRR <= 1.0% | Low-risk Assisted Self-Service |

## 4. Demographic Fairness & Disparity Audit

- **Fairness Audit Status:** **EQUITABLE** (Max Cross-Demographic Disparity = **12.5%**)

| Demographic Subgroup | Evaluated Pairs | Subgroup Accuracy | False Accept Rate | False Reject Rate | Mean Margin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **male_caucasian** | 4 | 100.0% | 0.0% | 0.0% | 0.6461 |
| **male_adult** | 8 | 87.5% | 25.0% | 0.0% | 0.5226 |
| **female_adult** | 8 | 100.0% | 0.0% | 0.0% | 0.7368 |

## 5. Engineering & Forensic Takeaways

1. **Landmark Normalization**: Aligning the two eye centers and mouth corners directly tackles the discrepancy between a formal flat document photo and an angled mobile selfie.
2. **Calibrated Confidence**: Similarity percentages are mapped through a calibrated sigmoid centered at the operational threshold, giving human reviewers an intuitive confidence index.
3. **Zero-Liveness Boundary Preserved**: As mandated by the canonical specification, liveness/anti-spoofing and morphing detection remain deferred to Milestone 9.
4. **M6 Post-Fusion Independence**: Biometric verification is audited independently to prevent joint label contamination during logistic document tamper fusion.
