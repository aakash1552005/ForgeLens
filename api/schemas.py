"""
ForgeLens-X — Milestone 8: Pydantic v2 API Schemas & Data Contracts
====================================================================
Enforces strict schema validation for synchronous REST requests,
streaming responses, batch screening ledgers, and health probes.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    status: str = Field(default="HEALTHY", description="Overall microservice status.")
    version: str = Field(default="1.1.0", description="ForgeLens-X API version.")
    engine: str = Field(default="CONCURRENT_ASYNC_M8", description="Engine architecture.")
    models_loaded: List[str] = Field(
        default=["RapidOCR", "ArcFace_SFace", "ELA_MultiQ", "FFT_Spectral", "M6_Risk_Fusion"],
        description="Active deep learning models in memory.",
    )
    timestamp_utc: str = Field(description="ISO-8601 UTC timestamp of probe check.")


class SuspiciousRegionModel(BaseModel):
    field: str = Field(description="Tampered document field or security zone.")
    bbox: List[int] = Field(description="Bounding box [x1, y1, x2, y2].")
    source: str = Field(description="Forensic detector that triggered flag.")
    confidence: float = Field(description="Confidence probability [0.0 - 1.0].")
    evidence: str = Field(description="Detailed physical or logical anomaly description.")


class RiskDriverModel(BaseModel):
    feature: str = Field(description="Feature vector metric contributing to risk.")
    contribution_log_odds: float = Field(description="SHAP-style log-odds contribution.")
    description: str = Field(description="Plain-language examiner explanation.")


class FaceVerifyResponse(BaseModel):
    verified: bool = Field(description="True if live selfie matches document portrait.")
    distance: Optional[float] = Field(default=None, description="ArcFace cosine distance.")
    similarity_pct: Optional[float] = Field(default=None, description="Cosine similarity percentage.")
    status: str = Field(default="COMPLETED", description="Biometric execution status.")


class ScreeningResponse(BaseModel):
    schema_version: str = Field(default="1.1", description="ForgeLens-X schema version.")
    document_id: str = Field(description="Unique credential or transaction identifier.")
    decision: str = Field(description="Operational verdict: VERIFIED | MANUAL_REVIEW | HIGH_RISK.")
    risk_score: float = Field(description="Calibrated risk index [0.0 - 100.0].")
    fraud_probability: float = Field(description="Calibrated fraud probability [0.0 - 1.0].")
    risk_tier: str = Field(description="Risk band: LOW | ELEVATED | HIGH | CRITICAL.")
    attack_type_guess: str = Field(description="Diagnosed attack hypothesis: none | date_edit | text_edit | photo_swap | copy_move.")
    suspicious_regions: List[SuspiciousRegionModel] = Field(default=[], description="Pinpointed manipulation bounding boxes.")
    risk_drivers: List[RiskDriverModel] = Field(default=[], description="Top calibrated risk factors.")
    pipeline_latency_ms: float = Field(description="Total end-to-end screening latency in milliseconds.")
    fields: Optional[Dict[str, Any]] = Field(default=None, description="Extracted OCR text fields.")
    liveness: Optional[Dict[str, Any]] = Field(default=None, description="Presentation attack liveness detection audit (if selfie provided).")
    morphing: Optional[Dict[str, Any]] = Field(default=None, description="Facial morphing forensics detection audit.")


class BatchItemSummary(BaseModel):
    filename: str = Field(description="Source image filename.")
    document_id: str = Field(description="Assigned document identifier.")
    decision: str = Field(description="Operational verdict.")
    risk_score: float = Field(description="Calibrated risk score.")
    fraud_probability: float = Field(description="Calibrated fraud probability.")
    attack_type: str = Field(description="Primary attack hypothesis.")
    suspicious_count: int = Field(description="Number of flagged regions.")
    latency_ms: float = Field(description="Screening latency in milliseconds.")
    status: str = Field(description="PASS or FAIL.")


class BatchScreeningResponse(BaseModel):
    total_screened: int = Field(description="Total credentials screened in batch.")
    total_verified: int = Field(description="Count of authentic credentials verified.")
    total_flagged: int = Field(description="Count of fraudulent credentials intercepted.")
    mean_risk_score: float = Field(description="Average risk score across batch.")
    mean_latency_ms: float = Field(description="Average latency per credential in milliseconds.")
    results: List[BatchItemSummary] = Field(description="Per-credential forensic audit summaries.")


# --- Milestone 9: Presentation Attack Detection & Morphing Schemas ---

class LivenessResponse(BaseModel):
    is_live: bool = Field(description="True if presenting subject is confirmed live human.")
    liveness_score: float = Field(description="Calibrated liveness confidence [0.0 - 1.0].")
    spoof_risk: float = Field(description="Presentation attack risk [0.0 - 1.0].")
    spoof_tier: str = Field(description="GENUINE_LIVE_SUBJECT | SUSPECT_PRINT | SUSPECT_SCREEN_REPLAY | SYNTHETIC_DEEPFAKE.")
    spoof_type_guess: str = Field(description="none | print_attack | screen_replay | synthetic_deepfake.")
    signals: Dict[str, Any] = Field(description="Diagnostic signals (Fourier moire, LBP texture, color gamut, corneal symmetry).")
    face_bbox: Optional[List[int]] = Field(default=None, description="Detected face bounding box [x, y, w, h].")


class MorphingResponse(BaseModel):
    morphing_detected: bool = Field(description="True if document photo shows composite morphing artifacts.")
    morphing_score: float = Field(description="Calibrated morphing risk index [0.0 - 100.0].")
    morph_tier: str = Field(description="GENUINE_SINGLE_IDENTITY | SUSPECT_MORPHED_COMPOSITE.")
    signals: Dict[str, Any] = Field(description="Component signals (S-MAD ghosting contours, bilateral gradient asymmetry, D-MAD residual).")
    face_bbox: Optional[List[int]] = Field(default=None, description="Detected portrait bounding box.")


class StreamingProgressEvent(BaseModel):
    processed: int = Field(description="Documents processed so far.")
    total: int = Field(description="Total documents in stream.")
    verified: int = Field(description="Count of verified authentic credentials.")
    flagged: int = Field(description="Count of flagged credentials.")
    current_fps: float = Field(description="Current processing throughput (docs/sec).")
    elapsed_sec: float = Field(description="Elapsed seconds since batch start.")
    eta_sec: float = Field(description="Estimated seconds remaining.")
    last_filename: Optional[str] = Field(default=None, description="Last processed filename.")
    last_decision: Optional[str] = Field(default=None, description="Last verdict.")

