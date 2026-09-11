"""
ForgeLens-X — Left Panel: Document Ingestion & Admissibility Gating
===================================================================
Handles document upload, optional live selfie ingestion, metadata inspection,
and scan quality admissibility gating.
"""

import hashlib
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from dashboard.components.header import FACE_PRESETS, PRESET_SAMPLES


def render_left_panel(
    active_preset: str,
    doc_image_path: Optional[str],
    face_image_path: Optional[str],
    report: Optional[Dict[str, Any]],
) -> Tuple[str, Optional[str]]:
    """
    Renders the left ingestion and metadata audit panel.
    Returns: (effective_doc_path, effective_face_path)
    """
    st.markdown("""
    <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <span>◈</span> DOCUMENT INGESTION & QUALITY
    </div>
    """, unsafe_allow_html=True)

    # Ingestion Source Mode Tab
    ingest_mode = st.radio(
        "Source Mode",
        options=["Benchmark Presets", "Custom Document Upload"],
        horizontal=True,
        label_visibility="collapsed",
    )

    selected_doc_path = doc_image_path
    selected_face_path = face_image_path

    if ingest_mode == "Benchmark Presets":
        st.markdown("<div style='font-size: 0.76rem; color: #94a3b8; margin-top: 4px;'>Selected Preset:</div>", unsafe_allow_html=True)
        preset_choice = st.selectbox(
            "Document Preset",
            options=list(PRESET_SAMPLES.keys()),
            format_func=lambda k: f"{PRESET_SAMPLES[k]['icon']} {PRESET_SAMPLES[k]['label']}",
            index=list(PRESET_SAMPLES.keys()).index(active_preset) if active_preset in PRESET_SAMPLES else 0,
            label_visibility="collapsed",
        )
        selected_doc_path = PRESET_SAMPLES[preset_choice]["path"]
        st.caption(f"_{PRESET_SAMPLES[preset_choice]['description']}_")

        # Optional Face Match Preset
        st.markdown("<div style='font-size: 0.76rem; color: #94a3b8; margin-top: 10px;'>Optional Biometric Verification:</div>", unsafe_allow_html=True)
        face_choice = st.selectbox(
            "Face Preset",
            options=list(FACE_PRESETS.keys()),
            format_func=lambda k: FACE_PRESETS[k]["label"],
            index=0,
            label_visibility="collapsed",
        )
        selected_face_path = FACE_PRESETS[face_choice]["path"]

    else:
        # Custom Document Upload
        uploaded_doc = st.file_uploader(
            "Upload Identity Document",
            type=["jpg", "jpeg", "png"],
            help="Upload an identity document for forensic tampering screening.",
        )
        if uploaded_doc is not None:
            temp_dir = os.path.join("data", "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            temp_doc_path = os.path.join(temp_dir, f"uploaded_{uploaded_doc.name}")
            with open(temp_doc_path, "wb") as f:
                f.write(uploaded_doc.getbuffer())
            selected_doc_path = temp_doc_path

        # Optional Face Selfie Upload
        uploaded_face = st.file_uploader(
            "Upload Live Face Selfie (Optional)",
            type=["jpg", "jpeg", "png"],
            help="Upload a reference live face selfie to verify against the document portrait.",
        )
        if uploaded_face is not None:
            temp_dir = os.path.join("data", "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            temp_face_path = os.path.join(temp_dir, f"uploaded_face_{uploaded_face.name}")
            with open(temp_face_path, "wb") as f:
                f.write(uploaded_face.getbuffer())
            selected_face_path = temp_face_path
        else:
            selected_face_path = None

    # Ingested Previews
    if selected_doc_path and os.path.exists(selected_doc_path):
        st.markdown("---")
        col_prev1, col_prev2 = st.columns([1, 1] if selected_face_path else [1, 0.01])
        with col_prev1:
            st.caption("Credential Ingested")
            st.image(selected_doc_path, use_container_width=True)
        if selected_face_path and os.path.exists(selected_face_path):
            with col_prev2:
                st.caption("Live Selfie Ingested")
                st.image(selected_face_path, use_container_width=True)

    # Document Metadata Audit
    if selected_doc_path and os.path.exists(selected_doc_path):
        doc_img = cv2.imread(selected_doc_path)
        if doc_img is not None:
            h, w = doc_img.shape[:2]
            fsize_kb = os.path.getsize(selected_doc_path) / 1024.0
            with open(selected_doc_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()[:16]

            st.markdown(f"""
            <div class="fl-glass-card" style="padding: 12px 14px; margin-top: 14px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;">
                    Metadata Provenance
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.75rem; line-height: 1.6; color: #cbd5e1;">
                    <div><b style="color: #38bdf8;">DIMENSIONS:</b> {w} × {h} px ({w/h:.2f} AR)</div>
                    <div><b style="color: #38bdf8;">FILE SIZE:</b> {fsize_kb:.1f} KB</div>
                    <div><b style="color: #38bdf8;">DIGEST:</b> {sha256}...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Scan Quality Admissibility Gating Card
    if report:
        quality = report.get("quality", {})
        blur_score = quality.get("blur_score", 0.0)
        res_ok = quality.get("resolution_ok", True)
        rel_tier = quality.get("analysis_reliability", "HIGH")
        ocr_conf = quality.get("ocr_mean_confidence", 0.85)

        # Admissibility status
        if rel_tier == "HIGH":
            gate_badge = '<span class="badge-pass" style="font-size: 0.78rem;"><span class="status-dot-green"></span> ADMISSIBLE</span>'
        elif rel_tier == "MEDIUM":
            gate_badge = '<span class="badge-fail" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border-color: #f59e0b; font-size: 0.78rem;"><span class="status-dot-amber"></span> SUB-OPTIMAL</span>'
        else:
            gate_badge = '<span class="badge-fail" style="font-size: 0.78rem;"><span class="status-dot-red"></span> INSUFFICIENT EVIDENCE</span>'

        st.markdown(f"""
        <div class="fl-glass-card" style="padding: 12px 14px; margin-top: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">
                    Quality Gating
                </span>
                {gate_badge}
            </div>
            <div style="font-size: 0.78rem; line-height: 1.6;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8;">Sharpness (Blur):</span>
                    <span style="font-family: var(--font-mono); color: #f1f5f9;">{blur_score:.1f} (≥ 45.0)</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8;">Resolution Gate:</span>
                    <span style="font-family: var(--font-mono); color: {'#34d399' if res_ok else '#f87171'};">{'PASS' if res_ok else 'FAIL'}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8;">OCR Text Fidelity:</span>
                    <span style="font-family: var(--font-mono); color: #f1f5f9;">{ocr_conf*100:.1f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8;">Field Completeness:</span>
                    <span style="font-family: var(--font-mono); color: #f1f5f9;">{quality.get('field_completeness_ratio', 1.0)*100:.0f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return (selected_doc_path or "", selected_face_path)
