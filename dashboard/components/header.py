"""
ForgeLens-X — Header & Preset Quick-Bar Component
=================================================
Renders 21st.dev Magic UI header banner with live system telemetry
and 1-click preset sample selection buttons.
"""

import os
from typing import Callable, Optional
import streamlit as st

PRESET_SAMPLES = {
    "genuine": {
        "label": "Genuine Document",
        "icon": "🟢",
        "path": "data/generated/images/src_0000_genuine.jpg",
        "description": "Authentic identity credential with zero physical or semantic tamper signals.",
    },
    "date_edit": {
        "label": "Date Tampering",
        "icon": "🟠",
        "path": "data/generated/images/src_0000_date_edit.jpg",
        "description": "Altered expiry/issue dates displaying localized JPEG recompression discontinuity.",
    },
    "text_edit": {
        "label": "Text Modification",
        "icon": "🟠",
        "path": "data/generated/images/src_0000_text_edit.jpg",
        "description": "Modified surname or document number with typography & semantic contradictions.",
    },
    "photo_swap": {
        "label": "Photo Swap",
        "icon": "🔴",
        "path": "data/generated/images/src_0000_photo_swap.jpg",
        "description": "Substituted portrait photo with face boundary artifact and biometric mismatch.",
    },
    "copy_move": {
        "label": "Copy-Move Cloning",
        "icon": "🟣",
        "path": "data/generated/images/src_0000_copy_move.jpg",
        "description": "Duplicated official emblem/stamp motif with matching ORB keypoint vectors.",
    },
}

FACE_PRESETS = {
    "none": {
        "label": "None (Document Screening Only)",
        "path": None,
    },
    "genuine": {
        "label": "Genuine Matching Selfie",
        "path": "data/face_pairs/images/pair_0001_genuine_live.jpg",
    },
    "impostor": {
        "label": "Impostor Mismatched Selfie",
        "path": "data/face_pairs/images/pair_0011_imposter_live.jpg",
    },
}


def render_header(
    active_preset: str,
    on_preset_selected: Callable[[str], None],
) -> None:
    """Render the top cyber-forensic header with quick preset selector."""
    # Glassmorphism Top Banner
    st.markdown("""
    <div class="fl-header-container">
        <div>
            <div class="fl-brand-title">
                FORGELENS-X
                <span class="telemetry-pill">
                    <span class="status-dot-green"></span> PIPELINE ONLINE
                </span>
            </div>
            <div class="fl-brand-subtitle">
                INTELLIGENT DOCUMENT FORENSICS & BORDER ADMISSIBILITY CONSOLE · COMPETITION EDITION
            </div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span class="telemetry-pill"> RapidOCR </span>
            <span class="telemetry-pill"> SFace / ArcFace </span>
            <span class="telemetry-pill"> Isotonic M6 Fusion </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1-Click Preset Selection Bar
    st.markdown("<div style='margin-bottom: 6px; font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;'>Quick-Load Benchmark Presets</div>", unsafe_allow_html=True)
    
    cols = st.columns(len(PRESET_SAMPLES))
    for col, (preset_key, pdata) in zip(cols, PRESET_SAMPLES.items()):
        is_active = (active_preset == preset_key)
        btn_label = f"{pdata['icon']} {pdata['label']}"
        if col.button(btn_label, key=f"preset_btn_{preset_key}", use_container_width=True, type="primary" if is_active else "secondary"):
            on_preset_selected(preset_key)
