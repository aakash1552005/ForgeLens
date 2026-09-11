"""
ForgeLens-X — Milestone 7: Intelligent Document Forensics Console
================================================================
The flagship competition interactive examiner console.
Built with Streamlit, 21st.dev Magic UI design, and multi-modal forensic fusion.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from dashboard.components.bottom_panel import render_bottom_panel
from dashboard.components.center_panel import render_center_panel
from dashboard.components.header import PRESET_SAMPLES, render_header
from dashboard.components.left_panel import render_left_panel
from dashboard.components.right_panel import render_right_panel
from dashboard.styles import apply_custom_styles
from dashboard.utils import run_screening_pipeline


def main() -> None:
    # 1. Page Configuration (Cyber-Forensic Wide Console)
    st.set_page_config(
        page_title="ForgeLens-X · Forensics Console",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # 2. Inject 21st.dev / Magic UI Custom Styles
    st.markdown(apply_custom_styles(), unsafe_allow_html=True)

    # 3. Session State Initialization
    if "active_preset" not in st.session_state:
        st.session_state["active_preset"] = "date_edit"
    if "doc_path" not in st.session_state:
        st.session_state["doc_path"] = PRESET_SAMPLES["date_edit"]["path"]
    if "face_path" not in st.session_state:
        st.session_state["face_path"] = None
    if "examiner_id" not in st.session_state:
        st.session_state["examiner_id"] = "EXAMINER-409"
    if "action_taken" not in st.session_state:
        st.session_state["action_taken"] = "PENDING_REVIEW"
    if "examiner_notes" not in st.session_state:
        st.session_state["examiner_notes"] = ""

    def on_preset_selected(preset_key: str) -> None:
        st.session_state["active_preset"] = preset_key
        st.session_state["doc_path"] = PRESET_SAMPLES[preset_key]["path"]
        st.session_state["action_taken"] = "PENDING_REVIEW"

    # 4. Render Ambient Header & Preset Quick-Bar
    render_header(
        active_preset=st.session_state["active_preset"],
        on_preset_selected=on_preset_selected,
    )

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # 5. Execute Forensic Screening Pipeline
    current_doc_path = st.session_state["doc_path"]
    current_face_path = st.session_state.get("face_path")

    report = None
    if current_doc_path and os.path.exists(current_doc_path):
        try:
            report = run_screening_pipeline(
                image_path=current_doc_path,
                reference_face_path=current_face_path,
            )
        except Exception as e:
            st.error(f"Forensic Pipeline Exception: {str(e)}")
            st.info("Ensure all model weights and dependencies are accessible.")

    # 6. Three-Panel Primary Layout
    col_left, col_center, col_right = st.columns([1.05, 2.2, 1.15], gap="medium")

    # LEFT PANEL: Ingestion & Admissibility Gating
    with col_left:
        selected_doc, selected_face = render_left_panel(
            active_preset=st.session_state["active_preset"],
            doc_image_path=current_doc_path,
            face_image_path=current_face_path,
            report=report,
        )
        # If user changed paths via left panel, update state
        if selected_doc != current_doc_path or selected_face != current_face_path:
            st.session_state["doc_path"] = selected_doc
            st.session_state["face_path"] = selected_face
            st.rerun()

    # CENTER PANEL: Multi-Modal Visual Forensic Viewer
    with col_center:
        if report and current_doc_path and os.path.exists(current_doc_path):
            render_center_panel(
                doc_path=current_doc_path,
                face_path=current_face_path,
                report=report,
            )
        else:
            st.info("Ingest or select an identity credential to view visual evidence.")

    # RIGHT PANEL: Calibrated Risk, Decision Policy & Audit Telemetry
    with col_right:
        if report:
            render_right_panel(
                report=report,
                face_path=current_face_path,
            )
        else:
            st.info("Awaiting pipeline report.")

    # 7. BOTTOM PANEL: Plain-Language Findings & Examiner Sign-Off
    if report and current_doc_path:
        render_bottom_panel(
            report=report,
            doc_path=current_doc_path,
        )


if __name__ == "__main__":
    main()
