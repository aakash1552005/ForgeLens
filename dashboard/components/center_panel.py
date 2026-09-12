"""
ForgeLens-X — Center Panel: Multi-Modal Visual Forensic Viewer
==============================================================
Interactive forensic canvas with real-time layer toggles, ELA heatmap
opacity blending, 4 inspection tabs, and deep zoom suspicious crop inspector.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np
import streamlit as st

from dashboard.utils import (
    extract_region_crops,
    generate_copy_move_matches_bgr,
    generate_ela_heatmap_bgr,
    get_cached_diagnostic_card,
    render_dynamic_overlay,
)


def render_center_panel(
    doc_path: str,
    face_path: Optional[str],
    report: Dict[str, Any],
) -> None:
    """Render the central multi-modal visual forensic viewer."""
    st.markdown("""
    <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <span>◈</span> MULTI-MODAL VISUAL FORENSIC VIEWER
    </div>
    """, unsafe_allow_html=True)

    doc_bgr = cv2.imread(doc_path)
    if doc_bgr is None:
        st.error(f"Failed to read image at: {doc_path}")
        return

    # Pre-generate or retrieve ELA heatmap for blending & tabs
    ela_heat_bgr = generate_ela_heatmap_bgr(doc_path, report=report)

    # 4 Main Forensic Inspection Tabs
    tab_spatial, tab_ela, tab_copymove, tab_master = st.tabs([
        "🔍 Spatial Canvas & Layers",
        "🔥 ELA Residue Heatmap",
        "🧬 Copy-Move Vectors",
        "📋 Master Diagnostic Card",
    ])

    # Tab 1: Interactive Spatial Evidence Canvas
    with tab_spatial:
        # Layer Visibility Controls Card
        with st.expander("🛠️ Evidence Layer Controls & Heatmap Blending", expanded=True):
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            with col_t1:
                show_ocr = st.checkbox("OCR Fields", value=True, help="Cyan bounding boxes for extracted text fields.")
            with col_t2:
                show_ela = st.checkbox("ELA Splicing", value=True, help="Amber box highlighting localized recompression anomaly.")
            with col_t3:
                show_cm = st.checkbox("Copy-Move", value=True, help="Purple vectors connecting cloned image patches.")
            with col_t4:
                show_susp = st.checkbox("Suspicious Badges", value=True, help="Crimson alert callouts for detected fraud regions.")

            ela_opacity = st.slider(
                "Heatmap Blend Opacity (Alpha)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Slide to blend the ELA compression error heatmap directly over the document canvas.",
            )

        # Dynamic Canvas Rendering
        overlay_rgb = render_dynamic_overlay(
            doc_bgr=doc_bgr,
            report=report,
            show_ocr=show_ocr,
            show_ela=show_ela,
            show_copy_move=show_cm,
            show_suspicious=show_susp,
            ela_opacity=ela_opacity,
            ela_heatmap_bgr=ela_heat_bgr,
        )

        st.image(overlay_rgb, use_container_width=True)

    # Tab 2: Error Level Analysis (ELA) Heatmap
    with tab_ela:
        st.caption("Error Level Analysis (ELA) visualizes JPEG recompression residual differences. High-intensity pixels indicate localized post-processing, splicing, or text alterations.")
        heat_rgb = cv2.cvtColor(ela_heat_bgr, cv2.COLOR_BGR2RGB)
        st.image(heat_rgb, use_container_width=True)

        # ELA Telemetry Metrics
        ela_data = report.get("tamper_signals", {}).get("ela", {})
        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("Anomaly Energy", f"{ela_data.get('anomaly_energy', 0.0):.1f}")
        col_e2.metric("Mean Residual", f"{ela_data.get('mean_error', 0.0):.2f}")
        col_e3.metric("Candidate Splicing", "DETECTED" if ela_data.get("detected") else "NONE")

    # Tab 3: Copy-Move ORB Keypoint Matches
    with tab_copymove:
        st.caption("ORB keypoint feature matching & RANSAC homography estimation detect cloned visual motifs (seals, emblems, or signatures) within the same document.")
        cm_canvas_bgr = generate_copy_move_matches_bgr(doc_path)
        cm_canvas_rgb = cv2.cvtColor(cm_canvas_bgr, cv2.COLOR_BGR2RGB)
        st.image(cm_canvas_rgb, use_container_width=True)

        cm_data = report.get("tamper_signals", {}).get("copy_move", {})
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Verified Matches", f"{cm_data.get('num_matches', 0)}")
        col_c2.metric("RANSAC Inliers", f"{cm_data.get('num_inliers', 0)}")
        col_c3.metric("Cloned Motif", "DETECTED" if cm_data.get("detected") else "NONE")

    # Tab 4: Master 4-Panel Screening Card
    with tab_master:
        st.caption("Publication-grade 1280x870 4-panel diagnostic card synthesizing spatial, ELA, copy-move, and biometric evidence.")
        try:
            card_rgb = get_cached_diagnostic_card(doc_path, face_path)
            st.image(card_rgb, use_container_width=True)
        except Exception as e:
            st.warning(f"Diagnostic card renderer notice: {e}")

    # Deep Zoom Suspicious Region Inspector (Below tabs)
    susp_regions = report.get("suspicious_regions", [])
    if susp_regions:
        st.markdown("""
        <div style="font-size: 0.88rem; font-weight: 700; color: #ef4444; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 18px; margin-bottom: 8px;">
            🚨 DEEP ZOOM EVIDENCE INSPECTOR (SUSPICIOUS REGIONS)
        </div>
        """, unsafe_allow_html=True)

        for idx, sreg in enumerate(susp_regions):
            bbox = sreg.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            field_name = sreg.get("field", "region")
            source_mod = sreg.get("source", "forensic").upper()
            conf = sreg.get("confidence", 0.0)
            evidence_desc = sreg.get("evidence", "Anomaly detected in region.")

            with st.expander(f"📍 Region #{idx+1}: [{source_mod}] {field_name} (Confidence: {conf*100:.0f}%)", expanded=(idx == 0)):
                orig_crop, ela_crop = extract_region_crops(doc_bgr, ela_heat_bgr, bbox)

                c_crop1, c_crop2, c_crop3 = st.columns([1.2, 1.2, 2.0])
                with c_crop1:
                    st.caption("Credential Patch")
                    st.image(orig_crop, use_container_width=True)
                with c_crop2:
                    st.caption("ELA Compression Residue")
                    st.image(ela_crop, use_container_width=True)
                with c_crop3:
                    st.markdown(f"""
                    <div style="font-size: 0.8rem; line-height: 1.6; color: #cbd5e1;">
                        <div><b style="color: #38bdf8;">FIELD:</b> {field_name}</div>
                        <div><b style="color: #38bdf8;">MODALITY:</b> {source_mod}</div>
                        <div><b style="color: #38bdf8;">COORDINATES:</b> [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]</div>
                        <div style="margin-top: 6px; padding: 6px 8px; background: rgba(239, 68, 68, 0.12); border-left: 3px solid #ef4444; border-radius: 4px; color: #fca5a5;">
                            {evidence_desc}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
