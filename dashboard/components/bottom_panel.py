"""
ForgeLens-X — Bottom Panel: Plain-Language Anomaly Explanation & Examiner Sign-Off
==================================================================================
Synthesizes natural-language explanations for document examiners, reinforces
human-decision-support ethics, and provides examiner clearance action controls.
"""

import json
from typing import Any, Dict

import streamlit as st

from dashboard.utils import (
    format_plain_language_explanations,
    generate_audit_certificate_dict,
)


def render_bottom_panel(
    report: Dict[str, Any],
    doc_path: str,
) -> None:
    """Render the bottom plain-language explanation and examiner sign-off panel."""
    st.markdown("---")

    # Section Title
    st.markdown("""
    <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <span>◈</span> EXPLAINABLE FORENSIC FINDINGS & EXAMINER DECISION SUPPORT
    </div>
    """, unsafe_allow_html=True)

    # 1. Plain-Language Anomalies (Natural Language Synthesis)
    explanations = format_plain_language_explanations(report)

    col_exp1, col_exp2 = st.columns([2.2, 1.0])

    with col_exp1:
        st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #cbd5e1; margin-bottom: 8px;'>Synthesized Examiner Findings:</div>", unsafe_allow_html=True)
        for item in explanations:
            sev = item.get("severity", "MODERATE")
            cat = item.get("category", "Finding")
            txt = item.get("text", "")

            if sev in ["HIGH", "CRITICAL"]:
                border_col = "#ef4444"
                bg_col = "rgba(239, 68, 68, 0.08)"
                badge = f'<span class="badge-fail">{cat}</span>'
            elif sev == "MODERATE":
                border_col = "#f59e0b"
                bg_col = "rgba(245, 158, 11, 0.08)"
                badge = f'<span class="badge-fail" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: #f59e0b;">{cat}</span>'
            else:
                border_col = "#10b981"
                bg_col = "rgba(16, 185, 129, 0.08)"
                badge = f'<span class="badge-pass">{cat}</span>'

            st.markdown(f"""
            <div style="background: {bg_col}; border-left: 3px solid {border_col}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.82rem; line-height: 1.5; color: #e2e8f0;">
                <div style="margin-bottom: 4px;">{badge}</div>
                <div>{txt}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_exp2:
        # Legal / Decision-Support Ethics Notice
        st.markdown("""
        <div class="fl-glass-card" style="padding: 14px; border-color: rgba(56, 189, 248, 0.25);">
            <div style="font-size: 0.78rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 6px;">
                ⚖️ Human Review Mandate
            </div>
            <div style="font-size: 0.74rem; color: #94a3b8; line-height: 1.5;">
                ForgeLens-X is an <b>AI-assisted decision-support system</b>, not an autonomous legal authority.
                <br><br>
                A <b style="color: #f87171;">HIGH_RISK</b> verdict signifies that corroborating forensic anomalies require <b>mandatory physical review</b> by an accredited examiner before any identity clearance or detention action.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Examiner Workflow & Operational Sign-off Bar
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px 20px; margin-top: 14px;">
        <div style="font-size: 0.84rem; font-weight: 700; color: #f1f5f9; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 10px;">
            ✍️ Examiner Inspection Sign-Off & Audit Logging
        </div>
    """, unsafe_allow_html=True)

    col_note, col_action = st.columns([2.0, 1.2])

    with col_note:
        examiner_notes = st.text_input(
            "Examiner Inspection Notes",
            value=st.session_state.get("examiner_notes", ""),
            placeholder="e.g. Watermark inspected under UV light; microprint intact on reverse.",
            key="examiner_notes_input",
            help="Notes will be stamped onto the digital audit certificate.",
        )
        st.session_state["examiner_notes"] = examiner_notes

    with col_action:
        st.markdown("<div style='font-size: 0.76rem; color: #94a3b8; margin-bottom: 4px;'>Official Clearance Action:</div>", unsafe_allow_html=True)
        col_b1, col_b2, col_b3 = st.columns(3)
        if col_b1.button("✓ Clear", use_container_width=True, help="Approve document and issue border clearance."):
            st.session_state["action_taken"] = "CLEARED_BY_EXAMINER"
            st.success("Document cleared by examiner.")
        if col_b2.button("⚠ Review", use_container_width=True, help="Escalate for Level 2 physical forensic examination."):
            st.session_state["action_taken"] = "ESCALATED_PHYSICAL_REVIEW"
            st.warning("Flagged for physical inspection.")
        if col_b3.button("✕ Detain", use_container_width=True, help="Issue fraud detention alert."):
            st.session_state["action_taken"] = "FRAUD_DETENTION_ISSUED"
            st.error("Detention alert logged.")

    # 3. Export Actions
    st.markdown("<div style='margin-top: 10px; display: flex; gap: 10px; align-items: center;'>", unsafe_allow_html=True)
    col_dl1, col_dl2 = st.columns([1, 1])

    # Certificate JSON
    current_action = st.session_state.get("action_taken", "PENDING_REVIEW")
    audit_cert = generate_audit_certificate_dict(
        report=report,
        examiner_notes=examiner_notes,
        examiner_id=st.session_state.get("examiner_id", "EXAMINER-409"),
        action_taken=current_action,
    )
    cert_json_str = json.dumps(audit_cert, indent=2)

    with col_dl1:
        st.download_button(
            label="📥 Download Signed Audit Certificate (JSON)",
            data=cert_json_str,
            file_name=f"audit_certificate_{report.get('document_id', 'doc')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_dl2:
        # Full report JSON
        st.download_button(
            label="📄 Download Full Forensic Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"unified_report_{report.get('document_id', 'doc')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("</div></div>", unsafe_allow_html=True)
