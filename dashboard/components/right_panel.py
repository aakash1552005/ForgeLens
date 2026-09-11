"""
ForgeLens-X — Right Panel: Calibrated Risk, Decision Policy & Audit
===================================================================
Renders calibrated risk score gauge, fraud probability, operational decision
banner, explainable top-3 risk drivers, fired signals, semantic checks,
and biometric face verification card.
"""

from typing import Any, Dict, List, Optional

import streamlit as st


def render_right_panel(
    report: Dict[str, Any],
    face_path: Optional[str],
) -> None:
    """Render the right risk telemetry, explainable drivers, and decision policy panel."""
    st.markdown("""
    <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <span>◈</span> RISK SCORE & DECISION POLICY
    </div>
    """, unsafe_allow_html=True)

    risk_score = float(report.get("risk_score", 0.0))
    fraud_prob = float(report.get("fraud_probability", 0.0))
    decision = str(report.get("decision", "MANUAL_REVIEW")).upper()
    risk_tier = str(report.get("risk_tier", "LOW")).upper()

    # Operational Decision Banner
    if decision in ["VERIFIED", "CLEAR_AUTHENTIC"]:
        dec_class = "decision-verified"
        dec_icon = '<span class="status-dot-green"></span>'
        sub_text = "CREDENTIAL AUTHENTIC · CLEARED"
    elif decision in ["HIGH_RISK", "CRITICAL_FRAUD"]:
        dec_class = "decision-risk"
        dec_icon = '<span class="status-dot-red"></span>'
        sub_text = "HIGH RISK · REQUIRES EXPERT HUMAN REVIEW"
    elif decision in ["MANUAL_REVIEW", "SUSPECT_TAMPERING"]:
        dec_class = "decision-review"
        dec_icon = '<span class="status-dot-amber"></span>'
        sub_text = "EVIDENCE CONFLICT · SECONDARY INSPECTION"
    else:
        dec_class = "decision-insufficient"
        dec_icon = '<span class="status-dot-amber"></span>'
        sub_text = "INSUFFICIENT FORENSIC EVIDENCE"

    st.markdown(f"""
    <div class="decision-badge {dec_class}">
        {dec_icon}
        <div>
            <div>{decision}</div>
            <div style="font-size: 0.68rem; font-weight: 500; letter-spacing: 0.06em; opacity: 0.85;">
                {sub_text}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk Score & Fraud Probability KPI Cards
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        # Calibrated Risk Score [0 - 100]
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-card-title">Calibrated Risk Score</div>
            <div class="kpi-card-value" style="color: {'#34d399' if risk_score < 30 else '#fbbf24' if risk_score < 60 else '#f87171'};">
                {risk_score:.1f}<span style="font-size: 0.85rem; color: #94a3b8;">/100</span>
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">Tier: <b>{risk_tier}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col_k2:
        # Calibrated Fraud Probability [0.0 - 1.0]
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-card-title">Fraud Probability</div>
            <div class="kpi-card-value" style="color: {'#34d399' if fraud_prob < 0.3 else '#fbbf24' if fraud_prob < 0.6 else '#f87171'};">
                {fraud_prob*100:.1f}<span style="font-size: 0.85rem; color: #94a3b8;">%</span>
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">P(Fraud) = {fraud_prob:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk Meter Progress Bar
    meter_color = "#10b981" if risk_score < 30 else "#f59e0b" if risk_score < 60 else "#ef4444"
    st.markdown(f"""
    <div style="width: 100%; background: rgba(255, 255, 255, 0.08); height: 6px; border-radius: 3px; margin: 12px 0 16px 0; overflow: hidden;">
        <div style="width: {min(100.0, max(0.0, risk_score))}%; background: {meter_color}; height: 100%; transition: width 0.4s ease;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Explainable Top-3 Risk Drivers (Linear Log-Odds Decomposition)
    risk_drivers = report.get("risk_drivers", [])
    if risk_drivers:
        st.markdown("""
        <div style="font-size: 0.82rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">
            Explainable Risk Drivers (Log-Odds Impact)
        </div>
        """, unsafe_allow_html=True)

        for driver in risk_drivers[:3]:
            feat_name = driver.get("feature", "feature")
            contrib = float(driver.get("contribution_log_odds", 0.0))
            desc = driver.get("description", feat_name)
            sign_char = "+" if contrib >= 0 else ""

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; font-size: 0.78rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <span style="font-weight: 600; color: #f1f5f9;">{feat_name}</span>
                    <span style="font-family: var(--font-mono); color: {'#f87171' if contrib > 0 else '#34d399'}; font-weight: 700;">
                        {sign_char}{contrib:.3f} log-odds
                    </span>
                </div>
                <div style="color: #94a3b8; font-size: 0.73rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Biometric Face Verification Card
    face_res = report.get("face_verification", {})
    if face_res.get("has_face_check"):
        is_verified = face_res.get("verified", False)
        dist = face_res.get("distance", 1.0)
        thresh = face_res.get("threshold", 0.40)
        sim_pct = face_res.get("similarity_pct", 0.0)

        st.markdown(f"""
        <div class="fl-glass-card" style="padding: 12px 14px; margin-top: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #38bdf8; text-transform: uppercase;">
                    Biometric Face Verification
                </span>
                <span class="{'badge-pass' if is_verified else 'badge-fail'}">
                    {'MATCH' if is_verified else 'MISMATCH'}
                </span>
            </div>
            <div style="font-size: 0.78rem; line-height: 1.6;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8;">Cosine Distance:</span>
                    <span style="font-family: var(--font-mono);">{dist:.4f} (Thresh: {thresh:.2f})</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8;">Similarity Score:</span>
                    <span style="font-family: var(--font-mono); color: {'#34d399' if is_verified else '#f87171'}; font-weight: 700;">
                        {sim_pct:.1f}%
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94a3b8;">Biometric Gate:</span>
                    <span style="color: #cbd5e1;">{face_res.get('evidence_influence', 'Gated')}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 8px 12px; margin-top: 10px; font-size: 0.74rem; color: #94a3b8; text-align: center;">
            Biometric Face Gate: <i>Optional Live Face Not Provided</i>
        </div>
        """, unsafe_allow_html=True)

    # Semantic & Integrity Rules Checklist
    with st.expander("📋 Semantic & Document Integrity Rules", expanded=False):
        semantic_data = report.get("semantic_checks", [])
        if isinstance(semantic_data, list) and semantic_data:
            for citem in semantic_data:
                cname = citem.get("check", "rule")
                passed = (citem.get("status") == "PASS")
                badge_html = '<span class="badge-pass">PASS</span>' if passed else '<span class="badge-fail">FAIL</span>'
                clean_name = cname.replace("_", " ").title()
                st.markdown(f"""
                <div class="rule-check-row">
                    <span style="color: #cbd5e1;">{clean_name}</span>
                    {badge_html}
                </div>
                """, unsafe_allow_html=True)
        elif isinstance(semantic_data, dict) and semantic_data.get("checks"):
            for cname, cdata in semantic_data.get("checks", {}).items():
                passed = cdata.get("passed", True)
                badge_html = '<span class="badge-pass">PASS</span>' if passed else '<span class="badge-fail">FAIL</span>'
                clean_name = cname.replace("_", " ").title()
                st.markdown(f"""
                <div class="rule-check-row">
                    <span style="color: #cbd5e1;">{clean_name}</span>
                    {badge_html}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No semantic rules evaluated.")

    # Fired Tamper Signals Summary
    with st.expander("⚡ Fired Tamper Modalities", expanded=False):
        tamper = report.get("tamper_signals", {})
        ela_fire = tamper.get("ela", {}).get("detected", False)
        cm_fire = tamper.get("copy_move", {}).get("detected", False)
        font_fire = report.get("font_forensics", {}).get("is_anomalous", False)
        meta_fire = report.get("metadata_forensics", {}).get("is_tampered", False)

        signals_list = [
            ("ELA Recompression Splicing", ela_fire),
            ("Copy-Move Motif Cloning", cm_fire),
            ("Font Typography Inconsistency", font_fire),
            ("EXIF Software Manipulation", meta_fire),
        ]

        for s_title, s_active in signals_list:
            b_html = '<span class="badge-fail">ALERT</span>' if s_active else '<span class="badge-pass">CLEAR</span>'
            st.markdown(f"""
            <div class="rule-check-row">
                <span style="color: #cbd5e1;">{s_title}</span>
                {b_html}
            </div>
            """, unsafe_allow_html=True)
