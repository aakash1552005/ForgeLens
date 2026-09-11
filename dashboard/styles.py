"""
ForgeLens-X — 21st.dev / Magic UI Forensic Design System
=========================================================
Custom CSS and HTML injection for enterprise-grade cyber-forensic aesthetics.
Includes glassmorphic panels, glowing status badges, pulse animations,
and typography refinements.
"""

CSS_STYLES = """
<style>
/* Import modern typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Base Root Tweaks */
:root {
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --bg-dark: #080b12;
    --bg-card: rgba(17, 24, 39, 0.75);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-cyan: rgba(56, 189, 248, 0.35);
    --cyan-glow: 0 0 20px -3px rgba(56, 189, 248, 0.25);
    --emerald-glow: 0 0 20px -3px rgba(16, 185, 129, 0.25);
    --amber-glow: 0 0 20px -3px rgba(245, 158, 11, 0.25);
    --crimson-glow: 0 0 20px -3px rgba(239, 68, 68, 0.25);
}

html, body, [class*="css"] {
    font-family: var(--font-sans);
    color: #f1f5f9;
}

/* Subtle dark background with radial depth */
.stApp {
    background: radial-gradient(circle at 50% 0%, #111827 0%, #080b12 75%, #05070c 100%);
    background-attachment: fixed;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #080b12;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(56, 189, 248, 0.4);
}

/* 21st.dev Glassmorphism Cards */
.fl-glass-card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px -2px rgba(0, 0, 0, 0.5);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.fl-glass-card:hover {
    border-color: rgba(255, 255, 255, 0.14);
    box-shadow: 0 6px 30px -2px rgba(0, 0, 0, 0.65);
}

/* Glowing Top Header */
.fl-header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 24px;
    margin-bottom: 20px;
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 14px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), var(--cyan-glow);
}

.fl-brand-title {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #38bdf8 55%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.fl-brand-subtitle {
    font-size: 0.82rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* Telemetry Micro-Pills */
.telemetry-pill {
    font-family: var(--font-mono);
    font-size: 0.74rem;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #38bdf8;
    border-radius: 6px;
    padding: 3px 8px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

/* Pulsing Status Dot Animation */
@keyframes pulse-dot {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

@keyframes pulse-dot-red {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

@keyframes pulse-dot-amber {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(245, 158, 11, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
}

.status-dot-green {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-dot 2s infinite;
}

.status-dot-red {
    width: 8px;
    height: 8px;
    background: #ef4444;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-dot-red 1.5s infinite;
}

.status-dot-amber {
    width: 8px;
    height: 8px;
    background: #f59e0b;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-dot-amber 1.8s infinite;
}

/* Operational Decision Banners */
.decision-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 14px 20px;
    border-radius: 10px;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    margin-bottom: 14px;
}

.decision-verified {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
    border: 1.5px solid #10b981;
    color: #34d399;
    box-shadow: var(--emerald-glow);
}

.decision-review {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.25) 100%);
    border: 1.5px solid #f59e0b;
    color: #fbbf24;
    box-shadow: var(--amber-glow);
}

.decision-risk {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.3) 100%);
    border: 1.5px solid #ef4444;
    color: #f87171;
    box-shadow: var(--crimson-glow);
}

.decision-insufficient {
    background: linear-gradient(135deg, rgba(148, 163, 184, 0.12) 0%, rgba(100, 116, 139, 0.2) 100%);
    border: 1.5px solid #94a3b8;
    color: #cbd5e1;
}

/* Metric KPI Stat Cards */
.kpi-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}

.kpi-card-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}

.kpi-card-value {
    font-family: var(--font-mono);
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
}

/* Suspicious Region Inspector Card */
.inspector-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}

.inspector-card:hover {
    border-color: rgba(239, 68, 68, 0.6);
    box-shadow: 0 4px 18px -2px rgba(239, 68, 68, 0.2);
}

/* Tab Styling Enhancements */
div[data-baseweb="tab-list"] {
    background-color: rgba(15, 23, 42, 0.5) !important;
    padding: 4px 8px !important;
    border-radius: 10px !important;
    border: 1px solid var(--border-subtle) !important;
    gap: 4px !important;
}

div[data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(56, 189, 248, 0.15) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
}

/* Buttons Enhancement */
.stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    border: 1px solid var(--border-subtle) !important;
    background: rgba(30, 41, 59, 0.7) !important;
}

.stButton button:hover {
    border-color: rgba(56, 189, 248, 0.5) !important;
    background: rgba(56, 189, 248, 0.15) !important;
    box-shadow: 0 0 14px -2px rgba(56, 189, 248, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* Primary buttons */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    border: 1px solid #38bdf8 !important;
    box-shadow: 0 0 16px -2px rgba(56, 189, 248, 0.4) !important;
}

.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important;
    box-shadow: 0 0 22px 0 rgba(56, 189, 248, 0.6) !important;
}

/* Custom Table / Rule Check Styling */
.rule-check-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 0.82rem;
}

.rule-check-row:last-child {
    border-bottom: none;
}

.badge-pass {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
}

.badge-fail {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
}

/* Examiner Action Bar */
.action-bar-container {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
}
</style>
"""


def apply_custom_styles() -> str:
    """Return the CSS style block to inject into Streamlit via st.markdown."""
    return CSS_STYLES
