"""
ForgeLens-X — Enterprise Forensic Console Stylesheet
=====================================================
Built according to 21st.dev and UI/UX Pro Max design intelligence:
- True zinc neutral dark system (#09090b / #121215)
- Inter / Geist typography paired with JetBrains Mono numbers
- Zero cartoon emojis: crisp SVG vector icons & live pulsating micro-dots
- Segmented control pills and Bento grid forensic signals
- Ultra-responsive, high-contrast, professional cybersecurity aesthetic
"""

ENTERPRISE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, 'SF Mono', monospace;
    
    /* True Zinc Neutral Palette (Linear / 21st.dev style) */
    --bg-base: #09090b;
    --bg-surface: #121215;
    --bg-elevated: #18181b;
    --bg-muted: #27272a;
    
    /* Precise 1px Subtle Borders */
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(255, 255, 255, 0.16);
    --border-focus: rgba(56, 189, 248, 0.4);
    
    /* Text Hierarchy */
    --text-primary: #f4f4f5;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --text-accent: #38bdf8;
    
    /* Semantic Status Accents */
    --status-verified-text: #34d399;
    --status-verified-bg: rgba(16, 185, 129, 0.07);
    --status-verified-border: rgba(16, 185, 129, 0.22);
    
    --status-tampered-text: #fb7185;
    --status-tampered-bg: rgba(244, 63, 94, 0.07);
    --status-tampered-border: rgba(244, 63, 94, 0.25);
    
    --status-review-text: #fbbf24;
    --status-review-bg: rgba(245, 158, 11, 0.07);
    --status-review-border: rgba(245, 158, 11, 0.22);
}

/* Base resets & layout */
html, body, [class*="css"] {
    font-family: var(--font-sans);
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background-color: var(--bg-base);
}

/* Ensure top header never overlaps with Streamlit toolbar */
header[data-testid="stHeader"] {
    background: rgba(9, 9, 11, 0.8) !important;
    backdrop-filter: blur(8px) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

.main .block-container {
    padding-top: 4.2rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1480px !important;
}

/* Top Navigation Bar */
.nav-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0 16px 0;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border-subtle);
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nav-logo-icon {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}

.nav-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.015em;
    color: var(--text-primary);
}

.nav-divider {
    color: var(--border-hover);
    font-size: 0.9rem;
    margin: 0 4px;
}

.nav-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 400;
}

.nav-telemetry {
    display: flex;
    align-items: center;
    gap: 10px;
}

.nav-badge {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    padding: 4px 10px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Live Pulsating Micro-Dots */
.dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.dot-emerald {
    background: #10b981;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.25);
    animation: pulse-emerald 2s infinite ease-in-out;
}

.dot-rose {
    background: #f43f5e;
    box-shadow: 0 0 0 2px rgba(244, 63, 94, 0.25);
    animation: pulse-rose 1.6s infinite ease-in-out;
}

.dot-amber {
    background: #f59e0b;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.25);
    animation: pulse-amber 1.8s infinite ease-in-out;
}

@keyframes pulse-emerald {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
    70% { transform: scale(1); box-shadow: 0 0 0 5px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

@keyframes pulse-rose {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.4); }
    70% { transform: scale(1); box-shadow: 0 0 0 5px rgba(244, 63, 94, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }
}

@keyframes pulse-amber {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4); }
    70% { transform: scale(1); box-shadow: 0 0 0 5px rgba(245, 158, 11, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
}

/* Section Header Labels */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Layer Explanation Badge */
.layer-desc-badge {
    font-size: 0.76rem;
    color: var(--text-secondary);
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 5px 12px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.4;
}

/* Primary Verdict Card */
.verdict-card {
    background: var(--bg-surface);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
    border: 1px solid var(--border-subtle);
    position: relative;
    overflow: hidden;
}

.verdict-card-verified {
    background: var(--status-verified-bg);
    border-color: var(--status-verified-border);
}

.verdict-card-tampered {
    background: var(--status-tampered-bg);
    border-color: var(--status-tampered-border);
}

.verdict-card-review {
    background: var(--status-review-bg);
    border-color: var(--status-review-border);
}

.verdict-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 10px;
}

.verdict-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.verdict-card-verified .verdict-badge { color: var(--status-verified-text); }
.verdict-card-tampered .verdict-badge { color: var(--status-tampered-text); }
.verdict-card-review .verdict-badge { color: var(--status-review-text); }

.verdict-score-block {
    text-align: right;
}

.verdict-score-num {
    font-family: var(--font-mono);
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1;
    color: var(--text-primary);
}

.verdict-score-label {
    font-size: 0.70rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-top: 3px;
}

.verdict-narrative {
    font-size: 0.86rem;
    line-height: 1.5;
    color: var(--text-secondary);
    margin-bottom: 12px;
}

.verdict-metrics-row {
    display: flex;
    gap: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    padding-top: 10px;
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--text-muted);
}

.verdict-metric-item b {
    color: var(--text-primary);
}

/* Anomaly Callout Pill */
.anomaly-callout {
    background: rgba(244, 63, 94, 0.08);
    border: 1px solid rgba(244, 63, 94, 0.22);
    border-left: 3px solid #f43f5e;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 0.82rem;
    color: #fca5a5;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    line-height: 1.45;
}

.anomaly-callout-icon {
    flex-shrink: 0;
    margin-top: 1px;
}

/* Bento Grid Forensic Signals */
.bento-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}

.bento-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 12px 14px;
    transition: border-color 0.15s ease;
}

.bento-card:hover {
    border-color: var(--border-hover);
}

.bento-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.bento-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 6px;
}

.bento-status {
    font-size: 0.66rem;
    font-weight: 600;
    font-family: var(--font-mono);
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.bento-status-pass {
    background: rgba(16, 185, 129, 0.1);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.25);
}

.bento-status-fail {
    background: rgba(244, 63, 94, 0.1);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.28);
}

.bento-status-info {
    background: rgba(56, 189, 248, 0.1);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.25);
}

.bento-value {
    font-family: var(--font-mono);
    font-size: 1.10rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 2px;
}

.bento-desc {
    font-size: 0.76rem;
    color: var(--text-secondary);
    line-height: 1.35;
}

/* Biometric Special Tile */
.biometric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.biometric-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Streamlit Native Segmented Control Customization (21st.dev Style) */
div[data-testid="stSegmentedControl"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 9px !important;
    padding: 3px !important;
    margin-bottom: 8px !important;
}

div[data-testid="stSegmentedControl"] button {
    border-radius: 6px !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    border: none !important;
    background: transparent !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
}

div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: var(--bg-muted) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.5) !important;
}

/* Clean Button Styling */
.stButton button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    border: 1px solid var(--border-subtle) !important;
    background: var(--bg-surface) !important;
    color: var(--text-secondary) !important;
    padding: 6px 12px !important;
    transition: all 0.15s ease !important;
}

.stButton button:hover {
    border-color: var(--border-hover) !important;
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

.stButton button[kind="primary"] {
    background: #0284c7 !important;
    border-color: #38bdf8 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

.stButton button[kind="primary"]:hover {
    background: #0369a1 !important;
    border-color: #7dd3fc !important;
}

/* Download Buttons */
.stDownloadButton button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    border: 1px solid var(--border-subtle) !important;
    background: var(--bg-surface) !important;
    color: var(--text-secondary) !important;
    padding: 7px 14px !important;
    transition: all 0.15s ease !important;
}

.stDownloadButton button:hover {
    border-color: var(--border-hover) !important;
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

/* Dropzone Styling */
div[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-surface) !important;
    border: 1.5px dashed var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 14px !important;
    transition: border-color 0.2s ease !important;
}

div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--border-focus) !important;
}

/* Tabs */
div[data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    gap: 8px !important;
    padding-bottom: 2px !important;
    margin-bottom: 14px !important;
}

div[data-baseweb="tab"] {
    padding: 6px 14px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    margin-top: 10px !important;
}

div[data-testid="stExpander"] details summary {
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    padding: 8px 12px !important;
}

/* Monospace Pre & Code */
pre, code {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    background: #09090b !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
}
</style>
"""


def apply_custom_styles() -> str:
    """Return the professional 21st.dev / UI-UX Pro Max CSS block."""
    return ENTERPRISE_CSS
