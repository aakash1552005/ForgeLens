---
name: 21st-dev
description: 21st.dev, Magic UI, and modern React/shadcn design component patterns. Enforces Bento grid layouts, micro-interactions, segmented pill toggles, and clean enterprise cybersecurity styling.
---

# 21st.dev & Magic UI — Design Component Guidelines

## 1. Bento Grid Architecture
Organize disparate forensic signals into an asymmetric or balanced 2x2 / 3-column Bento grid.
- Each tile represents an independent forensic signal:
  1. `Compression & Noise (ELA)`: Artifact residue ratio, thermal anomaly delta.
  2. `Clone & Splicing (ORB)`: Matched keypoints and cluster centroids.
  3. `Typography & MRZ`: Font stroke width variance, ICAO Doc 9303 checksum parity.
  4. `Biometric Face Match`: ArcFace cosine distance, selfie identity similarity.
- Cards maintain consistent padding (`14px - 18px`), subtle borders (`rgba(255, 255, 255, 0.08)`), and dark backgrounds (`#121215`).

## 2. Segmented Pill Controls
- Replace clunky native radio buttons or bulky tabs with sleek segmented pill strips:
  - Container: `display: inline-flex; background: #18181b; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 3px;`
  - Pill Item: `padding: 6px 14px; font-size: 12px; font-weight: 500; border-radius: 6px; color: #a1a1aa; cursor: pointer; transition: all 0.15s ease;`
  - Active Pill: `background: #27272a; color: #f4f4f5; box-shadow: 0 1px 2px rgba(0,0,0,0.4);`

## 3. Inline Vector Iconography
- Instead of unicode emojis, use inline SVG icons with clean geometric strokes (`stroke-width: 1.75`, `fill: none`, `stroke: currentColor`):
  - **Shield**: Identity integrity and overall verification.
  - **AlertTriangle**: Tamper detection, high risk warning.
  - **CheckCircle**: Passed checks, valid checksums, verified biometrics.
  - **Eye / Layers**: Visual evidence layers, heatmap toggles.
  - **Download / FileText**: Forensic report and audit certificate exports.
  - **Cpu / Activity**: Real-time telemetry, model status, inference latency.

## 4. Pulsing Micro-Dot Indicators
- Minimalist 6px-8px live status indicators:
  ```css
  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
  }
  .status-dot-emerald {
    background: #10b981;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    animation: pulse-emerald 2s infinite ease-in-out;
  }
  .status-dot-rose {
    background: #f43f5e;
    box-shadow: 0 0 0 2px rgba(244, 63, 94, 0.2);
    animation: pulse-rose 1.5s infinite ease-in-out;
  }
  ```
