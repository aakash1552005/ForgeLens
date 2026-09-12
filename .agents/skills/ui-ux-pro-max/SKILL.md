---
name: ui-ux-pro-max
description: Design intelligence and professional UI/UX rules for building human-crafted, enterprise-grade web and dashboard interfaces. Eliminates generic AI aesthetics, enforces typography hierarchy, zinc dark mode, high WCAG contrast, and clean layout patterns.
---

# UI/UX Pro Max — Design Intelligence & Engineering Rules

## 1. Core Principles: Eradicating AI-Generated UI Tropes
Generic AI-generated interfaces share telltale flaws that immediately look unpolished and amateurish. Apply these hard constraints:
- **No Emoji Clutter**: Never use cartoon emojis (🚨, ⚠️, ✅, ❌, 🔥, 🔍, 📥) as UI icons, button labels, or status badges in security, fintech, or forensic software. Always use crisp inline SVG vector icons (Lucide/Heroicons style) or clean typographical badges (`• VERIFIED`, `• TAMPER DETECTED`).
- **No Harsh Glowing Drop-Shadows**: Avoid saturated neon glows (`box-shadow: 0 0 25px rgba(239, 68, 68, 0.8)`). Use subtle, refined elevation shadows (`box-shadow: 0 1px 3px rgba(0,0,0,0.5), 0 1px 2px rgba(0,0,0,0.4)`).
- **No Garish Rainbow Gradients**: Prefer solid, deep neutral surfaces with subtle 1px borders (`rgba(255, 255, 255, 0.08)`).
- **Restrained Semantic Accents**: Colors must have a clear purpose. Reserve emerald for clean/verified, crimson/rose for threats/tampering, amber for manual inspection, and sky/cyan for interactive focus.

## 2. Color System: Zinc Neutral Dark Theme
- **Canvas Base**: `#09090b` (zinc-950)
- **Primary Cards & Containers**: `#121215` (dark surface) with `border: 1px solid rgba(255, 255, 255, 0.08)`
- **Hover & Secondary Surface**: `#18181b` (zinc-900)
- **Borders & Dividers**: `rgba(255, 255, 255, 0.08)` / `#27272a`
- **Text Primary**: `#f4f4f5` (zinc-100) — high contrast, authoritative
- **Text Secondary**: `#a1a1aa` (zinc-400) — legible labels and descriptions
- **Text Muted**: `#71717a` (zinc-500) — metadata and timestamps

### Semantic Status Palette
- **Verified / Safe**:
  - Background: `rgba(16, 185, 129, 0.08)`
  - Border: `rgba(16, 185, 129, 0.25)`
  - Text: `#34d399`
  - Indicator: `●` live pulse dot in `#10b981`
- **Tampered / High Risk**:
  - Background: `rgba(244, 63, 94, 0.08)`
  - Border: `rgba(244, 63, 94, 0.28)`
  - Text: `#fb7185`
  - Indicator: `●` live pulse dot in `#f43f5e`
- **Review / Suspicious**:
  - Background: `rgba(245, 158, 11, 0.08)`
  - Border: `rgba(245, 158, 11, 0.25)`
  - Text: `#fbbf24`
  - Indicator: `●` live pulse dot in `#f59e0b`

## 3. Typography Hierarchy & Font Pairings
- **Primary UI Font**: `Inter`, `Geist`, or `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto`
- **Data & Telemetry Font**: `JetBrains Mono`, `Geist Mono`, `ui-monospace, "SF Mono", monospace`
- **Scale**:
  - Section Badges: `10px - 11px`, `text-transform: uppercase`, `letter-spacing: 0.08em`, `font-weight: 600`
  - Card Titles: `13px - 14px`, `font-weight: 600`, color `zinc-300`
  - Metrics / Numbers: `24px - 32px`, `font-family: var(--font-mono)`, `font-weight: 700`, color `zinc-100`
  - Body / Explanations: `12px - 13.5px`, `line-height: 1.5`, color `zinc-400`

## 4. Layout Architecture: Enterprise Forensic Console
- **Top Navigation Bar**: Brand identity, breadcrumb path, real-time subsystem telemetry badges, and quick reset action.
- **2-Column Split Workspace**:
  - **Left (Evidence Canvas)**: High-resolution visual view with segmented pill controls for instantaneous layer switching (`Tamper Regions`, `Compression Thermal`, `Original Baseline`, `Full Audit Card`). Inline anomaly callout below canvas.
  - **Right (Forensic Ledger)**: Top verdict card with pulsating status dot, risk score, fraud probability, 2x2 Bento grid metrics, export action buttons, and technical telemetry drawer.
- **Fast Loading & Responsive Interaction**:
  - Lazy or pre-computed layer memoization in memory.
  - Zero layout shifts (CLS < 0.05).
  - Transition animations restricted to opacity and transforms (`0.15s ease-out`).
