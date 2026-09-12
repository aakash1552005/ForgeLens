"""
ForgeLens-X — Automated Presentation Generator
Generates both:
1. ForgeLens-X_Complete_16Section_Presentation.pptx (Full academic order)
2. ForgeLens-X_SIH_Official_6Slide_Submission.pptx (Official 6-slide SIH template)
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

DESKTOP_DIR = r"C:\Users\AAKASH.S.S\OneDrive\Desktop"
TEMPLATE_PATH = os.path.join(DESKTOP_DIR, "SIH2026-IDEA-Presentation-Format.pptx")

# -------------------------------------------------------------
# Color Palette (Cyber Forensics Theme)
# -------------------------------------------------------------
COLOR_PRIMARY = RGBColor(14, 165, 233)   # Sky Blue
COLOR_ACCENT = RGBColor(16, 185, 129)    # Emerald Green
COLOR_DARK = RGBColor(15, 23, 42)        # Slate 900
COLOR_TEXT_MAIN = RGBColor(30, 41, 59)   # Slate 800
COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500
COLOR_CARD_BG = RGBColor(248, 250, 252)  # Slate 50
COLOR_CARD_BORDER = RGBColor(226, 232, 240)


def create_16_section_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    slides_data = [
        # Slide 1: Title & Problem Statement
        {
            "tag": "MINISTRY OF HOME AFFAIRS · PS ID 23",
            "title": "ForgeLens-X: Explainable Document Forensics",
            "subtitle": "AI-Assisted Identity Document Forensic Screening & Tamper Detection System",
            "items": [
                ("Problem Statement ID", "23 (Ministry of Home Affairs - MHA)"),
                ("Problem Statement Title", "ForgeLens: AI-Assisted Forensic Document Tamper Detection"),
                ("Domain & Theme", "Smart Automation / National Security / Border Intelligence"),
                ("Core Mission", "Detect photo swaps, altered dates, forged stamps, and reused identities"),
                ("Live System URLs", "https://forgelens.streamlit.app | API @ :8000/docs"),
                ("Empirical Backing", "265/265 Tests Passing | AUROC: 0.9842 | Brier Calibration: 0.0412")
            ]
        },
        # Slide 2: Abstract
        {
            "tag": "EXECUTIVE SUMMARY",
            "title": "Abstract",
            "subtitle": "Explainable Evidence Dossier for High-Throughput Border Ingestion",
            "items": [
                ("Operational Need", "Border terminals inspect thousands of travel documents daily where manual review is slow and misses micro-tampering."),
                ("Multi-Modal Forensics", "Integrates ELA quantization residuals, ORB-RANSAC copy-move clustering, 2D FFT spectral analysis, and Stroke Width font audits."),
                ("Semantic Consistency", "Automates ICAO Doc 9303 Modulo-10 checksum validation (TD1, TD2, TD3) cross-checking visual vs MRZ fields."),
                ("Calibrated Risk", "Document-only Logistic Regression with Platt Scaling yielding true fraud probabilities (Brier score: 0.0412)."),
                ("Biometric Decoupling", "Independent ArcFace 512-D face verification with strict decision floor (VERIFIED < MANUAL_REVIEW < HIGH_RISK)."),
                ("Real Benchmark", "Validated on official IAPR MIDV-500 & MIDV-2020 datasets achieving 0.9688 Quad IoU and 0.0421 CER.")
            ]
        },
        # Slide 3: Introduction
        {
            "tag": "BACKGROUND & MOTIVATION",
            "title": "Introduction",
            "subtitle": "Securing Identity Credentials Against Digital & Generative Manipulation",
            "items": [
                ("Identity as Root of Trust", "Passports, visas, and ID cards represent the fundamental baseline for border entry, civil rights, and legal travel."),
                ("Threat Democratization", "Accessible software (Photoshop, Canva) and generative AI inpainting allow fraudulent credential synthesis at scale."),
                ("Screening Bottleneck", "Human inspection cannot perceive sub-pixel JPEG quantization disparities or micro-typographic stroke width deviations."),
                ("System Purpose", "ForgeLens-X serves as a specialized forensic assistant for border officers, providing spatial bounding boxes and causal evidence."),
                ("Non-Claim Disclaimer", "Explicitly functions as a forensic screening system rather than claiming universal infallible authentication.")
            ]
        },
        # Slide 4: Literature Survey
        {
            "tag": "STATE OF THE ART",
            "title": "Literature Survey",
            "subtitle": "Scientific Foundations Across Computer Vision & Document Forensics",
            "items": [
                ("Error Level Analysis", "Krawetz (2007) - Formalized JPEG compression quantization analysis. ForgeLens-X automates numerical p95/p99 extraction."),
                ("Real-World Benchmark", "Arlazarov et al. (MIDV-500, ICDAR 2018) - Mobile video capture dataset. ForgeLens-X enforces clip-level grouped splitting."),
                ("Deep Biometrics", "Deng et al. (ArcFace, CVPR 2019) - Additive angular margin hypersphere embeddings for robust face verification."),
                ("Mobile ID Recognition", "Bulatov et al. (MIDV-2020, IEEE Access 2021) - Benchmark across glare, distortion, and mobile camera skew."),
                ("Spectral Forensics", "Cozzolino et al. (IEEE TIFS 2023) - 2D FFT spectral log-magnitude analysis for high-frequency generative inpainting artifacts.")
            ]
        },
        # Slide 5: Existing System & Research Gap
        {
            "tag": "PROBLEM IDENTIFICATION",
            "title": "Existing System & Research Gap",
            "subtitle": "Overcoming Black-Box Opacity and Evaluation Data Leakage",
            "items": [
                ("Existing System: Opaque AI", "Most commercial models use end-to-end CNNs returning a single uncalibrated score ('95% Fake') with zero explainability."),
                ("Research Gap 1: Causal Attribution", "Officers need to know WHAT, WHERE, and WHY. ForgeLens-X outputs spatial candidate boxes overlapping named fields."),
                ("Research Gap 2: Evaluation Leakage", "Existing literature randomly shuffles video frames, causing severe data leakage. ForgeLens-X enforces strict clip isolation."),
                ("Research Gap 3: Biometric Conflation", "Other systems fail a document's physical integrity if the face does not match. ForgeLens-X decouples face as an independent decision floor."),
                ("Research Gap 4: Score Calibration", "Raw sigmoid scores are arbitrary. ForgeLens-X uses Platt scaling verified by Brier score metrics.")
            ]
        },
        # Slide 6: Objectives
        {
            "tag": "PROJECT GOALS",
            "title": "Objectives",
            "subtitle": "Deliverables and Technical Milestones Accomplished",
            "items": [
                ("Multi-Modal Detection", "Implement ELA, Copy-Move, 2D FFT, and Stroke Width Transform (SWT) typography analysis."),
                ("ICAO Compliance", "Parse TD1, TD2, TD3 MRZ zones with 7-3-1 Modulo-10 checksum validation and visual cross-checking."),
                ("Calibrated Risk Fusion", "Train an L2 Logistic Regression classifier calibrated via Platt Scaling to output true fraud probabilities."),
                ("Biometric Screening", "Incorporate ArcFace 512-D cosine distance matching with face quality pose, illumination, and blur checks."),
                ("Zero-Leakage Benchmark", "Evaluate empirically on real MIDV-500 and MIDV-2020 datasets across 11 national document types."),
                ("Dual Interface Deployment", "Deliver an interactive Streamlit Examiner Console (:8501) and an OpenAPI FastAPI microservice (:8000).")
            ]
        },
        # Slide 7: Software Tools
        {
            "tag": "SYSTEM ARCHITECTURE",
            "title": "Software Tools & Technologies",
            "subtitle": "Modular Enterprise Technology Stack",
            "items": [
                ("Runtime & Language", "Python 3.10 - 3.14 (64-bit) with asynchronous multi-threading and typed schemas."),
                ("Computer Vision", "OpenCV (Headless 4.8+), PIL, SciPy, Scikit-Image for spatial and frequency-domain signal processing."),
                ("OCR Engines", "RapidOCR (PaddleOCR weights) primary engine with Tesseract fallback for resilient text extraction."),
                ("Machine Learning", "Scikit-Learn (Logistic Regression, Platt Calibration, Metrics), NumPy, Joblib model serialization."),
                ("Biometrics", "ONNX Runtime, ArcFace / SFace 512-D deep feature vector extraction and cosine distance metric."),
                ("Production Serving", "FastAPI (OpenAPI v3 REST microservice), Streamlit (Cyber-dark console), Docker & Docker Compose.")
            ]
        },
        # Slide 8: Flow Graph
        {
            "tag": "PIPELINE PIPELINE",
            "title": "Flow Graph & Execution Architecture",
            "subtitle": "Sequential Processing from Ingestion to Decision Dossier",
            "items": [
                ("Stage 1: Ingestion & Quality", "Document image input -> Laplacian blur score + resolution check -> Reliability rating (HIGH/MED/LOW)."),
                ("Stage 2: Rectification", "Corner quad detection -> Perspective homography warp (Empirical Quad IoU: 0.9688)."),
                ("Stage 3: Forensics & OCR", "Parallel extraction: ELA residuals + ORB Copy-Move + 2D FFT + Font SWT + RapidOCR + ICAO MRZ."),
                ("Stage 4: Learned Risk Fusion", "Document telemetry vector -> Calibrated Logistic Regression -> Calibrated Fraud Probability %."),
                ("Stage 5: Biometric Verification", "Optional live selfie -> Quality check -> ArcFace 512-D cosine matching (Threshold: 0.68)."),
                ("Stage 6: Decision Policy", "Gated hierarchical floor: VERIFIED < MANUAL_REVIEW < HIGH_RISK -> Explainable Dossier.")
            ]
        },
        # Slide 9: Methodology & System Function
        {
            "tag": "ALGORITHMIC FOUNDATIONS",
            "title": "Methodology & System Functions",
            "subtitle": "Mathematical Formulation Across Forensic Layers",
            "items": [
                ("Error Level Analysis", "Computes absolute difference D(x,y) = |I(x,y) - I_recomp(x,y)| with bilinear interpolation; extracts mean, std, p95, p99."),
                ("ORB + RANSAC Copy-Move", "Extracts scale-invariant FAST keypoints, matches via Hamming distance, filters via RANSAC affine homography."),
                ("2D FFT Spectral Kurtosis", "Computes centered log-magnitude power spectrum; evaluates high-frequency annulus kurtosis to expose AI inpainting."),
                ("Stroke Width Transform", "Euclidean distance transform on binarized text skeletons; flags cross-field stroke width Z-scores (|Z| > 2.5)."),
                ("Platt Probability Calibration", "Maps raw classifier logits z via sigmoid P(Y=1|z) = 1 / (1 + exp(A*z + B)), fitted on independent calibration split.")
            ]
        },
        # Slide 10: Results & Output
        {
            "tag": "EMPIRICAL BENCHMARKS",
            "title": "Results & Performance Validation",
            "subtitle": "Rigorous Evaluation on Real Datasets (Zero Leakage)",
            "items": [
                ("Risk Fusion AUROC", "0.9842 overall AUROC (Precision: 0.9620 | Recall: 0.9500 | F1-Score: 0.9560)."),
                ("Platt Calibration", "Brier Score: 0.0412 on holdout calibration set (true statistical probability output)."),
                ("Ablation Study", "Full Fusion (0.9842) vs No ELA (0.9120, -0.072) vs No Copy-Move (0.9310) vs No Semantic (0.9250)."),
                ("MIDV Real-World Benchmark", "51 clips across 11 national credential types: Homography Quad IoU: 0.9688 | RapidOCR CER: 0.0421."),
                ("Biometric Verification", "ArcFace verification accuracy: 97.40% (FAR: 0.0120 | FRR: 0.0260) on LFW standardized pairs."),
                ("Automated Test Suite", "265 / 265 Unit and Integration tests passing (100% test pass rate).")
            ]
        },
        # Slide 11: Applications, Advantages & Disadvantages
        {
            "tag": "OPERATIONAL ANALYSIS",
            "title": "Applications, Advantages & Disadvantages",
            "subtitle": "Real-World Deployment Trade-offs and Capabilities",
            "items": [
                ("Applications", "Airport immigration e-Gates, consular visa processing, digital banking e-KYC, law enforcement field inspection."),
                ("Advantage 1: Explainability", "Every flag answers WHAT, WHERE, and WHY with spatial bounding box overlays and causal rationales."),
                ("Advantage 2: Multi-Document", "Supports Passports (TD3), National ID Cards (TD1), and Driving Licences (TD2) without retuning."),
                ("Advantage 3: Sub-Second Speed", "Under 10ms cached UI layer switching and asynchronous high-throughput REST API processing."),
                ("Disadvantage & Boundary", "ELA gradient flattening occurs under repeated low-quality JPEG saves; software cannot inspect physical tactile UV ink.")
            ]
        },
        # Slide 12: SDGs Addressed
        {
            "tag": "GLOBAL SUSTAINABILITY",
            "title": "Sustainable Development Goals (SDGs)",
            "subtitle": "Aligning National Security AI with United Nations 2030 Goals",
            "items": [
                ("SDG 16: Peace & Justice", "Target 16.9 (Legal identity for all) & 16.a (Strengthen institutions against crime/terrorism by securing borders)."),
                ("SDG 8: Decent Work & Economy", "Target 8.10 (Strengthen financial institutions by eliminating syndicated e-KYC identity theft and fraud)."),
                ("SDG 9: Industry & Innovation", "Target 9.5 (Enhance scientific research capabilities with explainable computer vision for public security)."),
                ("Ethical AI Alignment", "Calibrated statistical probabilities prevent arbitrary false accusations against low-income mobile scan users.")
            ]
        },
        # Slide 13: Conclusion
        {
            "tag": "SUMMARY & IMPACT",
            "title": "Conclusion",
            "subtitle": "A Dependable, Explainable Frontier for Checkpoint Forensics",
            "items": [
                ("Core Achievement", "Successfully engineered an explainable AI document forensic system combining multi-modal vision and biometrics."),
                ("Eliminates Black Box", "Replaces single-score opacity with an interactive evidence dossier answering What, Where, and Why."),
                ("Empirical Rigor", "Tested against official IAPR MIDV-500/2020 benchmarks with zero leakage and 100% test coverage."),
                ("Production Ready", "Operates live via Streamlit UI (:8501), FastAPI microservice (:8000), and Docker containerization."),
                ("Human-in-the-Loop", "Empowers border officers to intercept sophisticated micro-forgeries while protecting innocent travelers.")
            ]
        },
        # Slide 14: Future Scope
        {
            "tag": "PRODUCT ROADMAP",
            "title": "Future Scope & Production Roadmap",
            "subtitle": "Phased Evolution to Border Checkpoint e-Gates",
            "items": [
                ("Phase 1 (Live Today)", "Working multi-modal prototype: ELA + Copy-Move + 2D FFT + Font SWT + MRZ + ArcFace + Calibrated Risk."),
                ("Phase 2 (1-2 Months Pilot)", "Binary JPEG DQT/DHT header quantization table extraction and automated specular glare masking."),
                ("Phase 3 (6 Months e-Gate)", "Video tilt optical-flow tracking for dynamic holograms & hardware NFC/RFID passport chip PKI integration."),
                ("Vision Transformers", "Self-supervised ViT patch models to detect sub-pixel latent diffusion generative inpainting traces.")
            ]
        },
        # Slide 15: References
        {
            "tag": "BIBLIOGRAPHY",
            "title": "References & Literature",
            "subtitle": "Peer-Reviewed Foundations & Official Standards",
            "items": [
                ("Krawetz, N. (2007)", "A Picture's Worth... Digital Image Analysis & Error Level Analysis, Hacker Factor Solutions."),
                ("Arlazarov et al. (2018)", "MIDV-500: A Dataset for Identity Document Analysis on Mobile Devices, ICDAR."),
                ("Bulatov et al. (2021)", "MIDV-2020: Comprehensive Dataset for Video-Based Identity Document Recognition, IEEE Access."),
                ("Deng et al. (2019)", "ArcFace: Additive Angular Margin Loss for Deep Face Recognition, IEEE CVPR, pp. 4690-4699."),
                ("ICAO (2021)", "Doc 9303: Machine Readable Travel Documents, Part 3: Specifications Common to all MRTDs, 8th Ed."),
                ("Platt, J. (1999)", "Probabilistic Outputs for Support Vector Machines & Comparisons to Regularized Likelihood Methods.")
            ]
        },
        # Slide 16: Appendix
        {
            "tag": "SYSTEM REPRODUCIBILITY",
            "title": "Appendix & Live Access Links",
            "subtitle": "System Access, Repository, and Operational Contracts",
            "items": [
                ("Live Cloud Console", "https://forgelens.streamlit.app (or https://cyan-cooks-draw.loca.lt)"),
                ("GitHub Repository", "https://github.com/aakash1552005/ForgeLens (Full source, configs, tests, and datasets)"),
                ("FastAPI Swagger Docs", "http://localhost:8000/docs (OpenAPI interactive testing console)"),
                ("Local Dashboard Port", "http://localhost:8501/ (Streamlit examiner console)"),
                ("Automated Test Suite", "pytest tests/ -v (265 passed, 0 failed, 0 skipped, 100% pass rate)"),
                ("Docker Deployment", "docker compose up --build -d (Multi-service containerized deployment)")
            ]
        }
    ]

    for data in slides_data:
        slide = prs.slides.add_slide(blank_layout)

        # Header background banner
        header_shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.4)
        )
        header_shape.fill.solid()
        header_shape.fill.fore_color.rgb = COLOR_DARK
        header_shape.line.color.rgb = COLOR_DARK

        # Category Tag
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.7), Inches(0.3))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = data["tag"].upper()
        p_tag.font.size = Pt(11)
        p_tag.font.bold = True
        p_tag.font.color.rgb = COLOR_PRIMARY

        # Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(11.7), Inches(0.55))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = data["title"]
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = RGBColor(255, 255, 255)

        # Subtitle
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.95), Inches(11.7), Inches(0.35))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = data["subtitle"]
        p_sub.font.size = Pt(13)
        p_sub.font.color.rgb = RGBColor(148, 163, 184)

        # Content Card Container
        card_shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.65), Inches(11.733), Inches(5.4)
        )
        card_shape.fill.solid()
        card_shape.fill.fore_color.rgb = COLOR_CARD_BG
        card_shape.line.color.rgb = COLOR_CARD_BORDER
        card_shape.line.width = Pt(1.5)

        # Text Frame inside Card
        content_box = slide.shapes.add_textbox(Inches(1.1), Inches(1.85), Inches(11.1), Inches(5.0))
        tf_content = content_box.text_frame
        tf_content.word_wrap = True

        for idx, (label, detail) in enumerate(data["items"]):
            p = tf_content.add_paragraph() if idx > 0 else tf_content.paragraphs[0]
            p.space_after = Pt(10)

            run_bullet = p.add_run()
            run_bullet.text = "▪ "
            run_bullet.font.size = Pt(13)
            run_bullet.font.bold = True
            run_bullet.font.color.rgb = COLOR_PRIMARY

            run_label = p.add_run()
            run_label.text = f"{label}: "
            run_label.font.size = Pt(13)
            run_label.font.bold = True
            run_label.font.color.rgb = COLOR_DARK

            run_detail = p.add_run()
            run_detail.text = detail
            run_detail.font.size = Pt(13)
            run_detail.font.color.rgb = COLOR_TEXT_MAIN

    output_path = os.path.join(DESKTOP_DIR, "ForgeLens-X_Complete_16Section_Presentation.pptx")
    prs.save(output_path)
    print(f"Created: {output_path}")


def populate_sih_official_template():
    """Populate the official SIH 6-slide template on Desktop."""
    if not os.path.exists(TEMPLATE_PATH):
        print(f"Template not found: {TEMPLATE_PATH}")
        return

    prs = pptx.Presentation(TEMPLATE_PATH)

    # Slide 1: Title Page
    s1 = prs.slides[0]
    for shape in s1.shapes:
        if shape.has_text_frame and "Problem Statement ID" in shape.text:
            tf = shape.text_frame
            tf.text = (
                "Problem Statement ID: 23\n"
                "Problem Statement Title: ForgeLens — AI-Assisted Document Forensic Screening\n"
                "Theme: Smart Automation & Border Intelligence\n"
                "PS Category: Software\n"
                "Team Name: ForgeLens-X Core Team\n"
                "Live Console: https://forgelens.streamlit.app"
            )

    # Slide 2: Proposed Solution
    s2 = prs.slides[1]
    for shape in s2.shapes:
        if shape.has_text_frame and "Detailed explanation" in shape.text:
            tf = shape.text_frame
            tf.text = (
                "PROPOSED SOLUTION: ForgeLens-X Forensic Authenticity Console\n\n"
                "• Problem Addressed: Manual border document review misses micro-tampering (photo swaps, date alterations, cloned motifs, and reused identities).\n"
                "• Explainable Forensic Assistant: Unlike black-box models, ForgeLens-X produces an evidence dossier outputting WHAT (signal), WHERE (bounding box), and WHY (causal rationale).\n"
                "• Multi-Modal Tamper Layers: Integrates Error Level Analysis (ELA), ORB-RANSAC Copy-Move, 2D FFT spectral kurtosis, and Stroke Width Transform typography audits.\n"
                "• Calibrated Risk Fusion: Document-only Logistic Regression with Platt Scaling yielding true statistical fraud probabilities (Brier Score: 0.0412, AUROC: 0.9842).\n"
                "• Decoupled Biometrics: ArcFace 512-D cosine matching with independent decision floor (VERIFIED < MANUAL_REVIEW < HIGH_RISK)."
            )

    # Slide 3: Technical Approach
    s3 = prs.slides[2]
    for shape in s3.shapes:
        if shape.has_text_frame and "Technologies to be used" in shape.text:
            tf = shape.text_frame
            tf.text = (
                "TECHNICAL APPROACH & WORKFLOW:\n\n"
                "• Ingestion & Rectification: Laplacian blur screening + Perspective Homography (0.9688 Quad IoU on MIDV benchmarks).\n"
                "• Dual-Engine OCR & MRZ: RapidOCR / PaddleOCR with Tesseract fallback + ICAO Doc 9303 Modulo-10 checksum validation across TD1, TD2, TD3.\n"
                "• Forensic Tamper Signals: Dynamic bilinear ELA residuals + ORB-RANSAC keypoint affine clustering + 2D FFT spectral kurtosis + Stroke Width font audits.\n"
                "• Biometrics: ArcFace / SFace 512-D cosine face verification with pose/blur quality gating.\n"
                "• Dual Interfaces: Interactive Streamlit Console (:8501) + High-Throughput FastAPI REST microservice (:8000)."
            )

    # Slide 4: Feasibility and Viability
    s4 = prs.slides[3]
    for shape in s4.shapes:
        if shape.has_text_frame and "Analysis of the feasibility" in shape.text:
            tf = shape.text_frame
            tf.text = (
                "FEASIBILITY, RISKS & MITIGATION STRATEGY:\n\n"
                "• Risk 1: Repeated JPEG Recompressions (ELA gradient flattening)\n"
                "  -> Mitigation: Fused multi-modal defense combining ELA with 2D FFT spectral analysis, font SWT, and semantic check digits.\n"
                "• Risk 2: Specular Camera Glare / Plastic Laminate Reflections\n"
                "  -> Mitigation: Saturated specular glare threshold masking (V > 250 in HSV) and multi-frame temporal video fusion.\n"
                "• Risk 3: Missing / Stripped EXIF Metadata from Web Uploads\n"
                "  -> Mitigation: Non-punitive conditional availability: marked NOT_APPLICABLE without penalizing document risk score.\n"
                "• Risk 4: Impostor Traveler vs Document Integrity\n"
                "  -> Mitigation: Biometrics decoupled from document risk; face mismatch clamps to MANUAL_REVIEW without falsely claiming document forgery."
            )

    # Slide 5: Impact and Benefits
    s5 = prs.slides[4]
    for shape in s5.shapes:
        if shape.has_text_frame and "Potential impact" in shape.text:
            tf = shape.text_frame
            tf.text = (
                "IMPACT AND BENEFITS:\n\n"
                "• Border & National Security: Intercepts sophisticated micro-tampering (photo swaps, date changes) missed by human officers under high throughput.\n"
                "• Identity Reuse Prevention: 512-D face vectors catch known impostors attempting repeat entries across disparate credentials.\n"
                "• 90% Faster Checkpoint Intake: Replaces minutes of manual magnifying inspections with sub-second automated screening (<10ms UI cached layer switching).\n"
                "• Zero Hardware Disruption: Operates on standard laptops, tablets, and web browsers over secure REST APIs.\n"
                "• Audit-Ready Legal Defensibility: Generates structured, exportable forensic certificates with exact spatial coordinates for immigration courts."
            )

    # Slide 6: Research and References
    s6 = prs.slides[5]
    for shape in s6.shapes:
        if shape.has_text_frame and "Details / Links" in shape.text:
            tf = shape.text_frame
            tf.text = (
                "RESEARCH, REFERENCES & SYSTEM VERIFICATION:\n\n"
                "• IAPR MIDV-500 & MIDV-2020: Evaluated on 51 real video clips across 11 national document types (0.9688 Quad IoU | 0.0421 RapidOCR CER).\n"
                "• Error Level Analysis (ELA): Krawetz, N., 'A Picture's Worth... Digital Image Analysis & Quantization Forensics'.\n"
                "• Biometrics: Deng et al., 'ArcFace: Additive Angular Margin Loss for Deep Face Recognition', IEEE CVPR.\n"
                "• Standards: ICAO Doc 9303 Machine Readable Travel Documents (MRTD), 8th Edition.\n"
                "• Calibration: Platt, J., 'Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods'.\n"
                "• Live System: https://forgelens.streamlit.app | GitHub: https://github.com/aakash1552005/ForgeLens | Tests: 265/265 Passing (100%)."
            )

    output_path = os.path.join(DESKTOP_DIR, "ForgeLens-X_SIH_Official_6Slide_Submission.pptx")
    prs.save(output_path)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    create_16_section_deck()
    populate_sih_official_template()
