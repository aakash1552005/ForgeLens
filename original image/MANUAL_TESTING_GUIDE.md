# ForgeLens-X · Manual Testing Dataset Guide
## 5 Curated Passport Test Pairs with Real Human Portraits

Each pair contains one **Authentic Document** in `original image/` and one **Tampered Document** in `tamper image/`.

| Pair | Citizen Name | Authentic File (`original image/`) | Tampered File (`tamper image/`) | Attack Vector | Expected Authentic Score | Expected Tampered Score | Forensic Evidence Triggered |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **Pair 1** | Arun Kumar Verma | `passport_01_original.jpg` | `passport_01_tampered_date_edit.jpg` | **Date Tampering** | `VERIFIED (0.0)` | `HIGH_RISK (98.0)` | Visual date `09/05/2034` contradicts ICAO MRZ `240509` (Primary: `date_edit`) |
| **Pair 2** | Rajesh Mehra | `passport_02_original.jpg` | `passport_02_tampered_text_edit.jpg` | **Text Alteration** | `VERIFIED (0.0)` | `HIGH_RISK (98.0)` | Visual surname `Singhania` contradicts MRZ `MEHRA` (Primary: `text_edit`) |
| **Pair 3** | Siddharth Iyer | `passport_03_original.jpg` | `passport_03_tampered_cloned_stamp.jpg` | **Security Stamp Cloning** | `VERIFIED (0.0)` | `HIGH_RISK (98.0)` | ORB keypoint Copy-Move: 416 verified cluster correspondences (Primary: `copy_move`) |
| **Pair 4** | Priya Sharma | `passport_04_original.jpg` | `passport_04_tampered_photo_swap.jpg` | **Photo Swap** | `VERIFIED (0.0)` | `HIGH_RISK (95-96.0)` | Physical cut seam detected autonomously; 1:1 ArcFace mismatch when selfie uploaded (`photo_swap`) |
| **Pair 5** | Manoj Joshi | `passport_05_original.jpg` | `passport_05_tampered_docnum_edit.jpg` | **Doc Number Tampering**| `VERIFIED (0.0)` | `HIGH_RISK (98.0)` | ICAO Doc 9303 Modulo-10 checksum failure + MRZ contradiction (`text_edit`) |

---

### Step-by-Step Testing in ForgeLens-X Console (`http://localhost:8501`):

1. **Test Authentic Passports (Zero False Alarms):**
   - Click the **Upload Custom Credential** tab.
   - Drag and drop any image from `original image/` (e.g., `passport_01_original.jpg`).
   - Notice the verdict: **VERIFIED**, **Risk Index: 0.0**, **0 Suspicious Regions**, all semantic and MRZ checks pass with flying colors.

2. **Test Tampered Passports (Pinpointed Fraud):**
   - Drag and drop any image from `tamper image/` (e.g., `passport_01_tampered_date_edit.jpg`, `passport_03_tampered_cloned_stamp.jpg`, or `passport_05_tampered_docnum_edit.jpg`).
   - Notice the verdict: **HIGH RISK (96 - 98.0)**, **Manipulation Pattern Diagnosed**, with exact yellow/red highlight bounding boxes and explainable risk drivers.

3. **Testing Pair 4 (Photo Swap & Biometrics):**
   - **Mode A (Autonomous Document Forensics - No Selfie Needed):**
     - Upload `passport_04_tampered_photo_swap.jpg` alone into the document upload box.
     - ForgeLens-X detects the physical boundary cut seam and flags **HIGH RISK (96.0)**, diagnosing `photo_swap` over the portrait box.
   - **Mode B (1:1 Biometric Verification with Live Selfie):**
     - Upload `passport_04_tampered_photo_swap.jpg` into the document box.
     - Upload `passport_04_selfie_genuine.jpg` into the **Optional Live Selfie** box.
     - ArcFace SFace verifies 1:1 facial identity and flags **Biometric Mismatch (0% match, distance=0.85)** -> **HIGH RISK (95.0)**!

---

### Scaling to 100 or 1,000 Unseen Images:
- **Standardized ICAO 9303 Modulo-10 Engine**: Automatically parses and validates check digits for any standard TD1, TD2, or TD3 machine-readable credential worldwide.
- **Cross-Modal VIZ/MRZ Concordance**: Validates visual expiry, birth date, name, and document number against cryptographic MRZ text.
- **ORB-RANSAC Copy-Move**: Detects cloned motifs, stamps, and duplicated signatures on any image without requiring templates.
- **Multi-Modal Calibrated Risk Fusion**: Unifies physical, typographic, semantic, and biometric signals into a mathematically calibrated fraud probability.
