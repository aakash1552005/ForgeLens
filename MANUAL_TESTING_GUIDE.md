# ForgeLens-X · Manual Testing Dataset Guide
## 5 Curated Passport Test Pairs with Real Human Portraits

Each pair contains one **Authentic Document** in `original image/` and one **Tampered Document** in `tamper image/`.

| Pair | Citizen Name | Original File (`original image/`) | Tampered File (`tamper image/`) | Specific Attack Vector | Forensic Evidence Triggered |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Pair 1** | Arun Kumar Verma | `passport_01_original.jpg` | `passport_01_tampered_date_edit.jpg` | **Date Tampering** (10-yr extension) | ELA compression residue discontinuity + ICAO MRZ Expiry mismatch |
| **Pair 2** | Rajesh Mehra | `passport_02_original.jpg` | `passport_02_tampered_text_edit.jpg` | **Text Alteration** (Surname changed) | Typographic stroke-width anomaly + MRZ Name concordance failure |
| **Pair 3** | Siddharth Iyer | `passport_03_original.jpg` | `passport_03_tampered_cloned_stamp.jpg` | **Security Stamp Cloning** | ORB keypoint Copy-Move clustered vector correspondences |
| **Pair 4** | Priya Sharma | `passport_04_original.jpg` | `passport_04_tampered_photo_swap.jpg` | **Photo Swap** (Impostor portrait) | Splicing edge discontinuity + ArcFace biometric verification failure |
| **Pair 5** | Manoj Joshi | `passport_05_original.jpg` | `passport_05_tampered_docnum_edit.jpg` | **Document Number Tampering** | ICAO Modulo-10 checksum failure + VIZ-to-MRZ Document No. mismatch |

---

### How to Test in the ForgeLens-X Console:
1. Open your browser at `http://localhost:8501`.
2. In the top screening banner, click on the **Upload Document** tab.
3. Drag and drop any image from `original image/` to see the **Authentic / Verified** verdict and clean forensic layers.
4. Drag and drop the corresponding image from `tamper image/` to see the **Tampering Detected / High Risk** verdict with highlighted bounding boxes, thermal heatmaps, and plain-language driver explanations!
