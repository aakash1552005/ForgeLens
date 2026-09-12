import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.forensic_report import generate_unified_forensic_report

print('--- AUDITING 5 ORIGINAL IMAGES ---')
for orig_path in sorted(glob.glob('original image/passport_*_original.jpg')):
    rep = generate_unified_forensic_report(orig_path)
    dec = rep.get('decision')
    risk = rep.get('risk_score')
    att = rep.get('attack_type_guess')
    n_susp = len(rep.get('suspicious_regions', []))
    print(f"[{orig_path}] Decision={dec}, Risk={risk}, Attack={att}, Suspicious={n_susp}")
    assert dec == 'VERIFIED', f"Original failed: {orig_path} got {dec}"
    assert risk < 30.0, f"Original high risk: {orig_path} got {risk}"
    assert n_susp == 0, f"Original has suspicious regions: {n_susp}"

print('\n--- AUDITING 5 TAMPERED IMAGES ---')
# Pair 1: Date Edit
p1 = generate_unified_forensic_report('tamper image/passport_01_tampered_date_edit.jpg')
print(f"Pair 1 (Date Edit): Decision={p1['decision']}, Risk={p1['risk_score']}, Attack={p1['attack_type_guess']}")
assert p1['decision'] == 'HIGH_RISK', f"Pair 1 decision: {p1['decision']}"
assert p1['attack_type_guess'] == 'date_edit', f"Pair 1 attack: {p1['attack_type_guess']}"

# Pair 2: Text Edit
p2 = generate_unified_forensic_report('tamper image/passport_02_tampered_text_edit.jpg')
print(f"Pair 2 (Text Edit): Decision={p2['decision']}, Risk={p2['risk_score']}, Attack={p2['attack_type_guess']}")
assert p2['decision'] == 'HIGH_RISK', f"Pair 2 decision: {p2['decision']}"
assert p2['attack_type_guess'] == 'text_edit', f"Pair 2 attack: {p2['attack_type_guess']}"

# Pair 3: Cloned Stamp
p3 = generate_unified_forensic_report('tamper image/passport_03_tampered_cloned_stamp.jpg')
print(f"Pair 3 (Stamp Clone): Decision={p3['decision']}, Risk={p3['risk_score']}, Attack={p3['attack_type_guess']}")
assert p3['decision'] == 'HIGH_RISK', f"Pair 3 decision: {p3['decision']}"
assert p3['attack_type_guess'] == 'copy_move', f"Pair 3 attack: {p3['attack_type_guess']}"

# Pair 4A: Photo Swap (Autonomous Document Mode - No Selfie)
p4_auto = generate_unified_forensic_report('tamper image/passport_04_tampered_photo_swap.jpg')
print(f"Pair 4A (Photo Swap Autonomous): Decision={p4_auto['decision']}, Risk={p4_auto['risk_score']}, Attack={p4_auto['attack_type_guess']}")
assert p4_auto['decision'] == 'HIGH_RISK', f"Pair 4A decision: {p4_auto['decision']}"
assert p4_auto['attack_type_guess'] == 'photo_swap', f"Pair 4A attack: {p4_auto['attack_type_guess']}"

# Pair 4B: Photo Swap (1:1 Biometric Verification Mode - With Selfie)
p4_bio = generate_unified_forensic_report(
    'tamper image/passport_04_tampered_photo_swap.jpg',
    reference_face_path='tamper image/passport_04_selfie_genuine.jpg'
)
print(f"Pair 4B (Photo Swap Biometric): Decision={p4_bio['decision']}, Risk={p4_bio['risk_score']}, Attack={p4_bio['attack_type_guess']}")
assert p4_bio['decision'] == 'HIGH_RISK', f"Pair 4B decision: {p4_bio['decision']}"
assert p4_bio['attack_type_guess'] == 'photo_swap', f"Pair 4B attack: {p4_bio['attack_type_guess']}"

# Pair 5: Document Number Edit
p5 = generate_unified_forensic_report('tamper image/passport_05_tampered_docnum_edit.jpg')
print(f"Pair 5 (DocNum Edit): Decision={p5['decision']}, Risk={p5['risk_score']}, Attack={p5['attack_type_guess']}")
assert p5['decision'] == 'HIGH_RISK', f"Pair 5 decision: {p5['decision']}"
assert p5['attack_type_guess'] in ['text_edit', 'docnum_edit'], f"Pair 5 attack: {p5['attack_type_guess']}"

print('\n>>> ALL 10 PASSPORT TESTS (5 AUTHENTIC + 5 TAMPERED) PASSED WITH 100% FORENSIC ACCURACY! <<<')
