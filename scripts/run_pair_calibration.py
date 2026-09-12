import os
import sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from dashboard.utils import run_screening_pipeline

test_files = [
    ("original image/passport_01_original.jpg", "tamper image/passport_01_tampered_date_edit.jpg"),
]

for orig, tamp in test_files:
    if os.path.exists(orig):
        res_orig = run_screening_pipeline(orig)
        print(f"=== {orig} ===")
        print("Keys:", res_orig.keys())
        print(f"Risk Score: {res_orig.get('risk_score')} | Decision: {res_orig.get('decision') or res_orig.get('status') or res_orig.get('classification')}")
        print(f"Risk Drivers: {res_orig.get('risk_drivers', [])}")
        print(f"Semantic Fails: {[c['check'] + ': ' + c['detail'] for c in res_orig.get('semantic_checks', []) if c.get('status') == 'FAIL']}")
        print(f"Tamper Summary: {res_orig.get('tamper_summary')}")

    if os.path.exists(tamp):
        res_tamp = run_screening_pipeline(tamp)
        print(f"=== {tamp} ===")
        print(f"Risk Score: {res_tamp['risk_score']} | Verdict: {res_tamp['verdict']}")
        print(f"Risk Drivers: {res_tamp.get('risk_drivers', [])}")
        print(f"Semantic Fails: {[c['check'] + ': ' + c['detail'] for c in res_tamp.get('semantic_checks', []) if c['status'] == 'FAIL']}")
        print(f"Tamper Summary: {res_tamp.get('tamper_summary')}")
