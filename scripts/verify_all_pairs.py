import os
import sys
sys.path.insert(0, os.path.abspath("."))
from dashboard.utils import run_screening_pipeline

orig_files = sorted([os.path.join("original image", f) for f in os.listdir("original image") if f.endswith(".jpg")])
tamper_files = sorted([os.path.join("tamper image", f) for f in os.listdir("tamper image") if f.endswith(".jpg")])

print("=" * 80)
print("VERIFYING ORIGINAL IMAGES (EXPECTED: AUTHENTIC / LOW RISK)")
print("=" * 80)
for f in orig_files:
    r = run_screening_pipeline(f)
    fails = [c["check"] for c in r.get("semantic_checks", []) if c.get("status") == "FAIL"]
    print(f"{f:35} | Risk: {r['risk_score']:5.1f} | Decision: {r['decision']:12} | Semantic Fails: {fails}")

print("\n" + "=" * 80)
print("VERIFYING TAMPERED IMAGES (EXPECTED: TAMPERING DETECTED / HIGH RISK)")
print("=" * 80)
for f in tamper_files:
    r = run_screening_pipeline(f)
    drivers = [d["description"] for d in r.get("risk_drivers", [])][:2]
    fails = [c["check"] for c in r.get("semantic_checks", []) if c.get("status") == "FAIL"]
    print(f"{f:40} | Risk: {r['risk_score']:5.1f} | Decision: {r['decision']:12} | Drivers: {drivers} | Fails: {fails}")
