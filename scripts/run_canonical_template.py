import os
import sys
sys.path.insert(0, os.path.abspath("."))
from src.document_template import generate_document
from dashboard.utils import run_screening_pipeline

# Generate document with real face
face_path = "data/face_pairs/images/pair_0001_genuine_doc.jpg"
doc_dict = generate_document(source_id="test_doc_01", seed=42, face_photo=face_path)
test_out = "scripts/canonical_test_real_face.jpg"
doc_dict["image"].save(test_out, "JPEG", quality=95, subsampling=0)

report = run_screening_pipeline(test_out)
print("=== Canonical Template with Real Face ===")
print(f"Risk Score: {report.get('risk_score')}")
print(f"Risk Tier: {report.get('risk_tier')}")
print(f"Decision: {report.get('decision')}")
print(f"Risk Drivers: {report.get('risk_drivers', [])}")
print(f"Semantic Fails: {[c['check'] + ': ' + c['detail'] for c in report.get('semantic_checks', []) if c.get('status') == 'FAIL']}")
