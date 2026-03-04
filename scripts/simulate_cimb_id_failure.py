from app.ultimate_patterns_v3 import extract_all_fields_v3
import json

# Real OCR text from the latest upload
with open('debug_latest_cimb.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Testing extraction on REAL CIMB OCR text...")
results = extract_all_fields_v3(text)
print("\n--- RESULTS ---")
print(json.dumps(results, indent=2))
