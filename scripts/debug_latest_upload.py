import os
import glob
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

# Find latest upload
uploads = glob.glob('data/uploads/*.pdf') + glob.glob('data/uploads/*.png') + glob.glob('data/uploads/*.jpg')
if not uploads:
    print("No uploads found.")
    exit()

latest_file = max(uploads, key=os.path.getmtime)
print(f"Analyzing latest upload: {latest_file}")

pipeline = EnhancedOCRPipeline()
result = pipeline.process_file(latest_file)
text = result['text']

print("\n--- OCR TEXT ---")
print(text)

print("\n--- EXTRACTION RESULTS ---")
extraction = extract_all_fields_v3(text)
import json
print(json.dumps(extraction, indent=2))

with open('debug_latest_cimb.txt', 'w', encoding='utf-8') as f:
    f.write(text)
