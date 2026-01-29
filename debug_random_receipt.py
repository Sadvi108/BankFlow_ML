
import sys
import os
import json
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path.cwd()))

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

def debug_receipt():
    # Pick a receipt
    filename = "3243781e-77dd-4e74-a1f3-b1ee94bc43c3_DDCONTKUL1.pdf"
    file_path = Path("Receipts") / filename
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"Processing {filename}...")
    
    pipeline = EnhancedOCRPipeline()
    
    # Run pipeline
    result = pipeline.process_file(str(file_path))
    
    print("\n--- OCR/PDF Extraction Result ---")
    print(f"Method: {result.get('method')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Text Length: {len(result.get('text', ''))}")
    print(f"Text Preview: {result.get('text', '')[:200].replace(chr(10), ' ')}")
    
    # Run Pattern Matching
    print("\n--- Pattern Matching Result ---")
    extraction = extract_all_fields_v3(result.get('text', ''))
    
    print(json.dumps(extraction, indent=2))
    
    if extraction['bank_name'] == 'Unknown':
        print("\n❌ FAILURE: Bank not detected")
    else:
        print(f"\n✅ SUCCESS: Detected Bank: {extraction['bank_name']}")

if __name__ == "__main__":
    debug_receipt()
