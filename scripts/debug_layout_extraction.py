import os
import sys
import logging
import json
from pathlib import Path
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.layout_aware_extractor import layout_extractor
from app.ultimate_patterns_v3 import extract_all_fields_v3

# Initialize ocr_pipeline locally
ocr_pipeline = EnhancedOCRPipeline()

# Setup logging to console
logging.basicConfig(level=logging.DEBUG)

def debug_file(file_path):
    print(f"\n--- DEBUGGING: {file_path} ---")
    
    # 1. OCR
    try:
        import cv2
        import numpy as np
        
        if file_path.lower().endswith('.pdf'):
            result = ocr_pipeline.process_file(file_path)
        else:
            image = cv2.imread(file_path)
            result = ocr_pipeline.extract_text_with_confidence(image)
            
        print(f"OCR Success: {result.get('processed_successfully')}")
        print(f"OCR Confidence: {result.get('confidence')}")
        print("\n--- FULL EXTRACTED TEXT ---")
        print(result.get('text', ''))
        print("---------------------------\n")
        
    except Exception as e:
        print(f"OCR CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Layout Extraction
    try:
        print("\nRunning Layout-Aware Extractor...")
        layout_results = layout_extractor.extract(result)
        print(f"Layout Success: {layout_results.get('success')}")
        print(f"Layout Result: {json.dumps(layout_results, indent=2)}")
    except Exception as e:
        print(f"LAYOUT EXTRACTOR CRASHED: {e}")
        import traceback
        traceback.print_exc()

    # 3. Patterns
    try:
        print("\nRunning Pattern Fallback...")
        pattern_results = extract_all_fields_v3(result.get('text', ''))
        print(f"Pattern Result: {json.dumps(pattern_results, indent=2)}")
    except Exception as e:
        print(f"PATTERN MATCHER CRASHED: {e}")

if __name__ == "__main__":
    # Test with a few Receipts if available
    receipts = list(Path("Receipts").glob("*.pdf")) + list(Path("Receipts").glob("*.jpg")) + list(Path("Receipts").glob("*.png"))
    
    if len(sys.argv) > 1:
        debug_file(sys.argv[1])
    elif receipts:
        # Try the first one
        debug_file(str(receipts[0]))
    else:
        print("No receipts found in Receipts/ folder.")
