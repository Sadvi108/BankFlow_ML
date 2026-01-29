
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

def test_secured_pdf():
    # Find the HANA file
    receipts_dir = Path("Receipts")
    hana_files = list(receipts_dir.glob("*HANA*SECURED*.pdf"))
    
    if not hana_files:
        print("Could not find the HANA SECURED file pattern.")
        # Try generic search
        hana_files = list(receipts_dir.glob("*HANA*.pdf"))
        
    if not hana_files:
        print("No HANA files found.")
        return

    target_file = hana_files[0]
    print(f"Testing file: {target_file}")
    
    pipeline = EnhancedOCRPipeline()
    
    try:
        result = pipeline.process_file(str(target_file))
        print("Pipeline finished successfully.")
        print(f"Method: {result.get('method')}")
        print(f"Confidence: {result.get('confidence')}")
        text = result.get('text', '')
        print(f"Text length: {len(text)}")
        print(f"Preview: {text[:200]}")
        
        extraction = extract_all_fields_v3(text)
        print(f"Detected Bank: {extraction.get('bank_name')}")
        print(f"Detected ID: {extraction.get('transaction_id')}")
        
    except Exception as e:
        print(f"❌ CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_secured_pdf()
