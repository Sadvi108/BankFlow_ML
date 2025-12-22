
import sys
import os
from pathlib import Path
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

# Fix encoding for Windows console
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def debug_failures():
    receipts_dir = Path("Receipts")
    failing_files = [
        "38c11069-2dc5-4e91-a37d-25cbc8518f68.pdf",
        "AFFIN BANK -CUSTOMER REF.pdf",
        "DuitNow_Transaction_Report - INVOICE.pdf",
        "PUBLIC BANK - INVOICE.pdf",
        "UOB- INV REF.pdf"
    ]
    
    ocr_pipeline = EnhancedOCRPipeline()
    
    for filename in failing_files:
        file_path = receipts_dir / filename
        if not file_path.exists():
            print(f"File not found: {filename}")
            continue
            
        print(f"\n{'='*80}")
        print(f"DEBUGGING: {filename}")
        print(f"{'='*80}")
        
        try:
            # Process file
            result = ocr_pipeline.process_file(str(file_path))
            text = result['text']
            method = result.get('method', 'unknown')
            
            print(f"Method: {method}")
            print(f"Text Length: {len(text)}")
            print("-" * 40)
            print("EXTRACTED TEXT START")
            print("-" * 40)
            print(text[:2000]) # Print first 2000 chars
            print("-" * 40)
            print("EXTRACTED TEXT END")
            print("-" * 40)
            
            # Try extraction
            extraction = extract_all_fields_v3(text)
            print(f"Bank Detected: {extraction.get('bank_name')}")
            print(f"IDs Found: {extraction.get('all_ids')}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_failures()
