
import sys
import os
import logging
from pathlib import Path

# Disable logging to stdout
logging.getLogger('app.enhanced_ocr_pipeline').setLevel(logging.ERROR)
logging.getLogger('app.ocr_pipeline').setLevel(logging.ERROR)
logging.getLogger('app.ultimate_patterns_v3').setLevel(logging.ERROR)

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline

def debug_to_file():
    receipts_dir = Path("Receipts")
    failing_files = [
        "38c11069-2dc5-4e91-a37d-25cbc8518f68.pdf",
        "AFFIN BANK -CUSTOMER REF.pdf",
        "DuitNow_Transaction_Report - INVOICE.pdf",
        "PUBLIC BANK - INVOICE.pdf",
        "UOB- INV REF.pdf"
    ]
    
    ocr_pipeline = EnhancedOCRPipeline()
    
    with open("debug_output.txt", "w", encoding="utf-8") as f:
        for filename in failing_files:
            file_path = receipts_dir / filename
            if not file_path.exists():
                f.write(f"File not found: {filename}\n")
                continue
                
            f.write(f"\n{'='*80}\n")
            f.write(f"FILE: {filename}\n")
            f.write(f"{'='*80}\n")
            
            try:
                # Process file
                result = ocr_pipeline.process_file(str(file_path))
                text = result['text']
                method = result.get('method', 'unknown')
                
                f.write(f"Method: {method}\n")
                f.write(f"Text Length: {len(text)}\n")
                f.write("-" * 40 + "\n")
                f.write(text)
                f.write("\n" + "-" * 40 + "\n")
                
            except Exception as e:
                f.write(f"Error: {e}\n")

if __name__ == "__main__":
    debug_to_file()
