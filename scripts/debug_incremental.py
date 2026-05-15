
import os
import sys
import glob
import json
from pathlib import Path
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

def test_incremental():
    receipts_dir = Path("Receipts")
    files = sorted(list(receipts_dir.glob("*.pdf"))) + sorted(list(receipts_dir.glob("*.jpg"))) + sorted(list(receipts_dir.glob("*.jpeg")))
    
    print(f"Found {len(files)} files to test.")
    
    ocr_pipeline = EnhancedOCRPipeline()
    results = []
    
    for i, file_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing {file_path.name}...", end='', flush=True)
        try:
            # Process
            res = ocr_pipeline.process_file(str(file_path))
            text = res.get('text', '')
            
            # Extract
            extraction = extract_all_fields_v3(text)
            ids = extraction.get('all_ids', [])
            bank = extraction.get('bank_name', 'Unknown')
            
            status = "SUCCESS" if ids else "FAILED"
            print(f" {status}")
            
            # Save failing result immediately/details
            result_entry = {
                "file": file_path.name,
                "status": status,
                "ids": ids,
                "bank": bank,
                "text_snippet": text[:200] if text else ""
            }
            results.append(result_entry)
            
            # write to file incrementally
            with open("incremental_results.json", "w") as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f" ERROR: {str(e)}")
            results.append({
                "file": file_path.name,
                "status": "ERROR",
                "error": str(e)
            })

    # Summary
    failed = [r for r in results if r['status'] != 'SUCCESS']
    print("\n\nFAILURES:")
    for f in failed:
        print(f"- {f['file']} (Bank: {f.get('bank')})")

if __name__ == "__main__":
    test_incremental()
