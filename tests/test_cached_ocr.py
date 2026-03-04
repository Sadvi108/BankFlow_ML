
import sys
import os
from pathlib import Path
import glob

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.ultimate_patterns_v3 import extract_all_fields_v3

def test_on_cached_files():
    ocr_dir = Path("debug_ocr_output")
    files = list(ocr_dir.glob("*_ocr.txt"))
    
    with open("cached_results.txt", "w", encoding="utf-8") as out:
        out.write(f"Testing extraction on {len(files)} cached OCR files...\n\n")
        
        passed = 0
        failed = 0
        
        for file_path in files:
            out.write(f"File: {file_path.name}\n")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract fields
                result = extract_all_fields_v3(content)
                
                # Check for fields
                has_date = result['date'] is not None
                has_amount = result['amount'] is not None
                has_ids = result['all_ids'] and len(result['all_ids']) > 0
                
                status = []
                if has_date: status.append(f"Date: OK ({result['date']})")
                else: status.append(f"Date: MISSING ({result['date']})")
                
                if has_amount: status.append(f"Amount: OK ({result['amount']})")
                else: status.append(f"Amount: MISSING ({result['amount']})")
                
                if has_ids: status.append(f"IDs: OK ({result['all_ids']})")
                else: status.append("IDs: MISSING")
                
                out.write(f"  {' | '.join(status)}\n")
                
                if has_date and has_amount and has_ids:
                    out.write("  SUCCESS\n")
                    passed += 1
                else:
                    out.write("  FAILURE\n")
                    failed += 1
                    
            except Exception as e:
                out.write(f"  ERROR: {e}\n")
                failed += 1
            out.write("-" * 50 + "\n")

        out.write(f"\nSummary: {passed} Passed, {failed} Failed\n")
        print(f"Results written to cached_results.txt (Passed: {passed}, Failed: {failed})")

if __name__ == "__main__":
    test_on_cached_files()
