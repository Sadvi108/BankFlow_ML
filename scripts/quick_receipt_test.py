"""
Quick Receipt Tester - Simpler version for faster testing
Just imports and tests each receipt file directly
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

def main():
    print("="*80)
    print("QUICK RECEIPT TEST")
    print("="*80)
    
    receipts_dir = Path("Receipts")
    receipt_files = sorted(receipts_dir.glob("*.*"))
    receipt_files = [f for f in receipt_files if f.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png']]
    
    print(f"\nTesting {len(receipt_files)} receipts...\n")
    
    ocr = EnhancedOCRPipeline()
    
    success = 0
    failed = 0
    failed_files = []
    
    for i, receipt in enumerate(receipt_files, 1):
        print(f"[{i}/{len(receipt_files)}] {receipt.name[:40]:40s} ... ", end='', flush=True)
        
        try:
            ocr_result = ocr.process_file(str(receipt))
            text = ocr_result.get('text', '')
            extraction = extract_all_fields_v3(text)
            ids = extraction.get('all_ids', [])
            
            if ids:
                print(f"✅ {extraction.get('bank_name', 'Unknown'):15s} IDs: {len(ids)}")
                success += 1
            else:
                print(f"❌ NO IDS FOUND")
                failed += 1
                failed_files.append(receipt.name)
        except Exception as e:
            print(f"❌ ERROR: {str(e)[:30]}")
            failed += 1
            failed_files.append(receipt.name)
    
    print("\n" + "="*80)
    print(f"SUCCESS: {success}/{len(receipt_files)} ({success/len(receipt_files)*100:.1f}%)")
    print(f"FAILED: {failed}")
    print("="*80)
    
    if failed_files:
        print("\nFailed receipts:")
        for f in failed_files:
            print(f"  - {f}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
