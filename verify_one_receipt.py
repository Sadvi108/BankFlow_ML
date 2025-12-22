
import sys
import os
from pathlib import Path

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

def test_real_receipt():
    # Pick a specific receipt
    receipt_name = "470d7184-a47c-4c29-bb2d-4bea83576eb5_11739919 payment slip .pdf"
    receipt_path = Path("Receipts") / receipt_name
    
    if not receipt_path.exists():
        # Fallback to finding any PDF
        pdfs = list(Path("Receipts").glob("*.pdf"))
        if pdfs:
            receipt_path = pdfs[0]
        else:
            print("No receipts found!")
            return

    print(f"Testing real receipt: {receipt_path}")
    
    pipeline = EnhancedOCRPipeline()
    result = pipeline.process_file(str(receipt_path))
    
    with open("verify_result.txt", "w", encoding="utf-8") as f:
        f.write(f"OCR Confidence: {result['confidence']:.2%}\n")
        f.write(f"Text length: {len(result['text'])}\n")
        
        extraction = extract_all_fields_v3(result['text'])
        
        f.write("\nExtraction Results:\n")
        f.write(f"  Bank: {extraction['bank_name']}\n")
        f.write(f"  Transaction IDs: {extraction['transaction_ids']}\n")
        f.write(f"  Date: {extraction['date']}\n")
        f.write(f"  Amount: {extraction['amount']}\n")
        
        if extraction['date'] and extraction['amount']:
            f.write("\nSUCCESS: Date and Amount extracted!\n")
        else:
            f.write("\nFAILURE: Missing fields.\n")
            f.write("Text dump:\n")
            f.write(result['text'])
            
    print("Verification complete. Results written to verify_result.txt")

if __name__ == "__main__":
    test_real_receipt()
