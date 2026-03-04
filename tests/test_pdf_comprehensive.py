#!/usr/bin/env python3
"""
Test PDF processing with various PDF files to ensure comprehensive functionality
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.ocr_enhanced import EnhancedOCRProcessor
from app.main_enhanced import extract_receipt_content
from app.ultimate_patterns_v2 import UltimatePatternMatcherV2

def test_multiple_pdfs():
    """Test PDF processing with various PDF files."""
    print("Testing PDF processing with multiple files...")
    
    # Find PDF files in Receipts directory
    receipts_dir = Path(__file__).parent / "Receipts"
    pdf_files = list(receipts_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in Receipts directory")
        return False
    
    print(f"Found {len(pdf_files)} PDF files")
    
    processor = EnhancedOCRProcessor()
    matcher = UltimatePatternMatcherV2()
    
    success_count = 0
    total_tested = min(5, len(pdf_files))  # Test up to 5 files
    
    for i, test_pdf in enumerate(pdf_files[:total_tested]):
        print(f"\n--- Testing file {i+1}: {test_pdf.name} ---")
        
        try:
            # Test the complete pipeline
            file_id = f"test_pdf_{i}"
            text, tokens, processed_path = extract_receipt_content(test_pdf, file_id)
            print(f"✅ Pipeline successful: {len(text)} characters, {len(tokens)} tokens")
            
            if text.strip():
                # Test pattern extraction
                result = matcher.extract_all_fields(text)
                print(f"📊 Extraction results:")
                print(f"  Bank: {result.get('bank_name', 'Unknown')}")
                print(f"  Transaction IDs: {result.get('transaction_ids', [])}")
                print(f"  Reference Numbers: {result.get('reference_numbers', [])}")
                print(f"  Amounts: {result.get('amounts', [])}")
                print(f"  Confidence: {result.get('confidence', 0)}")
                
                # Check if we got meaningful results
                if (result.get('bank_name') != 'Unknown' or 
                    result.get('transaction_ids') or 
                    result.get('reference_numbers') or
                    result.get('amounts')):
                    success_count += 1
                    print("✅ Meaningful extraction achieved")
                else:
                    print("⚠️  No meaningful extraction (poor OCR quality)")
            else:
                print("❌ No text extracted")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n--- Summary ---")
    print(f"Files tested: {total_tested}")
    print(f"Successful extractions: {success_count}")
    print(f"Success rate: {success_count/total_tested*100:.1f}%")
    
    if success_count > 0:
        print("✅ PDF processing is working correctly")
        return True
    else:
        print("⚠️  PDF processing needs improvement")
        return False

if __name__ == "__main__":
    success = test_multiple_pdfs()
    sys.exit(0 if success else 1)