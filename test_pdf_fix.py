#!/usr/bin/env python3
"""
Test PDF processing fix to ensure PDF files are handled correctly
"""

import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.ocr_enhanced import EnhancedOCRProcessor
from app.main_enhanced import extract_receipt_content

def test_pdf_processing():
    """Test PDF processing with the fixed pipeline."""
    print("Testing PDF processing fix...")
    
    # Find PDF files in Receipts directory
    receipts_dir = Path(__file__).parent / "Receipts"
    pdf_files = list(receipts_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in Receipts directory")
        return False
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Test with first PDF file
    test_pdf = pdf_files[0]
    print(f"Testing with: {test_pdf.name}")
    
    try:
        # Test direct PDF extraction
        processor = EnhancedOCRProcessor()
        text, tokens = processor.extract_from_pdf(test_pdf)
        print(f"Direct PDF extraction successful: {len(text)} characters, {len(tokens)} tokens")
        
        if text.strip():
            print("Sample text:", text[:200] + "..." if len(text) > 200 else text)
        
        # Test the complete pipeline
        file_id = "test_pdf_123"
        text, tokens, processed_path = extract_receipt_content(test_pdf, file_id)
        print(f"Complete pipeline successful: {len(text)} characters, {len(tokens)} tokens")
        print(f"Processed path: {processed_path}")
        
        if text.strip():
            print("Pipeline text sample:", text[:200] + "..." if len(text) > 200 else text)
            
            # Test pattern extraction
            from app.ultimate_patterns_v2 import UltimatePatternMatcherV2
            matcher = UltimatePatternMatcherV2()
            result = matcher.extract_all_fields(text)
            print(f"Pattern extraction results:")
            print(f"  Bank: {result.get('bank_name', 'Unknown')}")
            print(f"  Transaction IDs: {result.get('transaction_ids', [])}")
            print(f"  Reference Numbers: {result.get('reference_numbers', [])}")
            print(f"  Amounts: {result.get('amounts', [])}")
            print(f"  Confidence: {result.get('confidence', 0)}")
        
        print("✅ PDF processing test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pdf_processing()
    sys.exit(0 if success else 1)