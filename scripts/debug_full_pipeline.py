#!/usr/bin/env python3
"""
Comprehensive debugging script to test the entire OCR pipeline
"""
import sys
import os
from pathlib import Path

# Fix encoding for Windows console
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.ocr_pipeline import OCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

def test_receipt(receipt_path):
    """Test a single receipt through the entire pipeline"""
    print(f"\n{'='*80}")
    print(f"Testing: {receipt_path.name}")
    print(f"{'='*80}")
    
    # Step 1: OCR Extraction
    print("\n[Step 1] OCR Extraction...")
    ocr_pipeline = OCRPipeline()
    try:
        ocr_result = ocr_pipeline.process_file(str(receipt_path))
        text = ocr_result['text']
        confidence = ocr_result['confidence']
        
        print(f"  OCR Confidence: {confidence:.2%}")
        print(f"  Text Length: {len(text)} characters")
        print(f"\n  Extracted Text (first 500 chars):")
        print(f"  {'-'*76}")
        print(f"  {text[:500]}")
        if len(text) > 500:
            print(f"  ... (truncated, total {len(text)} chars)")
        print(f"  {'-'*76}")
        
    except Exception as e:
        print(f"  ❌ OCR Failed: {e}")
        return None
    
    # Step 2: Pattern Matching
    print("\n[Step 2] Pattern Matching with V3...")
    try:
        extraction_result = extract_all_fields_v3(text)
        
        print(f"  Bank Name: {extraction_result.get('bank_name', 'Unknown')}")
        print(f"  Transaction IDs: {extraction_result.get('all_ids', [])}")
        print(f"  Confidence: {extraction_result.get('confidence', 0):.2%}")
        
        if extraction_result.get('all_ids'):
            print(f"  ✅ SUCCESS: Found {len(extraction_result['all_ids'])} transaction ID(s)")
        else:
            print(f"  ❌ FAILED: No transaction IDs found")
            
        return extraction_result
        
    except Exception as e:
        print(f"  ❌ Pattern Matching Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Test multiple receipts"""
    receipts_dir = Path("Receipts")
    
    if not receipts_dir.exists():
        print(f"❌ Receipts directory not found: {receipts_dir}")
        return
    
    # Test a variety of receipts
    test_files = [
        "MAYBANK - REF.pdf",
        "CIMB - DIFF REF FRM STATEMENT.pdf",
        "PUBLIC BANK - INVOICE.pdf",
        "RHB - CUSTOMER REF.pdf",
        "HSBC - REF NAME.pdf",
    ]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE OCR PIPELINE DEBUG TEST")
    print("="*80)
    
    results = []
    for filename in test_files:
        receipt_path = receipts_dir / filename
        if receipt_path.exists():
            result = test_receipt(receipt_path)
            results.append({
                'file': filename,
                'result': result
            })
        else:
            print(f"\n⚠️  File not found: {filename}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = 0
    failed = 0
    
    for item in results:
        if item['result'] and item['result'].get('all_ids'):
            successful += 1
            print(f"✅ {item['file']}: {item['result']['all_ids']}")
        else:
            failed += 1
            print(f"❌ {item['file']}: No IDs found")
    
    print(f"\nTotal: {successful}/{len(results)} successful")
    print(f"Success Rate: {successful/len(results)*100:.1f}%")
    
    if failed > 0:
        print(f"\n⚠️  {failed} receipts failed to extract transaction IDs")
        print("This suggests either:")
        print("  1. OCR is not extracting readable text")
        print("  2. The extracted text doesn't match our patterns")
        print("  3. The receipt format is different from test cases")

if __name__ == "__main__":
    main()
