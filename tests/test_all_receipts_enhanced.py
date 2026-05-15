#!/usr/bin/env python3
"""
Comprehensive test of enhanced extraction on all receipts
"""
import sys
import os
from pathlib import Path

# Fix encoding for Windows console
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3


def test_all_receipts():
    """Test all receipts with enhanced extraction"""
    receipts_dir = Path("Receipts")
    
    if not receipts_dir.exists():
        print(f"❌ Receipts directory not found")
        return
    
    # Get all PDF files
    pdf_files = sorted(list(receipts_dir.glob("*.pdf")))
    
    print("\n" + "="*80)
    print(f"TESTING ENHANCED EXTRACTION ON {len(pdf_files)} RECEIPTS")
    print("="*80)
    
    ocr_pipeline = EnhancedOCRPipeline()
    
    successful = 0
    failed = 0
    results = []
    
    for pdf_file in pdf_files:
        print(f"\n{pdf_file.name}")
        print("-" * 80)
        
        try:
            # Extract text
            result = ocr_pipeline.process_file(str(pdf_file))
            text = result['text']
            confidence = result['confidence']
            method = result.get('method', 'unknown')
            
            print(f"  Method: {method}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Text Length: {len(text)} chars")
            
            # Extract fields
            extraction = extract_all_fields_v3(text)
            ids = extraction.get('all_ids', [])
            bank = extraction.get('bank_name', 'Unknown')
            
            if ids:
                print(f"  ✅ Bank: {bank}")
                print(f"  ✅ IDs: {ids}")
                successful += 1
                results.append((pdf_file.name, 'SUCCESS', ids, bank))
            else:
                print(f"  ❌ No IDs found (Bank: {bank})")
                failed += 1
                results.append((pdf_file.name, 'FAILED', [], bank))
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
            results.append((pdf_file.name, 'ERROR', [], 'Unknown'))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for filename, status, ids, bank in results:
        if status == 'SUCCESS':
            print(f"✅ {filename}: {ids}")
        else:
            print(f"❌ {filename}: No IDs")
    
    print(f"\n" + "="*80)
    print(f"Success Rate: {successful}/{len(pdf_files)} ({successful/len(pdf_files)*100:.1f}%)")
    print(f"Improvement from baseline: 10% -> {successful/len(pdf_files)*100:.1f}%")
    print("="*80)


if __name__ == "__main__":
    test_all_receipts()
