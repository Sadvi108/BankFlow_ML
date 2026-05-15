#!/usr/bin/env python3
"""
Comprehensive test of enhanced extraction on uploaded receipts
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


def test_uploads_folder():
    """Test all receipts in data/uploads with enhanced extraction"""
    uploads_dir = Path("data/uploads")
    
    if not uploads_dir.exists():
        print(f"❌ Uploads directory not found: {uploads_dir}")
        return
    
    # Get all supported files
    supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = [p for p in uploads_dir.iterdir() if p.suffix.lower() in supported_extensions]
    files.sort(key=lambda x: x.name)
    
    print("\n" + "="*80)
    print(f"TESTING ENHANCED EXTRACTION ON {len(files)} UPLOADED RECEIPTS")
    print("="*80)
    
    ocr_pipeline = EnhancedOCRPipeline()
    
    successful = 0
    failed = 0
    results = []
    
    for file_path in files:
        print(f"\n{file_path.name}")
        print("-" * 80)
        
        try:
            # Extract text
            result = ocr_pipeline.process_file(str(file_path))
            text = result['text']
            confidence = result.get('confidence', 0)
            method = result.get('method', 'unknown')
            
            print(f"  Method: {method}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Text Length: {len(text)} chars")
            
            # Extract fields
            extraction = extract_all_fields_v3(text)
            ids = extraction.get('all_ids', [])
            bank = extraction.get('bank_name', 'Unknown')
            
            if ids:
                print(f"  ✅ Bank: {bank}")
                print(f"  ✅ IDs: {ids}")
                successful += 1
                results.append((file_path.name, 'SUCCESS', ids, bank))
            else:
                print(f"  ❌ No IDs found (Bank: {bank})")
                print(f"  OCR Text Sample: {text[:200].replace(chr(10), ' ')}...")
                failed += 1
                results.append((file_path.name, 'FAILED', [], bank))
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
            results.append((file_path.name, 'ERROR', [], 'Unknown'))
    
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
    print(f"Success Rate: {successful}/{len(files)} ({successful/len(files)*100:.1f}%)")
    print("="*80)


if __name__ == "__main__":
    test_uploads_folder()
