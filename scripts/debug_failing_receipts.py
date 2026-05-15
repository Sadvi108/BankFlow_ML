#!/usr/bin/env python3
"""
Diagnostic script to analyze failing receipts
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

def diagnose_failures():
    receipts_dir = Path("Receipts")
    output_dir = Path("debug_failures")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("DIAGNOSING FAILING RECEIPTS")
    print("="*80)
    
    ocr_pipeline = EnhancedOCRPipeline()
    pdf_files = sorted(list(receipts_dir.glob("*.pdf")))
    
    failures = []
    
    for pdf_file in pdf_files:
        try:
            result = ocr_pipeline.process_file(str(pdf_file))
            text = result['text']
            method = result.get('method', 'unknown')
            
            extraction = extract_all_fields_v3(text)
            ids = extraction.get('all_ids', [])
            
            if not ids:
                print(f"\n❌ FAILED: {pdf_file.name}")
                print(f"   Method: {method}")
                print(f"   Text Length: {len(text)}")
                print(f"   First 200 chars: {text[:200].replace(chr(10), ' ')}")
                
                # Save text for inspection
                out_file = output_dir / f"{pdf_file.stem}_debug.txt"
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(f"File: {pdf_file.name}\n")
                    f.write(f"Method: {method}\n")
                    f.write("="*80 + "\n")
                    f.write(text)
                
                failures.append(pdf_file.name)
                
        except Exception as e:
            print(f"❌ ERROR processing {pdf_file.name}: {e}")
            failures.append(pdf_file.name)
            
    print("\n" + "="*80)
    print(f"Total Failures: {len(failures)}/{len(pdf_files)}")
    print("="*80)
    for f in failures:
        print(f"- {f}")

if __name__ == "__main__":
    diagnose_failures()
