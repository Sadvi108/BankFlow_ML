#!/usr/bin/env python3
"""
Enhanced debugging - extract and save OCR text for manual inspection
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

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline

def save_ocr_text(receipt_path, output_dir="debug_ocr_output"):
    """
    Process receipt and save OCR text to file
    Returns tuple (text, result_dict)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        receipt_path = Path(receipt_path)
        
        # Initialize pipeline
        pipeline = EnhancedOCRPipeline()
        
        # Process file
        print(f"\nProcessing: {receipt_path.name}")
        ocr_result = pipeline.process_file(str(receipt_path))
        text = ocr_result['text']
        confidence = ocr_result['confidence']
        
        # Save to file
        output_file = output_path / f"{receipt_path.stem}_ocr.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"File: {receipt_path.name}\n")
            f.write(f"OCR Confidence: {confidence:.2%}\n")
            f.write(f"Text Length: {len(text)} characters\n")
            f.write("="*80 + "\n")
            f.write(text)
        
        print(f"  ✅ Saved OCR text to: {output_file}")
        print(f"  OCR Confidence: {confidence:.2%}, Length: {len(text)} chars")
        
        # Try extraction
        result = extract_all_fields_v3(text)
        if result.get('all_ids'):
            print(f"  ✅ Found IDs: {result['all_ids']}")
        else:
            print(f"  ❌ No IDs found (Bank: {result.get('bank_name', 'Unknown')})")
            
        return text, result
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None

def main():
    """Extract OCR text from all receipts"""
    receipts_dir = Path("Receipts")
    
    print("\n" + "="*80)
    print("EXTRACTING OCR TEXT FROM ALL RECEIPTS")
    print("="*80)
    
    # Get all PDF files
    pdf_files = list(receipts_dir.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDF files")
    
    successful = 0
    failed = 0
    
    for pdf_file in sorted(pdf_files):  # Process ALL files
        text, result = save_ocr_text(pdf_file)
        if result and result.get('all_ids'):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "="*80)
    print(f"SUMMARY: {successful} successful, {failed} failed")
    print(f"OCR text saved to: debug_ocr_output/")
    print("="*80)

if __name__ == "__main__":
    main()
