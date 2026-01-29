#!/usr/bin/env python3
"""
Enhanced OCR pipeline with direct PDF text extraction
This tries to extract text directly from PDFs before falling back to OCR
"""
import sys
import os
from pathlib import Path
import re

# Fix encoding for Windows console
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("⚠️  PyPDF2 not available, will use OCR only")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️  pdfplumber not available, will use OCR only")

from app.ocr_pipeline import OCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3


def extract_text_from_pdf_pypdf2(pdf_path):
    """Extract text directly from PDF using PyPDF2"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n=== PAGE {page_num + 1} ===\n{page_text}"
        return text.strip()
    except Exception as e:
        print(f"  PyPDF2 extraction failed: {e}")
        return None


def extract_text_from_pdf_pdfplumber(pdf_path):
    """Extract text directly from PDF using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n=== PAGE {page_num + 1} ===\n{page_text}"
        return text.strip()
    except Exception as e:
        print(f"  pdfplumber extraction failed: {e}")
        return None


def extract_with_hybrid_approach(pdf_path):
    """Try multiple extraction methods and return the best result"""
    results = []
    
    print(f"\n{'='*80}")
    print(f"Testing: {pdf_path.name}")
    print(f"{'='*80}")
    
    # Method 1: PyPDF2
    if PYPDF2_AVAILABLE:
        print("\n[Method 1] PyPDF2 Text Extraction...")
        text = extract_text_from_pdf_pypdf2(pdf_path)
        if text and len(text) > 50:
            print(f"  ✅ Extracted {len(text)} characters")
            result = extract_all_fields_v3(text)
            if result.get('all_ids'):
                print(f"  ✅ Found IDs: {result['all_ids']}")
                results.append(('PyPDF2', text, result))
            else:
                print(f"  ⚠️  No IDs found (Bank: {result.get('bank_name', 'Unknown')})")
                results.append(('PyPDF2', text, result))
        else:
            print(f"  ❌ No text extracted or too short")
    
    # Method 2: pdfplumber
    if PDFPLUMBER_AVAILABLE:
        print("\n[Method 2] pdfplumber Text Extraction...")
        text = extract_text_from_pdf_pdfplumber(pdf_path)
        if text and len(text) > 50:
            print(f"  ✅ Extracted {len(text)} characters")
            result = extract_all_fields_v3(text)
            if result.get('all_ids'):
                print(f"  ✅ Found IDs: {result['all_ids']}")
                results.append(('pdfplumber', text, result))
            else:
                print(f"  ⚠️  No IDs found (Bank: {result.get('bank_name', 'Unknown')})")
                results.append(('pdfplumber', text, result))
        else:
            print(f"  ❌ No text extracted or too short")
    
    # Method 3: OCR (fallback)
    print("\n[Method 3] OCR Extraction (fallback)...")
    ocr_pipeline = OCRPipeline()
    try:
        ocr_result = ocr_pipeline.process_file(str(pdf_path))
        text = ocr_result['text']
        confidence = ocr_result['confidence']
        print(f"  OCR Confidence: {confidence:.2%}, Length: {len(text)} chars")
        
        result = extract_all_fields_v3(text)
        if result.get('all_ids'):
            print(f"  ✅ Found IDs: {result['all_ids']}")
            results.append(('OCR', text, result))
        else:
            print(f"  ⚠️  No IDs found (Bank: {result.get('bank_name', 'Unknown')})")
            results.append(('OCR', text, result))
    except Exception as e:
        print(f"  ❌ OCR failed: {e}")
    
    # Choose best result
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print(f"{'='*80}")
    
    best_method = None
    best_result = None
    
    for method, text, result in results:
        ids = result.get('all_ids', [])
        print(f"{method:15} - IDs: {ids if ids else 'None'}")
        
        if ids and (best_result is None or len(ids) > len(best_result.get('all_ids', []))):
            best_method = method
            best_result = result
    
    if best_method:
        print(f"\n✅ Best Method: {best_method}")
        print(f"   IDs: {best_result.get('all_ids', [])}")
        print(f"   Bank: {best_result.get('bank_name', 'Unknown')}")
        return best_method, best_result
    else:
        print(f"\n❌ No method successfully extracted IDs")
        return None, None


def main():
    """Test hybrid extraction on CIMB receipt"""
    
    # Test on the specific CIMB receipt that has reference number 202510310309176178
    test_file = Path("Receipts/CIMB - DIFF REF FRM STATEMENT.pdf")
    
    if not test_file.exists():
        print(f"❌ File not found: {test_file}")
        return
    
    print("\n" + "="*80)
    print("ENHANCED PDF TEXT EXTRACTION TEST")
    print("Expected Reference Number: 202510310309176178")
    print("="*80)
    
    method, result = extract_with_hybrid_approach(test_file)
    
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    
    if result and result.get('all_ids'):
        expected = "202510310309176178"
        found_ids = result.get('all_ids', [])
        
        if expected in found_ids:
            print(f"🎉 SUCCESS! Found expected reference number: {expected}")
        else:
            print(f"⚠️  Found IDs but not the expected one:")
            print(f"   Expected: {expected}")
            print(f"   Found: {found_ids}")
    else:
        print(f"❌ FAILED: Could not extract reference number")
        print(f"   This means the PDF might not have embedded text")
        print(f"   Will need to improve OCR quality instead")


if __name__ == "__main__":
    main()
