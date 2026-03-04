#!/usr/bin/env python3
"""
Debug script to see what text is actually extracted from the CIMB PDF
"""
import sys
import os

# Fix encoding for Windows console
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pdfplumber
from pathlib import Path

pdf_path = Path("Receipts/CIMB - DIFF REF FRM STATEMENT.pdf")

print("="*80)
print("EXTRACTED PDF TEXT")
print("="*80)

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages):
        text = page.extract_text()
        print(f"\n=== PAGE {page_num + 1} ===")
        print(text)
        print("\n" + "-"*80)

print("\n" + "="*80)
print("Searching for reference number: 202510310309176178")
print("="*80)

with pdfplumber.open(pdf_path) as pdf:
    all_text = ""
    for page in pdf.pages:
        all_text += page.extract_text() or ""
    
    if "202510310309176178" in all_text:
        print("✅ FOUND! The reference number exists in the PDF text")
        # Find context around it
        idx = all_text.index("202510310309176178")
        start = max(0, idx - 100)
        end = min(len(all_text), idx + 100)
        print(f"\nContext:")
        print(all_text[start:end])
    else:
        print("❌ NOT FOUND in extracted text")
        print("\nSearching for partial matches...")
        # Try to find parts of it
        for i in range(len("202510310309176178") - 5):
            partial = "202510310309176178"[i:i+10]
            if partial in all_text:
                print(f"  Found partial: {partial}")
