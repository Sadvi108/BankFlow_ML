
import fitz
import os
from pathlib import Path

def extract_images():
    receipts_dir = Path("Receipts")
    output_dir = Path("debug_failures_images")
    output_dir.mkdir(exist_ok=True)
    
    failing_files = [
        "38c11069-2dc5-4e91-a37d-25cbc8518f68.pdf",
        "AFFIN BANK -CUSTOMER REF.pdf",
        "DuitNow_Transaction_Report - INVOICE.pdf",
        "PUBLIC BANK - INVOICE.pdf",
        "UOB- INV REF.pdf"
    ]
    
    for filename in failing_files:
        file_path = receipts_dir / filename
        if not file_path.exists():
            continue
            
        print(f"Processing {filename}...")
        try:
            doc = fitz.open(file_path)
            page = doc.load_page(0) # Just first page
            
            # Render at high DPI (300 DPI = ~4.16x of 72 DPI)
            mat = fitz.Matrix(4.0, 4.0) 
            pix = page.get_pixmap(matrix=mat)
            
            output_path = output_dir / f"{filename}.png"
            pix.save(output_path)
            print(f"Saved {output_path} ({pix.width}x{pix.height})")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    extract_images()
