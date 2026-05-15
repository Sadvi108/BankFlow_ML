
import sys
import os
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

# Check if app module can be found
sys.path.insert(0, str(Path.cwd()))

# Reduce logging
logging.basicConfig(level=logging.ERROR)

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

def process_file(file_path):
    try:
        pipeline = EnhancedOCRPipeline()
        result = pipeline.process_file(file_path)
        if not result.get('text'):
             return (os.path.basename(file_path), "no_text", 0)
             
        extraction = extract_all_fields_v3(result['text'])
        bank = extraction.get('bank_name', 'Unknown')
        
        status = "success" if bank != "Unknown" else "failed"
        return (os.path.basename(file_path), status, bank)
        
    except Exception as e:
        return (os.path.basename(file_path), "error", str(e))

def audit_all():
    files = glob.glob("Receipts/*.*")
    print(f"Found {len(files)} files. Auditing...")
    
    # Process only first 20 for speed in this interaction, or all if feasible?
    # Let's do 50 to get a good sample including the failing ones hopefully.
    # User said "Many", so picking 50 might hit them.
    # Actually, let's target images especially since user uploaded an image last time.
    
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    
    # Mix: 20 images + 30 PDFs
    target_files = image_files[:20] + pdf_files[:30]
    
    print(f"Auditing sample of {len(target_files)} files...")
    
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, target_files))
        
    for fname, status, info in results:
        if status == "failed" or status == "error":
            failed_files.append(fname)
            print(f"❌ {fname}: {info}")
        else:
            print(f"✅ {fname}: {info}")
            
    print("\n" + "="*50)
    print(f"Summary: {len(results) - len(failed_files)} passed, {len(failed_files)} failed.")
    print("="*50)

if __name__ == "__main__":
    audit_all()
