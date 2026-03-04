#!/usr/bin/env python3
"""
Ingest new receipts into the training dataset
"""
import sys
import os
from pathlib import Path

# Fix encoding for Windows console
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app import dataset, classify, extract
from app.ocr_pipeline import OCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3
import uuid

def ingest_receipts():
    """Process all receipts in the Receipts folder and add to dataset."""
    print("\n" + "="*80)
    print("INGESTING RECEIPTS INTO TRAINING DATASET")
    print("="*80)
    
    # Ensure dataset directories exist
    dataset.ensure_dirs()
    
    # Get existing annotations to avoid duplicates
    existing_annotations = dataset.read_annotations()
    existing_filenames = {ann.get('filename') for ann in existing_annotations}
    
    print(f"\nExisting annotations: {len(existing_annotations)}")
    print(f"Existing filenames: {len(existing_filenames)}")
    
    # Get all receipt files
    receipts_dir = Path("Receipts")
    if not receipts_dir.exists():
        print(f"❌ Receipts directory not found: {receipts_dir}")
        return
    
    supported_exts = {".pdf", ".png", ".jpg", ".jpeg"}
    files = [p for p in receipts_dir.iterdir() if p.suffix.lower() in supported_exts]
    
    print(f"\nFound {len(files)} receipt files")
    
    # Initialize OCR pipeline
    ocr_pipeline = OCRPipeline()
    
    processed = 0
    skipped = 0
    errors = 0
    
    for receipt_file in sorted(files):
        filename = receipt_file.name
        
        # Skip if already processed
        if filename in existing_filenames:
            print(f"⏭️  Skipping (already in dataset): {filename}")
            skipped += 1
            continue
        
        print(f"\n📄 Processing: {filename}")
        
        try:
            # Generate unique ID
            file_id = uuid.uuid4().hex
            
            # Extract OCR text
            ocr_result = ocr_pipeline.process_file(str(receipt_file))
            text = ocr_result['text']
            tokens = ocr_result.get('tokens', [])
            confidence = ocr_result['confidence']
            
            if not text.strip():
                print(f"  ⚠️  Warning: No text extracted")
                errors += 1
                continue
            
            print(f"  OCR: {len(text)} chars, {confidence:.2%} confidence")
            
            # Extract fields using ultimate patterns v3
            extraction_result = extract_all_fields_v3(text)
            
            # Classify bank
            bank_name = extraction_result.get('bank_name', 'unknown')
            bank_confidence = 1.0 if bank_name != 'unknown' else 0.0
            
            # Get all extracted IDs
            all_ids = extraction_result.get('all_ids', [])
            
            # Build annotation entry
            entry = {
                "id": file_id,
                "filename": filename,
                "bank": {
                    "name": bank_name,
                    "confidence": bank_confidence
                },
                "fields": {
                    "reference_number": extraction_result.get('reference_number'),
                    "transaction_id": extraction_result.get('transaction_id'),
                    "transaction_number": extraction_result.get('transaction_number'),
                    "duitnow_reference_number": extraction_result.get('duitnow_reference_number'),
                    "invoice_number": extraction_result.get('invoice_number'),
                    "amount": extraction_result.get('amount'),
                    "date": extraction_result.get('date'),
                    "boxes": extraction_result.get('boxes', {}),
                    "meta": extraction_result.get('meta', {})
                },
                "ocr_text": text,
                "meta": {
                    "source_path": str(receipt_file.absolute()),
                    "processed_path": str(receipt_file.absolute()),
                    "dataset_image_path": str(dataset.IMAGES_DIR / f"{file_id}.png"),
                    "ocr_tokens": len(tokens)
                },
                "ground_truth": {
                    "bank_name": None,
                    "reference_number": None,
                    "transaction_id": None,
                    "invoice_number": None,
                    "amount": None,
                    "date": None
                }
            }
            
            # Save to dataset
            dataset.append_annotation(entry)
            
            # Print extraction results
            print(f"  ✅ Bank: {bank_name}")
            if all_ids:
                print(f"  ✅ Extracted IDs: {all_ids}")
            else:
                print(f"  ⚠️  No IDs extracted")
            
            processed += 1
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors += 1
    
    print("\n" + "="*80)
    print("INGESTION SUMMARY")
    print("="*80)
    print(f"Total files found: {len(files)}")
    print(f"Already in dataset: {skipped}")
    print(f"Newly processed: {processed}")
    print(f"Errors: {errors}")
    print(f"\nTotal annotations now: {len(existing_annotations) + processed}")
    print("="*80)
    
    # Show dataset summary
    summary = dataset.summary()
    print(f"\nDataset Summary:")
    print(f"  Total: {summary['total']}")
    print(f"  Per bank: {summary['per_bank']}")

if __name__ == "__main__":
    ingest_receipts()
