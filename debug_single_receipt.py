
import sys
import os
import logging
from pathlib import Path
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ultimate_patterns_v3 import extract_all_fields_v3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_single():
    file_path = "data/uploads/3c6b2dac-7944-4985-a444-0d8445844758.pdf"
    
    print(f"Testing extraction on: {file_path}")
    pipeline = EnhancedOCRPipeline()
    
    try:
        # Process file
        result = pipeline.process_file(file_path)
        
        print("\n--- OCR Result ---")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Method: {result.get('method', 'unknown')}")
        print(f"Text Preview: {result['text'][:200]}...")
        
        # Extract fields
        extraction = extract_all_fields_v3(result['text'])
        print("\n--- Extraction ---")
        print(f"IDs: {extraction.get('all_ids')}")
        print(f"Date: {extraction.get('date')}")
        print(f"Amount: {extraction.get('amount')}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    debug_single()
