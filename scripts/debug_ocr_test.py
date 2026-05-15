
import os
import sys
import logging
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.ocr_pipeline import OCRPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_pdf(pdf_path):
    print(f"Processing {pdf_path}...")
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # 1. Try EnhancedOCRPipeline
    print("\n--- EnhancedOCRPipeline ---")
    enhanced_pipeline = EnhancedOCRPipeline()
    try:
        result = enhanced_pipeline.process_file(pdf_path)
        print(f"Success: {result.get('processed_successfully', 'Unknown')}")
        print(f"Confidence: {result.get('confidence')}")
        text = result.get('text', '')
        print(f"Text length: {len(text)}")
        print(f"Text full:\n{text}")
    except Exception as e:
        print(f"EnhancedOCRPipeline failed: {e}")

    # 2. Try OCRPipeline directly
    print("\n--- OCRPipeline ---")
    ocr_pipeline = OCRPipeline()
    try:
        result = ocr_pipeline.process_file(pdf_path)
        print(f"Success: {result.get('processed_successfully', 'Unknown')}")
        print(f"Confidence: {result.get('confidence')}")
        text = result.get('text', '')
        print(f"Text length: {len(text)}")
        print(f"Text full:\n{text}")
    except Exception as e:
        print(f"OCRPipeline failed: {e}")

if __name__ == "__main__":
    pdf_path = "Receipts/3243781e-77dd-4e74-a1f3-b1ee94bc43c3_DDCONTKUL1.pdf"
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    debug_pdf(pdf_path)
