
import logging
import sys
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_rotation():
    pipeline = EnhancedOCRPipeline()
    file_path = "Receipts/38c11069-2dc5-4e91-a37d-25cbc8518f68.pdf"
    
    print(f"Testing {file_path}...")
    result = pipeline.process_file(file_path)
    
    print("\nFINAL RESULT:")
    print(f"Method: {result.get('method')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Text len: {len(result.get('text', ''))}")
    print("Preview:")
    print(result.get('text', '')[:500])

if __name__ == "__main__":
    test_rotation()
