
import logging
import sys
from app.enhanced_ocr_pipeline import EnhancedOCRPipeline

# Configure logging to stdout
logging.basicConfig(
    level=logging.ERROR, # Less noise
    handlers=[logging.StreamHandler(sys.stdout)]
)

def inspect_receipt():
    pipeline = EnhancedOCRPipeline()
    file_path = "Receipts/98c61712-9ceb-43c2-b83f-b25a154a3a83_dnd 10.60.pdf"
    
    print(f"Inspecting {file_path}...")
    result = pipeline.process_file(file_path)
    
    print("\nFULL TEXT START:")
    print("-" * 40)
    print(result.get('text', ''))
    print("-" * 40)
    print("FULL TEXT END")

if __name__ == "__main__":
    inspect_receipt()
