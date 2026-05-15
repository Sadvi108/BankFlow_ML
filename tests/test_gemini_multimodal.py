import os
import sys
from app.gemini_extractor import gemini_extractor
import glob
from pathlib import Path

def test_gemini():
    # Check for API Key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found in environment.")
        print("Please run: export GOOGLE_API_KEY='your-api-key'")
        return

    # Find some receipts to test
    receipts = glob.glob('Receipts/*.pdf') + glob.glob('Receipts/*.png') + glob.glob('Receipts/*.jpg')
    if not receipts:
        print("No receipts found in Receipts/ folder.")
        return

    # Pick the first image receipt (multimodal works best on images)
    image_receipts = [r for r in receipts if not r.lower().endswith('.pdf')]
    if not image_receipts:
        print("No image receipts found. Multimodal test skipped (PDFs need rendering).")
        return

    test_file = image_receipts[0]
    print(f"Testing Gemini 1.5 Flash on: {test_file}")

    with open(test_file, 'rb') as f:
        content = f.read()

    results = gemini_extractor.extract_from_image(content, filename=Path(test_file).name)
    
    import json
    print("\n--- GEMINI EXTRACTION RESULTS ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    test_gemini()
