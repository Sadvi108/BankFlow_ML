
import sys
import os
import cv2
import pytesseract
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path.cwd()))

from app.ocr_pipeline import OCRPipeline

def debug_ocr():
    pipeline = OCRPipeline()
    
    # Paths provided in metadata
    image_paths = [
        r"C:/Users/User/.gemini/antigravity/brain/a0caa328-c5df-43eb-97b0-fdd10f9ecfc3/uploaded_image_0_1769070516427.png",
        r"C:/Users/User/.gemini/antigravity/brain/a0caa328-c5df-43eb-97b0-fdd10f9ecfc3/uploaded_image_1_1769070516427.png"
    ]
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
            
        print(f"\nProcessing {img_path}...")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load image")
            continue
            
        # 1. Test Raw Image
        print("\n[TEST 1] Raw Image OCR:")
        custom_config = r'--oem 1 --psm 6'
        text_raw = pytesseract.image_to_string(img, config=custom_config)
        print(f"Raw Text Preview: {text_raw[:200].replace('\n', ' ')}")

        # 2. Test Grayscale + Scale
        print("\n[TEST 2] Grayscale + 2x Resize OCR:")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        text_gray = pytesseract.image_to_string(gray, config=custom_config)
        print(f"Gray+Scale Text Preview: {text_gray[:200].replace('\n', ' ')}")
        
        # 3. Pipeline
        print("\n[TEST 3] Full Pipeline OCR:")
        result = pipeline.extract_text_with_confidence(img)
        text = result['text']
        
        print("-" * 50)
        print(f"Pipeline Text Preview:")
        print(text[:500])
        print("-" * 50)
        
        # Check targets in ALL results
        all_texts = [text_raw, text_gray, text]
        found = False
        target_id_correct = "C001987200"
        target_id_wrong = "C003987200"
        
        for i, t in enumerate(["Raw", "Gray", "Pipeline"]):
            txt = all_texts[i]
            if target_id_correct in txt:
                print(f"SUCCESS in {t}: Found correct ID: {target_id_correct}")
                found = True
            elif target_id_wrong in txt:
                 print(f"FAILURE in {t}: Found wrong ID: {target_id_wrong}")
        
        if not found:
             print(f"WARNING: Neither ID found in any mode.")
             import re
             matches = re.findall(r'C00[0-9]{7}', text_gray) # search in gray as it's likely best
             print(f"Potential candidates in Gray: {matches}")

if __name__ == "__main__":
    debug_ocr()
