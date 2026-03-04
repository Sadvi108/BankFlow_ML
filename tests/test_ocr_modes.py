
import pytesseract
import cv2
import os
import sys

# Fix encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def test_ocr_modes():
    image_path = "debug_failures_images/PUBLIC BANK - INVOICE.pdf.png"
    if not os.path.exists(image_path):
        print("Image not found")
        return

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing (simple)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    modes = [3, 4, 6, 11, 12]
    
    print(f"Testing OCR on {image_path}")
    print("=" * 80)
    
    for mode in modes:
        print(f"\nTesting PSM {mode}...")
        config = f'--oem 1 --psm {mode}'
        try:
            text = pytesseract.image_to_string(thresh, config=config)
            print(f"Length: {len(text)}")
            print("-" * 20)
            print(text[:500]) # First 500 chars
            print("-" * 20)
            
            # Check for key terms
            if "Reference" in text or "Ref" in text:
                print("✅ Found 'Reference' keyword")
            else:
                print("❌ 'Reference' keyword NOT found")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_ocr_modes()
