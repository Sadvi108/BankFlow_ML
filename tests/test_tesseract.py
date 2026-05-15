import pytesseract
from PIL import Image
import os
import sys

def test():
    print("Tesseract Version:", pytesseract.get_tesseract_version())
    
    # Create a small image with text
    from PIL import ImageDraw, ImageFont
    img = Image.new('RGB', (200, 50), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10,10), "TEST 123", fill=(0,0,0))
    img.save('test_ocr.png')
    
    print("Running OCR on test image...")
    text = pytesseract.image_to_string(img)
    print(f"Result: '{text.strip()}'")
    
    if "TEST" in text.upper():
        print("Tesseract is working correctly!")
    else:
        print("Tesseract FAILED to read simple text.")

if __name__ == "__main__":
    test()
