#!/usr/bin/env python3
"""
Debug OCR pipeline functionality
"""

import sys
sys.path.append('.')

from app.ocr_pipeline import OCRPipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Create a simple test image
def create_simple_test_image():
    """Create a simple test image with clear text"""
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Simple test text
    text = """MAYBANK TRANSACTION
Reference: M2U12345678
Amount: RM 100.00
Transaction ID: MBB98765432
Date: 31/10/2025"""
    
    lines = text.split('\n')
    y_pos = 50
    for line in lines:
        draw.text((50, y_pos), line, fill='black', font=font)
        y_pos += 40
    
    img.save('debug_test.png')
    return 'debug_test.png'

def test_ocr_pipeline():
    """Test the OCR pipeline with a simple image"""
    print("🧪 Testing OCR Pipeline...")
    
    # Create test image
    image_path = create_simple_test_image()
    print(f"Created test image: {image_path}")
    
    # Initialize OCR pipeline
    ocr = OCRPipeline()
    print("Initialized OCR pipeline")
    
    # Process the image
    result = ocr.process_file(image_path)
    
    print(f"OCR Result:")
    print(f"  Text extracted: {len(result['text'])} characters")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Success: {result['processed_successfully']}")
    print(f"  Text preview: {result['text'][:200]}...")
    
    if result['text']:
        print("\n📋 Full extracted text:")
        print(result['text'])
    else:
        print("\n❌ No text extracted!")
    
    return result['processed_successfully'] and len(result['text']) > 50

if __name__ == "__main__":
    success = test_ocr_pipeline()
    if success:
        print("\n✅ OCR pipeline is working correctly")
    else:
        print("\n❌ OCR pipeline needs debugging")