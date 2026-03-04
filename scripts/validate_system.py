#!/usr/bin/env python3
"""
Simple validation script to test the ML system components
"""

import sys
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports work."""
    print("Testing basic imports...")
    
    try:
        from app import utils, ocr, classify, extract, dataset
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_ocr_functionality():
    """Test OCR functionality."""
    print("Testing OCR functionality...")
    
    try:
        from app import ocr
        
        # Create a simple test image
        img = Image.new('RGB', (100, 50), color='white')
        
        # Test image processing (will return empty for blank image)
        text, tokens = ocr.image_to_text_and_tokens(img)
        
        print(f"✅ OCR processing successful (text length: {len(text)}, tokens: {len(tokens)})")
        return True
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

def test_classification():
    """Test bank classification."""
    print("Testing bank classification...")
    
    try:
        from app import classify
        
        # Test with Maybank text
        text = "This is a Maybank receipt with reference ABC123"
        bank, confidence = classify.bank_from_text(text)
        
        if bank == "Maybank" and confidence > 0:
            print(f"✅ Classification successful: {bank} (confidence: {confidence:.2f})")
            return True
        else:
            print(f"❌ Classification unexpected: {bank} (confidence: {confidence:.2f})")
            return False
            
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False

def test_extraction():
    """Test transaction ID extraction."""
    print("Testing transaction extraction...")
    
    try:
        from app import extract
        
        # Test with sample text
        text = "Reference: ABC123456 Transaction ID: TXN789"
        fields = extract.extract_fields(text, [], None)
        
        if fields.get("reference_number") == "ABC123456":
            print(f"✅ Extraction successful: {fields['reference_number']}")
            return True
        else:
            print(f"❌ Extraction failed: {fields}")
            return False
            
    except Exception as e:
        print(f"❌ Extraction test failed: {e}")
        return False

def test_ml_models():
    """Test ML model components."""
    print("Testing ML model components...")
    
    try:
        from app.ml_models import create_receipt_features
        
        # Create test data
        img = Image.new('RGB', (224, 224), color='white')
        text = "Test receipt from Maybank Reference: ABC123"
        tokens = [{"text": "Test", "left": 10, "top": 10, "width": 50, "height": 20}]
        
        features = create_receipt_features(img, text, tokens)
        
        if (features.image_features.shape == (3, 224, 224) and 
            features.text_features == text and
            len(features.ocr_tokens) == 1):
            print("✅ ML features creation successful")
            return True
        else:
            print("❌ ML features creation failed")
            return False
            
    except Exception as e:
        print(f"❌ ML models test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 Bank Receipt ML System Validation")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_ocr_functionality,
        test_classification,
        test_extraction,
        test_ml_models
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! System is ready for training.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())