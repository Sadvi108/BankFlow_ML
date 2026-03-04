#!/usr/bin/env python3
"""
Test the simplified web interface functionality
"""

import requests
import json
import os
from pathlib import Path

def test_simple_interface():
    """Test the simplified web interface."""
    base_url = "http://localhost:8081"
    
    print("Testing simplified web interface...")
    
    # Test health endpoint
    try:
        health_resp = requests.get(f"{base_url}/health", timeout=5)
        if health_resp.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {health_resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test main page loads
    try:
        page_resp = requests.get(f"{base_url}/", timeout=10)
        if page_resp.status_code == 200:
            print("✅ Simple interface page loads successfully")
            # Check if simplified content is present
            if "Bank Receipt Extractor" in page_resp.text and "Upload your receipt" in page_resp.text:
                print("✅ Simplified content verified")
            else:
                print("⚠️  Page loads but content might not be simplified")
        else:
            print(f"❌ Interface page failed: {page_resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Interface page failed: {e}")
        return False
    
    # Test file upload functionality
    receipts_dir = Path(__file__).parent / "Receipts"
    pdf_files = list(receipts_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files for testing")
        return False
    
    test_pdf = receipts_dir / "AMBANK - CUSTOMER REF.pdf"
    if not test_pdf.exists():
        test_pdf = pdf_files[0]
    
    print(f"Testing upload with: {test_pdf.name}")
    
    try:
        with open(test_pdf, 'rb') as f:
            files = {'file': (test_pdf.name, f, 'application/pdf')}
            response = requests.post(f"{base_url}/extract_enhanced", files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Upload and extraction successful")
                
                # Check simplified results
                bank = result.get('bank', {}).get('name', 'Unknown')
                extraction = result.get('extraction', result.get('enhanced_extraction', {}))
                confidence = result.get('confidence', 0)
                
                print(f"📊 Results:")
                print(f"  Bank: {bank}")
                print(f"  Transaction ID: {extraction.get('transaction_id', 'Not found')}")
                print(f"  Reference Number: {extraction.get('reference_number', 'Not found')}")
                print(f"  Amount: {extraction.get('amount', 'Not found')}")
                print(f"  Confidence: {confidence * 100:.1f}%")
                
                if bank != 'Unknown':
                    print("✅ Meaningful extraction achieved")
                    return True
                else:
                    print("⚠️  Processing successful but bank not detected")
                    return True
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_interface()
    exit(0 if success else 1)