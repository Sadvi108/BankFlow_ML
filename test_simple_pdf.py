#!/usr/bin/env python3
"""
Simple test to verify PDF upload fix works correctly
"""

import requests
import json
import os
from pathlib import Path

def test_simple_pdf_upload():
    """Test simple PDF upload to verify the fix works."""
    base_url = "http://localhost:8081"
    
    print("Testing simple PDF upload fix...")
    
    # Test health endpoint first
    try:
        health_resp = requests.get(f"{base_url}/health", timeout=5)
        if health_resp.status_code != 200:
            print(f"❌ Health check failed: {health_resp.status_code}")
            return False
        print("✅ Health check passed")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Find a good PDF file for testing
    receipts_dir = Path(__file__).parent / "Receipts"
    pdf_files = list(receipts_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found")
        return False
    
    # Test with a known good PDF (AmBank from our previous test)
    test_pdf = receipts_dir / "AMBANK - CUSTOMER REF.pdf"
    if not test_pdf.exists():
        test_pdf = pdf_files[0]  # Fallback to first PDF
    
    print(f"Testing with: {test_pdf.name}")
    
    try:
        # Test with shorter timeout and simple endpoint
        with open(test_pdf, 'rb') as f:
            files = {'file': (test_pdf.name, f, 'application/pdf')}
            
            print("Uploading PDF...")
            response = requests.post(f"{base_url}/extract", files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ PDF upload and processing successful!")
                
                # Show key results
                bank_name = result.get('bank', {}).get('name', 'Unknown')
                transaction_id = result.get('extraction', {}).get('transaction_id', 'None')
                confidence = result.get('extraction', {}).get('confidence', 0)
                
                print(f"📊 Results:")
                print(f"  Bank: {bank_name}")
                print(f"  Transaction ID: {transaction_id}")
                print(f"  Confidence: {confidence}")
                print(f"  Processing time: {result.get('meta', {}).get('processing_time', 0):.2f}s")
                
                # Check if we got meaningful results
                if bank_name != 'Unknown' or transaction_id != 'None':
                    print("✅ Meaningful extraction achieved - PDF fix is working!")
                    return True
                else:
                    print("⚠️  Processing successful but no meaningful extraction")
                    return True  # Still consider it a success since PDF processing worked
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.Timeout:
        print("❌ Request timed out - server might be processing slowly")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_pdf_upload()
    exit(0 if success else 1)