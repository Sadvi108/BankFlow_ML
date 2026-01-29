#!/usr/bin/env python3
"""
Test web interface API with both image and PDF files to ensure complete functionality
"""

import requests
import json
import os
from pathlib import Path

def test_web_interface_api():
    """Test the web interface API with various file types."""
    base_url = "http://localhost:8080"
    
    print("Testing web interface API...")
    
    # Test health endpoint
    try:
        health_resp = requests.get(f"{base_url}/health", timeout=10)
        if health_resp.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {health_resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Find test files
    receipts_dir = Path(__file__).parent / "Receipts"
    
    # Test with different file types
    test_files = []
    
    # Find PDF files
    pdf_files = list(receipts_dir.glob("*.pdf"))
    if pdf_files:
        test_files.append(("PDF", pdf_files[0]))
    
    # Find image files
    img_files = list(receipts_dir.glob("*.png")) + list(receipts_dir.glob("*.jpg")) + list(receipts_dir.glob("*.jpeg"))
    if img_files:
        test_files.append(("Image", img_files[0]))
    
    if not test_files:
        print("❌ No test files found")
        return False
    
    print(f"Testing with {len(test_files)} file types")
    
    success_count = 0
    
    for file_type, file_path in test_files:
        print(f"\n--- Testing {file_type}: {file_path.name} ---")
        
        try:
            # Test standard extraction endpoint
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/octet-stream')}
                
                # Test standard extraction
                response = requests.post(f"{base_url}/extract", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    print(f"✅ Standard extraction successful")
                    print(f"  Bank: {result.get('bank', {}).get('name', 'Unknown')}")
                    print(f"  Transaction ID: {result.get('extraction', {}).get('transaction_id', 'None')}")
                    print(f"  Reference Number: {result.get('extraction', {}).get('reference_number', 'None')}")
                    print(f"  Amount: {result.get('extraction', {}).get('amount', 'None')}")
                    print(f"  Confidence: {result.get('extraction', {}).get('confidence', 0)}")
                    print(f"  Processing time: {result.get('meta', {}).get('processing_time', 0):.2f}s")
                    
                    # Test enhanced extraction
                    f.seek(0)  # Reset file pointer
                    enhanced_response = requests.post(f"{base_url}/extract_enhanced", files=files, timeout=30)
                    
                    if enhanced_response.status_code == 200:
                        enhanced_result = enhanced_response.json()
                        print(f"✅ Enhanced extraction successful")
                        print(f"  Enhanced confidence: {enhanced_result.get('confidence', 0)}")
                        print(f"  Method: {enhanced_result.get('method', 'unknown')}")
                        
                        # Check if we got meaningful results
                        extraction = result.get('extraction', {})
                        if (extraction.get('transaction_id') or 
                            extraction.get('reference_number') or 
                            extraction.get('amount') or
                            result.get('bank', {}).get('name') != 'Unknown'):
                            success_count += 1
                            print("✅ Meaningful extraction achieved")
                        else:
                            print("⚠️  No meaningful extraction")
                    else:
                        print(f"❌ Enhanced extraction failed: {enhanced_response.status_code}")
                        
                else:
                    print(f"❌ Standard extraction failed: {response.status_code}")
                    if response.status_code == 500:
                        print(f"Error: {response.text}")
                    
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n--- Summary ---")
    print(f"Files tested: {len(test_files)}")
    print(f"Successful extractions: {success_count}")
    print(f"Success rate: {success_count/len(test_files)*100:.1f}%")
    
    if success_count > 0:
        print("✅ Web interface API is working correctly")
        return True
    else:
        print("⚠️  Web interface API needs attention")
        return False

if __name__ == "__main__":
    success = test_web_interface_api()
    exit(0 if success else 1)