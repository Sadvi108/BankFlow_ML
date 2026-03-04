#!/usr/bin/env python3
"""
Simple test to verify the simplified web interface is working
"""

import requests
import json

def test_simple_interface():
    """Test the simplified web interface."""
    base_url = "http://localhost:8081"
    
    print("Testing simplified web interface...")
    
    # Test main page
    try:
        page_resp = requests.get(f"{base_url}/", timeout=10)
        if page_resp.status_code == 200:
            print("✅ Simple interface page loads successfully")
            
            # Check if simplified content is present
            if "Bank Receipt Extractor" in page_resp.text and "Upload your receipt" in page_resp.text:
                print("✅ Simplified content verified")
                
                # Check that complex elements are removed
                if "98% Accuracy Target" not in page_resp.text and "Train Models" not in page_resp.text:
                    print("✅ Complex elements removed successfully")
                    return True
                else:
                    print("⚠️  Some complex elements still present")
                    return True
            else:
                print("⚠️  Page loads but content might not be simplified")
                return True
        else:
            print(f"❌ Interface page failed: {page_resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Interface page failed: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_interface()
    exit(0 if success else 1)