#!/usr/bin/env python3
"""
Test script to demonstrate the web interface functionality
"""

import requests
import json

def test_web_interface():
    """Test the web interface endpoints."""
    
    print("🌐 Testing Web Interface")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API Health: {health_data['status']}")
            print(f"✅ ML Models Loaded: {health_data['ml_models_loaded']}")
            print(f"✅ OCR Available: {health_data['ocr_available']}")
        else:
            print("❌ Health check failed")
            return
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        return
    
    # Test web interface endpoint
    try:
        response = requests.get(f"{base_url}/web")
        if response.status_code == 200:
            print(f"✅ Web interface available at: {base_url}/web")
            print(f"📄 HTML content length: {len(response.text)} characters")
            
            # Check if key elements are present
            if "Bank Receipt Processing System" in response.text:
                print("✅ Title found in web interface")
            if "upload-area" in response.text:
                print("✅ Upload area found")
            if "process_receipt" in response.text:
                print("✅ API integration found")
            
            print("\n🎯 Web Interface Features:")
            print("• 📁 Drag & drop file upload")
            print("• 📋 Multiple file support")
            print("• 🔍 Real-time processing results")
            print("• 💎 Transaction ID extraction")
            print("• 💰 Amount and date extraction")
            print("• 📊 Processing time display")
            print("• 🎯 API connection status")
            
        else:
            print(f"❌ Web interface returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error accessing web interface: {e}")
    
    print("\n" + "=" * 50)
    print("🚀 Ready for browser testing!")
    print(f"🌐 Open this URL in your browser: {base_url}/web")
    print("\n📋 Instructions:")
    print("1. Click or drag & drop bank receipts")
    print("2. Click 'Upload Receipts' to process")
    print("3. View extracted transaction IDs and details")
    print("4. Test with your own receipts!")

if __name__ == "__main__":
    test_web_interface()