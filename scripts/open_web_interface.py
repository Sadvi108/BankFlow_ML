#!/usr/bin/env python3
"""
Script to open the web interface in your default browser
"""

import webbrowser
import time
import requests
import sys

def check_server():
    """Check if the server is running."""
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("🌐 Opening Bank Receipt Processing Web Interface...")
    print("=" * 60)
    
    # Check if server is running
    if not check_server():
        print("❌ API server is not running on localhost:8000")
        print("📝 Please start the server first with:")
        print("   python -m uvicorn app.main_enhanced:app --reload --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    print("✅ API server is running")
    print("🚀 Opening web interface in your browser...")
    
    # Wait a moment for the browser to open
    time.sleep(1)
    
    # Open the web interface
    url = "http://localhost:8000/web"
    webbrowser.open(url)
    
    print(f"🌐 Web interface opened: {url}")
    print("\n📋 How to use:")
    print("1. 📁 Click or drag & drop bank receipts (PDF, JPG, PNG)")
    print("2. 📤 Click 'Upload Receipts' to process")
    print("3. 🔍 View extracted transaction IDs and details")
    print("4. 🧪 Test with multiple receipts from different banks")
    print("\n💡 Tips:")
    print("- Supports multiple file upload")
    print("- Shows processing time and confidence scores")
    print("- Displays extracted transaction IDs, amounts, and dates")
    print("- API connection status shown in top-right corner")
    
    print("\n🎯 The system is ready for testing!")
    print("=" * 60)

if __name__ == "__main__":
    main()