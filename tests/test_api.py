import requests
import json
from pathlib import Path

def test_receipt_extraction():
    """Test the enhanced receipt extraction API"""
    
    # Test with different receipt files
    test_files = [
        "MAYBANK - CUSTOMER REF.pdf",
        "CIMB - INV REF.pdf", 
        "PUBLIC BANK - INVOICE.pdf",
        "HSBC - REF- INV NUMBER.pdf",
        "DUITNOW - INV REF.pdf"
    ]
    
    base_url = "http://localhost:8001"
    
    print("🧪 Testing Enhanced Receipt Extraction API")
    print("=" * 50)
    
    for filename in test_files:
        file_path = Path("Receipts") / filename
        
        if not file_path.exists():
            print(f"⚠️  File not found: {filename}")
            continue
            
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f, 'application/pdf')}
                response = requests.post(f"{base_url}/extract", files=files)
                
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n📄 {filename}")
                print(f"🏦 Bank: {result.get('bank', {}).get('name', 'Unknown')} "
                      f"(confidence: {result.get('bank', {}).get('confidence', 0):.2f})")
                
                transaction_result = result.get('transaction_id', {})
                if transaction_result:
                    print(f"💳 Transaction ID: {transaction_result.get('best_match', 'None')}")
                    print(f"🎯 Method: {transaction_result.get('method', 'unknown')}")
                    print(f"📊 Confidence: {transaction_result.get('confidence', 0):.2f}")
                    
                    # Show all extracted IDs
                    all_ids = transaction_result.get('transaction_ids', [])
                    if all_ids:
                        print(f"🔍 All IDs found: {all_ids}")
                    
                    pattern_matches = transaction_result.get('pattern_matches', [])
                    if pattern_matches:
                        print(f"📝 Pattern matches: {pattern_matches}")
                else:
                    print("❌ No transaction ID found")
                    
                processing_time = result.get('meta', {}).get('processing_time', 0)
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                
            else:
                print(f"❌ Error processing {filename}: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception with {filename}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Testing completed!")

if __name__ == "__main__":
    test_receipt_extraction()