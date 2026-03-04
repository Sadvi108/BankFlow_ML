import requests
import json
from pathlib import Path

def test_receipt_extraction():
    """Test the enhanced receipt extraction API"""
    
    # Get actual receipt files
    receipts_dir = Path("Receipts")
    receipt_files = list(receipts_dir.glob("*.pdf"))[:3]  # Test first 3 PDFs
    
    if not receipt_files:
        print("❌ No PDF receipt files found in Receipts directory")
        return
    
    base_url = "http://localhost:8001"
    
    print("🧪 Testing Enhanced Receipt Extraction API")
    print("=" * 50)
    print(f"Testing {len(receipt_files)} receipt files...")
    
    for file_path in receipt_files:
        filename = file_path.name
            
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
                    best_match = transaction_result.get('best_match')
                    if best_match:
                        print(f"💳 Transaction ID: {best_match}")
                        print(f"🎯 Method: {transaction_result.get('method', 'unknown')}")
                        print(f"📊 Confidence: {transaction_result.get('confidence', 0):.2f}")
                    else:
                        print("❌ No transaction ID extracted")
                        
                    # Show all extracted IDs
                    all_ids = transaction_result.get('transaction_ids', [])
                    if all_ids:
                        print(f"🔍 All IDs found: {[id['text'] for id in all_ids]}")
                    
                    pattern_matches = transaction_result.get('pattern_matches', [])
                    if pattern_matches:
                        print(f"📝 Pattern matches: {pattern_matches}")
                else:
                    print("❌ No transaction ID found")
                    
                processing_time = result.get('meta', {}).get('processing_time', 0)
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                
                # Show other extracted fields
                fields = result.get('fields', {})
                if fields:
                    print(f"📋 Other fields: {list(fields.keys())}")
                
            else:
                print(f"❌ Error processing {filename}: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception with {filename}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Testing completed!")
    print("🌐 Web interface available at: http://localhost:8001")
    print("📊 Try uploading receipts through the web interface!")

if __name__ == "__main__":
    test_receipt_extraction()