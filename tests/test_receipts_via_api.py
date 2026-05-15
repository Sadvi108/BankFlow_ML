"""
Test all receipts by uploading to the web API
This script tests each receipt individually via the running server
"""

import requests
import os
from pathlib import Path
import json
from datetime import datetime

RECEIPTS_DIR = Path("Receipts")
API_URL = "http://localhost:8081/extract"

def test_receipt(filepath):
    """Upload and test a single receipt"""
    print(f"\n{'='*80}")
    print(f"Testing: {filepath.name}")
    print(f"{'='*80}")
    
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (filepath.name, f)}
            response = requests.post(API_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                data = result.get('data', {})
                print(f"✅ SUCCESS")
                print(f"  Bank: {data.get('bank_name', 'Unknown')}")
                print(f"  All IDs: {data.get('all_ids', [])}")
                print(f"  Transaction IDs: {data.get('transaction_ids', [])}")
                print(f"  Reference IDs: {data.get('reference_ids', [])}")
                print(f"  Amount: {data.get('amount')}")
                print(f"  Date: {data.get('date')}")
                print(f"  Confidence: {data.get('confidence', 0):.2%}")
                
                return {
                    'filename': filepath.name,
                    'success': True,
                    'data': data
                }
            else:
                print(f"❌ FAILED - {result.get('error', 'Unknown error')}")
                return {
                    'filename': filepath.name,
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }
        else:
            print(f"❌ HTTP Error {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return {
                'filename': filepath.name,
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text[:100]}"
            }
    
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {
            'filename': filepath.name,
            'success': False,
            'error': str(e)
        }

def main():
    # Check if server is running
    try:
        health = requests.get("http://localhost:8081/health", timeout=5)
        if health.status_code != 200:
            print("❌ Server not responding properly")
            print("Please start the server: python simple_server.py")
            return
    except:
        print("❌ Server is not running!")
        print("Please start the server with: python simple_server.py")
        return
    
    print("✅ Server is running\n")
    
    # Find all receipts
    receipts = sorted(RECEIPTS_DIR.glob("*.*"))
    receipts = [r for r in receipts if r.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png']]
    
    print(f"Found {len(receipts)} receipts to test\n")
    
    results = []
    success_count = 0
    failure_count = 0
    
    # Test each receipt
    for receipt in receipts:
        result = test_receipt(receipt)
        results.append(result)
        
        if result['success']:
            success_count += 1
        else:
            failure_count += 1
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {len(receipts)}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failures: {failure_count}")
    print(f"Accuracy: {(success_count/len(receipts)*100):.2f}%")
    print(f"{'='*80}\n")
    
    # List failures
    if failure_count > 0:
        print("\nFAILED RECEIPTS:")
        print(f"{'='*80}")
        for result in results:
            if not result['success']:
                print(f"❌ {result['filename']}")
                print(f"   Error: {result.get('error', 'Unknown')}")
    
    # Save results
    output_file = f"api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_receipts': len(receipts),
            'success_count': success_count,
            'failure_count': failure_count,
            'accuracy': success_count/len(receipts)*100,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    if failure_count > 0:
        print(f"\n⚠️  NOT AT 100% ACCURACY - Need to fix {failure_count} receipts")
    else:
        print(f"\n🎉 100% ACCURACY ACHIEVED!")

if __name__ == "__main__":
    main()
