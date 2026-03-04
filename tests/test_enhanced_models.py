#!/usr/bin/env python3
"""
Test the enhanced ML models for transaction ID extraction
"""

import requests
import json
import os
from pathlib import Path

def test_enhanced_models():
    """Test the enhanced models with all receipts."""
    base_url = "http://localhost:8000"
    
    # Test receipts directory
    receipts_dir = Path("Receipts")
    
    if not receipts_dir.exists():
        print("Receipts directory not found!")
        return
    
    results = []
    total_tests = 0
    successful_extractions = 0
    
    print("Testing enhanced ML models...")
    print("=" * 60)
    
    # Test each receipt
    for receipt_file in receipts_dir.glob("*"):
        if receipt_file.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']:
            print(f"\nTesting: {receipt_file.name}")
            
            try:
                # Upload file
                with open(receipt_file, 'rb') as f:
                    files = {'file': f}
                    response = requests.post(f"{base_url}/extract", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract transaction ID
                    transaction_id = None
                    if result.get('transaction_number'):
                        transaction_id = result['transaction_number']
                    elif result.get('reference_number'):
                        transaction_id = result['reference_number']
                    elif result.get('transaction_id'):
                        transaction_id = result['transaction_id']
                    
                    total_tests += 1
                    
                    if transaction_id and len(str(transaction_id)) > 3:  # Valid ID found
                        successful_extractions += 1
                        print(f"✅ SUCCESS: Found transaction ID: {transaction_id}")
                        status = "SUCCESS"
                    else:
                        print(f"❌ FAILED: No transaction ID found")
                        status = "FAILED"
                    
                    results.append({
                        'filename': receipt_file.name,
                        'status': status,
                        'transaction_id': transaction_id,
                        'full_result': result
                    })
                    
                else:
                    print(f"❌ ERROR: HTTP {response.status_code}")
                    results.append({
                        'filename': receipt_file.name,
                        'status': 'ERROR',
                        'error': response.text
                    })
                    
            except Exception as e:
                print(f"❌ EXCEPTION: {str(e)}")
                results.append({
                    'filename': receipt_file.name,
                    'status': 'EXCEPTION',
                    'error': str(e)
                })
    
    # Calculate success rate
    if total_tests > 0:
        success_rate = (successful_extractions / total_tests) * 100
        print(f"\n{'='*60}")
        print(f"ENHANCED MODEL TEST RESULTS:")
        print(f"Total receipts tested: {total_tests}")
        print(f"Successful extractions: {successful_extractions}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"{'='*60}")
        
        # Show detailed results
        print("\nDETAILED RESULTS:")
        for result in results:
            if result['status'] == 'SUCCESS':
                print(f"✅ {result['filename']}: {result['transaction_id']}")
            elif result['status'] == 'FAILED':
                print(f"❌ {result['filename']}: No ID found")
            else:
                print(f"⚠️  {result['filename']}: {result.get('error', 'Unknown error')}")
        
        # Save results
        with open('enhanced_test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'successful_extractions': successful_extractions,
                    'success_rate': success_rate
                },
                'detailed_results': results
            }, f, indent=2)
        
        print(f"\nResults saved to enhanced_test_results.json")
        
        return success_rate >= 90.0
    else:
        print("No receipts were tested successfully!")
        return False

if __name__ == "__main__":
    # Ensure server is running
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("Server is running, starting enhanced model tests...")
            test_enhanced_models()
        else:
            print("Server health check failed!")
    except:
        print("Cannot connect to server. Please ensure it's running on localhost:8000")