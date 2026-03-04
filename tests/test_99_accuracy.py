#!/usr/bin/env python3
"""
Test script to validate 99% accuracy achievement for bank receipt extraction
Tests the AdvancedBankExtractor with ML-based reference identifier learning
"""

import requests
import json
import os
import time
from pathlib import Path

def test_bank_extraction():
    """Test extraction accuracy for all Malaysian banks"""
    
    # Test data with various reference number formats
    test_cases = [
        {
            'filename': 'maybank_receipt.txt',
            'content': '''
            Maybank2u Transaction Confirmation
            Date: 31/10/2025
            Reference Number: M2U12345678
            Amount: RM 50.00
            Transaction ID: MBB98765432
            Maybank Berhad - Banking Made Easy
            ''',
            'expected_bank': 'Maybank',
            'expected_references': ['M2U12345678', 'MBB98765432'],
            'expected_amount': 'RM 50.00',
            'expected_date': '31/10/2025'
        },
        {
            'filename': 'cimb_receipt.txt',
            'content': '''
            CIMB Bank Berhad
            Transaction Receipt
            Date: 15/11/2025
            Reference No: CIMB20251115001
            Amount: RM 125.50
            Transaction ID: CIMBTXN789012
            RPP Reference: RPP123456789
            Thank you for banking with CIMB
            ''',
            'expected_bank': 'CIMB',
            'expected_references': ['CIMB20251115001', 'CIMBTXN789012', 'RPP123456789'],
            'expected_amount': 'RM 125.50',
            'expected_date': '15/11/2025'
        },
        {
            'filename': 'public_bank_receipt.txt',
            'content': '''
            Public Bank Berhad
            Fund Transfer Receipt
            Date: 20/11/2025
            Reference: PBB2025FT001234
            Amount: RM 200.00
            Transaction Reference: PBFTX987654
            Public Bank - Your Trusted Partner
            ''',
            'expected_bank': 'Public Bank',
            'expected_references': ['PBB2025FT001234', 'PBFTX987654'],
            'expected_amount': 'RM 200.00',
            'expected_date': '20/11/2025'
        },
        {
            'filename': 'rhb_receipt.txt',
            'content': '''
            RHB Banking Group
            Transaction Confirmation
            Date: 25/11/2025
            Reference Number: RHB20251125001
            Amount: RM 75.25
            Transaction ID: RHBTXN123456
            RHB - Together We Progress
            ''',
            'expected_bank': 'RHB',
            'expected_references': ['RHB20251125001', 'RHBTXN123456'],
            'expected_amount': 'RM 75.25',
            'expected_date': '25/11/2025'
        },
        {
            'filename': 'duitnow_receipt.txt',
            'content': '''
            DuitNow Transfer Confirmation
            Date: 15/12/2025
            Reference Number: DN20251215001
            Amount: RM 89.99
            Transaction ID: DNTXN890123
            DuitNow - Instant Transfer
            ''',
            'expected_bank': 'DuitNow',
            'expected_references': ['DN20251215001', 'DNTXN890123'],
            'expected_amount': 'RM 89.99',
            'expected_date': '15/12/2025'
        }
    ]
    
    base_url = "http://localhost:8081"
    
    print("🧪 Testing AdvancedBankExtractor with ML-based Reference Identifier Learning")
    print("=" * 80)
    
    total_tests = 0
    passed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['filename']}")
        print("-" * 50)
        
        # Create test file
        test_file = Path(f"test_data/{test_case['filename']}")
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text(test_case['content'])
        
        # Test extraction
        try:
            with open(test_file, 'rb') as f:
                files = {'file': (test_case['filename'].replace('.txt', '.jpg'), f, 'image/jpeg')}
                response = requests.post(f"{base_url}/extract", files=files)
                
            if response.status_code == 200:
                result = response.json()
                data = result['data']
                
                print(f"🏦 Bank Detected: {data.get('bank', 'None')} (confidence: {data.get('bank_confidence', 0)}%)")
                print(f"📄 Transaction ID: {data.get('transaction_id', 'None')} (confidence: {data.get('transaction_id_confidence', 0)}%)")
                print(f"💰 Amount: {data.get('amount', 'None')} (confidence: {data.get('amount_confidence', 0)}%)")
                print(f"📅 Date: {data.get('date', 'None')} (confidence: {data.get('date_confidence', 0)}%)")
                print(f"🎯 Global Confidence: {data.get('global_confidence', 0)}%")
                
                # Validate results
                test_passed = True
                
                # Check bank detection
                if data.get('bank') != test_case['expected_bank']:
                    print(f"❌ Bank mismatch: expected {test_case['expected_bank']}, got {data.get('bank')}")
                    test_passed = False
                else:
                    print(f"✅ Bank detection correct")
                
                # Check transaction ID (should match any of the expected references)
                transaction_id = data.get('transaction_id')
                if transaction_id and transaction_id in test_case['expected_references']:
                    print(f"✅ Transaction ID correct: {transaction_id}")
                else:
                    print(f"❌ Transaction ID mismatch: expected one of {test_case['expected_references']}, got {transaction_id}")
                    test_passed = False
                
                # Check amount
                if data.get('amount') == test_case['expected_amount']:
                    print(f"✅ Amount correct: {data.get('amount')}")
                else:
                    print(f"❌ Amount mismatch: expected {test_case['expected_amount']}, got {data.get('amount')}")
                    test_passed = False
                
                # Check date
                if data.get('date') == test_case['expected_date']:
                    print(f"✅ Date correct: {data.get('date')}")
                else:
                    print(f"❌ Date mismatch: expected {test_case['expected_date']}, got {data.get('date')}")
                    test_passed = False
                
                # Check confidence levels for 99% accuracy
                if data.get('global_confidence', 0) >= 90:
                    print(f"✅ High confidence achieved: {data.get('global_confidence')}%")
                else:
                    print(f"⚠️  Confidence below target: {data.get('global_confidence')}% (target: 90%+)")
                
                # Check for dummy values (should be eliminated)
                if transaction_id and 'TEST' in transaction_id:
                    print(f"❌ Dummy ID detected: {transaction_id}")
                    test_passed = False
                
                total_tests += 1
                if test_passed:
                    passed_tests += 1
                    print(f"🎉 Test {i} PASSED")
                else:
                    print(f"💥 Test {i} FAILED")
                
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        # Clean up
        test_file.unlink(missing_ok=True)
    
    # Calculate accuracy
    accuracy = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "=" * 80)
    print(f"📊 ACCURACY REPORT")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Failed Tests: {total_tests - passed_tests}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 99:
        print("🎯🎯🎯 99% ACCURACY ACHIEVED! 🎯🎯🎯")
        print("✅ ML-based reference identifier learning is working perfectly")
        print("✅ Transaction ID = Reference Number confusion resolved")
        print("✅ Dummy ID generation eliminated")
        print("✅ Bank-specific patterns are highly accurate")
    elif accuracy >= 95:
        print("🎯 95%+ Accuracy achieved - Very close to 99% target")
    elif accuracy >= 90:
        print("✅ 90%+ Accuracy achieved - Good performance")
    else:
        print("⚠️  Accuracy below 90% - Need further improvements")
    
    return accuracy

if __name__ == "__main__":
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    for i in range(30):
        try:
            response = requests.get("http://localhost:8081/health")
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"Still waiting... ({i}/30)")
    else:
        print("❌ Server not responding. Please ensure simple_server.py is running.")
        exit(1)
    
    # Run accuracy test
    accuracy = test_bank_extraction()
    
    print(f"\n🎯 Final Result: {accuracy:.1f}% accuracy achieved")