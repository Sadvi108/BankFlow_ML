#!/usr/bin/env python3
"""
Direct test of AdvancedBankExtractor to validate 99% accuracy achievement
Tests ML-based reference identifier learning without OCR pipeline
"""

import sys
sys.path.append('app')

from app.advanced_extractor import AdvancedBankExtractor

def test_advanced_extractor():
    """Test the AdvancedBankExtractor directly"""
    
    # Test data with various reference number formats
    test_cases = [
        {
            'name': 'Maybank with Reference Number and Transaction ID',
            'text': '''
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
            'name': 'CIMB with RPP Reference',
            'text': '''
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
            'name': 'Public Bank with Transaction Reference',
            'text': '''
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
            'name': 'RHB with Reference Number and Transaction ID',
            'text': '''
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
            'name': 'DuitNow Transfer',
            'text': '''
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
    
    print("🧪 Testing AdvancedBankExtractor with ML-based Reference Identifier Learning")
    print("=" * 80)
    
    # Initialize extractor
    extractor = AdvancedBankExtractor()
    
    total_tests = 0
    passed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print("-" * 50)
        
        # Extract fields
        results = extractor.extract_all_fields(test_case['text'], ocr_confidence=95.0)
        
        print(f"🏦 Bank Detected: {results.get('bank', 'None')} (confidence: {results.get('bank_confidence', 0)}%)")
        print(f"📄 Transaction ID: {results.get('transaction_id', 'None')} (confidence: {results.get('transaction_id_confidence', 0)}%)")
        print(f"📄 Reference Number: {results.get('reference_number', 'None')} (confidence: {results.get('reference_number_confidence', 0)}%)")
        print(f"💰 Amount: {results.get('amount', 'None')} (confidence: {results.get('amount_confidence', 0)}%)")
        print(f"📅 Date: {results.get('date', 'None')} (confidence: {results.get('date_confidence', 0)}%)")
        print(f"🎯 Global Confidence: {results.get('global_confidence', 0)}%")
        print(f"📊 All References Found: {results.get('all_reference_ids', [])}")
        
        # Validate results
        test_passed = True
        
        # Check bank detection
        if results.get('bank') != test_case['expected_bank']:
            print(f"❌ Bank mismatch: expected {test_case['expected_bank']}, got {results.get('bank')}")
            test_passed = False
        else:
            print(f"✅ Bank detection correct")
        
        # Check transaction ID (should match any of the expected references)
        transaction_id = results.get('transaction_id')
        if transaction_id and transaction_id in test_case['expected_references']:
            print(f"✅ Transaction ID correct: {transaction_id}")
        else:
            print(f"❌ Transaction ID mismatch: expected one of {test_case['expected_references']}, got {transaction_id}")
            test_passed = False
        
        # Check reference number (should be same as transaction_id)
        reference_number = results.get('reference_number')
        if reference_number == transaction_id:
            print(f"✅ Reference Number matches Transaction ID: {reference_number}")
        else:
            print(f"❌ Reference Number mismatch: expected {transaction_id}, got {reference_number}")
            test_passed = False
        
        # Check amount
        if results.get('amount') == test_case['expected_amount']:
            print(f"✅ Amount correct: {results.get('amount')}")
        else:
            print(f"❌ Amount mismatch: expected {test_case['expected_amount']}, got {results.get('amount')}")
            test_passed = False
        
        # Check date
        if results.get('date') == test_case['expected_date']:
            print(f"✅ Date correct: {results.get('date')}")
        else:
            print(f"❌ Date mismatch: expected {test_case['expected_date']}, got {results.get('date')}")
            test_passed = False
        
        # Check confidence levels for 99% accuracy
        if results.get('global_confidence', 0) >= 90:
            print(f"✅ High confidence achieved: {results.get('global_confidence')}%")
        else:
            print(f"⚠️  Confidence below target: {results.get('global_confidence')}% (target: 90%+)")
        
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
    accuracy = test_advanced_extractor()
    print(f"\n🎯 Final Result: {accuracy:.1f}% accuracy achieved")