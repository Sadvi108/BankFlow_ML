#!/usr/bin/env python3
"""
Comprehensive Test Script for All Malaysian Banks
Tests all banks with various receipt formats to ensure 98%+ accuracy
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ultimate_patterns_v2 import UltimatePatternMatcherV2, extract_all_fields_v2
import json
from datetime import datetime

# Test cases for all Malaysian banks with realistic receipt formats
test_cases = [
    # Maybank test cases
    {
        "bank": "Maybank",
        "text": """
        MAYBANK BERHAD
        Transaction Reference: MYCN20251031123456
        Amount: RM 1,250.00
        Date: 31/10/2025
        M2U Ref: M2U123456789
        Status: Successful
        """,
        "expected_transaction_ids": ["MYCN20251031123456", "M2U123456789"],
        "expected_amount": "RM 1250.00",
        "expected_date": "31/10/2025"
    },
    {
        "bank": "Maybank",
        "text": """
        Maybank2u Transaction
        Ref: MBB20251031ABC123
        Payment: RM 850.50
        Transfer Date: 31-10-2025
        Transaction ID: FPX20251031XYZ789
        """,
        "expected_transaction_ids": ["MBB20251031ABC123", "FPX20251031XYZ789"],
        "expected_amount": "RM 850.50",
        "expected_date": "31-10-2025"
    },
    
    # CIMB test cases
    {
        "bank": "CIMB",
        "text": """
        CIMB BANK BERHAD
        Reference Number: B10-2025-ABC123456
        Transaction Amount: RM 2,340.00
        Transaction Date: 31/10/2025
        CIMB Clicks Ref: CBC20251031XYZ
        """,
        "expected_transaction_ids": ["B10-2025-ABC123456", "CBC20251031XYZ"],
        "expected_amount": "RM 2340.00",
        "expected_date": "31/10/2025"
    },
    {
        "bank": "CIMB",
        "text": """
        CIMB Bank Transaction
        Transfer Reference: CIMB20251031REF123
        Amount Transfer: RM 1,500.75
        Date: 31-10-2025
        IBG Reference: IBG20251031CIMB
        """,
        "expected_transaction_ids": ["CIMB20251031REF123", "IBG20251031CIMB"],
        "expected_amount": "RM 1500.75",
        "expected_date": "31-10-2025"
    },
    
    # Public Bank test cases
    {
        "bank": "Public Bank",
        "text": """
        PUBLIC BANK BERHAD
        Transaction Reference: PBB20251031REF789
        Payment Amount: RM 3,200.00
        Payment Date: 31 October 2025
        PBeBank Ref: PBE20251031PB123
        """,
        "expected_transaction_ids": ["PBB20251031REF789", "PBE20251031PB123"],
        "expected_amount": "RM 3200.00",
        "expected_date": "31 October 2025"
    },
    {
        "bank": "Public Bank",
        "text": """
        Public Bank Online Transfer
        Reference: PB20251031ONLINE456
        Transfer Amount: RM 950.25
        Transaction Date: 31/10/2025
        """,
        "expected_transaction_ids": ["PB20251031ONLINE456"],
        "expected_amount": "RM 950.25",
        "expected_date": "31/10/2025"
    },
    
    # RHB test cases
    {
        "bank": "RHB",
        "text": """
        RHB BANK BERHAD
        Transaction ID: RHB20251031TRANS123
        Amount: RM 4,100.00
        Date: 31-10-2025
        RHB Now Ref: RHBNOW20251031
        """,
        "expected_transaction_ids": ["RHB20251031TRANS123", "RHBNOW20251031"],
        "expected_amount": "RM 4100.00",
        "expected_date": "31-10-2025"
    },
    {
        "bank": "RHB",
        "text": """
        RHB Bank Transfer
        Reference Number: RHB20251031REF456
        Payment: RM 2,750.50
        Transfer Date: 31/10/2025
        """,
        "expected_transaction_ids": ["RHB20251031REF456"],
        "expected_amount": "RM 2750.50",
        "expected_date": "31/10/2025"
    },
    
    # Hong Leong Bank test cases
    {
        "bank": "Hong Leong Bank",
        "text": """
        HONG LEONG BANK BERHAD
        Transaction Reference: HLB20251031HL123
        Transaction Amount: RM 1,800.00
        Date: 31/10/2025
        HL Online Ref: HLO20251031REF
        """,
        "expected_transaction_ids": ["HLB20251031HL123", "HLO20251031REF"],
        "expected_amount": "RM 1800.00",
        "expected_date": "31/10/2025"
    },
    {
        "bank": "Hong Leong Bank",
        "text": """
        Hong Leong Bank Payment
        Ref: HONG20251031PAY789
        Amount: RM 1,200.25
        Payment Date: 31-10-2025
        """,
        "expected_transaction_ids": ["HONG20251031PAY789"],
        "expected_amount": "RM 1200.25",
        "expected_date": "31-10-2025"
    },
    
    # AmBank test cases
    {
        "bank": "AmBank",
        "text": """
        AMBANK BERHAD
        Reference: AMB20251031AMB456
        Amount: RM 2,500.00
        Transaction Date: 31/10/2025
        AmBank Online: AMBA20251031
        """,
        "expected_transaction_ids": ["AMB20251031AMB456", "AMBA20251031"],
        "expected_amount": "RM 2500.00",
        "expected_date": "31/10/2025"
    },
    {
        "bank": "AmBank",
        "text": """
        AmBank Transfer
        Transaction Ref: AM20251031TRANS123
        Transfer Amount: RM 1,650.75
        Date: 31-10-2025
        """,
        "expected_transaction_ids": ["AM20251031TRANS123"],
        "expected_amount": "RM 1650.75",
        "expected_date": "31-10-2025"
    },
    
    # HSBC test cases
    {
        "bank": "HSBC",
        "text": """
        HSBC BANK MALAYSIA BERHAD
        Transaction Reference: HSBC20251031HS123
        Payment Amount: RM 3,800.00
        Payment Date: 31 October 2025
        """,
        "expected_transaction_ids": ["HSBC20251031HS123"],
        "expected_amount": "RM 3800.00",
        "expected_date": "31 October 2025"
    },
    {
        "bank": "HSBC",
        "text": """
        HSBC International Transfer
        Ref: HSB20251031INT456
        Amount: RM 5,200.50
        Date: 31/10/2025
        """,
        "expected_transaction_ids": ["HSB20251031INT456"],
        "expected_amount": "RM 5200.50",
        "expected_date": "31/10/2025"
    },
    
    # UOB test cases
    {
        "bank": "UOB",
        "text": """
        UNITED OVERSEAS BANK (MALAYSIA) BHD
        Transaction ID: UOB20251031UOB123
        Amount: RM 2,100.00
        Transaction Date: 31/10/2025
        """,
        "expected_transaction_ids": ["UOB20251031UOB123"],
        "expected_amount": "RM 2100.00",
        "expected_date": "31/10/2025"
    },
    {
        "bank": "UOB",
        "text": """
        UOB Bank Transfer
        Reference: UOBK20251031REF456
        Transfer Amount: RM 1,350.25
        Date: 31-10-2025
        """,
        "expected_transaction_ids": ["UOBK20251031REF456"],
        "expected_amount": "RM 1350.25",
        "expected_date": "31-10-2025"
    },
    
    # Standard Chartered test cases
    {
        "bank": "Standard Chartered",
        "text": """
        STANDARD CHARTERED BANK MALAYSIA BERHAD
        Transaction Reference: SCB20251031SC123
        Amount: RM 4,500.00
        Date: 31/10/2025
        """,
        "expected_transaction_ids": ["SCB20251031SC123"],
        "expected_amount": "RM 4500.00",
        "expected_date": "31/10/2025"
    },
    {
        "bank": "Standard Chartered",
        "text": """
        Standard Chartered Online Payment
        Ref: STAN20251031ONLINE789
        Payment Amount: RM 2,800.50
        Payment Date: Oct 31, 2025
        """,
        "expected_transaction_ids": ["STAN20251031ONLINE789"],
        "expected_amount": "RM 2800.50",
        "expected_date": "Oct 31, 2025"
    },
    
    # DuitNow test cases
    {
        "bank": "DuitNow",
        "text": """
        DUITNOW INSTANT TRANSFER
        DuitNow Reference: DN20251031DN123
        Transfer Amount: RM 1,500.00
        Transfer Date: 31/10/2025
        Instant Transfer ID: INSTANT20251031
        """,
        "expected_transaction_ids": ["DN20251031DN123", "INSTANT20251031"],
        "expected_amount": "RM 1500.00",
        "expected_date": "31/10/2025"
    },
    {
        "bank": "DuitNow",
        "text": """
        Real Time Transfer via DuitNow
        Reference: DUITNOW20251031RT456
        Amount: RM 950.75
        Date: 31-10-2025
        """,
        "expected_transaction_ids": ["DUITNOW20251031RT456"],
        "expected_amount": "RM 950.75",
        "expected_date": "31-10-2025"
    }
]

def run_comprehensive_tests():
    """Run comprehensive tests for all banks."""
    print("🚀 Starting Comprehensive Bank Receipt Extraction Tests")
    print("=" * 60)
    
    matcher = UltimatePatternMatcherV2()
    results = []
    total_tests = len(test_cases)
    passed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/{total_tests}: {test_case['bank']}")
        print("-" * 40)
        
        # Run extraction
        result = matcher.extract_all_fields(test_case['text'])
        
        # Validate results
        validation = validate_extraction(result, test_case)
        
        # Print results
        print(f"🏦 Detected Bank: {result['bank_name']}")
        print(f"🎯 Transaction IDs: {result['transaction_ids']}")
        print(f"📄 Reference Numbers: {result['reference_numbers']}")
        print(f"💰 Amount: {result['amount']}")
        print(f"📅 Date: {result['date']}")
        print(f"⭐ Confidence: {result['confidence']:.1%}")
        
        # Print validation results
        print(f"✅ Validation: {'PASSED' if validation['passed'] else 'FAILED'}")
        if not validation['passed']:
            print(f"❌ Issues: {validation['issues']}")
        else:
            passed_tests += 1
        
        # Store results
        results.append({
            'test_case': test_case,
            'result': result,
            'validation': validation
        })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    accuracy = (passed_tests / total_tests) * 100
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Failed Tests: {total_tests - passed_tests}")
    print(f"Overall Accuracy: {accuracy:.1f}%")
    
    # Confidence analysis
    confidences = [r['result']['confidence'] for r in results]
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    
    print(f"\n📈 Confidence Analysis:")
    print(f"Average Confidence: {avg_confidence:.1%}")
    print(f"Minimum Confidence: {min_confidence:.1%}")
    print(f"Maximum Confidence: {max_confidence:.1%}")
    
    # Bank-specific performance
    print(f"\n🏦 Bank-Specific Performance:")
    bank_performance = {}
    for result in results:
        bank = result['test_case']['bank']
        if bank not in bank_performance:
            bank_performance[bank] = {'total': 0, 'passed': 0}
        bank_performance[bank]['total'] += 1
        if result['validation']['passed']:
            bank_performance[bank]['passed'] += 1
    
    for bank, stats in bank_performance.items():
        bank_accuracy = (stats['passed'] / stats['total']) * 100
        print(f"{bank}: {bank_accuracy:.1f}% ({stats['passed']}/{stats['total']})")
    
    # Target achievement
    print(f"\n🎯 Target Achievement:")
    if accuracy >= 98.0:
        print("✅ SUCCESS: 98%+ accuracy achieved!")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: {98.0 - accuracy:.1f}% short of 98% target")
    
    if avg_confidence >= 0.95:
        print("✅ SUCCESS: High confidence maintained!")
    else:
        print(f"⚠️  NEEDS IMPROVEMENT: Confidence could be higher")
    
    return results, accuracy

def validate_extraction(result, test_case):
    """Validate extraction results against expected values."""
    issues = []
    
    # Validate bank detection
    if result['bank_name'] != test_case['bank']:
        issues.append(f"Bank mismatch: expected {test_case['bank']}, got {result['bank_name']}")
    
    # Validate transaction IDs
    expected_ids = set(test_case['expected_transaction_ids'])
    found_ids = set(result['transaction_ids'])
    
    if not expected_ids.intersection(found_ids):
        issues.append(f"No expected transaction IDs found. Expected: {expected_ids}, Found: {found_ids}")
    
    # Validate amount (allow some format flexibility)
    if test_case['expected_amount']:
        expected_amount = test_case['expected_amount'].replace('RM ', '').replace(',', '')
        found_amount = (result['amount'] or '').replace('RM ', '').replace(',', '')
        if expected_amount != found_amount:
            issues.append(f"Amount mismatch: expected {test_case['expected_amount']}, got {result['amount']}")
    
    # Validate date (allow some format flexibility)
    if test_case['expected_date']:
        expected_date = test_case['expected_date']
        found_date = result['date'] or ''
        if expected_date not in found_date and found_date not in expected_date:
            issues.append(f"Date mismatch: expected {test_case['expected_date']}, got {result['date']}")
    
    # Validate confidence
    if result['confidence'] < 0.8:
        issues.append(f"Low confidence: {result['confidence']:.1%}")
    
    return {
        'passed': len(issues) == 0,
        'issues': issues
    }

def test_real_world_scenarios():
    """Test with real-world receipt scenarios."""
    print("\n" + "=" * 60)
    print("🌍 REAL-WORLD SCENARIO TESTS")
    print("=" * 60)
    
    real_world_cases = [
        # Mixed format receipt
        {
            "name": "Mixed Format Receipt",
            "text": """
            RECEIPT CONFIRMATION
            Bank: Maybank
            Transaction Reference: MYCN20251031123456
            Reference Number: REF20251031ABC
            DuitNow Ref: DN20251031DN123
            Amount: RM 1,500.00
            Date: 31/10/2025
            Status: SUCCESSFUL
            Transaction completed successfully
            """
        },
        # Poor quality OCR text
        {
            "name": "Poor OCR Quality",
            "text": """
            MAYBANK
            TRX REF: MYCN 2025 1031 123456
            AMT: RM 850.50
            DT: 31/10/2025
            STAT: COMPLETED
            """
        },
        # Multiple banks mentioned
        {
            "name": "Multiple Banks Mentioned",
            "text": """
            FUND TRANSFER
            From: Maybank Account
            To: CIMB Account
            Transaction ID: TRANS20251031REF123
            Amount: RM 2,000.00
            Transfer Date: 31/10/2025
            Reference: REF20251031MULTI
            """
        }
    ]
    
    matcher = UltimatePatternMatcherV2()
    
    for case in real_world_cases:
        print(f"\n📄 Testing: {case['name']}")
        result = matcher.extract_all_fields(case['text'])
        
        print(f"🏦 Bank: {result['bank_name']}")
        print(f"🎯 Transaction IDs: {result['transaction_ids']}")
        print(f"📄 References: {result['reference_numbers']}")
        print(f"💰 Amount: {result['amount']}")
        print(f"📅 Date: {result['date']}")
        print(f"⭐ Confidence: {result['confidence']:.1%}")

if __name__ == "__main__":
    # Run comprehensive tests
    results, accuracy = run_comprehensive_tests()
    
    # Run real-world scenario tests
    test_real_world_scenarios()
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'overall_accuracy': accuracy,
            'test_results': results[0] if isinstance(results, tuple) else results
        }, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    if accuracy >= 98.0:
        print("\n🎉 SUCCESS: System ready for production with 98%+ accuracy!")
    else:
        print(f"\n⚠️  System needs improvement to reach 98% accuracy target")