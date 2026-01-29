#!/usr/bin/env python3
"""
Test suite to validate 100% accuracy on all test cases.
Tests the enhanced pattern matcher V3 against all failing cases.
"""

import sys
import os

# Fix encoding for Windows console
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.append('.')

from app.ultimate_patterns_v3 import extract_all_fields_v3


def test_failing_cases():
    """Test the 5 previously failing test cases."""
    
    test_cases = [
        {
            'id': 8,
            'text': 'PBeBank Reference: PBB251031580390 Amount: RM500',
            'expected': 'PBB251031580390',
            'description': 'PBeBank Reference: with space after colon'
        },
        {
            'id': 11,
            'text': 'RHB Bank Reference: RHB251031943944 Status: OK',
            'expected': 'RHB251031943944',
            'description': 'RHB Bank Reference: format'
        },
        {
            'id': 27,
            'text': 'Payment Reference : PBB251031111111 Status : Done',
            'expected': 'PBB251031111111',
            'description': 'Payment Reference : with spaces around colon'
        },
        {
            'id': 28,
            'text': 'Bank Transfer-> RHB251031222222 | Amount: RM500',
            'expected': 'RHB251031222222',
            'description': 'Bank Transfer-> with arrow separator'
        },
        {
            'id': 35,
            'text': 'PBB 251031999999 Public Bank transaction',
            'expected': 'PBB251031999999',
            'description': 'PBB with space in ID'
        }
    ]
    
    print("=" * 80)
    print("TESTING PREVIOUSLY FAILING CASES")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        result = extract_all_fields_v3(test['text'])
        extracted_ids = result.get('all_ids', [])
        
        # Check if expected ID is in extracted IDs
        success = test['expected'] in extracted_ids
        
        status = "✅ PASS" if success else "❌ FAIL"
        
        print(f"\nTest {test['id']}: {status}")
        print(f"  Description: {test['description']}")
        print(f"  Text: {test['text']}")
        print(f"  Expected: {test['expected']}")
        print(f"  Extracted: {extracted_ids}")
        print(f"  Bank: {result.get('bank_name', 'Unknown')}")
        print(f"  Confidence: {result.get('confidence', 0):.2%}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(test_cases)} tests passed")
    print("=" * 80)
    
    return passed == len(test_cases)


def test_all_35_cases():
    """Test all 35 test cases from the validation suite."""
    
    all_test_cases = [
        ("Maybank Transfer Reference: MYCN251031853500 Status: Successful", "MYCN251031853500"),
        ("M2U Maybank Ref No: MYCN251031291491 Amount: RM100.00", "MYCN251031291491"),
        ("Maybank2u.com Transaction ID: MYCN251031842914", "MYCN251031842914"),
        ("CIMB Transaction ID: B10-2510-625105 Amount: RM250.00", "B10-2510-625105"),
        ("CIMB Clicks Reference: B10-2510-240816 Status: Completed", "B10-2510-240816"),
        ("CIMB Bank Transfer Ref: B10-2510-409674", "B10-2510-409674"),
        ("Public Bank Payment PBB251031758202 Completed", "PBB251031758202"),
        ("PBeBank Reference: PBB251031580390 Amount: RM500", "PBB251031580390"),
        ("Public Bank Transaction ID: PBB251031407081", "PBB251031407081"),
        ("RHB Transfer RHB251031180741 Done", "RHB251031180741"),
        ("RHB Bank Reference: RHB251031943944 Status: OK", "RHB251031943944"),
        ("RHB Now Transaction: RHB251031798726", "RHB251031798726"),
        ("HSBC Transaction HSBC251031529613 Processed", "HSBC251031529613"),
        ("HSBC Bank Transfer Ref: HSBC251031645651", "HSBC251031645651"),
        ("HSBC Reference Number: HSBC251031701530", "HSBC251031701530"),
        ("UOB Reference UOB251031200108 Status: Completed", "UOB251031200108"),
        ("UOB Bank Transaction ID: UOB251031806076", "UOB251031806076"),
        ("UOB Transfer Reference: UOB251031499003", "UOB251031499003"),
        ("Standard Chartered Payment SCB251031744606 Successful", "SCB251031744606"),
        ("SCB Transaction Ref: SCB251031911983 Amount: RM1000", "SCB251031911983"),
        ("Standard Chartered Bank ID: SCB251031295446", "SCB251031295446"),
        ("DuitNow Payment DN251031333313 Reference", "DN251031333313"),
        ("DuitNow Transfer ID: DN251031565451 Status: OK", "DN251031565451"),
        ("DuitNow Reference: DN251031915330", "DN251031915330"),
        ("Transfer completed. Ref: MYCN251031123456 Date: 31/10/25", "MYCN251031123456"),
        ("transaction id: b10-2510-987654 amount: rm100", "B10-2510-987654"),
        ("Payment Reference : PBB251031111111 Status : Done", "PBB251031111111"),
        ("Bank Transfer-> RHB251031222222 | Amount: RM500", "RHB251031222222"),
        ("#HSBC251031333333# Transaction Processed", "HSBC251031333333"),
        ("UOB:UOB251031444444:COMPLETED", "UOB251031444444"),
        ("SCB_Transfer_SCB251031555555_Success", "SCB251031555555"),
        ("DN251031666666 DuitNow Payment", "DN251031666666"),
        ("Reference No.MYCN251031777777 for your transfer", "MYCN251031777777"),
        ("CIMB ref#B10-2510-888888 completed successfully", "B10-2510-888888"),
        ("PBB 251031999999 Public Bank transaction", "PBB251031999999"),
    ]
    
    print("\n" + "=" * 80)
    print("TESTING ALL 35 TEST CASES")
    print("=" * 80)
    
    passed = 0
    failed = 0
    failed_cases = []
    
    for i, (text, expected) in enumerate(all_test_cases, 1):
        result = extract_all_fields_v3(text)
        extracted_ids = result.get('all_ids', [])
        
        success = expected in extracted_ids
        
        if success:
            passed += 1
            print(f"Test {i:2d}: ✅ PASS - {expected}")
        else:
            failed += 1
            failed_cases.append((i, text, expected, extracted_ids))
            print(f"Test {i:2d}: ❌ FAIL - Expected: {expected}, Got: {extracted_ids}")
    
    print("\n" + "=" * 80)
    print(f"OVERALL RESULTS: {passed}/{len(all_test_cases)} tests passed")
    print(f"Accuracy: {passed/len(all_test_cases)*100:.2f}%")
    print("=" * 80)
    
    if failed_cases:
        print("\nFAILED CASES DETAILS:")
        for case_num, text, expected, extracted in failed_cases:
            print(f"\nTest {case_num}:")
            print(f"  Text: {text}")
            print(f"  Expected: {expected}")
            print(f"  Extracted: {extracted}")
    
    return passed == len(all_test_cases)


if __name__ == "__main__":
    print("\n🧪 Bank Receipt OCR - 100% Accuracy Validation Test\n")
    
    # Test the 5 previously failing cases
    failing_cases_pass = test_failing_cases()
    
    # Test all 35 cases
    all_cases_pass = test_all_35_cases()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Previously Failing Cases (5): {'✅ ALL PASS' if failing_cases_pass else '❌ SOME FAILED'}")
    print(f"All Test Cases (35): {'✅ 100% ACCURACY ACHIEVED' if all_cases_pass else '❌ NOT 100% YET'}")
    print("=" * 80)
    
    if all_cases_pass:
        print("\n🎉 SUCCESS! 100% accuracy achieved on all test cases!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests still failing. Review the output above.")
        sys.exit(1)
