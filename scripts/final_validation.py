import json
import torch
from pathlib import Path
from enhanced_patterns import EnhancedPatternMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def final_validation_test():
    """Final comprehensive validation test"""
    pattern_matcher = EnhancedPatternMatcher()
    
    # Test with realistic receipt patterns from your actual data
    test_cases = [
        # Maybank patterns
        {"text": "Maybank Transfer Reference: MYCN251031853500 Status: Successful", "expected": "MYCN251031853500"},
        {"text": "M2U Maybank Ref No: MYCN251031291491 Amount: RM100.00", "expected": "MYCN251031291491"},
        {"text": "Maybank2u.com Transaction ID: MYCN251031842914", "expected": "MYCN251031842914"},
        
        # CIMB patterns  
        {"text": "CIMB Transaction ID: B10-2510-625105 Amount: RM250.00", "expected": "B10-2510-625105"},
        {"text": "CIMB Clicks Reference: B10-2510-240816 Status: Completed", "expected": "B10-2510-240816"},
        {"text": "CIMB Bank Transfer Ref: B10-2510-409674", "expected": "B10-2510-409674"},
        
        # Public Bank patterns
        {"text": "Public Bank Payment PBB251031758202 Completed", "expected": "PBB251031758202"},
        {"text": "PBeBank Reference: PBB251031580390 Amount: RM500", "expected": "PBB251031580390"},
        {"text": "Public Bank Transaction ID: PBB251031407081", "expected": "PBB251031407081"},
        
        # RHB patterns
        {"text": "RHB Transfer RHB251031180741 Done", "expected": "RHB251031180741"},
        {"text": "RHB Bank Reference: RHB251031943944 Status: OK", "expected": "RHB251031943944"},
        {"text": "RHB Now Transaction: RHB251031798726", "expected": "RHB251031798726"},
        
        # HSBC patterns
        {"text": "HSBC Transaction HSBC251031529613 Processed", "expected": "HSBC251031529613"},
        {"text": "HSBC Bank Transfer Ref: HSBC251031645651", "expected": "HSBC251031645651"},
        {"text": "HSBC Reference Number: HSBC251031701530", "expected": "HSBC251031701530"},
        
        # UOB patterns
        {"text": "UOB Reference UOB251031200108 Status: Completed", "expected": "UOB251031200108"},
        {"text": "UOB Bank Transaction ID: UOB251031806076", "expected": "UOB251031806076"},
        {"text": "UOB Transfer Reference: UOB251031499003", "expected": "UOB251031499003"},
        
        # Standard Chartered patterns
        {"text": "Standard Chartered Payment SCB251031744606 Successful", "expected": "SCB251031744606"},
        {"text": "SCB Transaction Ref: SCB251031911983 Amount: RM1000", "expected": "SCB251031911983"},
        {"text": "Standard Chartered Bank ID: SCB251031295446", "expected": "SCB251031295446"},
        
        # DuitNow patterns
        {"text": "DuitNow Payment DN251031333313 Reference", "expected": "DN251031333313"},
        {"text": "DuitNow Transfer ID: DN251031565451 Status: OK", "expected": "DN251031565451"},
        {"text": "DuitNow Reference: DN251031915330", "expected": "DN251031915330"},
        
        # Mixed and challenging cases
        {"text": "Transfer completed. Ref: MYCN251031123456 Date: 31/10/25", "expected": "MYCN251031123456"},
        {"text": "transaction id: b10-2510-987654 amount: rm100", "expected": "B10-2510-987654"},
        {"text": "Payment Reference : PBB251031111111 Status : Done", "expected": "PBB251031111111"},
        {"text": "Bank Transfer-> RHB251031222222 | Amount: RM500", "expected": "RHB251031222222"},
        {"text": "#HSBC251031333333# Transaction Processed", "expected": "HSBC251031333333"},
        {"text": "UOB:UOB251031444444:COMPLETED", "expected": "UOB251031444444"},
        {"text": "SCB_Transfer_SCB251031555555_Success", "expected": "SCB251031555555"},
        {"text": "DN251031666666 DuitNow Payment", "expected": "DN251031666666"},
        
        # Edge cases with variations
        {"text": "Reference No.MYCN251031777777 for your transfer", "expected": "MYCN251031777777"},
        {"text": "CIMB ref#B10-2510-888888 completed successfully", "expected": "B10-2510-888888"},
        {"text": "PBB 251031999999 Public Bank transaction", "expected": "PBB251031999999"},
    ]
    
    results = []
    total_tests = len(test_cases)
    
    logger.info(f"Running final validation with {total_tests} test cases...")
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        expected = test_case["expected"]
        
        # Extract using enhanced pattern matcher
        extracted_ids = pattern_matcher.extract_transaction_ids(text)
        
        # Check if expected ID was found
        found_expected = expected in extracted_ids
        
        # Also check if any valid transaction ID was found
        any_valid = len(extracted_ids) > 0
        
        results.append({
            'test_case': i,
            'text': text,
            'expected': expected,
            'extracted': extracted_ids,
            'found_expected': found_expected,
            'any_valid': any_valid,
            'success': found_expected
        })
        
        status = "✓" if found_expected else "✗"
        logger.info(f"{status} Test {i}: {extracted_ids}")
    
    # Calculate success rates
    exact_matches = sum(1 for r in results if r['found_expected'])
    any_id_found = sum(1 for r in results if r['any_valid'])
    
    exact_success_rate = (exact_matches / total_tests) * 100
    any_id_success_rate = (any_id_found / total_tests) * 100
    
    logger.info(f"\n📊 FINAL VALIDATION RESULTS:")
    logger.info(f"Exact Match Success Rate: {exact_success_rate:.1f}% ({exact_matches}/{total_tests})")
    logger.info(f"Any ID Found Success Rate: {any_id_success_rate:.1f}% ({any_id_found}/{total_tests})")
    
    # Detailed breakdown by bank
    banks = ['Maybank', 'CIMB', 'Public Bank', 'RHB', 'HSBC', 'UOB', 'Standard Chartered', 'DuitNow']
    for bank in banks:
        bank_tests = [r for r in results if bank.lower() in r['text'].lower()]
        if bank_tests:
            bank_success = sum(1 for r in bank_tests if r['success'])
            bank_rate = (bank_success / len(bank_tests)) * 100
            logger.info(f"{bank}: {bank_rate:.1f}% ({bank_success}/{len(bank_tests)})")
    
    # Save results
    results_data = {
        'total_tests': total_tests,
        'exact_success_rate': exact_success_rate,
        'any_id_success_rate': any_id_success_rate,
        'exact_matches': exact_matches,
        'any_id_found': any_id_found,
        'results': results,
        'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else '2025-11-14'
    }
    
    with open('final_validation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Return the key metric
    return exact_success_rate

if __name__ == "__main__":
    success_rate = final_validation_test()
    
    if success_rate >= 90.0:
        print(f"🎉 SUCCESS! Achieved {success_rate:.1f}% success rate (target: 90%)")
    else:
        print(f"⚠️  Below target: {success_rate:.1f}% (target: 90%)")
    
    print(f"Final Success Rate: {success_rate:.1f}%")