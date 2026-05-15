import json
from enhanced_patterns import EnhancedPatternMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_improved_patterns():
    """Test with improved fallback patterns"""
    pattern_matcher = EnhancedPatternMatcher()
    
    # Add fallback patterns to the existing patterns
    fallback_patterns = [
        r'\b[A-Z]{2,4}\d{8,15}\b',  # Generic bank format
        r'\b[A-Z]{2,4}\d{4}-\d{6,8}\b',  # CIMB-like format
        r'\bDN\d{8,15}\b',  # DuitNow
        r'\b(?:ref|reference|id|no|number)\s*[:#-]?\s*([A-Z]{2,4}\d{8,15})',  # Generic with prefix
        r'\b([A-Z]{2,4}\d{8,15})\s*(?:ref|reference|id|no|number)',  # Generic with suffix
    ]
    
    # Add fallback patterns to the pattern matcher
    pattern_matcher.patterns.extend(fallback_patterns)
    
    # Test the failed cases from previous run
    failed_cases = [
        {"text": "Payment Reference : PBB251031111111 Status : Done", "expected": "PBB251031111111"},
        {"text": "Bank Transfer-> RHB251031222222 | Amount: RM500", "expected": "RHB251031222222"},
        {"text": "PBB 251031999999 Public Bank transaction", "expected": "PBB251031999999"},
    ]
    
    improved = 0
    for test_case in failed_cases:
        text = test_case["text"]
        expected = test_case["expected"]
        
        extracted_ids = pattern_matcher.extract_transaction_ids(text)
        found = any(extracted_id['text'].upper().replace("-", "") == expected.upper().replace("-", "") for extracted_id in extracted_ids)
        
        if found:
            improved += 1
            logger.info(f"✓ FIXED: {text} -> {[id['text'] for id in extracted_ids]}")
        else:
            logger.info(f"✗ STILL FAILING: {text} -> {[id['text'] for id in extracted_ids]}")
    
    logger.info(f"Improved {improved}/{len(failed_cases)} cases with fallback patterns")
    
    # Now run the full test again
    test_cases = [
        {"text": "Payment Reference : PBB251031111111 Status : Done", "expected": "PBB251031111111"},
        {"text": "Bank Transfer-> RHB251031222222 | Amount: RM500", "expected": "RHB251031222222"},
        {"text": "PBB 251031999999 Public Bank transaction", "expected": "PBB251031999999"},
    ]
    
    results = []
    for test_case in test_cases:
        text = test_case["text"]
        expected = test_case["expected"]
        
        extracted_ids = pattern_matcher.extract_transaction_ids(text)
        found = any(extracted_id['text'].upper().replace("-", "") == expected.upper().replace("-", "") for extracted_id in extracted_ids)
        
        results.append({
            'text': text,
            'expected': expected,
            'extracted': [id['text'] for id in extracted_ids],
            'success': found
        })
    
    success_rate = sum(1 for r in results if r['success']) / len(results) * 100
    logger.info(f"Success rate with improvements: {success_rate:.1f}%")
    
    return results

if __name__ == "__main__":
    results = test_improved_patterns()