
import sys
import os
from pathlib import Path
from pprint import pprint

# Add app directory to path
sys.path.append(str(Path(__file__).parent / "app"))

from app.extract import extract_fields

def test_extraction():
    test_cases = [
        {
            "name": "Maybank with spaces in ID",
            "text": """
            Maybank2u.com
            Open 3rd Party Transfer
            Status: Successful
            Reference Number: 1234 5678 9012
            Transaction Date: 23 Jan 2026
            Amount: RM 1,234.50
            To: ABC Corp
            """,
            "expected_id": "123456789012"
        },
        {
            "name": "CIMB with arrow",
            "text": """
            CIMB Clicks
            Fund Transfer -> 9876543210
            Date: 23/01/2026
            Amount: RM50.00
            """,
            "expected_id": "9876543210"
        },
        {
            "name": "Public Bank PBB prefix",
            "text": """
            Public Bank Berhad
            PBB Reference: PBB12345678
            Amount: RM 100.00
            """,
            "expected_id": "PBB12345678"
        },
        {
            "name": "DuitNow with spaces",
            "text": """
            DuitNow Transfer
            Reference ID: DN 1234 5678
            Date: 2026-01-23
            """,
            "expected_id": "DN12345678"
        }
    ]

    print("Running V3 Extraction Tests...\n")
    
    for case in test_cases:
        print(f"Testing: {case['name']}")
        # Create dummy tokens (not used by V3 text extraction, only for boxes)
        tokens = [] 
        
        result = extract_fields(case['text'], tokens)
        
        extracted_ref = result.get('reference_number')
        extracted_txn = result.get('transaction_id')
        
        print(f"  Expected ID: {case['expected_id']}")
        print(f"  Extracted Ref: {extracted_ref}")
        print(f"  Extracted Txn: {extracted_txn}")
        print(f"  Detected Bank: {result.get('meta', {}).get('all_ids', [])}")
        
        if extracted_ref == case['expected_id'] or extracted_txn == case['expected_id']:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
        print("-" * 40)

if __name__ == "__main__":
    test_extraction()
