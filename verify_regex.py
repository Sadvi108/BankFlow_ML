
from app.ultimate_patterns_v3 import extract_all_fields_v3

def test_extraction():
    test_cases = [
        {
            "text": "Receipt for Payment\nDate: 25/11/2025\nAmount: RM 150.00\nRef: MBB12345678",
            "expected_date": "25/11/2025",
            "expected_amount": "150.00"
        },
        {
            "text": "Transfer Successful\n2023-12-01\nTotal Amount MYR 1,234.50\nTransaction ID: 99998888",
            "expected_date": "2023-12-01",
            "expected_amount": "1,234.50"
        },
        {
            "text": "Public Bank\nRef: PBB12345678\n12 Oct 2024\nRM 50.00",
            "expected_date": "12 Oct 2024",
            "expected_amount": "50.00"
        }
    ]

    print("Testing Regex Extraction...")
    passes = 0
    for i, case in enumerate(test_cases):
        print(f"\nCase {i+1}:")
        result = extract_all_fields_v3(case["text"])
        
        date_match = result['date'] == case['expected_date']
        amount_match = result['amount'] == case['expected_amount']
        
        print(f"  Date: {result['date']} (Expected: {case['expected_date']}) - {'OK' if date_match else 'FAIL'}")
        print(f"  Amount: {result['amount']} (Expected: {case['expected_amount']}) - {'OK' if amount_match else 'FAIL'}")
        
        if date_match and amount_match:
            passes += 1

    print(f"\nPassed {passes}/{len(test_cases)}")

if __name__ == "__main__":
    test_extraction()
