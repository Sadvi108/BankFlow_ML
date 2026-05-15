#!/usr/bin/env python3
"""
Test the web API with the V3 pattern matcher
"""
import requests
import json

# Test the extraction endpoint with sample text
def test_extraction():
    url = "http://localhost:8081/extract"
    
    # Test cases from our 100% accuracy test
    test_cases = [
        {
            'name': 'Maybank Test 1',
            'text': 'Maybank Transfer Reference: MYCN251031853500 Status: Successful'
        },
        {
            'name': 'CIMB Test',
            'text': 'CIMB Transaction ID: B10-2510-625105 Amount: RM250.00'
        },
        {
            'name': 'Public Bank Test',
            'text': 'PBeBank Reference: PBB251031580390 Amount: RM500'
        },
        {
            'name': 'RHB Test',
            'text': 'RHB Bank Reference: RHB251031943944 Status: OK'
        },
        {
            'name': 'Underscore Separator Test',
            'text': 'SCB_Transfer_SCB251031555555_Success'
        },
        {
            'name': 'No. Without Space Test',
            'text': 'Reference No.MYCN251031777777 for your transfer'
        }
    ]
    
    print("=" * 80)
    print("Testing Web API with V3 Pattern Matcher")
    print("=" * 80)
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        print(f"  Input: {test['text']}")
        
        # Create a text file with the test content
        with open('temp_test.txt', 'w', encoding='utf-8') as f:
            f.write(test['text'])
        
        # Upload the file
        try:
            with open('temp_test.txt', 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = requests.post(url, files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    data = result.get('data', {})
                    print(f"  ✅ Success!")
                    print(f"  Bank: {data.get('bank_name', 'Unknown')}")
                    print(f"  Transaction IDs: {data.get('all_ids', [])}")
                    print(f"  Confidence: {data.get('confidence', 0):.2%}")
                else:
                    print(f"  ❌ Failed: {result}")
            else:
                print(f"  ❌ HTTP Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_extraction()
