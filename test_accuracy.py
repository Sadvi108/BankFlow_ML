#!/usr/bin/env python3
"""
Comprehensive test script to validate >90% extraction accuracy
Tests all Malaysian banks with various receipt formats
"""

import requests
import json
import time
from pathlib import Path
import base64
from typing import Dict, List, Any
from PIL import Image, ImageDraw, ImageFont
import io

# Test server URL
BASE_URL = "http://localhost:8081"

# Test cases for different banks with expected results
TEST_CASES = [
    {
        "name": "Maybank Receipt 1",
        "text": """
        Maybank2u Transaction Confirmation
        Date: 31/10/2025
        Reference Number: M2U12345678
        Amount: RM 50.00
        Transaction ID: MBB98765432
        Maybank Berhad - Banking Made Easy
        """,
        "expected_bank": "Maybank",
        "expected_transaction_id": "MBB98765432",
        "expected_reference": "M2U12345678",
        "expected_amount": "RM 50.00"
    },
    {
        "name": "CIMB Receipt 1",
        "text": """
        CIMB Bank Berhad
        Transaction Receipt
        Date: 15/11/2025
        Reference No: CIMB20251115001
        Amount: RM 125.50
        Transaction ID: CIMBTXN789012
        Thank you for banking with CIMB
        """,
        "expected_bank": "CIMB",
        "expected_transaction_id": "CIMBTXN789012",
        "expected_reference": "CIMB20251115001",
        "expected_amount": "RM 125.50"
    },
    {
        "name": "Public Bank Receipt 1",
        "text": """
        Public Bank Berhad
        Fund Transfer Receipt
        Date: 20/11/2025
        Reference: PBB2025FT001234
        Amount: RM 200.00
        Transaction Reference: PBFTX987654
        Public Bank - Your Trusted Partner
        """,
        "expected_bank": "Public Bank",
        "expected_transaction_id": "PBFTX987654",
        "expected_reference": "PBB2025FT001234",
        "expected_amount": "RM 200.00"
    },
    {
        "name": "RHB Bank Receipt 1",
        "text": """
        RHB Banking Group
        Transaction Confirmation
        Date: 25/11/2025
        Reference Number: RHB20251125001
        Amount: RM 75.25
        Transaction ID: RHBTXN123456
        RHB - Together We Progress
        """,
        "expected_bank": "RHB",
        "expected_transaction_id": "RHBTXN123456",
        "expected_reference": "RHB20251125001",
        "expected_amount": "RM 75.25"
    },
    {
        "name": "HSBC Receipt 1",
        "text": """
        The Hongkong and Shanghai Banking Corporation
        Transaction Receipt
        Date: 30/11/2025
        Reference: HSBC2025TR001
        Amount: RM 300.00
        Transaction ID: HSBCTXN789012
        HSBC - The World's Local Bank
        """,
        "expected_bank": "HSBC",
        "expected_transaction_id": "HSBCTXN789012",
        "expected_reference": "HSBC2025TR001",
        "expected_amount": "RM 300.00"
    },
    {
        "name": "UOB Receipt 1",
        "text": """
        United Overseas Bank (Malaysia) Bhd
        Transaction Confirmation
        Date: 05/12/2025
        Reference No: UOB20251205001
        Amount: RM 150.75
        Transaction ID: UOBTXN456789
        UOB - Right By You
        """,
        "expected_bank": "UOB",
        "expected_transaction_id": "UOBTXN456789",
        "expected_reference": "UOB20251205001",
        "expected_amount": "RM 150.75"
    },
    {
        "name": "Standard Chartered Receipt 1",
        "text": """
        Standard Chartered Bank Malaysia Berhad
        Transaction Receipt
        Date: 10/12/2025
        Reference: SCB2025TX001
        Amount: RM 250.00
        Transaction ID: SCBTXN234567
        Standard Chartered - Here for good
        """,
        "expected_bank": "Standard Chartered",
        "expected_transaction_id": "SCBTXN234567",
        "expected_reference": "SCB2025TX001",
        "expected_amount": "RM 250.00"
    },
    {
        "name": "DuitNow Receipt 1",
        "text": """
        DuitNow Transfer Confirmation
        Date: 15/12/2025
        Reference Number: DN20251215001
        Amount: RM 89.99
        Transaction ID: DNTXN890123
        DuitNow - Instant Transfer
        """,
        "expected_bank": "DuitNow",
        "expected_transaction_id": "DNTXN890123",
        "expected_reference": "DN20251215001",
        "expected_amount": "RM 89.99"
    },
    {
        "name": "Ambank Receipt 1",
        "text": """
        AmBank (M) Berhad
        Transaction Confirmation
        Date: 20/12/2025
        Reference: AMB2025TR001
        Amount: RM 175.50
        Transaction ID: AMBTXN345678
        AmBank - Malaysia's Preferred Bank
        """,
        "expected_bank": "Ambank",
        "expected_transaction_id": "AMBTXN345678",
        "expected_reference": "AMB2025TR001",
        "expected_amount": "RM 175.50"
    },
    {
        "name": "Hong Leong Bank Receipt 1",
        "text": """
        Hong Leong Bank Berhad
        Fund Transfer Receipt
        Date: 25/12/2025
        Reference No: HLB20251225001
        Amount: RM 120.25
        Transaction ID: HLBTXN567890
        Hong Leong Bank - Building Relationships
        """,
        "expected_bank": "Hong Leong Bank",
        "expected_transaction_id": "HLBTXN567890",
        "expected_reference": "HLB20251225001",
        "expected_amount": "RM 120.25"
    }
]

def create_test_image(text: str, filename: str) -> str:
    """Create a test image with receipt text"""
    # Create image with white background
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use default font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Split text into lines and draw
    lines = text.strip().split('\n')
    y_position = 50
    
    for line in lines:
        line = line.strip()
        if line:
            draw.text((50, y_position), line, fill='black', font=font)
            y_position += 25
    
    # Save image
    img.save(filename, 'PNG')
    return filename

def test_extraction_accuracy():
    """Test extraction accuracy for all banks"""
    print("🧪 Starting comprehensive accuracy test for >90% extraction accuracy")
    print("=" * 60)
    
    results = []
    total_tests = len(TEST_CASES)
    passed_tests = 0
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n📋 Test {i}/{total_tests}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Create a test image with the receipt text - use bank name in filename
            bank_name = test_case['expected_bank'].lower().replace(' ', '_')
            test_file = f"{bank_name}_receipt_{i}.png"
            create_test_image(test_case['text'], test_file)
            
            # Test the extraction
            with open(test_file, 'rb') as f:
                files = {'file': (test_file, f, 'image/png')}
                response = requests.post(f"{BASE_URL}/extract", files=files)
            
            if response.status_code == 200:
                result = response.json()
                data = result.get('data', {})
                
                # Check bank detection
                detected_bank = data.get('bank', '').lower()
                expected_bank = test_case['expected_bank'].lower()
                bank_match = expected_bank in detected_bank or detected_bank in expected_bank
                
                # Check transaction ID
                transaction_id = data.get('transaction_id', '')
                transaction_match = transaction_id == test_case['expected_transaction_id']
                
                # Check reference number
                all_refs = data.get('all_reference_ids', [])
                reference_match = test_case['expected_reference'] in all_refs or any(
                    test_case['expected_reference'] in ref for ref in all_refs
                )
                
                # Check amount
                amount = data.get('amount', '')
                amount_match = amount == test_case['expected_amount']
                
                # Calculate confidence
                confidence = data.get('global_confidence', 0)
                
                # Overall test result
                test_passed = bank_match and (transaction_match or reference_match) and amount_match and confidence >= 90
                
                print(f"🏦 Bank Detection: {'✅' if bank_match else '❌'}")
                print(f"   Expected: {test_case['expected_bank']}")
                print(f"   Detected: {data.get('bank', 'None')}")
                
                print(f"🆔 Transaction ID: {'✅' if transaction_match else '❌'}")
                print(f"   Expected: {test_case['expected_transaction_id']}")
                print(f"   Detected: {transaction_id}")
                
                print(f"📋 Reference: {'✅' if reference_match else '❌'}")
                print(f"   Expected: {test_case['expected_reference']}")
                print(f"   All References: {all_refs}")
                
                print(f"💰 Amount: {'✅' if amount_match else '❌'}")
                print(f"   Expected: {test_case['expected_amount']}")
                print(f"   Detected: {amount}")
                
                print(f"📊 Confidence: {confidence}% {'✅' if confidence >= 90 else '❌'}")
                
                if test_passed:
                    passed_tests += 1
                    print(f"🎉 Test PASSED")
                else:
                    print(f"❌ Test FAILED")
                
                results.append({
                    'test_name': test_case['name'],
                    'passed': test_passed,
                    'bank_match': bank_match,
                    'transaction_match': transaction_match,
                    'reference_match': reference_match,
                    'amount_match': amount_match,
                    'confidence': confidence,
                    'detected_bank': data.get('bank', 'None'),
                    'detected_transaction': transaction_id,
                    'detected_references': all_refs,
                    'detected_amount': amount
                })
                
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                results.append({
                    'test_name': test_case['name'],
                    'passed': False,
                    'error': f"API Error: {response.status_code}"
                })
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            results.append({
                'test_name': test_case['name'],
                'passed': False,
                'error': str(e)
            })
        
        # Clean up test file
        try:
            import os
            os.remove(test_file)
        except:
            pass
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Calculate overall accuracy
    accuracy = (passed_tests / total_tests) * 100
    
    print("\n" + "=" * 60)
    print(f"📊 ACCURACY TEST RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Failed Tests: {total_tests - passed_tests}")
    print(f"Overall Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 90:
        print(f"🎉 SUCCESS: System achieves >90% accuracy target!")
    else:
        print(f"❌ FAILED: System needs improvement to reach >90% accuracy")
    
    # Detailed breakdown
    print("\n📋 Detailed Results:")
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status} - {result['test_name']}")
        if not result['passed'] and 'error' not in result:
            print(f"     Bank: {result['detected_bank']} (expected bank detection)")
            print(f"     Transaction: {result['detected_transaction']}")
            print(f"     References: {result['detected_references']}")
            print(f"     Amount: {result['detected_amount']}")
            print(f"     Confidence: {result['confidence']}%")
        elif 'error' in result:
            print(f"     Error: {result['error']}")
    
    return accuracy >= 90

if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Test server connectivity
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is ready")
            success = test_extraction_accuracy()
            exit(0 if success else 1)
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server: {str(e)}")
        print("Make sure the server is running on port 8081")
        exit(1)