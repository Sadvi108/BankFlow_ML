"""Quick manual test of the two failing patterns"""
import sys
import os

# Change to project directory
os.chdir(r'c:\Users\User\Documents\trae_projects\CLA_Training')
sys.path.insert(0, '.')

# Import the pattern matcher
from app.ultimate_patterns_v3 import extract_all_fields_v3

print("="*80)
print("MANUAL PATTERN TEST - Verifying Fixes")
print("="*80)

# Test 31: SCB underscore format
test31 = "SCB_Transfer_SCB251031555555_Success"
result31 = extract_all_fields_v3(test31)
print(f"\nTest 31: {test31}")
print(f"Expected: SCB251031555555")
print(f"Got: {result31.get('all_ids', [])}")
print(f"Bank: {result31.get('bank_name')}")
if 'SCB251031555555' in result31.get('all_ids', []):
    print("✅ PASS")
else:
    print("❌ FAIL")

# Test 33: No space after No.
test33 = "Reference No.MYCN251031777777 for your transfer"
result33 = extract_all_fields_v3(test33)
print(f"\nTest 33: {test33}")
print(f"Expected: MYCN251031777777")
print(f"Got: {result33.get('all_ids', [])}")
print(f"Bank: {result33.get('bank_name')}")
if 'MYCN251031777777' in result33.get('all_ids', []):
    print("✅ PASS")
else:
    print("❌ FAIL")

print("\n" + "="*80)
