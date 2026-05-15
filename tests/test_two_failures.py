"""Test the two failing cases directly"""
import sys
sys.path.insert(0, '.')

from app.ultimate_patterns_v3 import extract_all_fields_v3

# Test case 31: Underscore-separated SCB ID
test31_text = "SCB_Transfer_SCB251031555555_Success"
test31_expected = "SCB251031555555"

result31 = extract_all_fields_v3(test31_text)
ids31 = result31.get('all_ids', [])
test31_pass = test31_expected in ids31

print("Test 31: SCB_Transfer_SCB251031555555_Success")
print(f"Expected: {test31_expected}")
print(f"Extracted: {ids31}")
print(f"Result: {'✅ PASS' if test31_pass else '❌ FAIL'}")
print()

# Test case 33: No space after "No."
test33_text = "Reference No.MYCN251031777777 for your transfer"
test33_expected = "MYCN251031777777"

result33 = extract_all_fields_v3(test33_text)
ids33 = result33.get('all_ids', [])
test33_pass = test33_expected in ids33

print("Test 33: Reference No.MYCN251031777777")
print(f"Expected: {test33_expected}")
print(f"Extracted: {ids33}")
print(f"Result: {'✅ PASS' if test33_pass else '❌ FAIL'}")
print()

if test31_pass and test33_pass:
    print("🎉 Both tests PASSED! 100% accuracy achieved!")
    sys.exit(0)
else:
    print("⚠️ One or more tests failed")
    sys.exit(1)
