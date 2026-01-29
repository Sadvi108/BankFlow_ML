"""Debug the pattern matching for the two failing cases"""
import re

# Test the patterns directly

# Test case 31 patterns
test31_text = "SCB_Transfer_SCB251031555555_Success"
test31_text_upper = test31_text.upper()

# Original failing pattern
pattern31_old = r'\b(?:SCB|Standard\s*Chartered)_(?:Transfer|Payment|Txn)_(SCB[A-Z0-9]{8,20})_'
# My new pattern
pattern31_new1 = r'\b(?:SCB|SC|Standard)_(?:Transfer|Payment|Txn|Trx)_([A-Z0-9]{8,25})_'
pattern31_new2 = r'\b(?:[A-Z]{2,5})_(?:Transfer|Payment|Txn|Trx)_([A-Z0-9]{8,25})_'

print("Test 31: SCB_Transfer_SCB251031555555_Success")
print(f"Text upper: {test31_text_upper}")
print(f"Old pattern: {re.findall(pattern31_old, test31_text_upper)}")
print(f"New pattern 1 (SCB-specific): {re.findall(pattern31_new1, test31_text_upper)}")
print(f"New pattern 2 (generic): {re.findall(pattern31_new2, test31_text_upper)}")
print()

# Test case 33 patterns
test33_text = "Reference No.MYCN251031777777 for your transfer"
test33_text_upper = test33_text.upper()

# Patterns
pattern33_old = r'\b(?:Ref|Reference)\s*No\.([A-Z0-9]{8,20})\b'
pattern33_new = r'\b(?:Ref|Reference)\s*No\.\s*([A-Z0-9]{8,20})\b'

print("Test 33: Reference No.MYCN251031777777")
print(f"Text upper: {test33_text_upper}")
print(f"Old pattern: {re.findall(pattern33_old, test33_text_upper)}")
print(f"New pattern: {re.findall(pattern33_new, test33_text_upper)}")
print()

# Try alternative patterns
# For test 33 - the issue is \s* means zero or more spaces, so it should work
# Let's try without  word boundary at the end
pattern33_alt1 = r'(?:Ref|Reference)\s*No\.([A-Z0-9]{8,20})'
print(f"Alt pattern (no \\b): {re.findall(pattern33_alt1, test33_text_upper)}")

# Try with explicit no space
pattern33_alt2 = r'\b(?:Ref|Reference)\s*No\.([A-Z0-9]+)'
print(f"Alt pattern (no length limit): {re.findall(pattern33_alt2, test33_text_upper)}")
