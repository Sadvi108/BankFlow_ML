import re

def normalize_v3(text):
    # Old logic: re.sub(r'(.)\1{3,}', r'\1', text)
    # New logic: re.sub(r'(.)\1{3}', r'\1', text)
    text = re.sub(r'(.)\1{3}', r'\1', text)
    return text

test_text = "222200006666888888883333888855555555QQQQ"
expected = "206883855Q"
result = normalize_v3(test_text)

print(f"Input:    {test_text}")
print(f"Expected: {expected}")
print(f"Result:   {result}")
print(f"Success:  {result == expected}")

test_text2 = "RRRRMMMM 11115555....00000000"
expected2 = "RM 15.00"
result2 = normalize_v3(test_text2)
print(f"\nInput:    {test_text2}")
print(f"Expected: {expected2}")
print(f"Result:   {result2}")
print(f"Success:  {result2 == expected2}")
