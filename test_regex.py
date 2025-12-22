import re

text = "Transaction Reference’ No. + 2025'9310309000839"

patterns = [
    r'\b(?:Ref|Reference)\W*No\W*[:+]?\s*([A-Z0-9\']+){8,30}',
    r'Transaction\s*Reference\W*No\W*[\d\']+',
    r'\bReference\W+No\W+([A-Z0-9\']+)'
]

print(f"Text: {text}")

for i, p in enumerate(patterns):
    print(f"\nPattern {i+1}: {p}")
    match = re.search(p, text, re.IGNORECASE)
    if match:
        print(f"✅ Match: {match.group(0)}")
        if match.groups():
            print(f"   Group 1: {match.group(1)}")
    else:
        print("❌ No match")
