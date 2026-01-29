from app.ultimate_patterns_v3 import extract_all_fields_v3
import json

text = """
20251031PBBEMYKL010OR
          M10169596
"""

print("Testing full extraction on multiline ID...")
results = extract_all_fields_v3(text)
print("\n--- RESULTS ---")
print(json.dumps(results, indent=2))
