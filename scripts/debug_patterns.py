
import logging
from app.ultimate_patterns_v3 import UltimatePatternMatcherV3

logging.basicConfig(level=logging.INFO)

matcher = UltimatePatternMatcherV3()

text = """=== PAGE 1 ===
Open Interbank
Status: Successful
Reference number: 7901990048
Transaction date: 31 Oct 2025 12:09:52
Amount: RM10.60
Beneficiary Name : D&D CONTROL (M) SDN BHD
Receiving Bank : RHB BANK
Beneficiary Account Number : 21246660001343
Recipient reference: MXC12511726684
Other payment details:
Note: This receipt is computer generated and no signature is required."""

print("Detecting bank...")
bank = matcher.detect_bank(text)
print(f"Bank: {bank}")

print("\nExtracting IDs...")
# We need to access the internal method to see scores
matcher_ids = matcher.extract_transaction_ids(text, bank)
print(f"IDs found: {matcher_ids}")

# Let's debug the scoring manually by copying the logic from extract_transaction_ids
print("\n--- Detailed Scoring Debug ---")
text_normalized = matcher.normalize_text(text)
text_upper = text_normalized.upper()
candidates = []

# 1. Generic patterns
print("Checking generic patterns...")
for pattern in matcher.generic_patterns['transaction_ids']:
    import re
    matches = re.findall(pattern, text_upper, re.IGNORECASE)
    is_labeled = any(k in pattern for k in ['Ref', 'Reference', 'ID', 'No', 'Trx', 'Txn', 'Invoice', 'Bill'])
    base_score = 60 if is_labeled else 10
    
    for m in matches:
        val = m[0] if isinstance(m, tuple) else m
        if val:
            print(f"Match: '{val}' | Pattern: {pattern} | Base Score: {base_score}")
            candidates.append((val, base_score, "generic"))

# 2. Bank patterns
print(f"Checking {bank} patterns...")
if bank in matcher.bank_patterns:
    for pattern in matcher.bank_patterns[bank]['patterns']:
        matches = re.findall(pattern, text_upper, re.IGNORECASE)
        for m in matches:
            val = m[0] if isinstance(m, tuple) else m
            if val:
                print(f"Match: '{val}' | Pattern: {pattern} | Score: 80")
                candidates.append((val, 80, "bank_specific"))

# Scoring logic
print("\nScoring...")
valid_scored_ids = []
seen = set()
blacklist = set() # Assume empty for now

for tid, initial_score, source in candidates:
    tid_repaired = matcher._repair_ocr_digits(tid)
    tid_repaired = matcher._clean_id_suffix(tid_repaired)
    
    if matcher.is_valid_transaction_id(tid_repaired):
        tid_clean = tid_repaired.strip().upper().replace(' ', '').replace("'", "")
        if tid_clean not in seen:
            seen.add(tid_clean)
            final_score = initial_score
            
            if not tid_clean.isdigit():
                final_score += 10
                print(f"'{tid_clean}' +10 Alpha boost")
            
            if any(c in tid for c in ['-', ':', ' ']):
                final_score += 5
                print(f"'{tid_clean}' +5 Complex boost")
                
            if len(tid_clean) >= 12 and tid_clean.startswith('20') and tid_clean[2:4].isdigit():
                final_score += 25
                print(f"'{tid_clean}' +25 Date-ID boost")
                
            if tid_clean.isdigit() and 9 <= len(tid_clean) <= 17:
                 if tid_clean.startswith('20') and len(tid_clean) >= 12:
                     pass
                 elif source == "bank_specific" or (source == "generic" and initial_score >= 60):
                     pass
                 else:
                     final_score -= 30
                     print(f"'{tid_clean}' -30 Numeric penalty")
            
            prox_boost = matcher._label_proximity_boost(text_upper, tid_clean)
            if prox_boost > 0:
                final_score += prox_boost
                print(f"'{tid_clean}' +{prox_boost} Proximity boost")
                
            print(f"Candidate: {tid_clean} | Final Score: {final_score}")
            valid_scored_ids.append((tid_clean, final_score))

valid_scored_ids.sort(key=lambda x: x[1], reverse=True)
print(f"\nSorted Results: {valid_scored_ids}")
