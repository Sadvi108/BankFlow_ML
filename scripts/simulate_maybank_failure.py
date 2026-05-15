from app.ultimate_patterns_v3 import extract_all_fields_v3

text = """
--- PAGE 1 ---
SSSSccccaaaannnn &&&& PPPPaaaayyyy
Successful
Reference ID
14 Dec 2025, 7:08 PM
222200006666888888883333888855555555QQQQ
Recipient Name
MMMMaaaassssrrrruuuurrrr RRRRaaaahhhhmmmmaaaannnn SSSShhhhaaaahhhh
Recipient Account Number
555566662222222222227777444400001111777777775555
Amount
RRRRMMMM 11115555....00000000
Note: This receipt is computer generated and no
signature is required.
Malayan Banking Berhad (Co. Reg. : 196001000142)
Maybank Islamic Berhad (Co. Reg. : 200701029411)
"""

print("Testing extraction on Maybank problematic text...")
results = extract_all_fields_v3(text)
print(f"Bank: {results.get('bank_name')}")
print("\n--- RESULTS ---")
import json
print(json.dumps(results, indent=2))
