"""Regression tests for the Maybank M2E misextraction faults.

Reproduced from a real Maybank M2E "Cash Management System" payment PDF that the
production API got wrong on every field:

  reference -> "OOL12612032422"  (the Debit Description, not the Reference No.)
  amount    -> "512044514656"    (the Debit From Account No., not the amount)
  date      -> "04/08/2026"      (the browser print-header stamp, not the value date)
  merger    -> needs_review=False for a fully wrong record

The fixture below preserves the *layout* of that receipt — the label/value
ordering, the "(dd-mm-yyyy)" format hint, the per-page print footer — but every
account number, reference and party name has been replaced. Real receipts must
not be committed to the repo.

Run:  python tests/test_amount_date_faults.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ultimate_patterns_v3 import extract_all_fields_v3, validate_amount
from app.result_merger import merge


# Structure of the real receipt; values anonymised.
# MYIG2601011234567 keeps the I and G that the OCR digit-repair pass corrupts.
# OOL10000000001 keeps the leading letter O's for the same reason.
MAYBANK_M2E = """A. Transaction Details
B. Beneficiary Details

Payment
Debit From Account No.
500011112222 (MYR) ACME TRADING
SDN. BHD.
Payment Mode
Outward ACH
Destination Country
MY - MALAYSIA
Date and Time
03 Aug 2026 12:23:57 MY (UTC+08:00)
Reference No.
MYIG2601011234567
Customer Ref No.
ACME TRADING
Debit Reference
-
Debit Description
OOL10000000001
Transaction Currency
MYR
Transaction Amount
In Transaction Currency
10.60
Value Date (dd-mm-yyyy)
03 Aug 2026 MY (UTC+08:00)
Beneficiary Account No.
21200000000001
Beneficiary Name
ACME SUPPLIES (M) SDN BHD
04/08/2026, 07:57
Maybank M2E
https://www.maybank2e.com/m2e/portal/tran.view?do=ReLoad
1/4
"""

M2E_EXPECTED = {
    "bank_name": "Maybank",
    "transaction_id": "MYIG2601011234567",
    "date": "03 Aug 2026",
    "amount": "10.60",
}

# Receipts that already worked — these must keep working.
REGRESSION_GUARDS = [
    (
        "Maybank2u labelled amount",
        """Maybank2u.com
        Reference Number: 1234 5678 9012
        Transaction Date: 23 Jan 2026
        Amount: RM 1,234.50
        """,
        {"amount": "1,234.50", "date": "23 Jan 2026"},
    ),
    (
        "Transfer Amount with currency in parens",
        """CIMB Clicks
        Transaction Reference No. 20250101123456789012
        Transfer Amount (MYR) : 154.00
        Date: 01/01/2025
        """,
        {"amount": "154.00", "date": "01/01/2025"},
    ),
    (
        "Amount label then value on next line",
        """Public Bank
        Reference : PBB251031999999
        Amount (MYR)
        2,710.00
        Transaction Date : 31 Oct 2025
        """,
        {"amount": "2,710.00", "date": "31 Oct 2025"},
    ),
    (
        "Integer amount, no decimals",
        """RHB Now
        Reference No. RHB12345678
        Amount: RM 70
        Date: 15/03/2025
        """,
        {"amount": "70", "date": "15/03/2025"},
    ),
]


def check(label, actual, expected):
    ok = actual == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {actual!r}, expected {expected!r}")
    return ok


def main():
    failures = 0

    print("Maybank M2E cash management payment (embedded PDF text)")
    res = extract_all_fields_v3(MAYBANK_M2E, ocr_used=False)
    for field, want in M2E_EXPECTED.items():
        failures += not check(field, res[field], want)

    print("\nDigit repair must not corrupt letters in embedded (non-OCR) text")
    ocr_res = extract_all_fields_v3(MAYBANK_M2E, ocr_used=True)
    failures += not check("repair pass still active when ocr_used=True",
                          ocr_res["transaction_id"] != res["transaction_id"], True)

    print("\nA wrong record must not be reported as high-confidence")
    bad = merge({"bank_name": "Maybank", "transaction_id": "512044514656",
                 "date": "04/08/2026", "amount": "512044514656"}, None)
    failures += not check("needs_review for a 12-digit account no. as amount",
                          bad["needs_review"], True)
    failures += not check("validate_amount rejects an account number",
                          validate_amount("512044514656.00"), False)

    print("\nRegression guards — receipts that already worked")
    for name, text, expected in REGRESSION_GUARDS:
        got = extract_all_fields_v3(text)
        for field, want in expected.items():
            failures += not check(f"{name} :: {field}", got[field], want)

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
