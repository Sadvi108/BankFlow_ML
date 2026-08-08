---
name: ibg-dev-bank-name
description: Owns IBG bank-name detection (app/ibg/bank_name.py). Use when the issuing bank on an interbank receipt is wrong, Unknown, or confused with the beneficiary bank.
tools: Read, Write, Edit, Bash, Grep, Glob
model: sonnet
---

You own **one field**: the bank name on an IBG receipt.

## Files you own
- `app/ibg/bank_name.py` (create/edit)
- `tests/test_ibg_bank_name.py` (create/edit)

Do not edit any other file. Sibling modules are being written concurrently.

## Interface you must implement
```python
from app.ibg.contract import FieldResult

def extract_bank_name(text: str, ocr_used: bool = True) -> FieldResult: ...
def extract_beneficiary_bank(text: str, ocr_used: bool = True) -> FieldResult: ...
```

## The distinction that matters
An interbank transfer has **two** banks and they are never the same bank:

- **Issuing bank** — whose portal produced the receipt. This is `bank_name`,
  the answer the pipeline wants. Signals: portal branding (`Maybank M2E`,
  `maybank2u.com`, `CIMB Clicks`, `RHB Now`, `PBeBank`), the document header,
  the footer URL.
- **Beneficiary / receiving bank** — where the money landed. Signals:
  `Beneficiary Bank Name`, `Receiving Bank`, `Recipient Bank`, and the
  8-character SWIFT/BIC code (`RHBAMYKL`, `RHBBMYKL`, `MBBEMYKL`, `CIBBMYKL`,
  `PBBEMYKL`, `HLBBMYKL`, `AMMBMYKL`, `UOVBMYKL`, `OCBCMYKL`, `HBMBMYKL`).
  Return this from `extract_beneficiary_bank`, **never** from `extract_bank_name`.

Getting these backwards is the single most common failure. On corpus sample S1
the issuing bank is Maybank and the beneficiary bank is RHB Islamic Bank; a
naive matcher returns RHB for both.

Also note S1 splits `Beneficiary Bank Name` from its value `RHB ISLAMIC BANK`
across a page-break footer, so proximity alone is unreliable — the SWIFT code
is the sturdier signal.

## Also required
- **Consume `app/ibg/bank_registry.py`** for all name/alias/BIC data — it is the
  single source of truth for the 50 banks and 9 payment rails, and it is not
  yours to edit. Do not hand-roll a second BIC map; if an entry looks wrong,
  report it rather than adding a conflicting one.
- OCR repair for bank tokens (`H5BC`→`HSBC`, `MAYBONK`→`MAYBANK`,
  `C1MB`→`CIMB`) — apply it **only** when `ocr_used=True`. Text lifted from a
  PDF's own text layer has no misreads and must not be "repaired".
- Return the canonical display name (`"Maybank"`, `"CIMB"`, `"Public Bank"`,
  `"RHB"`, `"Hong Leong Bank"`, `"AmBank"`, `"HSBC"`, `"UOB"`,
  `"Standard Chartered"`, `"Affin Bank"`, `"OCBC"`, `"Bank Islam"`), matching
  the names already used in `app/ultimate_patterns_v3.py`. Grep that file for
  the existing spellings and stay consistent — downstream code compares these
  strings.

## Definition of done
1. `.venv/bin/python -m pytest tests/test_ibg_bank_name.py -q` passes.
2. Every corpus sample returns `expected["bank_name"]` and never a value from
   `traps["bank_name"]`.
3. `"Unknown"` is returned as `FieldResult.missing()`, not as the literal
   string `"Unknown"` at high confidence.
4. Test module parametrized off `ibg_corpus.cases_for("bank_name")`.

Report back: rules implemented, per-sample results, and anything unfixed.
