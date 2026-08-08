# IBG Extraction — Measured Baseline

Recorded 2026-08-07, before any team changes. This is the "before" column the
CTO and QA measure against. Every number here came from running the live
engine, not from reading the code.

Reproduce with:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from tests.fixtures.ibg_corpus import CORPUS
from app.ultimate_patterns_v3 import extract_all_fields_v3
for s in CORPUS:
    r = extract_all_fields_v3(s['text'], ocr_used=s['ocr_used'])
    print(s['id'], r.get('confidence'), r.get('transaction_id'), r.get('date'), r.get('amount'), r.get('bank_name'))"
```

## Scoreboard: 9 / 20 field checks pass

`TRAP` = returned a value the corpus explicitly lists as a known false positive.

| Sample | reference_id | bank_name | transaction_date | amount |
|---|---|---|---|---|
| S1 Maybank M2E ACH | PASS | PASS | FAIL | PASS |
| S2 M2E truncated | **TRAP** | PASS | FAIL | PASS |
| S3 Open Interbank | **TRAP** | PASS | FAIL | PASS |
| S4 GIRO/DuitNow | PASS | FAIL | FAIL | FAIL |
| S5 RHB degraded | FAIL | PASS | PASS | PASS |

## The five defects

### D1 — Account number returned as reference ID (S2)
`transaction_id = "510000000001"`, which is the **Debit From Account No.**
The document carries no reference number at all; the correct answer is `None`.
Owner: reference-id dev.

### D2 — Payer's own reference returned instead of the bank's (S3)
`transaction_id = "MXC12511726684"` — the `Recipient reference`, free text typed
by the payer. The correct answer is `7901990048`, labeled `Reference number:`.
Root cause shared with D1: the engine ranks candidates by the *shape* of the
token rather than by *which label introduced it*.
Owner: reference-id dev.

### D3 — Dates are never normalized (S1, S2, S3, S4 — all of them)
The engine returns the raw matched substring: `"03 Aug 2026"`, `"31 Oct 2025"`.
Different receipts yield different formats, and these strings flow straight into
the history CSV (`simple_server.py` `/export`) and into Supabase
(`db.append_annotation`). They are unsortable and uncomparable.
Every date must be ISO `YYYY-MM-DD`.
Owner: transaction-date dev.

### D4 — A payment rail returned as the bank name (S4)
`bank_name = "DuitNow"`. DuitNow is a transfer rail, not a bank; the issuing
bank on that receipt is Maybank. Related: on S1 the beneficiary bank (RHB) and
the issuing bank (Maybank) are conflated by a naive keyword match.
Owner: bank-name dev.

### D5 — Amount format is inconsistent (S4)
`amount = "1,144.80"` — thousands separator retained, where other samples return
`"10.60"` with none. Downstream `_normalize_amount` in `app/result_merger.py`
strips commas, but only on the merge path, so the raw engine output is
inconsistent for any other consumer.
Owner: amount dev.

## D6 — The systemic one: confidence is fabricated

**Every one of the five samples reports `confidence: 1.0`** — including S2,
which returns an account number, and S5, which returns `WH512511799` scraped
off a barely-legible scan.

The cause is [`app/ultimate_patterns_v3.py:1147`](../app/ultimate_patterns_v3.py):

```python
confidence = 0.95 if transaction_ids else 0.5
if bank_name != "Unknown":
    confidence += 0.05
```

Confidence is `1.0` whenever *any* ID matched and the bank is known. It never
checks whether the ID is plausible, and it ignores date and amount completely.

This is the defect that makes the other five dangerous. A wrong value reported
at `confidence: 1.0` is written to the ledger and never reviewed;
`simple_server.py` derives `needs_review` from these numbers, so nothing is
ever flagged. **Fixing extraction without fixing confidence leaves the system
silently trusting whatever it gets wrong next.**

Owner: CTO. Confidence must reflect evidence strength — a labeled match
outranks a shape match — and must account for all four fields, not just the ID.

## D7 — the regression safety net cannot fail

`tests/test_100_percent_accuracy.py` is this repo's headline validation and the
source of the README's "100% accuracy on the validation dataset (35/35
challenging cases)" claim. Both of its test functions end in a `return` rather
than an `assert`:

- line 90:  `return passed == len(test_cases)`
- line 169: `return passed == len(all_test_cases)`

Under pytest a test function that returns `False` still **passes** — it only
emits `PytestReturnNotNoneWarning`. Demonstrated minimally:

```python
def test_a(): return False   # PASSES
def test_b(): assert False   # fails
```
```
1 failed, 1 passed, 1 warning
```

So `pytest tests/test_100_percent_accuracy.py` reports `2 passed` regardless of
whether extraction works at all. Nothing has been guarding these 35 cases.

They do currently pass on their own merits — verified by calling the functions
directly rather than through pytest:

```
test_failing_cases() -> True
test_all_35_cases()  -> True
OVERALL RESULTS: 35/35 tests passed
```

That is the real baseline any integration must preserve. Check it with:

```bash
.venv/bin/python -c "
import tests.test_100_percent_accuracy as t
print(t.test_failing_cases(), t.test_all_35_cases())"
```

## Note on S5

`WH512511799` is what the engine returns; the underlying scan reads
`WHS12511799`. The OCR digit-repair pass turned `S` into `5`. The scan is too
degraded to confirm what the true reference is, which is exactly why returning
it at confidence 1.0 is the wrong behaviour. Low confidence, flagged for
review, is acceptable here.
