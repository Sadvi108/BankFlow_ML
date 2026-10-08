# Receipt accuracy verification — 8 October 2026

The production path is `simple_server.py` → bounded OCR → label-aware IBG
extraction. The older PyTorch/RandomForest training scripts are not used by
this server; retraining them would not change production receipt results.

## Changes

- Trained a local spelling-repair dictionary from 28 labeled development
  receipts and 1,650 generated OCR spelling examples. It retains 1,648
  unambiguous aliases across 73 supported label words and rejects conflicting
  aliases. The 15 holdout layouts are not used for training. This is a small
  label-repair model, not neural OCR retraining.
- Apply spelling repair to recognized OCR field labels, retaining original
  receipt text in the API response. Digital PDF spelling is authoritative.
  Unicode field separators are normalized on either path. Reference roles
  remain distinct, and customer-entered names/identifiers are not general
  spell-check targets. Repairs are reported in `label_repairs`.
- Recover OCR decimal commas only with a monetary label or currency evidence;
  keep amount, fees and total debit separate. Recover letter/digit confusion
  only inside explicitly labeled numeric dates, followed by calendar validation.
- Recognize the AM/PM transaction time immediately below a bank reference on
  mobile receipts. Browser/footer and explicitly printed dates remain excluded.
- Retry OCR when an explicit amount/date label has an unreadable value. Rank
  retries by recovered fields as well as OCR confidence, keeping existing
  document time budgets and pass limits. Explicitly unreadable fields require
  review and cannot be replaced by a legacy parser's unrelated guess.
- Audit the same PDF/OCR routing used by production. Its old Vision-only scan
  path incorrectly reported local Tesseract-readable files as unreadable.
- Add GitHub Actions regression checks and the project's installed TypeSafe
  skill/instructions. TypeSafe's guidance to keep exact rules in code and
  verify against source evidence informed these changes; runtime extraction
  continues locally without a TypeSafe API dependency.

## Measured results

The fixed benchmark has 28 development layouts plus 15 separate holdout layouts.
It checks exact `reference_id`, `bank_name`, `transaction_date` and `amount`
values. Five deterministic damage variants cover reference, transaction,
amount and date spelling, plus Unicode label colons. These variants simulate
OCR errors; they are not new, independently collected customer receipts.

| Version / branch | Clean core fields | OCR-label stress fields |
| --- | ---: | ---: |
| Previous `main` (`9ea1c90`) | 170/172 | 774/860 (90.00%) |
| This change | 170/172 | 850/860 (98.84%) |
| `codex/render-ibg-runtime` | 170/172 | 774/860 |
| `codex/fix-missing-reference-ids` | 170/172 | 774/860 |
| `fix/maybank-m2e-amount-date-extraction` | 69/172 | 319/860 |
| `fresh_deploy` | 66/172 | 316/860 |
| `clean-main` | 60/172 | 290/860 |

All 168 development checks across the six money/reference/bank/date scalar
fields still pass. The holdout passes 88/90 checks across those six fields.
Its two misses are issuer names on unbranded UOB/RHB layouts; the extractor
continues to abstain rather than infer the issuer from a beneficiary's bank.

The recent fix branches and Maybank branch are ancestors of `main`. The older
independent `fresh_deploy` and `clean-main` branches performed worse on identical
labeled inputs, so merging them would reintroduce failures. Comparisons use
isolated exports of each branch's application source, without starting servers
or relying on the current branch's extraction modules. The older branches use
the legacy V3 engine; modern branches use their IBG coordinator.

The production OCR audit read all **208 local receipts**. On the same cached OCR
text, amount coverage increased from 201 to 203 receipts and date coverage from
203 to 205. Numeric/date changes affect eight receipts when optional fees and
total debit are included. The OCBC amount `53.00`, the cheque receipt's printed
summary `0.00`, and the Maybank mobile timestamp were checked visually against
the originals. Coverage is not accuracy: some attached vouchers/slips omit
fields, and badly damaged photos still need review. The raw receipts and OCR
cache remain local and are excluded from the commit.

The customer's newly failing receipts were unavailable, so these results do not
establish their accuracy or promise perfect extraction on unfamiliar documents.

## Reproduce

```bash
python scripts/train_label_repair.py
python scripts/benchmark_extraction.py --branches --output docs/receipt_benchmark.json
python scripts/audit_receipts.py Receipts --cache --show-missing
python -m pytest -q tests/test_label_repair.py tests/test_receipt_numeric_recovery.py \
  tests/test_ibg_amount.py tests/test_ibg_bank_name.py tests/test_ibg_reference_id.py \
  tests/test_ibg_transaction_date.py tests/test_reference_recovery.py \
  tests/test_ibg_runtime.py tests/test_amount_date_faults.py \
  tests/test_api_reference_contract.py tests/test_ocr_adaptive.py tests/test_ocr_modes.py
python run_tests.py
```

The focused suite passes 728 tests; the legacy runner passes its 35 reference
examples. Full benchmark counts and bounded failure examples are recorded in
`receipt_benchmark.json`. A rerun after this commit will see a changed remote
`main`, so use the previous commit when comparing the original baseline.
