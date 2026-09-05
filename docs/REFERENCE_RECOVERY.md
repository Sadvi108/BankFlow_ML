# Reference recovery and PDF latency follow-up

The reported field is Reference ID. No account-number output feature is added.

## Confirmed failure paths

* A value printed first as Customer Reference and then as Bank Reference was
  deduplicated using its first label, leaving no bank primary. Issuer evidence
  now takes precedence when the same value repeats.
* HSBCnet can identify a payment inside a status sentence. That sentence now
  supplies a bank reference without promoting arbitrary customer notes.
* AmBank exports can put the payment-reference value before its label; bounded
  recovery handles this layout. Alliance inserts headings in its label/value
  blocks; those headings no longer shift the reference assignments.
* Transaction ID, deposit transaction/sequence labels, Malay transaction labels,
  and the truncated Recipient Reference / Customer label are recognized.
* Mixed PDFs previously returned the usable text layer and skipped scanned
  companion pages. Page routing now preserves every usable page, retries only
  unread pages, and flags partial results for review. Pages made of vector
  outlines also require OCR even though they contain no bitmap images.
* OCR could put a value before its label because their boxes differed slightly
  in height. Vision aligns rows by vertical centre, while Tesseract retains
  column gaps. Tesseract's zero-confidence words are retained as uncertain
  evidence; reference digits from such words require review.
* Missing Tesseract executables triggered unnecessary work before Vision.
  Backend selection now avoids that path. PDF rendering also avoids encoding
  and decoding PNG data and caps dimensions before allocating a full raster.
* OCR calls previously had no timeout, and a failed retry discarded the first
  read. Per-pass timeouts now bound calls and preserve earlier usable text.
* Vision silently stopped at ten PDF pages. Its default processes all requested
  pages. A PDF rotation retry is limited to pages with rotation metadata and no
  recognized references; the alternate view wins only if it recovers a reference.

## Verification

The focused regression suite includes the 28-receipt corpus, all 15 holdout
receipts, page-routing cases, API contracts, and OCR failure cases. The labeled
reference sets contain 121 expected references (72 corpus + 49 holdout); all are
recovered. Existing scalar, bank, amount, and date regression checks also pass.
These tests establish regression coverage, not universal extraction accuracy.

```sh
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/test_ibg_amount.py tests/test_ibg_bank_name.py \
  tests/test_ibg_reference_id.py tests/test_ibg_transaction_date.py \
  tests/test_amount_date_faults.py tests/test_normalization.py \
  tests/test_api_reference_contract.py tests/test_ocr_adaptive.py \
  tests/test_reference_recovery.py tests/test_comprehensive_banks.py \
  tests/test_v3_extraction.py
```

A local audit processed all 208 files in `Receipts/`; every file produced text.
Seven real receipts were then submitted through the FastAPI test client with
persistence disabled. Each returned its visually checked expected primary ID,
including the previously unreadable sideways deposit receipt. The audit is a
coverage check: most real files do not have hand-labeled complete reference sets.
Real receipt text and diagnostic exports were not added to the repository.

The local before/after comparison used the previous commit and the same macOS
runtime (Vision available, Tesseract absent), taking the median of two runs:

| Case | Previous OCR time | Updated OCR time |
| --- | ---: | ---: |
| Upright scan A | 0.401 s | 0.252 s |
| Upright scan B | 0.253 s | 0.191 s |
| Upright scan C | 0.375 s | 0.290 s |
| Sideways scan | 0.362 s, missing reference | 0.527 s, reference recovered |

The three upright scans improved by 23–37%. The sideways scan does an additional
bounded pass to recover its ID. These are local OCR timings, not measurements of
the company's deployed Linux server or its concurrent load. The Linux Tesseract
path is covered by control-flow/token regression tests; its production latency
still needs comparison using `timings.ocr_ms` and `timings.total_ms` after rollout.

Apple Vision needs access to macOS graphics services. It returned no result in
the restricted development sandbox and worked when the same diagnostic ran
outside it. This was a test-environment restriction, not a broken installation.

## Portal integration

Continue reading `reference_ids`/`all_reference_ids` for the complete list and
`references` for labels, roles, and confidence. The scalar primary remains a
bank reference; an official non-bank receipt number stays in the reference list
without being relabeled as a bank transaction. Honour `needs_review`,
`review_reasons`, and `ocr_details.unread_pages` for incomplete or uncertain reads.
