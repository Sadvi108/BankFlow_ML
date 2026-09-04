# Production OCR remediation plan

## Decision

Do not copy the full `text-extract-api` service into BankFlow. It is a generic
document extraction stack with additional workers and services; it does not
provide the bank-specific rule that distinguishes an account number from a
bank reference or returns every reference by role. BankFlow keeps its existing
FastAPI contract and uses a small, deterministic OCR and extraction path.

## Implemented in this change

1. **Bound OCR work**
   * Skip orientation detection on a clean first read; run it at most once for
     weak reads instead of once per preprocessing pass.
   * Use one fast pass and at most one normal retry by default.
   * Cap giant phone images at 2800 pixels on the longest edge.
   * Make the legacy heavy pass opt-in with `OCR_ENABLE_HEAVY_PASS=1`.
2. **Make PDFs predictable**
   * Keep embedded PDF text when it is readable, even if a legacy ID heuristic
     does not fire.
   * Render scanned PDFs one page at a time instead of holding every page in RAM.
   * Process every page, so later-page references are not lost to early exit.
   * Reuse the saved upload instead of writing the PDF a second time.
3. **Separate identifiers by evidence**
   * Use labels to classify bank-primary, bank-secondary, and payer-supplied
     references.
   * Exclude a number explicitly introduced by an account label from reference
     candidates, even when OCR layout is damaged.
   * Reject short OCR fragments and leaked form labels.
4. **Return all references to the portal**
   * Preserve `transaction_id` and `all_ids` for compatibility.
   * Add `primary_reference_id`, `reference_ids`, `bank_reference_ids`,
     `payer_reference_ids`, and the detailed `references` list.
5. **Expose latency**
   * Return upload, OCR, field-extraction, and total timing measurements plus
     OCR method/pass metadata.

## Verification completed

* 477 focused regression tests pass.
* 168/168 labeled scalar field checks pass.
* 72/72 expected references are captured.
* Exact multi-reference sets improved from 19/28 to 22/28 while keeping all
  expected references.
* The repository-wide `pytest` command is not a valid gate yet because legacy
  script-style test modules call `sys.exit()` during collection. The focused
  deterministic suite is the current release gate.

## Staging rollout

1. Deploy to staging with the default OCR settings from `render.yaml`.
2. Build a privacy-safe labeled sample from recent portal failures. For each
   document, record the correct primary reference, every reference, account
   decoys, and whether human review was required.
3. Compare current production and staging on:
   * p50 and p95 `timings.ocr_ms` and `timings.total_ms`;
   * primary-reference exact accuracy;
   * all-reference set recall and precision;
   * account-number-as-reference false positives (target: zero);
   * review rate and false-confidence rate.
4. Canary staging behavior to a small fraction of portal traffic, then expand
   only if accuracy is no worse and p95 latency improves.
5. If a hard subset still needs another engine, add it as an isolated fallback
   behind the same response contract and invoke it only for low-confidence,
   review-bound cases. Benchmark that subset before adding worker infrastructure.
