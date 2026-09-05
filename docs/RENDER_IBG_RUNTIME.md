# Render / IBG runtime follow-up

The report was reproduced with native Tesseract 5.5.3 installed locally and
Apple Vision disabled. This exercises the same OCR engine family as the Linux
container, but the local machine does not reproduce Render's CPU quota.

## Confirmed code failures

- A Tesseract IBG scan contained the correct ID after `Reference No,`, but the
  label matcher rejected the comma. Separator parsing was also anchored to the
  start of the document instead of the current field offset.
- A retry with a recovered reference could lose to a slightly higher average
  confidence over ordinary words. Selection now rewards reference recovery.
- A high-confidence receipt with no recognized references skipped its retry.
- Both normal passes used the same segmentation mode. The bounded retry now
  tries automatic layout segmentation; readable upright forms avoid an OSD
  invocation and unnecessary photo filtering. Damaged/photo reads retain the
  photo fallback. No heavy third pass is enabled by default.
- PDF reference pairing could cross page boundaries. Pages are now separate
  scopes; attached voucher reference columns do not replace a bank slip's ID.
- Synchronous OCR and persistence ran inside an async HTTP handler. That
  blocked health checks and other requests for the full extraction duration.
- Each OCR invocation had a timeout, but many pages could each consume fresh
  timeouts. They now share one document budget, retaining completed pages.
- The reported total excluded persistence, hiding Supabase/network delays.

## Controls and observability

| Setting | Default | Meaning |
| --- | ---: | --- |
| `WEB_CONCURRENCY` | 1 | Uvicorn workers; keep at one on the 512MB instance |
| `OMP_THREAD_LIMIT` | 1 | Prevent Tesseract thread oversubscription |
| `OCR_PASS_TIMEOUT_SECONDS` | 20 | Maximum per Tesseract invocation |
| `OCR_DOCUMENT_TIMEOUT_SECONDS` | 25 | Shared budget for Tesseract pages and retries |
| `EXTRACTION_QUEUE_TIMEOUT_SECONDS` | 8 | Admission wait before a retryable 503 |
| `SUPABASE_TIMEOUT_SECONDS` | 3 | Database HTTP timeout; a schema fallback can retry once |

The document budget checks time before rendering/preprocessing and limits each
Tesseract subprocess to the remaining time. It does not forcibly interrupt an
in-process renderer/filter and is not a hard HTTP deadline. Uploading, queueing,
embedded PDF inspection, field parsing, and persistence are separate work.

One admitted extraction runs in a worker thread per server process. Admission
waits do not occupy the event loop. Local history read/modify/write operations
are locked against concurrent history endpoints within that process.

`/health` and `/extract` expose `extraction_version: ibg-runtime-v2`; `/health`
also exposes Render's `RENDER_GIT_COMMIT` as `revision` when available.
`timings.queue_ms`, `ocr_ms`, `field_extraction_ms`, `persistence_ms`, and
`total_ms` distinguish the stages. `total_ms` starts at endpoint entry, so it
excludes the hosting platform's cold start and multipart parsing before entry.

When busy, `/extract` returns 503 with `Retry-After: 3`. Retry sequentially with
backoff; do not immediately launch parallel retries. On budget exhaustion,
usable pages are returned with `needs_review` and `ocr_details.unread_pages`;
`ocr_details.budget_exhausted` explains why work stopped. If no usable text was
read, a 422 asks for fewer pages or a clearer scan.

## Verification and limitations

Regression coverage includes the labeled corpus and holdout references, a
health request during slow OCR, busy admission, a shared multi-page deadline,
subprocess timeouts, persistence timing, reference punctuation, retry selection,
and keeping voucher numbers separate from bank references.

The focused suite passed 529 tests:

```sh
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/test_ibg_amount.py tests/test_ibg_bank_name.py \
  tests/test_ibg_reference_id.py tests/test_ibg_transaction_date.py \
  tests/test_amount_date_faults.py tests/test_normalization.py \
  tests/test_api_reference_contract.py tests/test_ocr_adaptive.py \
  tests/test_reference_recovery.py tests/test_ibg_runtime.py \
  tests/test_comprehensive_banks.py tests/test_v3_extraction.py
```

Seven real slips then returned their expected primary IDs through the FastAPI
test client, with Tesseract enabled and persistence disabled. Two additional
degraded scans were verified to return `needs_review` for uncertain OCR digits.

A local native-Tesseract audit processed all 208 receipt files; all returned
some text. Five documents had one or more unread pages and remain partial.
This is coverage, not a claim of complete reference accuracy. In particular,
degraded sideways scans still contain uncertain digits and require review.
Real receipts and raw diagnostic exports are not committed.

The previously missing reference on a tested IBG scan is now recovered by the
native path. Its local OCR took approximately 0.97s versus 1.54s before; that
single-case result must not be advertised as a Render performance guarantee.

The checked-in Render Blueprint uses the free compute plan. Render documents
[0.1 CPU and 512MB for Free](https://render.com/docs/compute-plans), and
[sleep after 15 idle minutes with roughly a minute to wake](https://render.com/docs/faq).
Those platform delays cannot be removed by reference regex changes. The
OpenMP limit follows [Tesseract's guidance](https://tesseract-ocr.github.io/tessdoc/FAQ.html#can-i-increase-speed-of-ocr).
The layout retry follows its [segmentation guidance](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#page-segmentation-method).

No Render plan upgrade was made. The exact service URL, deployed revision and
error logs are still needed to confirm the reported live failure. After rollout,
check `/health`, test one slip on a warm instance, and inspect the timing fields.
If logs show an out-of-memory kill or restart, increasing request timeouts will
not address that condition. A faster paid instance avoids free-tier sleep and
provides more CPU, but requires the owner's cost decision.
