"""Production-path regressions: concurrency, retry selection, and latency."""
import asyncio
import time

import anyio
import httpx
import numpy as np
import pytesseract

import simple_server
from app.ibg.reference_id import extract_reference_id
from app.ocr_pipeline import OCRPipeline


def _read(text, confidence):
    return dict(text=text, confidence=confidence, tokens=[], word_count=12,
                processed_successfully=True)


def test_ocr_comma_after_reference_number_does_not_drop_correct_digits():
    text = '| Reference No,  : 203012070343901619  noise\nAccount No: 1234567890'
    assert extract_reference_id(text).value == '203012070343901619'


def test_pipe_separator_preserves_multiple_reference_labels():
    from app.ibg.reference_id import extract_references
    assert [r.value for r in extract_references(
        'Reference No: BANK112233\nRecipient Reference: INV223344\n'
        'Other Payment Details | INV334455')] == ['BANK112233', 'INV223344', 'INV334455']


def test_retry_with_recovered_reference_beats_incomplete_confident_read(monkeypatch):
    pipeline = OCRPipeline()
    for name in ('preprocess_image_light', 'preprocess_for_photo'):
        monkeypatch.setattr(pipeline, name, lambda image, **kw: image)
    monkeypatch.setattr(pipeline, '_auto_rotate_text', lambda image: image)
    common = 'Public Bank\nIBG Payment\nDate: 01/02/2030\nAmount MYR 20.00\n'
    reads = iter([
        _read(common + 'Reference No:\nBeneficiary Account: 1234567890', 0.84),
        _read(common + 'Reference No: BANK112233\nBeneficiary Account: 1234567890', 0.76),
    ])
    monkeypatch.setattr(pipeline, '_run_tesseract', lambda *a, **kw: next(reads))
    result = pipeline.extract_text_with_confidence(np.zeros((20, 20), np.uint8))
    assert extract_reference_id(result['text']).value == 'BANK112233'


def test_receipt_without_any_readable_reference_does_not_skip_retry():
    result = _read('Public Bank\nIBG Payment\nDate: 01/02/2030\n'
                   'Amount MYR 20.00\nBeneficiary Account: 1234567890', 0.97)
    assert not OCRPipeline()._is_good_read(result)


def test_ocr_does_not_block_health_requests(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(simple_server, 'UPLOAD_DIR', tmp_path)
    monkeypatch.setattr(simple_server, 'db', None)
    monkeypatch.setattr(simple_server.history_manager, 'add_entry', lambda e: 'test')
    monkeypatch.setattr(simple_server, '_ocr_diagnostics', lambda: {})

    def slow_ocr(_path):
        events.append('ocr_started')
        time.sleep(0.3)
        events.append('ocr_finished')
        return dict(text='Public Bank\nReference No: BANK112233\n'
                    'Amount MYR 20.00\nDate 01/02/2030', tokens=[],
                    confidence=0.9, source='ocr')

    monkeypatch.setattr(simple_server, '_pdf_to_text', slow_ocr)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=simple_server.app),
                                    base_url='http://test') as client:
            upload = asyncio.create_task(client.post('/extract', files={
                'file': ('receipt.pdf', b'placeholder', 'application/pdf')}))
            while not events:
                await asyncio.sleep(0.005)
            assert (await client.get('/health')).status_code == 200
            events.append('health_returned')
            assert (await upload).status_code == 200

    asyncio.run(run())
    assert events == ['ocr_started', 'health_returned', 'ocr_finished']


def test_busy_upload_returns_retryable_error_without_running_ocr(monkeypatch):
    limiter = anyio.CapacityLimiter(1)
    monkeypatch.setattr(simple_server, 'extraction_limiter', limiter)
    monkeypatch.setattr(simple_server, 'EXTRACTION_QUEUE_TIMEOUT', 0.02)

    def unexpected(*args):
        raise AssertionError('Busy uploads must not enter OCR')

    monkeypatch.setattr(simple_server, '_extract_receipt_sync', unexpected)

    async def run():
        active_upload = object()
        await limiter.acquire_on_behalf_of(active_upload)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=simple_server.app),
                                        base_url='http://test') as client:
                response = await client.post('/extract', files={
                    'file': ('receipt.pdf', b'placeholder', 'application/pdf')})
                assert response.status_code == 503
                assert response.headers['Retry-After'] == '3'
        finally:
            limiter.release_on_behalf_of(active_upload)
        assert limiter.borrowed_tokens == 0

    asyncio.run(run())


def test_document_budget_retains_first_page_and_skips_later_work(monkeypatch):
    pipeline = OCRPipeline()
    pipeline.document_timeout = 0.02
    monkeypatch.setattr(pipeline, '_iter_pdf_images', lambda path: iter([
        (0, np.zeros((20, 20), np.uint8)), (1, np.zeros((20, 20), np.uint8))]))
    calls = []

    def read(image):
        calls.append(True)
        time.sleep(0.03)
        return _read('Reference No: BANK112233', 0.9)

    monkeypatch.setattr(pipeline, 'extract_text_with_confidence', read)
    result = pipeline.process_pdf_pages('not-opened.pdf')
    assert calls == [True]
    assert result['budget_exhausted']
    assert result['pages_processed'] == 1
    assert result['page_texts'][0]['text'] == 'Reference No: BANK112233'


def test_subprocess_timeout_cannot_exceed_remaining_document_budget(monkeypatch):
    pipeline = OCRPipeline()
    pipeline.document_timeout = 0.2
    recorded = []

    def read(image, **kwargs):
        recorded.append(kwargs['timeout'])
        return dict(text=[], conf=[], line_num=[], top=[], left=[], width=[], height=[])

    monkeypatch.setattr(pytesseract, 'image_to_data', read)
    with pipeline._time_budget():
        pipeline._run_tesseract(np.zeros((20, 20), np.uint8))
    assert 0 < recorded[0] <= 0.2


def test_lone_scan_border_does_not_hide_next_line_reference():
    assert extract_reference_id('Reference Number  Date\n‘\n'
                                'BANK112233  01/02/2030').value == 'BANK112233'


def test_references_do_not_pair_with_values_on_other_pages():
    from app.ibg.reference_id import extract_references
    refs = extract_references('=== PAGE 1 ===\nReference No:\n'
                              '=== PAGE 2 ===\nINV223344\nCustomer Ref: INV334455')
    assert [ref.value for ref in refs] == ['INV334455']


def test_multiple_bank_slips_keep_all_their_bank_references():
    from app.ibg.reference_id import extract_references
    refs = extract_references('=== PAGE 1 ===\nReference No: BANK112233\n'
                              '=== PAGE 2 ===\nReference No: BANK223344')
    assert [(ref.value, ref.role) for ref in refs] == [
        ('BANK112233', 'bank_primary'), ('BANK223344', 'bank_secondary')]


def test_attached_voucher_reference_does_not_replace_the_bank_slip_id():
    text = ('=== PAGE 1 ===\nPAYMENT VOUCHER\nReference No: 112233\n'
            '=== PAGE 2 ===\nFund Transfer\nReference Number: BANK445566')
    assert extract_reference_id(text).value == 'BANK445566'


def test_response_timing_includes_persistence_delay(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(simple_server, 'UPLOAD_DIR', tmp_path)
    monkeypatch.setattr(simple_server, 'db', None)
    monkeypatch.setattr(simple_server.history_manager, 'add_entry', lambda e: time.sleep(0.04))
    monkeypatch.setattr(simple_server, '_pdf_to_text', lambda p: dict(
        text='Public Bank\nReference No: BANK112233\nAmount MYR 20.00\nDate 01/02/2030',
        tokens=[], confidence=0.9, source='embedded'))
    with TestClient(simple_server.app) as client:
        response = client.post('/extract', files={
            'file': ('receipt.pdf', b'placeholder', 'application/pdf')})
    assert response.status_code == 200
    data = response.json()['data']
    assert data['timings']['persistence_ms'] >= 30
    assert data['timings']['total_ms'] >= data['timings']['persistence_ms']
    assert data['processing_time_ms'] == data['timings']['total_ms']
    assert data['extraction_version'] == simple_server.EXTRACTION_VERSION
