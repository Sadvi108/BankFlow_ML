"""Regressions for missing reference text, page loss, and uncertain OCR.

All document values are synthetic; no company receipts are persisted by tests.
"""
import cv2
import fitz
import numpy as np
import pytest
import pytesseract
from fastapi.testclient import TestClient

import simple_server
from app import ocr_vision
from app.ibg.extractor import extract_ibg_fields
from app.ibg.reference_id import extract_reference_id, extract_references
from app.ocr_pipeline import OCRPipeline


@pytest.mark.parametrize('label,value', [
    ('Transaction ID', 'TXN0011223344'),
    ('Trx No/Seq No', '0018 / 2034'),
    ('No Urus Niaga', '003451'),
    ('No Uus Niaga', '003464'),
])
def test_explicit_transaction_labels(label, value):
    assert extract_reference_id(label + ': ' + value, False).value == value


def test_status_sentence_without_any_other_reference_label():
    assert extract_reference_id(
        'The status for payment 48291ID07ABC is: Received by Beneficiary Bank',
        False,
    ).value == '48291ID07ABC'


def test_wrapped_recipient_customer_label_keeps_its_note():
    refs = extract_references(
        "Recipient's Reference / Customer\nCPC CHRGS - PACIFIC\nAmount MYR 12.00",
        False)
    assert [r.value for r in refs] == ['CPC CHRGS - PACIFIC']


def test_vision_retries_only_the_known_pdf_rotation(monkeypatch, tmp_path):
    path = tmp_path / 'rotated.pdf'
    _make_pdf(path, ['vector'])
    doc = fitz.open(path)
    doc[0].set_rotation(90)
    doc.saveIncr()
    doc.close()
    monkeypatch.setattr(ocr_vision, '_HAVE_VISION', True)
    attempts = iter([
        dict(text='garbled sideways payment receipt', confidence=0.6),
        dict(text='No Urus Niaga: 003464', confidence=0.85),
    ])
    monkeypatch.setattr(ocr_vision, 'image_to_text', lambda _path: next(attempts))
    result = ocr_vision.pdf_to_text(str(path))
    assert extract_reference_id(result['text']).value == '003464'


def test_vision_does_not_silently_stop_after_ten_pages(monkeypatch, tmp_path):
    path = tmp_path / 'eleven.pdf'
    _make_pdf(path, ['vector'] * 11)
    monkeypatch.setattr(ocr_vision, '_HAVE_VISION', True)
    monkeypatch.setattr(ocr_vision, 'image_to_text', lambda _path: dict(
        text='Reference No: MULTIPAGE123', confidence=0.9))
    result = ocr_vision.pdf_to_text(str(path), zoom=0.2)
    assert result['pages_processed'] == 11
    assert result['page_texts'][-1]['page'] == 11


def test_unread_bank_reference_is_sent_for_review():
    result = extract_ibg_fields(
        'Public Bank\nReference No:\nAmount: MYR 20.00\nDate: 01/02/2030', False)
    assert result['reference_id']['value'] is None
    assert result['needs_review']
    assert any('reference_id' in reason for reason in result['review_reasons'])


def test_vision_keeps_label_and_value_on_the_same_row():
    text = ocr_vision._lines_to_text([
        dict(text='Reference No.', x=0.05, y=0.60, w=0.2, h=0.026),
        dict(text=': BANK778899', x=0.4, y=0.604, w=0.3, h=0.020),
        dict(text='Amount: MYR 20.00', x=0.05, y=0.54, w=0.4, h=0.025),
    ])
    assert text.splitlines()[0] == 'Reference No.  : BANK778899'
    assert extract_reference_id(text, True).value == 'BANK778899'


def test_tesseract_keeps_zero_confidence_id_and_column_gaps(monkeypatch):
    words = ['Reference', 'No:', 'BANK998877', 'Customer', 'Ref:', 'INV112233']
    data = dict(text=words, conf=[98, 98, 0, 95, 95, 94],
                line_num=[1]*6, left=[0, 90, 125, 450, 535, 575],
                top=[10]*6, width=[80, 25, 100, 75, 30, 90], height=[20]*6)
    monkeypatch.setattr(pytesseract, 'image_to_data', lambda *_a, **_kw: data)
    result = OCRPipeline()._run_tesseract(np.zeros((100, 800), np.uint8))
    assert 'BANK998877  Customer' in result['text']
    assert result['tokens'][2]['conf'] == 0
    assert [r.value for r in extract_references(result['text'])] == [
        'BANK998877', 'INV112233']


def test_retry_timeout_preserves_first_read(monkeypatch):
    pipeline = OCRPipeline()
    monkeypatch.setattr(pipeline, 'preprocess_image_light', lambda image, **kw: image)
    monkeypatch.setattr(pipeline, 'preprocess_for_photo', lambda image, **kw: image)
    monkeypatch.setattr(pipeline, '_auto_rotate_text', lambda image: image)
    attempts = []

    def run(image, psm):
        attempts.append(psm)
        if len(attempts) == 2:
            raise RuntimeError('Tesseract process timeout')
        return dict(text='Reference No: RECOVER0011', confidence=0.6,
                    processed_successfully=True, tokens=[], word_count=3)

    monkeypatch.setattr(pipeline, '_run_tesseract', run)
    result = pipeline.extract_text_with_confidence(np.zeros((20, 20), np.uint8))
    assert 'RECOVER0011' in result['text']
    assert result['passes_used'] == 2


def _make_pdf(path, pages):
    doc = fitz.open()
    for kind in pages:
        page = doc.new_page()
        if kind == 'vector':
            page.draw_rect(fitz.Rect(20, 20, 40, 60), fill=(0, 0, 0))
        elif kind == 'image':
            ok, png = cv2.imencode('.png', np.full((40, 80), 255, np.uint8))
            assert ok
            page.insert_image(page.rect, stream=png.tobytes())
        elif kind == 'text':
            page.insert_text((72, 72), 'Reference No: DIGITAL112233\nAmount MYR 10.00')
    doc.save(path)
    doc.close()


def test_vector_page_is_not_mistaken_for_a_blank_page(monkeypatch, tmp_path):
    path = tmp_path / 'vectors.pdf'
    _make_pdf(path, ['vector'])
    core = simple_server.ocr_pipeline.ocr_pipeline
    monkeypatch.setattr(core, 'tesseract_available', lambda: True)
    monkeypatch.setattr(simple_server.ocr_vision, 'available', lambda: False)
    calls = []

    def ocr(_path, pages):
        calls.append(pages)
        return dict(page_texts=[dict(page=1, text='Reference No: VECTOR9988')],
                    confidence=91, tokens=[])

    monkeypatch.setattr(core, 'process_pdf_pages', ocr)
    result = simple_server._pdf_to_text(str(path))
    assert calls == [[0]]
    assert 'VECTOR9988' in result['text']


def test_partial_tesseract_result_only_retries_unread_pages(monkeypatch, tmp_path):
    path = tmp_path / 'mixed.pdf'
    _make_pdf(path, ['text', 'image', 'image'])
    core = simple_server.ocr_pipeline.ocr_pipeline
    monkeypatch.setattr(core, 'tesseract_available', lambda: True)
    monkeypatch.setattr(simple_server.ocr_vision, 'available', lambda: True)
    monkeypatch.setattr(core, 'process_pdf_pages', lambda *_a: dict(
        page_texts=[dict(page=2, text='Reference No: SCAN2222'), dict(page=3, text='')],
        confidence=90, tokens=[]))
    requests = []

    def vision(_path, page_indices):
        requests.append(page_indices)
        return dict(page_texts=[dict(page=3, text='Reference No: SCAN3333')],
                    confidence=0.92)

    monkeypatch.setattr(simple_server.ocr_vision, 'pdf_to_text', vision)
    result = simple_server._pdf_to_text(str(path))
    assert requests == [[2]]
    assert result['pages_processed'] == 3
    assert result['unread_pages'] == []
    assert {r.value for r in extract_references(result['text'])} == {
        'DIGITAL112233', 'SCAN2222', 'SCAN3333'}


def test_api_marks_missing_scan_page_for_review(monkeypatch, tmp_path):
    monkeypatch.setattr(simple_server, 'UPLOAD_DIR', tmp_path)
    monkeypatch.setattr(simple_server, 'db', None)
    monkeypatch.setattr(simple_server.history_manager, 'add_entry', lambda _e: 'test')
    monkeypatch.setattr(simple_server, '_pdf_to_text', lambda _path: dict(
        text='Public Bank\nReference No: KNOWN112233\nAmount MYR 10.00\nDate 01/02/2030',
        tokens=[], confidence=0.9, source='embedded_partial', unread_pages=[2]))
    with TestClient(simple_server.app) as client:
        response = client.post('/extract', files={
            'file': ('receipt.pdf', b'placeholder PDF', 'application/pdf')})
    assert response.status_code == 200
    data = response.json()['data']
    assert data['needs_review']
    assert data['ocr_details']['unread_pages'] == [2]
    assert any('page(s): 2' in reason for reason in data['review_reasons'])
