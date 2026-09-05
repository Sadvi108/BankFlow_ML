"""Control-flow tests for the bounded production OCR path.

The tests stub Tesseract itself: they verify how many expensive passes are
scheduled, independently of whether a Tesseract binary exists on the machine.
"""

import numpy as np
import cv2
import fitz

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline
from app.enhanced_ocr_pipeline import is_text_garbage
from app.ocr_pipeline import OCRPipeline
import simple_server


def _result(text, confidence):
    return {
        "text": text,
        "confidence": confidence,
        "tokens": [],
        "lines": text.splitlines(),
        "word_count": len(text.split()),
        "avg_word_confidence": confidence * 100,
        "processed_successfully": True,
        "width": 100,
        "height": 100,
    }


def _stub_preprocessing(pipeline):
    pipeline.preprocess_image_light = lambda image, skip_rotation=False: image
    pipeline.preprocess_for_photo = lambda image, skip_rotation=False: image
    pipeline.preprocess_image_heavy = lambda image, skip_rotation=False: image


def test_clean_receipt_skips_rotation_and_uses_one_ocr_pass():
    pipeline = OCRPipeline()
    _stub_preprocessing(pipeline)
    calls = {"rotation": 0, "ocr": 0}

    def rotate(image):
        calls["rotation"] += 1
        return image

    def run(image, psm=6):
        calls["ocr"] += 1
        return _result(
            "BANK PAYMENT Reference No TXN123456 Date 01/02/2030 "
            "Amount MYR 10.00 Beneficiary ACME",
            0.90,
        )

    pipeline._auto_rotate_text = rotate
    pipeline._run_tesseract = run
    result = pipeline.extract_text_with_confidence(
        np.zeros((100, 100), dtype=np.uint8))

    assert result["method"] == "light"
    assert result["passes_used"] == 1
    assert calls == {"rotation": 0, "ocr": 1}


def test_weak_read_is_bounded_to_two_passes_by_default():
    pipeline = OCRPipeline()
    pipeline.max_passes = 2
    _stub_preprocessing(pipeline)
    attempts = iter([
        _result("unstructured words with little useful content", 0.60),
        _result(
            "BANK TRANSFER Reference No TXN7654321 Date 01/02/2030 "
            "Amount MYR 10.00 Account 12345678",
            0.58,
        ),
    ])
    calls = []

    def run(image, psm=6):
        calls.append(psm)
        return next(attempts)

    rotations = []

    def rotate(image):
        rotations.append(True)
        return image

    pipeline._auto_rotate_text = rotate
    pipeline._run_tesseract = run
    result = pipeline.extract_text_with_confidence(
        np.zeros((100, 100), dtype=np.uint8))

    assert result["method"] == "photo"
    assert result["passes_used"] == 2
    assert calls == [6, 3]
    assert rotations == [True]


def test_large_phone_photo_is_downscaled_before_ocr():
    pipeline = OCRPipeline()
    pipeline.max_image_dimension = 100
    capped = pipeline._cap_image_size(
        np.zeros((300, 200, 3), dtype=np.uint8))
    assert max(capped.shape[:2]) == 100


def test_pdf_processing_keeps_references_from_every_page():
    pipeline = OCRPipeline()
    pages = [
        (0, np.zeros((20, 20), dtype=np.uint8)),
        (1, np.zeros((20, 20), dtype=np.uint8)),
    ]
    pipeline._iter_pdf_images = lambda _path: iter(pages)
    outputs = iter([
        _result("Reference No: FIRST123", 0.9),
        _result("Reference No: SECOND456", 0.9),
    ])
    pipeline.extract_text_with_confidence = lambda _image: next(outputs)

    result = pipeline.process_file("receipt.pdf")

    assert result["pages_processed"] == 2
    assert "FIRST123" in result["text"]
    assert "SECOND456" in result["text"]


def test_readable_pdf_text_does_not_require_a_legacy_id(monkeypatch, tmp_path):
    path = tmp_path / "receipt.pdf"
    path.write_bytes(b"not-read-by-this-test")
    pipeline = EnhancedOCRPipeline()
    text = "Readable payment advice without a conventional identifier. " * 2
    monkeypatch.setattr(pipeline, "_extract_pdf_text", lambda _path: text)

    def unexpected_ocr(_path):
        raise AssertionError("readable embedded text should not be OCR'd")

    monkeypatch.setattr(pipeline.ocr_pipeline, "process_file", unexpected_ocr)
    result = pipeline.process_file(str(path))
    assert result["text"] == text
    assert result["method"] == "pdf_text_extraction"


def test_short_meaningful_pdf_text_is_not_forced_through_ocr():
    assert not is_text_garbage("Reference No: AB123456789")
    assert is_text_garbage("cover page")


def test_mixed_pdf_ocrs_only_the_scanned_page(monkeypatch, tmp_path):
    path = tmp_path / "mixed.pdf"
    doc = fitz.open()
    digital = doc.new_page()
    digital.insert_text(
        (72, 72),
        "Payment receipt Reference No DIGITAL123456 Amount MYR 10.00",
    )
    scanned = doc.new_page()
    ok, encoded = cv2.imencode(
        ".png", np.full((40, 100, 3), 255, dtype=np.uint8))
    assert ok
    scanned.insert_image(scanned.rect, stream=encoded.tobytes())
    doc.save(path)
    doc.close()

    monkeypatch.setattr(
        simple_server.ocr_pipeline.ocr_pipeline,
        "tesseract_available",
        lambda: False,
    )
    requested = []

    def vision_result(_path, **kwargs):
        requested.extend(kwargs["page_indices"])
        return {
            "text": "Reference No: SCANNED654321",
            "confidence": 0.91,
            "page_texts": [
                {"page": 2, "text": "Reference No: SCANNED654321"}
            ],
        }

    monkeypatch.setattr(simple_server.ocr_vision, "available", lambda: True)
    monkeypatch.setattr(simple_server.ocr_vision, "pdf_to_text", vision_result)

    result = simple_server._pdf_to_text(str(path))
    assert requested == [1]
    assert result["source"] == "hybrid_vision"
    assert "DIGITAL123456" in result["text"]
    assert "SCANNED654321" in result["text"]
    assert result["unread_pages"] == []


def test_missing_tesseract_goes_directly_to_vision(monkeypatch, tmp_path):
    path = tmp_path / "scan.pdf"
    doc = fitz.open()
    page = doc.new_page()
    ok, encoded = cv2.imencode(
        ".png", np.full((40, 100, 3), 255, dtype=np.uint8))
    assert ok
    page.insert_image(page.rect, stream=encoded.tobytes())
    doc.save(path)
    doc.close()

    core = simple_server.ocr_pipeline.ocr_pipeline
    monkeypatch.setattr(core, "tesseract_available", lambda: False)
    monkeypatch.setattr(
        core,
        "process_pdf_pages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Tesseract path should be skipped")),
    )
    monkeypatch.setattr(simple_server.ocr_vision, "available", lambda: True)
    monkeypatch.setattr(
        simple_server.ocr_vision,
        "pdf_to_text",
        lambda _path, **_kwargs: {
            "text": "Reference No: VISION998877",
            "confidence": 0.9,
            "page_texts": [
                {"page": 1, "text": "Reference No: VISION998877"}
            ],
        },
    )

    result = simple_server._pdf_to_text(str(path))
    assert result["source"] == "vision"
    assert "VISION998877" in result["text"]
