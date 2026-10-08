"""Robustness and value-integrity tests for the trained production repair."""
import json
import re

import pytest

from app.ibg.extractor import extract_ibg_fields
from app.ibg.label_repair import MODEL_PATH, _aliases, repair_labels
from app.ocr_pipeline import OCRPipeline
from tests.fixtures.ibg_corpus import CORPUS
from tests.fixtures.ibg_holdout import HOLDOUT


@pytest.mark.parametrize("sample", CORPUS + HOLDOUT, ids=lambda s: s["id"])
@pytest.mark.parametrize("word,damaged", [
    ("Reference", "Reforence"), ("Transaction", "Transactlon"),
    ("Amount", "Arnount"), ("Date", "Oate"),
])
def test_noisy_labels_preserve_clean_extraction(sample, word, damaged):
    # Compare with the clean pipeline, rather than inventing evidence for a
    # bank name missing from an unbranded receipt. This includes unseen layouts.
    original = extract_ibg_fields(sample["text"], ocr_used=True)
    noisy = re.sub(word, damaged, sample["text"], flags=re.IGNORECASE)
    repaired = extract_ibg_fields(noisy, ocr_used=True)
    for field in ("reference_id", "bank_name", "transaction_date", "amount",
                  "fee", "total_debit"):
        assert repaired[field]["value"] == original[field]["value"]


@pytest.mark.parametrize("ocr_used", [False, True])
def test_values_and_party_names_are_never_spell_corrected(ocr_used):
    text = ("Customer Reference: Transactlon-Reforence-0091\n"
            "Beneficiary Name: Arnount Oate Trading\n"
            "Amount: 1,234.56\nReference No: RF0I112233\n")
    assert repair_labels(text, ocr_used)[0] == text


def test_clean_digital_text_is_authoritative():
    text = "Reforence No: AB123456\nArnount: MYR 12.34\n"
    assert repair_labels(text, ocr_used=False) == (text, [])
    assert repair_labels("DuitNow Reference No: BANK123456", ocr_used=False) == (
        "DuitNow Reference No: BANK123456", [])


@pytest.mark.parametrize("customer", ["Custorner", "Custonier"])
def test_secondary_and_customer_roles_survive_label_repair(customer):
    text = ("Public Bank\nTransaction Reforence No: BANK123456\n"
            "Service Reforence No: CLEAR123456\n" + customer + " Reforence: INV123456\n")
    result = extract_ibg_fields(text)
    assert [(r["value"], r["role"]) for r in result["references"]] == [
        ("BANK123456", "bank_primary"), ("CLEAR123456", "bank_secondary"),
        ("INV123456", "payer_supplied")]


def test_unicode_label_separator_preserves_value_punctuation():
    text = "Customer Reference： INV:RF0099\nTransaction Date： 01/02/2030 12:34:56\n"
    repaired, repairs = repair_labels(text, False)
    assert repaired == "Customer Reference: INV:RF0099\nTransaction Date: 01/02/2030 12:34:56\n"
    assert len(repairs) == 2


def test_prose_and_unlabeled_identifiers_are_unchanged():
    text = "Please use the Reforence from your email.\nARNOUNT0091\nA:Reforence\n"
    assert repair_labels(text)[0] == text


@pytest.mark.parametrize("value", ["Arnount Trading", "Transactlon-Reforence0091", "Reforence-0091"])
def test_multiline_customer_values_are_unchanged(value):
    text = "Customer Reference\n" + value + "\n"
    assert repair_labels(text)[0] == text


def test_api_keeps_unreadable_amount_empty_instead_of_using_legacy_guess(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import simple_server
    monkeypatch.setattr(simple_server, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(simple_server, "db", None)
    monkeypatch.setattr(simple_server.history_manager, "add_entry", lambda _entry: "test")
    monkeypatch.setattr(simple_server, "_pdf_to_text", lambda _path: {
        "text": "Public Bank\nReference No: BANK123456\nTransaction Date: 01/02/2030\nAmount: unreadable",
        "source": "embedded", "tokens": [], "confidence": 0.9})
    monkeypatch.setattr(simple_server, "extract_all_fields_v3", lambda *a, **kw: {
        "bank_name": "Public Bank", "transaction_id": "BANK123456",
        "amount": "999.99", "date": "2030-02-01"})
    response = TestClient(simple_server.app).post("/extract", files={
        "file": ("receipt.pdf", b"pdf-placeholder", "application/pdf")})
    assert response.status_code == 200
    assert response.json()["data"]["amount"] is None
    assert response.json()["data"]["needs_review"] is True


def test_model_missing_fails_open_for_clean_labels(monkeypatch, tmp_path):
    import app.ibg.label_repair as module
    monkeypatch.setattr(module, "MODEL_PATH", tmp_path / "missing.json")
    _aliases.cache_clear()
    try:
        assert repair_labels("Reference No: BANK123456")[0] == "Reference No: BANK123456"
    finally:
        _aliases.cache_clear()


def test_model_is_reproducible_and_does_not_train_on_holdout():
    from scripts.train_label_repair import train
    trained = train()
    assert trained == json.loads(MODEL_PATH.read_text())
    assert trained["training"]["receipts"] == len(CORPUS)
    assert all(value.isalpha() for value in trained["aliases"].values())


@pytest.mark.parametrize("unreadable", ["Amount: unreadable", "Transaction Date: unreadable"])
def test_explicit_unreadable_field_requires_review_and_ocr_retry(unreadable):
    text = "Public Bank\nIBG Transfer\nReference No: BANK123456\n" + unreadable
    result = extract_ibg_fields(text)
    assert result["needs_review"]
    assert "found the field but could not read it" in " ".join(result["review_reasons"])
    assert not OCRPipeline()._is_good_read({"text": text, "confidence": 0.99, "word_count": 10})


def test_retry_recovers_amount_without_losing_bank_reference(monkeypatch):
    import numpy as np
    pipeline = OCRPipeline()
    common = "Public Bank\nIBG Transfer\nReference No: BANK123456\nTransaction Date: 01/02/2030\n"
    reads = iter([
        {"text": common + "Amount: unreadable", "confidence": 0.97, "word_count": 20},
        {"text": common + "Arnount: MYR 12.34", "confidence": 0.91, "word_count": 20},
    ])
    monkeypatch.setattr(pipeline, "_run_tesseract", lambda *a, **kw: next(reads))
    monkeypatch.setattr(pipeline, "_auto_rotate_text", lambda image: image)
    result = pipeline.extract_text_with_confidence(np.zeros((20, 20), np.uint8))
    assert result["passes_used"] == 2
    assert extract_ibg_fields(result["text"])["amount"]["value"] == "12.34"
