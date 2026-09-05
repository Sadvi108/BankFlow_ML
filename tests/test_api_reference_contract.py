"""The portal-facing response must expose every reference without ambiguity."""

import cv2
import numpy as np
from fastapi.testclient import TestClient

import simple_server


def test_response_has_complete_reference_arrays_without_none(monkeypatch):
    monkeypatch.setattr(simple_server.history_manager, "add_entry", lambda _entry: "1")
    monkeypatch.setattr(simple_server, "db", None)
    merged = {
        "bank_name": "Example Bank",
        "transaction_id": "BANK-PRIMARY-1",
        "reference_number": "BANK-PRIMARY-1",
        "all_ids": [None, {"id": "BANK-PRIMARY-1", "score": 270}],
        "date": "2030-02-01",
        "amount": "10.00",
        "overall_confidence": 0.9,
        "field_confidence": {},
        "field_source": {},
        "method": "local",
        "needs_review": False,
    }
    ibg_fields = {
        key: {"value": None, "confidence": 0.0, "source": "none"}
        for key in (
            "reference_id", "bank_name", "beneficiary_bank",
            "transaction_date", "amount", "fee", "total_debit", "payer",
            "beneficiary", "payment_mode",
        )
    }
    ibg_fields.update({
        "references": [
            {"value": "BANK-PRIMARY-1", "label": "Reference No",
             "role": "bank_primary", "confidence": 0.95},
            {"value": "CLEARING-2", "label": "PayNet Reference No",
             "role": "bank_secondary", "confidence": 0.90},
            {"value": "INVOICE-3", "label": "Customer Ref",
             "role": "payer_supplied", "confidence": 0.85},
        ],
        "references_by_role": {},
        "reference_count": 3,
        "review_reasons": [],
        "fields_absent": [],
    })

    response = simple_server._build_response(
        "file-1", "receipt.pdf", merged, "raw", 0.9, False,
        ibg=ibg_fields,
    )["data"]

    assert response["primary_reference_id"] == "BANK-PRIMARY-1"
    assert response["reference_ids"] == [
        "BANK-PRIMARY-1", "CLEARING-2", "INVOICE-3"]
    assert response["all_ids"] == response["reference_ids"]
    assert response["bank_reference_ids"] == [
        "BANK-PRIMARY-1", "CLEARING-2"]
    assert response["payer_reference_ids"] == ["INVOICE-3"]
    assert response["reference_count"] == 3
    assert None not in response["all_ids"]


def test_extract_endpoint_returns_all_reference_roles_and_timings(monkeypatch,
                                                                  tmp_path):
    text = (
        "Public Bank Berhad\n"
        "Interbank GIRO\n"
        "Transaction Date: 01/02/2030\n"
        "Transaction Amount: MYR 10.00\n"
        "Beneficiary Account No: 21001234567890\n"
        "Transaction Reference No: TXN9988776655\n"
        "PayNet Reference No: CLEARING556677\n"
        "Customer Ref: INVOICE778899\n"
    )
    monkeypatch.setattr(simple_server, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(simple_server.history_manager, "add_entry", lambda _entry: "1")
    monkeypatch.setattr(simple_server, "db", None)
    monkeypatch.setattr(
        simple_server.ocr_pipeline.ocr_pipeline,
        "tesseract_available",
        lambda: True,
    )
    monkeypatch.setattr(
        simple_server.ocr_pipeline,
        "extract_text_with_confidence",
        lambda _image: {
            "text": text,
            "tokens": [],
            "confidence": 0.92,
            "source": "ocr",
            "method": "light",
            "passes_used": 1,
            "processed_successfully": True,
        },
    )
    encoded, png = cv2.imencode(
        ".png", np.zeros((20, 20, 3), dtype=np.uint8))
    assert encoded

    with TestClient(simple_server.app) as client:
        response = client.post(
            "/extract",
            files={"file": ("receipt.png", png.tobytes(), "image/png")},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["reference_ids"] == [
        "TXN9988776655", "CLEARING556677", "INVOICE778899"]
    assert data["bank_reference_ids"] == [
        "TXN9988776655", "CLEARING556677"]
    assert data["payer_reference_ids"] == ["INVOICE778899"]
    assert "21001234567890" not in data["reference_ids"]
    assert data["timings"]["ocr_ms"] >= 0
    assert data["processing_time_ms"] >= data["timings"]["ocr_ms"]
    assert data["ocr_details"]["passes_used"] == 1
