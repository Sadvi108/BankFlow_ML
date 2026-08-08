"""Coordinator: run every IBG field extractor and assemble one honest result.

Resolution order is not arbitrary. The bank must resolve first, because date
parsing is bank-dependent -- Citibank's CitiDirect prints US month-first while
every Malaysian portal is day-first, so `08/05/2026` means 5 August on one and
8 May on the others. Resolving the bank first is what lets the date module ask
the registry which order to use.

Confidence here is derived from what was actually found. The defect this
replaces reported `1.0` on every receipt, including ones returning a debit
account number as the transaction reference, which meant `needs_review` never
fired and wrong values reached the ledger silently.
"""
from typing import Any, Dict, Optional

from app.ibg.amount import extract_amount, extract_fee, extract_total_debit
from app.ibg.bank_name import extract_bank_name, extract_beneficiary_bank
from app.ibg.bank_registry import ALL_ENTITIES, is_rail
from app.ibg.contract import (
    ROLE_BANK_PRIMARY,
    ROLE_BANK_SECONDARY,
    ROLE_PAYER_SUPPLIED,
    looks_like_ibg,
)
from app.ibg.reference_id import extract_references
from app.ibg.transaction_date import extract_transaction_date

# Below this, a field is not trustworthy enough to post without a human look.
REVIEW_THRESHOLD = 0.6

# Fields that must be present for a receipt to be actionable. Fee and
# total_debit are genuinely absent on most receipts, so they never force review.
REQUIRED = ("reference_id", "bank_name", "transaction_date", "amount")


def _bank_key_for(name: Optional[str]) -> Optional[str]:
    """Registry key for a resolved display name, for the date-order lookup."""
    if not name:
        return None
    for entity in ALL_ENTITIES:
        if entity.name == name and not is_rail(entity):
            return entity.key
    return None


def extract_ibg_fields(text: str, ocr_used: bool = True) -> Dict[str, Any]:
    """Every field on an interbank receipt, with per-field confidence.

    Returns a dict with, for each field, `value` / `confidence` / `source`,
    plus the full roled `references` list and an overall `needs_review`.
    """
    bank = extract_bank_name(text, ocr_used=ocr_used)
    bank_key = _bank_key_for(bank.value)

    beneficiary = extract_beneficiary_bank(text, ocr_used=ocr_used)
    date = extract_transaction_date(text, ocr_used=ocr_used, bank_key=bank_key)
    amount = extract_amount(text, ocr_used=ocr_used)
    fee = extract_fee(text, ocr_used=ocr_used)
    total_debit = extract_total_debit(text, ocr_used=ocr_used)

    references = extract_references(text, ocr_used=ocr_used)
    primary = next(
        (r for r in references if r.role == ROLE_BANK_PRIMARY), None
    )

    fields = {
        "reference_id": primary,
        "bank_name": bank,
        "beneficiary_bank": beneficiary,
        "transaction_date": date,
        "amount": amount,
        "fee": fee,
        "total_debit": total_debit,
    }

    out: Dict[str, Any] = {}
    for name, result in fields.items():
        out[name] = {
            "value": result.value if result is not None else None,
            "confidence": round(result.confidence, 3) if result is not None else 0.0,
            "source": result.source if result is not None else "none",
        }

    out["references"] = [r.to_dict() for r in references]
    out["reference_count"] = len(references)
    out["references_by_role"] = {
        role: [r.to_dict() for r in references if r.role == role]
        for role in (ROLE_BANK_PRIMARY, ROLE_BANK_SECONDARY, ROLE_PAYER_SUPPLIED)
    }
    out["is_ibg"] = looks_like_ibg(text)

    # A field is a review trigger when it is required and either missing or
    # weakly evidenced. Optional fields never trigger on absence, only on a
    # weak value -- a fee we could not read at all is normal; a fee we read
    # badly is not.
    reasons = []
    for name in REQUIRED:
        entry = out[name]
        if entry["value"] is None:
            reasons.append("%s missing" % name)
        elif entry["confidence"] < REVIEW_THRESHOLD:
            reasons.append("%s low confidence (%.2f)" % (name, entry["confidence"]))
    for name in ("fee", "total_debit"):
        entry = out[name]
        if entry["value"] is not None and entry["confidence"] < REVIEW_THRESHOLD:
            reasons.append("%s low confidence (%.2f)" % (name, entry["confidence"]))

    scores = [out[n]["confidence"] for n in REQUIRED]
    out["overall_confidence"] = round(sum(scores) / len(scores), 3) if scores else 0.0
    out["needs_review"] = bool(reasons)
    out["review_reasons"] = reasons
    return out
