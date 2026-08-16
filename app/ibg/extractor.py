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
from app.ibg.party import (extract_beneficiary, extract_payer,
                           extract_payment_mode)
from app.ibg.reference_id import extract_references
from app.ibg.transaction_date import extract_transaction_date

# Below this, a field is not trustworthy enough to post without a human look.
REVIEW_THRESHOLD = 0.6

# Fields that must be present for a receipt to be actionable. Fee and
# total_debit are genuinely absent on most receipts, so they never force review.
REQUIRED = ("reference_id", "bank_name", "transaction_date", "amount")

# Everything the coordinator resolves. Which of these a given receipt actually
# carries varies by bank and document type, so absence is normal.
SCALAR_FIELDS = ("reference_id", "bank_name", "beneficiary_bank",
                 "transaction_date", "amount", "fee", "total_debit",
                 "payer", "beneficiary", "payment_mode")


import re

# Corporate suffixes carry no identifying weight, so "Bigbell industry" and
# "BIGBELL INDUSTRY SDN. BHD." must compare equal.
_CORP_SUFFIX_RE = re.compile(
    r"\b(?:SDN|BHD|BERHAD|PTE|LTD|LIMITED|COMPANY|CO|INC|CORP|"
    r"CORPORATION|GROUP|HOLDINGS|ENTERPRISE|MALAYSIA|M)\b\.?", re.IGNORECASE)


def _name_words(value):
    """Significant words of a party name, for echo comparison."""
    if not value:
        return set()
    stripped = _CORP_SUFFIX_RE.sub(" ", str(value).upper())
    return set(w for w in re.sub(r"[^A-Z0-9 ]", " ", stripped).split()
               if len(w) > 1)


def _drop_name_echoes(references, party_names):
    """Remove references that are just a party's own name written back.

    Only payer-supplied references are considered: a bank-issued reference that
    happens to share a word with the company name is still the bank's record
    and must survive.
    """
    parties = [_name_words(n) for n in party_names if n]
    if not parties:
        return references
    kept = []
    for ref in references:
        if ref.role != ROLE_PAYER_SUPPLIED:
            kept.append(ref)
            continue
        words = _name_words(ref.value)
        # A reference carrying digits is an identifier even if it also names
        # the company ("BIGBELL INV 4471"), so keep those.
        if not words or any(c.isdigit() for c in str(ref.value)):
            kept.append(ref)
            continue
        if any(p and (words <= p or len(words & p) >= max(2, len(words) * 0.6))
               for p in parties):
            continue
        kept.append(ref)
    return kept


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
    payer_result = extract_payer(text, ocr_used=ocr_used)
    beneficiary_result = extract_beneficiary(text, ocr_used=ocr_used)

    # Some payers type their own company name into the reference box, so the
    # slip carries "Customer Ref: Bigbell industry" next to a payer of
    # "BIGBELL INDUSTRY SDN. BHD.". That is a name, not a reference, and
    # surfacing it as one gives the reader a reference that identifies nothing.
    references = _drop_name_echoes(
        references, [payer_result.value, beneficiary_result.value])

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
        # Who the money moved between. Direction-aware: an inbound credit
        # advice names the payer as "Ordering Customer", an outbound slip as
        # "Debit From Account".
        "payer": payer_result,
        "beneficiary": beneficiary_result,
        # The rail the money moved on. Nothing extracted this before, so the
        # portal defaulted every transfer to IBG even when the slip said
        # DuitNow -- different rails with different clearing times.
        "payment_mode": extract_payment_mode(text, ocr_used=ocr_used),
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

    # Receipts differ: a DuitNow slip has no fee line, a recipient's copy names
    # no payer, and a truncated form carries no reference at all. A field that
    # is simply not on the document is not a failure and must not be flagged --
    # flagging it trains people to ignore the flag.
    #
    # What IS a failure is a field the document clearly carries that we could
    # not read. The extractors already distinguish the two in `source`:
    # "no ... label" means nothing to find, anything else means we found the
    # label and fell down on the value.
    def _absent_by_design(entry):
        src = (entry.get("source") or "").lower()
        return (
            src.startswith("none")
            or "no_reference_label" in src
            or "no_party_label" in src
            or "no_candidates" in src
            or "no_date_candidates" in src
            or "empty_text" in src
        )

    reasons = []
    for name in REQUIRED:
        entry = out[name]
        if entry["value"] is None:
            if not _absent_by_design(entry):
                # The label was there; we just could not resolve a value.
                reasons.append("%s: found the field but could not read it" % name)
        elif entry["confidence"] < REVIEW_THRESHOLD:
            reasons.append("%s low confidence (%.2f)" % (name, entry["confidence"]))
    for name in ("fee", "total_debit", "payer", "beneficiary",
                 "beneficiary_bank", "payment_mode"):
        entry = out[name]
        # Optional fields never flag on absence -- only on a value we distrust.
        if entry["value"] is not None and entry["confidence"] < REVIEW_THRESHOLD:
            reasons.append("%s low confidence (%.2f)" % (name, entry["confidence"]))

    out["fields_absent"] = [
        name for name in SCALAR_FIELDS
        if out[name]["value"] is None and _absent_by_design(out[name])
    ]

    # Average only over fields this receipt actually carries. Counting a field
    # the document never had as 0.0 dragged the score down for no reason and
    # made an otherwise clean extraction look doubtful.
    scores = [out[n]["confidence"] for n in REQUIRED
              if not _absent_by_design(out[n])]
    out["overall_confidence"] = round(sum(scores) / len(scores), 3) if scores else 0.0
    out["needs_review"] = bool(reasons)
    out["review_reasons"] = reasons
    return out
