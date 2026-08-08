"""IBG (Interbank GIRO) receipt field extraction.

One module per field, each independently testable:

    reference_id.py      -> extract_reference_id(text)
    bank_name.py         -> extract_bank_name(text)
    transaction_date.py  -> extract_transaction_date(text)
    amount.py            -> extract_amount(text)

`extractor.py` composes the four into a single `extract_ibg_fields(text)` call.
"""
from app.ibg.contract import (
    FieldResult,
    Reference,
    ROLE_BANK_PRIMARY,
    ROLE_BANK_SECONDARY,
    ROLE_PAYER_SUPPLIED,
    ROLES,
    looks_like_ibg,
    primary_of,
)

__all__ = [
    "FieldResult",
    "Reference",
    "ROLE_BANK_PRIMARY",
    "ROLE_BANK_SECONDARY",
    "ROLE_PAYER_SUPPLIED",
    "ROLES",
    "looks_like_ibg",
    "primary_of",
]
