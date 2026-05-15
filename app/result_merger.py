"""Merge local (regex+layout) and Gemini extraction results into one verified record.

Strategy per field:
  - both valid + equal  -> field_confidence = 1.0, source = "cross_validated"
  - both valid + differ -> take gemini, needs_review = True
  - only one valid      -> take it
  - neither valid       -> None, needs_review = True (if both were attempted)
"""
from typing import Any, Dict, Optional, Tuple
import logging
import re

from app.ultimate_patterns_v3 import validate_id, validate_amount, validate_date

logger = logging.getLogger(__name__)

_FIELD_VALIDATORS = {
    "transaction_id": validate_id,
    "date": validate_date,
    "amount": validate_amount,
}


def _normalize_id(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().upper()
    # Multi-line pattern captures sometimes append a trailing label
    # (e.g. "MYGP251111790268\nNOTIFY"). Keep only the first line.
    s = s.split("\n")[0].split("\r")[0].strip()
    return s.replace(" ", "")


def _normalize_amount(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().replace("RM", "").replace("MYR", "").strip()
    s = s.replace(",", "")
    try:
        return f"{float(s):.2f}"
    except (ValueError, TypeError):
        return s


def _normalize_date(v: Any) -> Optional[str]:
    if v is None:
        return None
    return str(v).strip()


_NORMALIZERS = {
    "transaction_id": _normalize_id,
    "amount": _normalize_amount,
    "date": _normalize_date,
}


def _merge_field(field: str, local: Any, llm: Any) -> Tuple[Any, float, str]:
    """Return (value, confidence, source) for a single field."""
    validator = _FIELD_VALIDATORS[field]
    normalizer = _NORMALIZERS[field]
    local_v = normalizer(local) if local else None
    llm_v = normalizer(llm) if llm else None
    local_ok = validator(local_v)
    llm_ok = validator(llm_v)

    if local_ok and llm_ok:
        if local_v == llm_v:
            return local_v, 1.0, "cross_validated"
        # Disagreement: trust LLM for date/amount; for ID, trust whichever is longer/more specific.
        if field == "transaction_id":
            chosen = llm_v if len(llm_v) >= len(local_v) else local_v
        else:
            chosen = llm_v
        return chosen, 0.7, "llm_preferred"
    if local_ok:
        return local_v, 0.85, "local"
    if llm_ok:
        return llm_v, 0.9, "llm"
    # Neither validates — return whichever is non-empty (best-effort), low confidence.
    fallback = llm_v or local_v
    return fallback, 0.3, "unvalidated"


def merge(local: Dict[str, Any], llm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge a local-pipeline result dict with an optional Gemini result dict.

    Both dicts use the keys: bank_name, transaction_id, date, amount.
    """
    llm = llm or {}

    bank_local = (local.get("bank_name") or "Unknown").strip()
    bank_llm = (llm.get("bank_name") or "Unknown").strip()
    if bank_local.lower() in ("", "unknown") and bank_llm.lower() not in ("", "unknown"):
        bank = bank_llm
    else:
        bank = bank_local

    fields = {}
    confidences = {}
    sources = {}
    for f in ("transaction_id", "date", "amount"):
        val, conf, src = _merge_field(f, local.get(f), llm.get(f))
        fields[f] = val
        confidences[f] = conf
        sources[f] = src

    used_llm = bool(llm)
    any_unvalidated = any(s == "unvalidated" for s in sources.values())
    any_missing = any(v is None for v in fields.values())
    needs_review = any_unvalidated or any_missing or (not bank or bank == "Unknown")

    if used_llm and not needs_review:
        method = "cross_validated" if all(s == "cross_validated" for s in sources.values()) else "llm_fallback"
    elif used_llm:
        method = "llm_fallback"
    else:
        method = "local"

    overall = (
        sum(confidences.values()) / len(confidences) if confidences else 0.0
    )
    if bank == "Unknown":
        overall *= 0.7

    return {
        "bank_name": bank,
        "transaction_id": fields["transaction_id"],
        "reference_number": fields["transaction_id"],
        "date": fields["date"],
        "amount": fields["amount"],
        "field_confidence": confidences,
        "field_source": sources,
        "method": method,
        "needs_review": needs_review,
        "overall_confidence": round(overall, 3),
    }
