"""Shared contract for the IBG field extractors.

Every field module under `app/ibg/` exposes one function that takes the receipt
text and returns a `FieldResult`. Nothing else is shared, so the four field
modules stay independent of each other.

A field that cannot be found returns `FieldResult.missing()`. Returning a wrong
value is worse than returning nothing: a `None` is flagged for review, whereas a
confident wrong value is silently trusted downstream.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- Reference roles -------------------------------------------------------
# A receipt prints several reference numbers and they are not interchangeable.
# The role says who issued the value, which is what decides whether it may be
# promoted to the scalar `reference_id` the rest of the system consumes.

ROLE_BANK_PRIMARY = "bank_primary"      # the bank's own transaction reference
ROLE_BANK_SECONDARY = "bank_secondary"  # service / clearing / batch / UETR
ROLE_PAYER_SUPPLIED = "payer_supplied"  # free text the payer typed

ROLES = (ROLE_BANK_PRIMARY, ROLE_BANK_SECONDARY, ROLE_PAYER_SUPPLIED)


@dataclass
class Reference:
    """One reference number found on a receipt, with the label that named it.

    Attributes:
        value:      The normalized reference.
        label:      The label as printed on the receipt, e.g.
                    "Service Reference No". Kept verbatim so a human reviewing
                    the extraction can see which line it came from.
        role:       One of ROLES. Drives which reference may become the scalar
                    `reference_id`: only `bank_primary` qualifies.
        confidence: 0.0-1.0.
        source:     Short slug naming the rule that matched, e.g.
                    "label:service_reference_no".
    """

    value: str
    label: str
    role: str
    confidence: float = 0.0
    source: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
            "role": self.role,
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


def primary_of(references: List["Reference"]) -> Optional["Reference"]:
    """The single bank-issued primary reference, or None.

    A payer-supplied reference is never promoted: the payer's own free text is
    not the bank's record of the transaction, and treating it as one is what
    made the pipeline report someone's invoice number as the transaction ID.
    """
    banked = [r for r in references if r.role == ROLE_BANK_PRIMARY]
    if not banked:
        return None
    return max(banked, key=lambda r: r.confidence)


@dataclass
class FieldResult:
    """One extracted field, with the evidence for how it was found.

    Attributes:
        value:      The normalized value, or None when not found.
        confidence: 0.0-1.0. Must be < 0.5 when `value` is None.
        source:     Short slug naming the rule that matched, e.g.
                    "label:reference_no". Used by QA to see *why* a value won.
        candidates: Every value considered, best first, as (value, score).
                    Kept for debugging and for the review UI.
    """

    value: Optional[str] = None
    confidence: float = 0.0
    source: str = "none"
    candidates: List[Any] = field(default_factory=list)

    @classmethod
    def missing(cls, source: str = "none") -> "FieldResult":
        return cls(value=None, confidence=0.0, source=source, candidates=[])

    @property
    def found(self) -> bool:
        return self.value is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "candidates": self.candidates[:10],
        }


# Markers that identify a receipt as an interbank transfer (IBG / Interbank
# GIRO and its modern equivalents). Any hit means the IBG path should run.
IBG_MARKERS = (
    "IBG",
    "INTERBANK",
    "INTER-BANK",
    "INTER BANK",
    "GIRO",
    "OUTWARD ACH",
    "INWARD ACH",
    "ACH TRANSFER",
    "DUITNOW",
    "INSTANT TRANSFER",
    "FUNDS TRANSFER",
    "RENTAS",
)


def looks_like_ibg(text: str) -> bool:
    """True when the receipt text carries any interbank-transfer marker."""
    if not text:
        return False
    upper = text.upper()
    return any(marker in upper for marker in IBG_MARKERS)
