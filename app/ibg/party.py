"""Who sent the money and who received it.

Every receipt names two parties, and which is which depends on the document's
direction. An outbound payment slip is written from the payer's side ("Debit
From Account", "From Account", "Applicant"); an inbound credit advice is
written from the beneficiary's side, where the payer appears as "Ordering
Customer" or "Remitter" instead.

Reading the wrong one puts the recipient's name in the payer column, which is
worse than leaving it blank: a reconciliation keyed on the payer would silently
match the wrong party.
"""
import re
from typing import List, Optional, Tuple

from app.ibg.contract import FieldResult

# Ordered most-specific first. Each entry is (pattern, confidence).
_PAYER_LABELS: List[Tuple[str, float]] = [
    (r"Ordering\s*Customer\s*Account\s*Name", 0.97),   # inbound credit advice
    (r"Remitter'?s?\s*Name", 0.97),
    (r"Debit\s*Acct\s*Name", 0.96),
    (r"Debit\s*Account\s*Name", 0.96),
    (r"Sender\s*Name", 0.96),
    (r"Applicant\s*Name", 0.95),
    (r"Account\s*Name", 0.88),
    (r"Company\s*Name", 0.88),
    (r"From\s*Account\s*Name", 0.95),
    # OCBC Velocity heads the payer block with "Your Account" and nothing else,
    # which is why the payer came back empty on every OCBC slip.
    (r"Your\s*Account", 0.90),
    (r"Debit\s*Acct\.?\s*No\.?", 0.85),
    (r"Source\s*Account", 0.90),
    (r"Debit\s*From\s*Account\s*No\.?", 0.90),
    (r"From\s*Account", 0.85),
    (r"Ordering\s*Customer", 0.90),
    # Citi CitiDirect names the payer "Ordering Party"; Maybank2E advices put
    # it under "Applicant Details / Contact Name". Neither was in the
    # vocabulary, so the payer came back empty on both -- and on the Maybank2E
    # advice the fallback reached the beneficiary's BANK instead.
    (r"Ordering\s*Party", 0.95),
    (r"Contact\s*Name", 0.88),
    (r"Payer\s*/?\s*Payee", 0.92),
    (r"Payer", 0.90),
    (r"Remitter", 0.95),
    (r"Applicant", 0.80),
]

_BENEFICIARY_LABELS: List[Tuple[str, float]] = [
    (r"Beneficiary\s*Name", 0.97),
    (r"Recipient\s*Name", 0.96),
    (r"Payee\s*Account\s*Name", 0.96),
    (r"Payee\s*Name", 0.95),
    (r"Payee'?s?\s*Nickname", 0.90),
    (r"Transfer\s*To", 0.90),
    (r"Credit\s*Account", 0.85),
    (r"Name\s*and\s*Addr\.?", 0.85),
]

_COMPILED_PAYER = [(re.compile(p, re.IGNORECASE), c) for p, c in _PAYER_LABELS]
_COMPILED_BENEF = [(re.compile(p, re.IGNORECASE), c) for p, c in _BENEFICIARY_LABELS]

_SEP_RE = re.compile(r"^[ \t]*[:\-]?[ \t]*")

# An organisation name, which is what a payer almost always is here.
_ORG_HINT_RE = re.compile(
    r"\b(?:SDN\.?\s*BHD|BHD|BERHAD|PTE|LTD|LIMITED|ENTERPRISE|TRADING|"
    r"LOGISTICS|SHIPPING|HOLDINGS|GROUP|CORPORATION|CORP|INC)\b",
    re.IGNORECASE)

# Account-number prefixes that lead a "From Account" value: the name follows.
_ACCOUNT_PREFIX_RE = re.compile(r"^[\d*X\-]{5,}\s*(?:\([A-Z]{3}\))?\s*[/|-]?\s*")
_MASKED_RE = re.compile(r"^[*X\d\s\-]+$", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"^[-–—.\s]*$")

_MAX_LEN = 80
_WINDOW = 160

# When a label's own value is blank the scan runs on into the next field and
# adopts ITS label as the party name -- "Transfer Mode", "Details", "To
# Account" were all being reported as payers. A party is an organisation or a
# person, never one of these.
_LABEL_LIKE_RE = re.compile(
    r"^(?:"
    r"(?:Transaction|Transfer|Payment|Beneficiary|Recipient|Payee|Payer|"
    r"Account|Bank|Debit|Credit|Value|Applicant|Ordering|Customer|Source|"
    r"Destination|Instruction|Additional|Reference|Service|Product|Status|"
    r"Charges?|Fee|Amount|Currency|Date|Details?|Mode|Type|Name|Number|No)"
    r"[\s.:/'’-]*)+$",
    re.IGNORECASE)


def _clean_party(raw: str) -> Optional[str]:
    """Trim a raw slice to an organisation/person name, or None."""
    if raw is None:
        return None
    value = raw.strip()
    value = re.split(r"\s{2,}", value)[0].strip()
    value = value.strip(" \t\r\n:;|")
    if not value or _PLACEHOLDER_RE.match(value):
        return None
    # "8000000002 / BLUE ORCHID LOGISTICS SDN BHD (MYR)" -> drop the account.
    value = _ACCOUNT_PREFIX_RE.sub("", value).strip()
    # "510000000001 (MYR) ACME LOGISTICS" leaves a currency marker behind.
    # Match the actual currency codes, not any three uppercase letters: the
    # loose form ate the first word of "ECO ACTION SDN BHD" and reported the
    # payer as "ACTION SDN BHD".
    value = re.sub(r"^\(?(?:MYR|RM|USD|SGD|EUR|GBP|AUD|JPY|CNY|HKD)\)?[\s-]+",
                   "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s*\((?:MYR|RM|USD|SGD|EUR|GBP|AUD|JPY|CNY|HKD)\)\s*$",
                   "", value, flags=re.IGNORECASE).strip()
    if not value or _MASKED_RE.match(value) or len(value) > _MAX_LEN:
        return None
    if not re.search(r"[A-Za-z]{3}", value):
        return None
    if _LABEL_LIKE_RE.match(value):
        return None
    # "Number: 8880000000002" -- the scan landed on the tail of the next
    # label rather than on a name.
    if re.match(r"^(?:No\.?|Number|ID|Code)\s*[:\-]", value, re.IGNORECASE):
        return None
    return value


def _value_after(text: str, start: int) -> Optional[str]:
    """Rest of the label's line, else the next non-empty line."""
    sep = _SEP_RE.match(text, start)
    cursor = sep.end() if sep else start
    limit = min(cursor + _WINDOW, len(text))
    line_end = text.find("\n", cursor)
    if line_end == -1:
        line_end = len(text)

    value = _clean_party(text[cursor:min(line_end, limit)])
    if value:
        return value

    pos = line_end + 1
    while pos < limit:
        nxt = text.find("\n", pos)
        if nxt == -1:
            nxt = len(text)
        candidate = _clean_party(text[pos:min(nxt, limit)])
        if candidate:
            return candidate
        if text[pos:nxt].strip():
            break
        pos = nxt + 1
    return None


def _extract(text: str, compiled, ocr_used: bool) -> FieldResult:
    if not text or not text.strip():
        return FieldResult.missing(source="missing:empty_text")

    best = None
    for regex, confidence in compiled:
        for m in regex.finditer(text):
            matched = text[m.start():m.end()]
            if matched.islower():        # prose, not a field label
                continue
            value = _value_after(text, m.end())
            if not value:
                continue
            score = confidence
            # An organisation name is far more likely to be the real party
            # than a stray line that happened to follow the label.
            if _ORG_HINT_RE.search(value):
                score += 0.02
            if ocr_used:
                score -= 0.05
            if best is None or score > best[1]:
                best = (value, score, matched.strip())
            break                        # first hit per label is enough

    if best is None:
        return FieldResult.missing(source="missing:no_party_label")
    value, score, label = best
    return FieldResult(
        value=value,
        confidence=round(max(0.5, min(0.99, score)), 3),
        source="label:" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_"),
        candidates=[],
    )


def extract_payer(text: str, ocr_used: bool = True) -> FieldResult:
    """The party the money came FROM."""
    return _extract(text, _COMPILED_PAYER, ocr_used)


def extract_beneficiary(text: str, ocr_used: bool = True) -> FieldResult:
    """The party the money went TO."""
    return _extract(text, _COMPILED_BENEF, ocr_used)


# ---------------------------------------------------------------------------
# Payment mode
# ---------------------------------------------------------------------------
# The portal was showing every transfer as "IBG" because nothing extracted the
# mode at all. Interbank GIRO and DuitNow are different rails with different
# clearing times, so the distinction is not cosmetic.
#
# Ordered most specific first: "DuitNow Transfer" must beat bare "Transfer",
# and a slip saying "Interbank GIRO (IBG)" must not also match "GIRO" alone.
_MODE_PATTERNS = [
    (r"Interbank\s*GIRO\s*\(\s*IBG\s*\)", "Interbank GIRO (IBG)"),
    (r"Interbank\s*GIRO", "Interbank GIRO (IBG)"),
    (r"\bIBG\b", "Interbank GIRO (IBG)"),
    (r"DuitNow\s*\(\s*Transfer\s*\)", "DuitNow Transfer"),
    (r"DuitNow\s*Transfer", "DuitNow Transfer"),
    (r"DuitNow\s*Payment", "DuitNow Payment"),
    (r"\bDuitNow\b", "DuitNow"),
    (r"Outward\s*ACH", "Outward ACH"),
    (r"Inward\s*ACH", "Inward ACH"),
    (r"ACH\s*Credit\s*/?\s*GIRO", "ACH Credit/GIRO"),
    (r"Automated\s*Clearing\s*House\s*\(\s*ACH\s*\)", "ACH"),
    (r"Instant\s*Transfer", "Instant Transfer"),
    (r"Immediate\s*Transfer", "Immediate Transfer"),
    (r"RENTAS", "RENTAS"),
    (r"Funds?\s*Transfer", "Fund Transfer"),
]
_COMPILED_MODE = [(re.compile(p, re.IGNORECASE), name) for p, name in _MODE_PATTERNS]

# Labels that introduce the mode. A hit next to one of these is authoritative;
# a bare mention anywhere on the page is weaker but still usable.
_MODE_LABEL_RE = re.compile(
    r"\b(?:Payment\s*Mode|Product\s*Type|Payment\s*Type|Transfer\s*Mode|"
    r"Transaction\s*Type|Services?|Payment\s*Method|Transfer\s*Type|"
    r"Financial\s*Txn\s*Type)\b", re.IGNORECASE)
_MODE_LABEL_WINDOW = 120


def extract_payment_mode(text, ocr_used=True):
    """The rail the money moved on -- IBG, DuitNow, ACH, RENTAS."""
    if not text or not text.strip():
        return FieldResult.missing(source="missing:empty_text")

    # A value sitting next to an explicit mode label wins.
    for lm in _MODE_LABEL_RE.finditer(text):
        window = text[lm.end():lm.end() + _MODE_LABEL_WINDOW]
        for regex, name in _COMPILED_MODE:
            m = regex.search(window)
            if m:
                conf = 0.95 - (0.05 if ocr_used else 0.0)
                return FieldResult(value=name, confidence=round(conf, 3),
                                   source="label:payment_mode", candidates=[])

    # Otherwise the most specific mode named anywhere on the receipt.
    for regex, name in _COMPILED_MODE:
        if regex.search(text):
            conf = 0.75 - (0.05 if ocr_used else 0.0)
            return FieldResult(value=name, confidence=round(conf, 3),
                               source="mention:payment_mode", candidates=[])

    return FieldResult.missing(source="missing:no_mode")
