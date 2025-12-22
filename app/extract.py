import re
from typing import Dict, List, Tuple, Optional


# Common patterns for fields across Malaysian bank slips
REF_PATTERNS = [
    r"\bRef(?:erence)?(?:\s*No\.?|#)?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
    r"\bIBG\s*Ref(?:erence)?\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
    r"\bReference\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
    r"\bCustomer\s+Ref(?:erence)?\b\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
    r"\bBank\s+Ref(?:erence)?\b\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
    r"\bRef\s*ID\b\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
]

TXN_PATTERNS = [
    r"\bTransaction(?:\s*ID|\s*No\.)?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
    r"\bTxn(?:\s*ID)?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
]

INVOICE_PATTERNS = [
    r"\bInvoice(?:\s*No\.?|#)?\s*[:\-]?\s*([A-Za-z0-9\-/]{4,})",
    r"\bInv\s*Ref\b\s*[:\-]?\s*([A-Za-z0-9\-/]{4,})",
]

# DuitNow-specific reference label patterns
DUITNOW_REF_PATTERNS = [
    r"\bDuit\s*Now\b.*?\bRef(?:erence)?(?:\s*No\.?|\s*Number|#)?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
    r"\bDuitNow\b.*?\bRef(?:erence)?(?:\s*No\.?|\s*Number|#)?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
    r"\bDuit\s*Now\s*Ref\b\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
    r"\bDuitNow\s*Ref\b\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
]

AMOUNT_PATTERNS = [
    r"\b(?:RM|MYR|\$)\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)\b",
    r"\bAmount\s*[:\-]?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)\b",
]

DATE_PATTERNS = [
    r"\b(\d{4}[\/-]\d{1,2}[\/-]\d{1,2})\b",  # yyyy-mm-dd
    r"\b(\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4})\b",  # dd/mm/yyyy
    r"\b(\d{1,2}\s*[A-Za-z]{3}\s*\d{2,4})\b",   # 12 Jan 2024
]


def _match_first(patterns: List[str], text: str) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            # return the first capturing group
            return m.group(1).strip()
    return None


def _find_token_box(tokens: List[Dict], value: str) -> Optional[Dict]:
    """Find a token whose text equals the value (case-insensitive) and return its box."""
    val_norm = value.strip().lower()
    for t in tokens:
        if t.get("text", "").strip().lower() == val_norm:
            return {
                "left": t["left"],
                "top": t["top"],
                "width": t["width"],
                "height": t["height"],
                "conf": t.get("conf", None),
            }
    return None


LABEL_PATTERNS = [
    r"^ref$", r"^ref\.$", r"^reference$", r"^refno$", r"^refno\.$",
    r"^cust$", r"^customer$", r"^bank$", r"^id$", r"^number$", r"^no$", r"^no\.$",
]


def _extract_by_label_tokens(tokens: List[Dict]) -> Optional[str]:
    """Heuristic: find a label token (e.g., 'Ref', 'Reference', 'Ref No'), then capture the
    next few tokens that look like an ID (alphanumeric, dashes, slashes)."""
    def is_label(tok: str) -> bool:
        t = tok.strip().lower()
        for pat in LABEL_PATTERNS:
            if re.match(pat, t):
                return True
        return False

    def is_value_token(tok: str) -> bool:
        # Must contain at least one digit and be reasonably long
        t = tok.strip()
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9\-/]+$", t):
            return False
        if sum(ch.isdigit() for ch in t) < 2:
            return False
        return len(t) >= 5

    n = len(tokens)
    for i in range(n):
        txt = tokens[i].get("text", "").strip()
        if not txt:
            continue
        if txt.endswith(":"):
            # direct value may follow after colon
            # e.g., "Reference:" <value>
            # check label nature before colon
            base = txt[:-1]
            if is_label(base):
                # look ahead
                values: List[str] = []
                for j in range(i + 1, min(i + 8, n)):
                    nxt = tokens[j].get("text", "").strip()
                    if is_value_token(nxt):
                        values.append(nxt)
                    else:
                        if values:
                            break
                if values:
                    return "".join(values)
        # patterns like "Ref" "No" ":" <value>
        if is_label(txt):
            # scan next few tokens to find colon and the value
            saw_colon = False
            values: List[str] = []
            for j in range(i + 1, min(i + 12, n)):
                nxt = tokens[j].get("text", "").strip()
                if not nxt:
                    continue
                if nxt in (":", "-", "–"):
                    saw_colon = True
                    continue
                if is_label(nxt):
                    # label continuation like "No" or "ID"; keep scanning
                    continue
                if is_value_token(nxt):
                    values.append(nxt)
                else:
                    if values:
                        break
            if values:
                return "".join(values)
            # If we saw a colon and next isn't a value, continue scanning
    return None


def _extract_duitnow_by_tokens(tokens: List[Dict]) -> Optional[str]:
    """Heuristic for DuitNow: find token 'DuitNow' (or 'Duit' 'Now'), then look
    ahead for a reference label and capture the following value token(s)."""
    def is_value_token(tok: str) -> bool:
        t = tok.strip()
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9\-/]+$", t):
            return False
        if sum(ch.isdigit() for ch in t) < 2:
            return False
        return len(t) >= 5

    n = len(tokens)
    i = 0
    while i < n:
        t = tokens[i].get("text", "").strip()
        low = t.lower()
        if low == "duitnow" or (low == "duit" and i + 1 < n and tokens[i+1].get("text", "").strip().lower() == "now"):
            # scan next few tokens for a ref-like label and a value
            start_j = i + 1 if low == "duitnow" else i + 2
            values: List[str] = []
            for j in range(start_j, min(start_j + 12, n)):
                nxt = tokens[j].get("text", "").strip()
                if not nxt:
                    continue
                if nxt in (":", "-", "–"):
                    continue
                # try to find 'Ref' like token or directly a value
                if re.match(r"^(ref|reference|refno|ref\.|no|no\.|number|id)$", nxt, flags=re.IGNORECASE):
                    continue
                if is_value_token(nxt):
                    values.append(nxt)
                else:
                    if values:
                        break
            if values:
                return "".join(values)
        i += 1
    return None


def extract_fields(text: str, tokens: List[Dict], bank_hint: Optional[str] = None) -> Dict:
    """Extract common fields using regex, optionally bank-specific tweaks.

    Returns a dict with values and optional bounding boxes.
    """
    result = {
        "reference_number": None,
        "transaction_id": None,
        "transaction_number": None,
        "duitnow_reference_number": None,
        "invoice_number": None,
        "amount": None,
        "date": None,
        "boxes": {},
        "meta": {},
    }

    # Bank-specific patterns can improve accuracy for certain templates
    bank_specific_ref_patterns: List[str] = []
    if bank_hint:
        bh = bank_hint.lower()
        if "public" in bh:
            bank_specific_ref_patterns.extend([
                r"\bTransaction\s*Reference(?:\s*No\.?|\s*Number)?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
                r"\bPayment\s*Reference(?:\s*No\.?|\s*Number)?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
            ])
        if "rhb" in bh:
            bank_specific_ref_patterns.extend([
                r"\bReference\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
            ])
        if "maybank" in bh:
            bank_specific_ref_patterns.extend([
                r"\bReference\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
                r"\bRef\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
            ])
        if "cimb" in bh:
            bank_specific_ref_patterns.extend([
                r"\bReference(?:\s*No\.?|\s*Number)?\s*[:\-]?\s*([A-Za-z0-9\-/]{6,})",
            ])

    ref = _match_first(bank_specific_ref_patterns + REF_PATTERNS, text)
    if not ref:
        # try token-based heuristic near labels
        ref = _extract_by_label_tokens(tokens)

    # DuitNow reference number
    dn_ref = _match_first(DUITNOW_REF_PATTERNS, text)
    if not dn_ref:
        dn_ref = _extract_duitnow_by_tokens(tokens)
    txn = _match_first(TXN_PATTERNS, text)
    inv = _match_first(INVOICE_PATTERNS, text)
    amt = _match_first(AMOUNT_PATTERNS, text)
    dt = _match_first(DATE_PATTERNS, text)

    result["reference_number"] = ref
    result["transaction_id"] = txn
    result["duitnow_reference_number"] = dn_ref
    # Unify: pick whichever exists as transaction_number
    if txn:
        result["transaction_number"] = txn
        result["meta"]["transaction_number_source"] = "transaction_id"
    elif ref:
        result["transaction_number"] = ref
        result["meta"]["transaction_number_source"] = "reference_number"
    result["invoice_number"] = inv
    result["amount"] = amt
    result["date"] = dt

    # Try to attach bounding boxes if exact token matches exist
    if ref:
        box = _find_token_box(tokens, ref)
        if box:
            result["boxes"]["reference_number"] = box
    if txn:
        box = _find_token_box(tokens, txn)
        if box:
            result["boxes"]["transaction_id"] = box
    if inv:
        box = _find_token_box(tokens, inv)
        if box:
            result["boxes"]["invoice_number"] = box
    if amt:
        box = _find_token_box(tokens, amt)
        if box:
            result["boxes"]["amount"] = box
    if dt:
        box = _find_token_box(tokens, dt)
        if box:
            result["boxes"]["date"] = box

    # Box for unified transaction_number
    if result["transaction_number"]:
        box = _find_token_box(tokens, result["transaction_number"])
        if box:
            result["boxes"]["transaction_number"] = box

    # Box for DuitNow reference
    if dn_ref:
        box = _find_token_box(tokens, dn_ref)
        if box:
            result["boxes"]["duitnow_reference_number"] = box

    return result