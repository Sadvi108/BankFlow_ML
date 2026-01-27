import re
import logging
from typing import Dict, List, Tuple, Optional
from .ultimate_patterns_v3 import ultimate_matcher_v3

# Set up logger
logger = logging.getLogger(__name__)

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
    try:
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

        # --- Step 1: Ultimate Pattern Matcher V3 (Primary) ---
        # Normalize text
        norm_text = ultimate_matcher_v3.normalize_text(text)
        
        # Determine bank: Prefer bank_hint if provided, else detect
        effective_bank = bank_hint if bank_hint and bank_hint.lower() != "unknown" else ultimate_matcher_v3.detect_bank(norm_text)
        
        # Extract using V3
        v3_ids = ultimate_matcher_v3.extract_transaction_ids(norm_text, effective_bank)
        v3_date = ultimate_matcher_v3.extract_date(norm_text)
        v3_amount = ultimate_matcher_v3.extract_amount(norm_text)
        
        # Populate result from V3
        if v3_ids:
            # V3 returns sorted list. Best one goes to main fields.
            primary_id = v3_ids[0]
            result["reference_number"] = primary_id
            result["transaction_id"] = primary_id
            result["transaction_number"] = primary_id
            result["meta"]["transaction_number_source"] = "ultimate_v3"
            result["meta"]["all_ids"] = v3_ids
            
            # Check for DuitNow specifically in the list or patterns
            # If any ID looks like DuitNow format, assign it
            # (For now, we just rely on primary_id)

        if v3_date:
            result["date"] = v3_date
            
        if v3_amount:
            result["amount"] = v3_amount

        # --- Step 2: Legacy / Specific Field Fallbacks ---
        
        # DuitNow specific pattern (legacy was good for this)
        dn_ref = _match_first(DUITNOW_REF_PATTERNS, text)
        if not dn_ref:
            dn_ref = _extract_duitnow_by_tokens(tokens)
        
        if dn_ref:
            result["duitnow_reference_number"] = dn_ref
            # If V3 didn't find anything, use this as main ID
            if not result["transaction_number"]:
                result["transaction_number"] = dn_ref
                result["reference_number"] = dn_ref
                result["meta"]["transaction_number_source"] = "duitnow_legacy"

        # Invoice Number (V3 doesn't explicitly separate Invoice No from other IDs)
        inv = _match_first(INVOICE_PATTERNS, text)
        if inv:
            result["invoice_number"] = inv
            
        # --- Step 3: Bounding Boxes ---
        # Try to find boxes for extracted values
        
        def try_find_box(value, key):
            if not value:
                return
            # Try exact match first
            box = _find_token_box(tokens, value)
            if box:
                result["boxes"][key] = box
                return
            
            # Try removing spaces from tokens to match value (since V3 removes spaces)
            # This is a simple heuristic: if value is "123456", and token is "123 456", we might not find it easily 
            # without complex logic. For now, we stick to simple lookup.
            
            # Try matching parts if value is long
            if len(value) > 10:
                 # partial match?
                 pass

        try_find_box(result["reference_number"], "reference_number")
        try_find_box(result["transaction_id"], "transaction_id")
        try_find_box(result["transaction_number"], "transaction_number")
        try_find_box(result["duitnow_reference_number"], "duitnow_reference_number")
        try_find_box(result["invoice_number"], "invoice_number")
        try_find_box(result["amount"], "amount")
        try_find_box(result["date"], "date")

        return result
    except Exception as e:
        logger.error(f"Error in extract_fields: {e}", exc_info=True)
        # Return partial result or re-raise?
        # If we re-raise, the API returns 400.
        # If we return partial, the user gets what we have.
        # Let's try to return what we have if 'result' exists, otherwise re-raise.
        if 'result' in locals():
            result['error'] = str(e)
            return result
        raise e