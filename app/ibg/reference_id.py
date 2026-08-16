"""IBG reference extraction -- every reference on the receipt, each with its role.

The bug this module exists to fix: the legacy engine ranks candidate values by
the *shape* of the token (length, alphanumeric mix, whether it "looks like" an
id) rather than by *which label introduced it*. That lets an account number, or
the payer's own free-text "Recipient reference", outscore the bank's actual
reference. The rule here is the opposite -- a value is only ever a candidate if
a known label introduced it, and the label decides the value's role.

A receipt carries several references and they are not interchangeable:

    Reference No.          1234567890123456   <- the bank's own record
    Service Reference No.  654321             <- the bank's internal clearing ref
    Recipient Reference    INV - ABC1234567   <- free text the payer typed

Only a `bank_primary` may be promoted to the scalar `reference_id` that the rest
of the system consumes. A payer-supplied value is never promoted, however
official it looks: that promotion is exactly how the pipeline came to report
someone's own invoice number as the bank's transaction id.

When no bank-issued label is present, `extract_reference_id` returns
`FieldResult.missing()`. A None the review queue catches beats a confident wrong
answer written silently to the ledger.

See `app/ibg/contract.py` for the `Reference` and `FieldResult` contracts.
"""
import re

from app.ibg.contract import (
    FieldResult,
    Reference,
    ROLE_BANK_PRIMARY,
    ROLE_BANK_SECONDARY,
    ROLE_PAYER_SUPPLIED,
    primary_of,
)

# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------
# Order matters only for readability; overlapping matches are resolved by
# preferring the LONGEST label at a given position, so "Service Reference No"
# always beats the bare "Reference No" nested inside it.
#
# Each entry: (pattern, role, canonical label used in the Reference.label field)

_P = ROLE_BANK_PRIMARY
_S = ROLE_BANK_SECONDARY
_C = ROLE_PAYER_SUPPLIED

_LABEL_SPECS = [
    # --- payer-supplied ---------------------------------------------------
    # Declared first purely so the reader meets the disqualifying ones first;
    # longest-match resolution is what actually protects them.
    # RHB prints "Recipient`s Reference" with a BACKTICK, not an apostrophe.
    # Matching only ' silently dropped every RHB recipient reference.
    (r"Recipient\s*(?:['’`´]s)?\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _C, "Recipient Reference"),
    (r"Recipient\s*Ref\.?\s*/\s*Customer\s*Ref\.?\s*No\.?", _C, "Recipient Ref./Customer Ref. No."),
    (r"Customer\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _C, "Customer Ref"),
    (r"Your\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _C, "Your Reference No"),
    (r"End\s*to\s*End\s*ID(?:\s*\([^)]*\))?", _C, "End to End ID"),
    (r"2nd\s*party\s*reference", _C, "2nd party reference"),
    (r"Other\s*Payment\s*Details?(?:\s*/\s*Advice)?", _C, "Other Payment Details"),
    # Hong Leong advices print a bare "Other Details", not "Other Payment
    # Details" -- without this the trailing reference on every HL slip is lost.
    (r"Other\s*Details?", _C, "Other Details"),
    (r"Payment\s*Details?", _C, "Payment Details"),
    (r"Debit\s*Reference", _C, "Debit Reference"),
    (r"Debit\s*Description", _C, "Debit Description"),
    # "Favourite Name" is the nickname on a saved payee ("DnD CONTROL"), not a
    # reference. Treating it as one surfaced the beneficiary's own name, and on
    # Public Bank slips the beneficiary BANK, as if they were references.
    (r"Remark", _C, "Remark"),

    # --- bank secondary ---------------------------------------------------
    (r"Service\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _S, "Service Reference No"),
    (r"PayNet\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _S, "PayNet Reference No"),
    (r"DuitNow\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _S, "DuitNow Reference No"),
    (r"Channel\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _S, "Channel Reference No"),
    (r"Batch\s*Ref(?:erence)?\.?\s*(?:Number|No\.?)?", _S, "Batch Reference"),
    (r"Import\s*Ref(?:erence)?\.?", _S, "Import Reference"),
    (r"Back\s*Office\s*Ref(?:erence)?\.?", _S, "Back Office Reference"),
    (r"Instruction\s*ref(?:erence)?\.?", _S, "Instruction reference"),
    (r"UTR\s*Ref(?:erence)?\.?", _S, "UTR Reference"),
    (r"(?:Tracking\s*Number\s*\(\s*UETR\s*\)|UETR(?:\s*Number)?)", _S, "UETR"),
    (r"RPP\s*Business\s*Message\s*ID", _S, "RPP Business Message ID"),
    (r"RPP\s*Simplified\s*ID", _S, "RPP Simplified ID"),
    # Without this, "RPP Reference Number" fell through to the generic
    # "Reference Number" pattern and was elected primary over UOB's own
    # "Bank Reference" -- swapping the two on every UOB DuitNow slip.
    (r"RPP\s*Ref(?:erence)?\.?\s*(?:Number|No\.?)?", _S, "RPP Reference Number"),
    (r"Entry\s*No\.?", _S, "Entry No."),

    # --- bank primary -----------------------------------------------------
    (r"Bank\s*Payment\s*Ref(?:erence)?\.?", _P, "Bank Payment Reference"),
    (r"Transaction\s*Ref(?:erence)?\.?\s*(?:Number|No\.?)?", _P, "Transaction Reference No"),
    (r"Advice\s*Ref(?:erence)?\.?\s*(?:no\.?|No\.?)?", _P, "Advice Reference no"),
    (r"OCBC\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _P, "OCBC Reference No"),
    (r"iGTB\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _P, "iGTB Reference"),
    (r"Txn\s*Ref\.?\s*No\.?", _P, "Txn Ref No"),
    (r"SCB\s*Ref\.?", _P, "SCB Ref"),
    (r"Group\s*No\.?", _P, "Group No."),
    # MUFG's GIRO advice letter uses neither "Reference No" nor any label the
    # vocabulary knew, so the file returned no reference at all and the portal
    # reported it as unreadable.
    (r"MUFG\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _P, "MUFG Reference"),
    (r"Payment\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _P, "Payment Reference"),
    # "Bank Reference No." is demoted to secondary later if a primary exists.
    (r"Bank\s*Ref(?:erence)?\.?\s*(?:No\.?)?", _P, "Bank Reference"),
    (r"Ref(?:erence)?\.?\s*(?:Number|No\.?|ID)", _P, "Reference No"),
]

# Every `\s*` inside a label is narrowed to horizontal whitespace before
# compiling. A label never wraps a line on these receipts, and letting `\s*`
# cross a newline glues unrelated lines together: on the AmBank advice it
# matched "Transfer **Advice**\n**Reference** Number", swallowing the line break
# and leaving "Number:" stranded inside the value.
_COMPILED = [
    (re.compile(pattern.replace(r"\s*", r"[ \t]*"), re.IGNORECASE), role, name)
    for pattern, role, name in _LABEL_SPECS
]

# A label is only a label when it is followed by a separator or a line break --
# otherwise "Reference" inside a sentence would open a field.
_SEPARATOR_RE = re.compile(r"^[ \t]*[:\-]?[ \t]*")

# Values we never accept, whatever label introduced them: placeholder dashes and
# masked account fragments. Receipts print both routinely.
_PLACEHOLDER_RE = re.compile(r"^[-–—.\s]*$")
_MASKED_RE = re.compile(r"[*Xx]{3,}")
# Portal footers sit right below the value block, so an over-running block
# pairing happily elected "https://www.alliancebizsmart.com.my/business/" as
# the bank's primary reference.
_URL_RE = re.compile(r"(?:https?://|www\.)|\.(?:com|my|net|org)\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")

_DATE_SHAPE_RES = (
    re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$"),
    re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$"),
    re.compile(r"^\d{1,2}[\s-][A-Za-z]{3,9}[\s-]\d{2,4}$"),
)

# Words that are never a reference value on their own -- they are status text or
# the next label leaking into the value slot.
_NON_VALUE_WORDS = frozenset([
    "SUCCESSFUL", "SUCCESS", "PROCESSED", "PENDING", "FAILED", "COMPLETED",
    "APPROVED", "RESIDENT", "YES", "NO", "NONE", "NIL", "MYR", "RM",
    "FUND TRANSFER", "FUNDS TRANSFER", "CURRENT ACCOUNT", "ACCOUNT NUMBER",
    "INTERNET PORTAL", "NEW TRANSFER", "DUITNOW TRANSFER", "OUTWARD ACH",
    # Product and rail names. A block-paired run that starts one slot early
    # elects the product type as the bank's reference -- "Interbank GIRO (IBG)"
    # was being returned as a bank_primary on Public Bank approval slips.
    "INTERBANK GIRO IBG", "INTERBANK GIRO", "DUITNOW", "GIRO", "IBG",
    "DUITNOW PAYMENT", "DUITNOW TRANSFER", "INSTANT TRANSFER", "INWARD ACH",
    "IMMEDIATE TRANSFER", "POST DATED TRANSFER", "FAVOURITE ACCOUNT TRANSFER",
    "THIRD PARTY TRANSFER", "ACH CREDIT GIRO", "PAYLINK GIRO",
    "AUTOMATED CLEARING HOUSE ACH", "ACCOUNT NO DUITNOW ID",
    # Section headers and form furniture that a scan can run into.
    "PAYMENT METHOD", "PAYMENT TYPE", "PAYMENT DETAILS", "SEARCH CRITERIA",
    "STATUS SUCCESS", "STATUS SUCCESSFUL", "STATUS PROCESSED", "SUCCESS",
    "DATE TIME", "TRANSFER TYPE", "TRANSFER MODE", "PRODUCT TYPE",
    "TRANSACTION DETAILS", "BENEFICIARY DETAILS", "PAYER DETAILS",
    "ADDITIONAL INFORMATION", "INSTRUCTION MODE", "APPROVAL STATUS",
])


def _normalise_phrase(value):
    """Uppercase, strip punctuation, collapse spaces -- for the noise check."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]+", " ", value)).strip().upper()

# Amount-shaped strings must never become references.
_MONEY_RE = re.compile(r"^(?:RM|MYR)?\s*[\d,]+\.\d{2}$", re.IGNORECASE)
# A value that *contains* a currency amount is a money field bleeding into the
# reference slot ("Debit Amount MYR 241.20"), never a reference.
_CONTAINS_MONEY_RE = re.compile(r"\b(?:RM|MYR)\s*[\d,]+\.\d{2}", re.IGNORECASE)
# Prose and section headers leak in when a label has no value of its own:
# "Please refer to the payment details below:" yields "below". A real reference
# either carries a digit or is an uppercase code (CPC, DEMURAGE).
_SINGLE_WORD_RE = re.compile(r"^[A-Za-z()]+$")
# Trailing form furniture that is part of the label, not the value.
_TRAILING_NOISE_RE = re.compile(r"^\(?\s*(?:optional|if\s+any|s)\s*\)?$",
                                re.IGNORECASE)

# Field labels that are not references but appear inside the same label blocks,
# so positional pairing must not mistake them for values.
_GENERIC_LABEL_RE = re.compile(
    r"^(?:Product\s*Type|Approval\s*Status|Transaction\s*Status(?:\s*Reason)?|"
    r"Transfer\s*(?:Type|Mode)|Payment\s*(?:Type|Date|Mode)|Value\s*Date|"
    r"From\s*Account|To\s*Account|Account\s*(?:No\.?|Number|Name)|"
    r"Beneficiary\s*(?:Bank|Name|Account.*)?|Recipient\s*Bank|Payee\s*(?:Name|Bank.*)?|"
    r"Amount(?:\s*\(MYR\))?|Fee|Total.*|Service\s*Charge.*|Status|Currency|"
    r"Transaction\s*Date(?:,\s*Time)?|Date\s*and\s*Time|Channel|"
    r"Recipient's\s*DuitNow\s*ID(?:\s*Type)?|Debit\s*Account|Credit\s*Account.*|"
    # Maybank M2E form labels. Their own values are printed as "-", so the
    # block pairing walked past the dash and adopted the NEXT label as a
    # reference -- surfacing "Purpose Of Transfer" as a payer reference.
    r"Charges\s*Borne\s*By.*|Category|Purpose\s*(?:Of\s*Transfer|Code.*)?|"
    r"Applicant(?:\s*(?:ID|Address|Details).*)?|Reason|User(?:\s*Activities)?|"
    r"Activities|Advice\s*(?:Detail|Date)|Instruction\s*(?:to\s*Bank|Mode)|"
    r"Send\s*Payment\s*Advice|Business\s*Reg\.?\s*No\.?|Zip/Postal\s*Code|"
    r"State/Province|Prefecture|City/District|Country|Remarks?|"
    r"Source\s*Of\s*Fund|Ultimate\s*(?:Debtor|Creditor)\s*Name|"
    r"Beneficiary\s*(?:Citizenship|Category|Bank\s*Branch)|Save\s*as\s*\w+|"
    r"Payee/?Beneficiary\s*Details|Transaction\s*Type|Resident|Maker|Channel)$",
    re.IGNORECASE)

# Same vocabulary, searched anywhere in a digit-free candidate rather than
# anchored, to catch runs of concatenated column headers.
_GENERIC_LABEL_SEARCH_RE = re.compile(
    r"\b(?:Transaction\s+(?:Status|Date|Type)|Approval\s+Status|Value\s+Date|"
    r"Payment\s+(?:Type|Date|Mode)|Transfer\s+(?:Type|Mode)|Product\s+Type|"
    r"Account\s+(?:No\.?|Number|Name)|Beneficiary\s+\w+|Recipient\s+Bank|"
    r"Service\s+Charge|Total\s+\w+)\b",
    re.IGNORECASE)

# Labels this module does not model as references but which still occupy a slot
# in a label-block-then-value-block layout. Anchored to line starts so a phrase
# inside a value never counts as a label.
_GENERIC_LABEL_RE_SCAN = re.compile(
    r"(?m)^[ \t]*(?:Product\s+Type|Approval\s+Status|Transaction\s+Status|"
    r"From\s+Account|To\s+Account|Transfer\s+(?:Mode|Type)|Favourite\s+Name|"
    r"Beneficiary\s+(?:Bank|Name|Account\s+No\.?)|Recipient's\s+DuitNow\s+ID"
    r"(?:/Account\s+No\.?|\s+Type)?|Recipient\s+Bank|Payee\s+Name|"
    r"Payment\s+(?:Type|Date|Mode)|Value\s+Date|Amount(?:\s*\(MYR\))?|Fee|"
    r"Total\s+\w+[\w\s]*|SMS\s+Fee|Service\s+Charge[\w\s()%-]*|Status|"
    r"Currency|Channel|Debit\s+Account|Credit\s+Account|Account\s+Number|"
    r"Resident\w*|Transaction\s+Date(?:,\s*Time)?|Fund\s+Transfer\s+Purpose)"
    r"[ \t]*:?[ \t]*$",
    re.IGNORECASE)

_MAX_VALUE_LEN = 60
_BLOCK_LOOKAHEAD = 400

# Sentinel: this label's value is present on the receipt and is explicitly
# blank ("-"). Distinct from None, which means "found nothing, keep looking".
_EMPTY = object()


class _Hit(object):
    """One label occurrence in the text."""

    __slots__ = ("start", "end", "role", "name")

    def __init__(self, start, end, role, name):
        self.start = start
        self.end = end
        self.role = role
        self.name = name


_COLON_VALUE_RE = re.compile(r"^\s*:\s*\S")


def _reflow_vertical_labels(text):
    """Rejoin labels that a narrow email column broke across several lines.

    Maybank2E payment advices render as:

        Advice
        Reference
        no
         : MYIG...

    so no label pattern matches anything. Where a line begins with a colon, the
    preceding run of colon-free lines is that label, split by the column width.
    Join them back onto the colon line and the ordinary matcher works.
    """
    lines = text.split("\n")
    out = []
    pending = []
    for line in lines:
        if _COLON_VALUE_RE.match(line) and pending:
            out.append(" ".join(p.strip() for p in pending) + " " + line.strip())
            pending = []
            continue
        if line.strip() and ":" not in line and len(line.strip()) <= 24:
            pending.append(line)
            if len(pending) > 4:          # not a wrapped label, just short lines
                out.extend(pending[:-4])
                pending = pending[-4:]
            continue
        out.extend(pending)
        pending = []
        out.append(line)
    out.extend(pending)
    return "\n".join(out)


def _find_labels(text):
    """Every label occurrence, de-overlapped by preferring the longest match."""
    hits = []
    for regex, role, name in _COMPILED:
        for m in regex.finditer(text):
            # Several patterns end in an optional "No."/"ID" preceded by \s*,
            # which greedily swallows the newline between a label and the value
            # printed beneath it. Wind the end back off any trailing whitespace
            # so the label never consumes its own line break.
            end = m.end()
            while end > m.start() and text[end - 1] in " \t\r\n":
                end -= 1
            # Require a separator or end-of-line right after the label, so a
            # label word inside prose is not treated as opening a field.
            # Bank of China marks mandatory fields with a trailing asterisk
            # ("Recipient's Reference*"), which otherwise fails this check.
            nxt = text[end:end + 1]
            if nxt not in ("", ":", "-", "\n", "\r", " ", "\t", ".", "*"):
                continue

            matched = text[m.start():end]
            # Prose, not a field label. "This Advice is for your reference only
            # and is not to be relied upon" matched "your reference" and bound
            # the next few words as the payer's reference. No bank prints a
            # field label in lowercase, so casing separates the two cleanly.
            if matched.islower():
                continue
            # A label mid-sentence is prose too: a real one begins its line or
            # its column, or follows a separator.
            line_start = text.rfind("\n", 0, m.start()) + 1
            before = text[line_start:m.start()]
            if before.strip() and not re.search(r"[:\-|]\s*$|\s{2,}$", before):
                # Allow a label that follows another label's value on the same
                # line (two-column layouts), but not one trailing prose.
                tail_word = re.search(r"([A-Za-z']+)\s*$", before)
                if tail_word and tail_word.group(1).islower():
                    continue

            hits.append(_Hit(m.start(), end, role, name))

    hits.sort(key=lambda h: (h.start, -(h.end - h.start)))

    kept = []
    for hit in hits:
        # Drop a hit that starts inside a hit we already kept, or that merely
        # extends one we kept from the same start.
        if kept and hit.start < kept[-1].end:
            continue
        kept.append(hit)
    return kept


def _matches_any_label(value):
    """True when `value` is itself a field label rather than a field's value."""
    probe = value.strip().rstrip(":").strip()
    for regex, _role, _name in _COMPILED:
        m = regex.match(probe)
        if m and m.end() >= len(probe):
            return True
    return bool(_GENERIC_LABEL_RE.match(probe))


def _clean_value(raw):
    """Trim a raw slice down to a usable reference value, or None."""
    if raw is None:
        return None
    value = raw.strip()
    # Cut at two-or-more spaces: on a two-column layout the next column's text
    # runs on after the value.
    value = re.split(r"\s{2,}", value)[0].strip()
    value = value.strip(" \t\r\n:;|")
    # A row in a multi-column table carries the reference, then other columns:
    # "ABC1234567890 - 01/02/2030" is reference, status-reason, date. Cut at
    # the first date-shaped token after the first, which ends the reference.
    # Payer references legitimately contain spaces and dashes
    # ("INV - ABC1234567", "ref 111111 222222"), so only a date cuts.
    tokens = value.split()
    for index in range(1, len(tokens)):
        if any(pat.match(tokens[index]) for pat in _DATE_SHAPE_RES):
            value = " ".join(tokens[:index]).strip(" -–—")
            break
    if not value or len(value) > _MAX_VALUE_LEN:
        return None
    if _PLACEHOLDER_RE.match(value) or _MASKED_RE.search(value):
        return None
    if _URL_RE.search(value) or _EMAIL_RE.search(value):
        return None
    if value.upper() in _NON_VALUE_WORDS:
        return None
    if _normalise_phrase(value) in _NON_VALUE_WORDS:
        return None
    if _MONEY_RE.match(value) or _CONTAINS_MONEY_RE.search(value):
        return None
    if _TRAILING_NOISE_RE.match(value):
        return None
    # A single alphabetic word with no digits is prose or a form label unless
    # it is an uppercase code. Kills "below", "Report", "Payment", "Citizen"
    # while keeping "CPC" and "DEMURAGE".
    if _SINGLE_WORD_RE.match(value) and not value.isupper():
        return None
    # A value that is itself one of our labels means the scan ran past this
    # field into the next one ("Payment Details" -> "Payment Method").
    if _matches_any_label(value):
        return None
    if any(pat.match(value) for pat in _DATE_SHAPE_RES):
        return None
    if not any(c.isalnum() for c in value):
        return None
    if _is_label_text(value):
        return None
    return value


def _is_label_text(value):
    """True when the candidate is itself a field label.

    On a label-block-then-value-block layout the positional pairing would
    otherwise happily adopt the next label in the block as a value, reporting
    "Approval Status" or "Payment Date" as a reference number.
    """
    stripped = value.strip().rstrip(":.").strip()
    for regex, _role, _name in _COMPILED:
        m = regex.match(stripped)
        if m and m.end() == len(stripped):
            return True
    if _GENERIC_LABEL_RE.match(stripped):
        return True
    # A digit-free string that still contains a label phrase is a run of column
    # headers, not a value -- e.g. the AmBank slip's
    # "Transaction Status Reason Transaction Date". Payer-supplied references
    # are often digit-free too (a bare invoice or payment note), so the
    # label phrase, not the absence of digits, is what disqualifies it.
    if not any(c.isdigit() for c in stripped):
        for regex, _role, _name in _COMPILED:
            m = regex.search(stripped)
            if m and (m.end() - m.start()) >= 8:
                return True
        if _GENERIC_LABEL_SEARCH_RE.search(stripped):
            return True
    return False


_STRONG_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/]{5,29}")


def _first_reference_token(span):
    """First token in `span` that can only be a reference, or None.

    Deliberately strict: it must carry a digit and be at least six characters,
    so ordinary words in a header run can never qualify. Used only after the
    plain rest-of-line read has already been rejected as column headers.
    """
    for m in _STRONG_TOKEN_RE.finditer(span):
        token = m.group(0).strip(" \t\r\n.,;:()[]{}'\"")
        if len(token) < 6 or not any(c.isdigit() for c in token):
            continue
        if any(pat.match(token) for pat in _DATE_SHAPE_RES):
            continue
        if _MONEY_RE.match(token) or _MASKED_RE.search(token):
            continue
        if _is_label_text(token):
            continue
        return token
    return None


def _value_after(text, hit, next_start):
    """Value for `hit`, taken from the rest of its line or the following line.

    Never reads past `next_start` (where the following label begins), so one
    field can never adopt another field's value.
    """
    sep = _SEPARATOR_RE.match(text, hit.end)
    cursor = sep.end() if sep else hit.end
    limit = min(next_start, len(text))

    # Rest of the label's own line.
    line_end = text.find("\n", cursor)
    if line_end == -1:
        line_end = len(text)

    # A field printed as "-" is explicitly empty, not merely unread. Say so, so
    # the caller can stop looking instead of scanning on and adopting the next
    # field's label as this field's value.
    same_line_raw = text[cursor:min(line_end, limit)].strip()
    if _PLACEHOLDER_RE.match(same_line_raw) and same_line_raw:
        return _EMPTY, cursor
    if not same_line_raw:
        nxt_end = text.find("\n", line_end + 1)
        nxt_end = len(text) if nxt_end == -1 else nxt_end
        following = text[line_end + 1:min(nxt_end, limit)].strip()
        if following and _PLACEHOLDER_RE.match(following):
            return _EMPTY, cursor

    if cursor < limit:
        same_line = text[cursor:min(line_end, limit)]
        value = _clean_value(same_line)
        if value:
            return value, cursor
        # The rest of the line was a run of other column headers, which happens
        # when a whole table renders onto one line: the label block runs first
        # and the value block follows it. Skip past the headers to the first
        # genuinely reference-shaped token still on that line.
        token = _first_reference_token(same_line)
        if token:
            return token, cursor

    # Otherwise the next non-empty line, provided it starts before the next label.
    pos = line_end + 1
    while pos < limit:
        nxt = text.find("\n", pos)
        if nxt == -1:
            nxt = len(text)
        candidate = _clean_value(text[pos:min(nxt, limit)])
        if candidate:
            return candidate, pos
        if text[pos:nxt].strip():
            break  # a non-empty line that is not a value: stop, don't skip past it
        pos = nxt + 1
    return None, cursor


def _slot_positions(text, hits, run, block_start):
    """Index of each run member within ALL labels in the label block.

    Returns {hit_index: slot}. The label block spans from the first run member
    to `block_start`; every label in it -- ours and the ones we do not model --
    consumes one slot in the value block that follows.
    """
    span_start = hits[run[0]].start
    span = text[span_start:block_start]
    positions = set()
    for regex, _role, _name in _COMPILED:
        for m in regex.finditer(span):
            positions.add(span_start + m.start())
    for m in _GENERIC_LABEL_RE_SCAN.finditer(span):
        positions.add(span_start + m.start())

    ordered = sorted(positions)
    slot_of = {}
    for hit_index in run:
        start = hits[hit_index].start
        # Count distinct labels that begin before this one.
        slot_of[hit_index] = sum(1 for p in ordered if p < start)
    return slot_of


def _pair_blocks(text, hits, resolved):
    """Positionally pair a run of consecutive value-less labels with the values
    that follow the run.

    Several corporate portals render as one block of labels followed by one
    block of values, in matching order -- Alliance BizSmart and the Public Bank
    approval slip both do it. Adjacency is meaningless there; index is the only
    signal.
    """
    i = 0
    while i < len(hits):
        if resolved.get(i) is not None:
            i += 1
            continue
        run = []
        j = i
        while j < len(hits) and resolved.get(j) is None:
            run.append(j)
            j += 1
        # A run that begins with an unresolved label absorbs the labels that
        # follow it on the same line. In a label-block-then-value-block layout
        # the first value belongs to the FIRST label, but a plain forward scan
        # hands it to whichever label happens to sit closest to it -- on the
        # OCR'd Maybank receipt that gave the bank's reference to
        # "Payment Details". Pool the run's values and reassign them in order.
        if run and j < len(hits):
            line_end = text.find("\n", hits[run[-1]].end)
            if line_end == -1:
                line_end = len(text)
            while j < len(hits) and hits[j].start < line_end:
                run.append(j)
                j += 1
        if len(run) >= 2:
            pooled = [resolved[k][0] for k in run if resolved.get(k)]
            for k in run:
                resolved[k] = None
            block_start = hits[run[-1]].end
            # The label block also contains labels this module does not model
            # (Product Type, Approval Status, From Account, ...). Their values
            # occupy slots in the value block all the same, so pairing only on
            # the reference labels shifts every assignment and hands the first
            # value -- the product type -- to the reference field. Count the
            # other labels too, and keep only the slots that land on ours.
            slot_of = _slot_positions(text, hits, run, block_start)
            block_end = min(len(text), block_start + _BLOCK_LOOKAHEAD)
            block = text[block_start:block_end]
            values = []
            for line in block.split("\n"):
                cleaned = _clean_value(line)
                if cleaned:
                    values.append(cleaned)
            if len(values) < len(run):
                # An OCR'd receipt can render an entire table onto a single
                # line, so there are no line breaks to pair against. Fall back
                # to the strong-token scan, which only accepts tokens that
                # cannot be anything but a reference.
                tokens = []
                pos = 0
                while pos < len(block) and len(tokens) < len(run):
                    token = _first_reference_token(block[pos:])
                    if not token:
                        break
                    tokens.append(token)
                    pos = block.index(token, pos) + len(token)
                if len(tokens) > len(values):
                    values = tokens
            # Values the run's own labels had already claimed come first: they
            # sit earlier in the document than anything after the last label.
            for value in values:
                if value not in pooled:
                    pooled.append(value)
            values = pooled
            for hit_index in run:
                offset = slot_of.get(hit_index, run.index(hit_index))
                if offset < len(values):
                    resolved[hit_index] = (values[offset], block_start)
        i = j
    return resolved


# How specific a primary label is. When a receipt carries two primary-ish
# labels, the more specific one is the bank's transaction reference and the
# other is a clearing artefact. `primary_of` elects on confidence, so this
# ordering is what actually decides.
_PRIMARY_SPECIFICITY = {
    "Transaction Reference No": 0.99,
    "Txn Ref No": 0.99,
    "Bank Payment Reference": 0.98,
    "OCBC Reference No": 0.98,
    "iGTB Reference": 0.97,
    "Advice Reference no": 0.97,
    "SCB Ref": 0.97,
    "Group No.": 0.96,
    "Bank Reference": 0.96,
    "Reference No": 0.95,
}


def _confidence_for(role, adjacent, ocr_used, label=None):
    if role == ROLE_BANK_PRIMARY:
        base = _PRIMARY_SPECIFICITY.get(label, 0.95)
    elif role == ROLE_BANK_SECONDARY:
        base = 0.9
    else:
        base = 0.85
    if not adjacent:
        base -= 0.1
    if ocr_used:
        base -= 0.05
    return round(max(0.55, min(0.99, base)), 3)


def _slug(name):
    return "label:" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def extract_references(text, ocr_used=True):
    """Every reference on the receipt, each carrying its label and role."""
    if not text or not text.strip():
        return []

    text = _reflow_vertical_labels(text)
    hits = _find_labels(text)
    if not hits:
        return []

    resolved = {}
    for index, hit in enumerate(hits):
        next_start = hits[index + 1].start if index + 1 < len(hits) else len(text)
        value, pos = _value_after(text, hit, next_start)
        if value is _EMPTY:
            resolved[index] = _EMPTY      # explicitly blank; not a block gap
        else:
            resolved[index] = (value, pos) if value else None

    _pair_blocks(text, hits, resolved)

    references = []
    seen = set()
    # Labels that name the bank's transaction reference on their own, but are a
    # clearing/system artefact whenever the receipt also prints a more specific
    # one. Bank of China shows both "Txn Ref No." and "iGTB Ref No."; the iGTB
    # value is the portal's handle, not the transaction's.
    _DEMOTABLE = ("Bank Reference", "iGTB Reference")
    has_primary = any(
        hits[i].role == ROLE_BANK_PRIMARY and resolved.get(i)
        for i in range(len(hits))
        if hits[i].name not in _DEMOTABLE
    )

    for index, hit in enumerate(hits):
        entry = resolved.get(index)
        if not entry or entry is _EMPTY:
            continue
        value, pos = entry
        role = hit.role
        # A generic "Bank Reference No." alongside a more specific primary is
        # the clearing reference, not the transaction reference.
        if hit.name in _DEMOTABLE and has_primary:
            role = ROLE_BANK_SECONDARY
        # Dedupe on the VALUE alone, not (value, role). A receipt that prints
        # its reference block twice -- once in the page header, once in the body
        # table -- otherwise yields the same number under two different roles,
        # and the block pairing shifts the second copy onto the wrong label.
        # First occurrence wins: the header is printed adjacent to its label,
        # so it is the better-evidenced one.
        key = value.upper()
        if key in seen:
            continue
        seen.add(key)
        adjacent = pos - hit.end < 40
        references.append(Reference(
            value=value,
            label=hit.name,
            role=role,
            confidence=_confidence_for(role, adjacent, ocr_used, hit.name),
            source=_slug(hit.name),
        ))

    # A clearing-network reference (PayNet, DuitNow, RPP) is secondary when the
    # bank also prints its own transaction reference -- but on portals that
    # print only the clearing reference, that IS the bank's record of the
    # transfer, and refusing to promote it would leave the receipt with no
    # reference at all. Promote the strongest bank-issued value in that case.
    if references and not any(r.role == ROLE_BANK_PRIMARY for r in references):
        secondaries = [r for r in references if r.role == ROLE_BANK_SECONDARY]
        if secondaries:
            best = max(secondaries, key=lambda r: r.confidence)
            best.role = ROLE_BANK_PRIMARY
            best.source = best.source + "+promoted_sole_bank_reference"

    return references


def extract_reference_id(text, ocr_used=True):
    """The single bank-issued primary reference, for the scalar `reference_id`.

    Returns `FieldResult.missing()` when the receipt carries no bank-issued
    reference label. A payer-supplied reference is never promoted.
    """
    if not text or not text.strip():
        return FieldResult.missing(source="missing:empty_text")

    references = extract_references(text, ocr_used=ocr_used)
    if not references:
        return FieldResult.missing(source="missing:no_reference_label")

    best = primary_of(references)
    candidates = [(r.value, r.confidence) for r in references]
    if best is None:
        return FieldResult(
            value=None,
            confidence=0.0,
            source="missing:no_bank_primary",
            candidates=candidates,
        )
    return FieldResult(
        value=best.value,
        confidence=best.confidence,
        source=best.source,
        candidates=candidates,
    )
