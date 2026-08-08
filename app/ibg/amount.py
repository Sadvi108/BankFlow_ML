"""IBG money-family extraction: transaction amount, fee, total debit.

Three numbers live on an interbank-transfer receipt and they are not
interchangeable:

    amount       -- what was actually transferred. This is what reconciles
                    against an invoice, and is the primary field.
    fee          -- total fees/charges the bank took (service charge,
                    transaction fee, GST/SST, commission...). Real money,
                    but never the transaction amount.
    total_debit  -- amount + fee, i.e. what actually left the account.
                    Only printed on some receipts.

Ranked, highest first, for `amount`:
    1. Amount / Transaction Amount / Transfer Amount / Payment Amount /
       Remittance Amount / Debit Amount / Gross Amount / Net Amount /
       bare Total Amount. The answer.
    2. Fees -- Service Charge, Fee, Charges, GST, Tax Amount. Real money,
       but never the answer -- and often "0.00", which makes a wrong pick
       look plausible.
    3. Total-debit-only phrasings -- Total Debit Amount, Total Amount to
       Debit, Total Debit, Total Amount(MYR). These restate amount + fee;
       never returned as `amount` even when no separate fee line exists.
    4. Not money at all -- account numbers, postal codes, reference digits.
       Must never be returned.

Two layout traps drive the amount/fee/total_debit label-to-value pairing:

    - S1 (intervening line): the label and its value can be separated by
      another line, e.g. "Transaction Amount" / "In Transaction Currency" /
      "42.42". A pattern that only allows a single newline between label
      and value misses this, so the scan looks forward across a *character*
      window, not "the next line".
    - S2 (column scramble): labels arrive as one whole block and values as
      a second whole block immediately after, so the value for "Amount" can
      sit several lines below it, right after an unrelated label ("Payment
      Date"). This is *not* solved by a wider adjacency window (nothing
      recognizable sits directly under "Amount" -- another label does).
      Instead: when two or more recognized labels are directly "touching"
      (no money value between them -- see `_find_touching_runs`), that is
      itself the signal that a label column, not a label/value pair, was
      just read. The matching value column is then found later in the
      document as the same number of money tokens sitting immediately next
      to each other, and paired back to the labels positionally.

The amount frequently carries no RM/MYR prefix at all (both traps above),
so no rule here requires a currency symbol -- a currency prefix is only a
last-resort fallback, used only for `amount`, when no label text survived
at all (e.g. OCR mangled "Amount" beyond recognition but "MYR 77.25" is
still legible).

Money-shape validation (`_is_plausible_amount`, `_looks_like_account_fragment`)
is at least as strict as `UltimatePatternMatcherV3._is_plausible_amount` in
`app/ultimate_patterns_v3.py`: both cap the integer part at 9 digits. This
module additionally rejects a value whose digits are actually a fragment of
a longer digit run in the source text (an account/reference number that
happens to contain a decimal-shaped sub-run), which the legacy matcher does
not check.
"""
import bisect
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
import re
from typing import Dict, List, NamedTuple, Optional, Tuple

from app.ibg.contract import FieldResult

# ---------------------------------------------------------------------------
# Money shape
# ---------------------------------------------------------------------------

# MYR receipts always print sen as two digits, so a money-shaped token
# requires exactly two decimal digits -- this alone rejects account numbers,
# postal codes and reference numbers, none of which carry a literal ".NN".
# Not wrapped in \b...\b: currency codes are often glued directly to the
# digits ("RM42.42", "MYR31.80"), and between a letter and a digit there is
# no regex word boundary character class change worth relying on here;
# boundary correctness is instead handled explicitly by
# `_looks_like_account_fragment`.
_MONEY_RE = re.compile(r'\d{1,3}(?:,\d{3})+\.\d{2}|\d{1,9}\.\d{2}')

_STRICT_AMOUNT_RE = re.compile(
    r'^\d{1,3}(?:,\d{3})*\.\d{2}$|^\d{1,9}\.\d{2}$'
)

# ---------------------------------------------------------------------------
# Label families
# ---------------------------------------------------------------------------
#
# total_debit is matched first and its spans are carved out of every other
# family below it -- "Total Debit Amount" / "Total Amount to Debit" would
# otherwise also look like a primary "...Amount" label, and "Total Charges"
# would otherwise also look like a bare fee label with a different span.

_TOTAL_DEBIT_LABEL_RE = re.compile(
    r'\bTotal\s*Debit\s*Amount\b'
    r'|\bTotal\s*Amount\s*to\s*Debit\b'
    r'|\bTotal\s*Amount\s*\(\s*(?:MYR|RM)\s*\)'
    r'|\bTotal\s*Debit\b(?!\s*Amount)',
    re.IGNORECASE,
)

# "Total Fee Charges" / "Total Charges" are a restated grand total of the
# fee lines beneath them, not one more fee to add on top -- flagged
# separately (`_TOTAL_FEE_RE`) so extract_fee prefers that single number
# instead of summing every component fee into it a second time.
_FEE_LABEL_RE = re.compile(
    r'\b(?:Total\s*Fee\s*Charges?|Total\s*Charges?|SMS\s*Fees?|'
    r'Service\s*Tax\s*Charges?|Service\s*Charges?|'
    r'Fund\s*Transfer\s*Charges?|Processing\s*Fees?|Handling\s*Fees?|'
    r'Bank\s*Charges?|Transaction\s*Fees?|Transfer\s*Fees?|'
    r'Tax\s*Amount|GST\s*Amount|SST\s*Amount|Remittance\s*Commission|'
    r'Charges?|Fees?|GST|SST|Commission|Levy|Stamp\s*Duty)\b',
    re.IGNORECASE,
)

_TOTAL_FEE_RE = re.compile(
    r'\bTotal\s*(?:Fee\s*Charges?|Charges?)\b', re.IGNORECASE,
)

# Primary tier for `amount`. Bare "Total Amount" belongs here, not in the
# total-debit family above: on this corpus it *is* the answer whenever it
# appears (a "GIRO Payment Details Report" prints "Total Amount MYR - 752.60"
# and a Straight2Bank report prints "Total Amount MYR 241.20", both equal to
# the debit amount because no separate fee line exists on either receipt).
# The phrasings that mean something else was added on top of the transfer
# -- "Total Debit Amount", "...to Debit", "Total Amount(MYR)" -- are carved
# out by `_TOTAL_DEBIT_LABEL_RE` above and never reach this pattern.
_STRONG_LABEL_RE = re.compile(
    r'\b(?:Transaction\s*Amount|Transfer\s*Amount|Payment\s*Amount|'
    r'Remittance\s*Amount|Debit\s*Amount|Gross\s*Amount|Net\s*Amount|'
    r'Total\s*Amount)\b',
    re.IGNORECASE,
)

# Bare "Amount" on its own, e.g. "Amount RM 1,200.50". Occurrences that are
# really part of a fee/total-debit compound ("Tax Amount", "Total Debit
# Amount") are excluded by checking span containment before use.
_WEAK_LABEL_RE = re.compile(r'\bAmount\b', re.IGNORECASE)

# Last-resort fallback for `amount` only, when no label text survived at
# all (e.g. OCR mangles "Amount" into gibberish but "MYR 77.25" is still
# legible).
_CURRENCY_RE = re.compile(r'\b(?:RM|MYR)\b', re.IGNORECASE)

_LABEL_WINDOW = 300
_FEE_WINDOW = 150
_CURRENCY_WINDOW = 25

# Block-scramble tuning (the S2 trap: labels arrive as one block, values as
# a second block immediately after). A "touching" run is two or more
# recognized labels with nothing money-shaped between them -- see
# `_find_touching_runs`. Once such a run is found, the matching value run is
# searched for as that many money tokens sitting immediately next to each
# other, within a generous window (a whole block of unrelated reference
# numbers and free text routinely separates the label block from its value
# block on these receipts) but a tight gap between consecutive members of
# the value chain itself (a real value column has nothing but a newline or
# a currency code between its entries).
_RUN_GAP_MAX = 60
_RUN_CHAIN_WINDOW = 1000
_RUN_CHAIN_GAP_MAX = 25

_SCORE_BLOCK_RUN = 98.0
_SCORE_STRONG = 97.0
_SCORE_WEAK = 90.0
_SCORE_CURRENCY = 65.0
_SCORE_FEE_DIRECT = 90.0
_SCORE_FEE_BLOCK_RUN = 95.0
_SCORE_TOTAL_DEBIT_DIRECT = 90.0
_SCORE_TOTAL_DEBIT_BLOCK_RUN = 95.0

_OCR_CONFIDENCE_PENALTY = 0.93

KIND_STRONG = "strong"
KIND_WEAK = "weak"
KIND_FEE = "fee"
KIND_TOTAL_DEBIT = "total_debit"

FAMILY_AMOUNT = "amount"
FAMILY_FEE = "fee"
FAMILY_TOTAL_DEBIT = "total_debit"

_KIND_TO_FAMILY = {
    KIND_STRONG: FAMILY_AMOUNT,
    KIND_WEAK: FAMILY_AMOUNT,
    KIND_FEE: FAMILY_FEE,
    KIND_TOTAL_DEBIT: FAMILY_TOTAL_DEBIT,
}


class _Label(NamedTuple):
    start: int
    end: int
    kind: str


class _Candidate(NamedTuple):
    value: str          # formatted, e.g. "42.42"
    score: float
    source: str
    start: int


# ---------------------------------------------------------------------------
# Structural / contextual filters
# ---------------------------------------------------------------------------

def _is_plausible_amount(raw: str) -> bool:
    """Reject anything that cannot be a real amount/fee/total-debit value.

    At least as strict as `UltimatePatternMatcherV3._is_plausible_amount`:
    every candidate here already carries a literal two-digit decimal (by
    construction of `_MONEY_RE`), so the integer part is capped the same
    way -- above RM 999,999,999 it is a reference number wearing a comma,
    not an amount.
    """
    s = raw.strip()
    if not _STRICT_AMOUNT_RE.match(s):
        return False
    int_part = s.split('.')[0].replace(',', '')
    if len(int_part) > 9:
        return False
    return True


def _looks_like_account_fragment(text: str, start: int, end: int) -> bool:
    """True when the money-shaped match at text[start:end] is actually a
    fragment of a longer run of digits in the source text -- an account or
    reference number that happens to contain a sub-run shaped like an
    amount -- not a standalone amount.

    Digit groups joined by a single space, as in an OCR'd account number
    like "7788 3341 0099 2205", count as one run. This also catches a
    malformed three-decimal value such as "MYR -31.800": the regex match
    only ever consumes two decimal digits ("31.80"), but the extra trailing
    digit ("0") is still immediately adjacent, so it is walked and the
    match is correctly rejected as a fragment rather than accepted as a
    genuine "31.80".
    """
    n = len(text)

    left = start
    while left > 0:
        if text[left - 1].isdigit():
            left -= 1
            continue
        if text[left - 1] == ' ' and left - 2 >= 0 and text[left - 2].isdigit():
            left -= 2
            continue
        break

    right = end
    while right < n:
        if text[right].isdigit():
            right += 1
            continue
        if text[right] == ' ' and right + 1 < n and text[right + 1].isdigit():
            right += 2
            continue
        break

    return left < start or right > end


def _format_amount(raw: str) -> Optional[str]:
    """Normalize a matched money token to the contract's output format: a
    plain decimal string, exactly two places, no thousands separator, no
    currency. Money is never round-tripped through float here -- only
    `decimal.Decimal`.
    """
    cleaned = raw.strip().replace(',', '')
    try:
        d = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None
    d = d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return str(d)


def _span_inside_any(span: Tuple[int, int], spans: List[Tuple[int, int]]) -> bool:
    s, e = span
    return any(a <= s and e <= b for a, b in spans)


# ---------------------------------------------------------------------------
# Label collection
# ---------------------------------------------------------------------------

def _collect_labels(text: str) -> List[_Label]:
    """Every recognized amount/fee/total-debit label in the document, in
    document order, tagged with which family it belongs to.

    total_debit is resolved first and carved out of the fee and strong-tier
    scans below it; strong is resolved next and carved out of the weak
    bare-"Amount" scan -- this is what keeps "Tax Amount" from also
    registering as a bare "Amount" label, and "Total Debit Amount" from
    also registering as a "Debit Amount" one.
    """
    total_debit_spans = [m.span() for m in _TOTAL_DEBIT_LABEL_RE.finditer(text)]

    fee_spans: List[Tuple[int, int]] = []
    for m in _FEE_LABEL_RE.finditer(text):
        if _span_inside_any(m.span(), total_debit_spans):
            continue
        fee_spans.append(m.span())

    strong_spans: List[Tuple[int, int]] = []
    for m in _STRONG_LABEL_RE.finditer(text):
        span = m.span()
        if _span_inside_any(span, total_debit_spans) or _span_inside_any(span, fee_spans):
            continue
        strong_spans.append(span)

    weak_spans: List[Tuple[int, int]] = []
    for m in _WEAK_LABEL_RE.finditer(text):
        span = m.span()
        if (_span_inside_any(span, total_debit_spans)
                or _span_inside_any(span, fee_spans)
                or _span_inside_any(span, strong_spans)):
            continue
        weak_spans.append(span)

    labels = (
        [_Label(s, e, KIND_TOTAL_DEBIT) for s, e in total_debit_spans]
        + [_Label(s, e, KIND_FEE) for s, e in fee_spans]
        + [_Label(s, e, KIND_STRONG) for s, e in strong_spans]
        + [_Label(s, e, KIND_WEAK) for s, e in weak_spans]
    )
    labels.sort(key=lambda l: l.start)
    return labels


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------

def _first_money_after(
    text: str, start: int, window: int,
    labels: List[_Label], label_starts: List[int],
) -> Optional[Tuple[str, int, int]]:
    """First plausible money token after position `start`, bounded by
    `window` characters and cut short at the next recognized label (of any
    family) so that one label's search cannot steal a value belonging to
    the next.

    Values that fail the plausibility or account-fragment checks are
    skipped in favor of the next candidate in the window rather than
    aborting the scan outright -- this is what lets a decoy digit run
    sitting between the label and its value be walked past.
    """
    limit = min(start + window, len(text))
    idx = bisect.bisect_left(label_starts, start)
    if idx < len(labels) and labels[idx].start < limit:
        limit = labels[idx].start
    for m in _MONEY_RE.finditer(text, start, limit):
        if _looks_like_account_fragment(text, m.start(), m.end()):
            continue
        if not _is_plausible_amount(m.group(0)):
            continue
        return m.group(0), m.start(), m.end()
    return None


def _find_touching_runs(text: str, labels: List[_Label]) -> List[List[_Label]]:
    """Maximal runs of two or more consecutive recognized labels with no
    money-shaped value between them and only a short gap ("(MYR) :\n" and
    the like) -- the signature of a label column printed before its value
    column, rather than an ordinary label directly above/before its value.
    """
    runs: List[List[_Label]] = []
    current: List[_Label] = []
    for lbl in labels:
        if not current:
            current = [lbl]
            continue
        prev = current[-1]
        gap = text[prev.end:lbl.start]
        if len(gap) <= _RUN_GAP_MAX and _MONEY_RE.search(gap) is None:
            current.append(lbl)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [lbl]
    if len(current) >= 2:
        runs.append(current)
    return runs


def _resolve_block_run(
    text: str, run: List[_Label],
) -> Optional[List[Tuple[str, int, int]]]:
    """For a touching label run of length N, find N money tokens sitting
    immediately next to each other later in the document, and return them
    in order so `run[i]` pairs with the i-th token.

    Every token in the chain must independently pass the same
    plausibility/account-fragment checks as any other candidate; a single
    failure aborts the whole run rather than guessing.
    """
    n = len(run)
    pos = run[-1].end
    limit = min(pos + _RUN_CHAIN_WINDOW, len(text))

    m = _MONEY_RE.search(text, pos, limit)
    if m is None:
        return None

    chain: List[Tuple[str, int, int]] = []
    while True:
        if _looks_like_account_fragment(text, m.start(), m.end()):
            return None
        if not _is_plausible_amount(m.group(0)):
            return None
        chain.append((m.group(0), m.start(), m.end()))
        if len(chain) == n:
            return chain
        next_m = _MONEY_RE.search(text, m.end(), limit)
        if next_m is None:
            return None
        if len(text[m.end():next_m.start()]) > _RUN_CHAIN_GAP_MAX:
            return None
        m = next_m


# ---------------------------------------------------------------------------
# Candidate gathering, shared by all three public functions
# ---------------------------------------------------------------------------

class _FeeEntry(NamedTuple):
    value: str
    span: Tuple[int, int]
    score: float
    source: str
    is_total_flavored: bool


class _Gathered(NamedTuple):
    amount_candidates: List[_Candidate]
    fee_entries: List[_FeeEntry]
    total_debit_candidates: List[_Candidate]


def _gather(text: str) -> _Gathered:
    labels = _collect_labels(text)
    label_starts = [l.start for l in labels]

    # Block-scramble resolution first, so direct per-label scans below can
    # be skipped in favor of the positionally-paired value where one was
    # found.
    block_values: Dict[Tuple[int, int], Tuple[str, int, int]] = {}
    for run in _find_touching_runs(text, labels):
        chain = _resolve_block_run(text, run)
        if chain is None:
            continue
        for lbl, token in zip(run, chain):
            block_values[(lbl.start, lbl.end)] = token

    total_fee_spans = [m.span() for m in _TOTAL_FEE_RE.finditer(text)]

    amount_candidates: List[_Candidate] = []
    fee_entries: List[_FeeEntry] = []
    total_debit_candidates: List[_Candidate] = []

    for lbl in labels:
        key = (lbl.start, lbl.end)
        block = block_values.get(key)

        if lbl.kind in (KIND_STRONG, KIND_WEAK):
            if block is not None:
                raw, s, e = block
                formatted = _format_amount(raw)
                if formatted is not None:
                    amount_candidates.append(_Candidate(
                        formatted, _SCORE_BLOCK_RUN, "block_run:amount", s,
                    ))
                continue
            found = _first_money_after(text, lbl.end, _LABEL_WINDOW, labels, label_starts)
            if found is None:
                continue
            raw, s, e = found
            formatted = _format_amount(raw)
            if formatted is None:
                continue
            score = _SCORE_STRONG if lbl.kind == KIND_STRONG else _SCORE_WEAK
            source = "label:" + lbl.kind
            amount_candidates.append(_Candidate(formatted, score, source, s))

        elif lbl.kind == KIND_FEE:
            is_total_flavored = _span_inside_any(key, total_fee_spans)
            if block is not None:
                raw, s, e = block
                formatted = _format_amount(raw)
                if formatted is not None:
                    fee_entries.append(_FeeEntry(
                        formatted, (s, e), _SCORE_FEE_BLOCK_RUN,
                        "block_run:fee", is_total_flavored,
                    ))
                continue
            found = _first_money_after(text, lbl.end, _FEE_WINDOW, labels, label_starts)
            if found is None:
                continue
            raw, s, e = found
            formatted = _format_amount(raw)
            if formatted is None:
                continue
            fee_entries.append(_FeeEntry(
                formatted, (s, e), _SCORE_FEE_DIRECT, "label:fee", is_total_flavored,
            ))

        elif lbl.kind == KIND_TOTAL_DEBIT:
            if block is not None:
                raw, s, e = block
                formatted = _format_amount(raw)
                if formatted is not None:
                    total_debit_candidates.append(_Candidate(
                        formatted, _SCORE_TOTAL_DEBIT_BLOCK_RUN,
                        "block_run:total_debit", s,
                    ))
                continue
            found = _first_money_after(text, lbl.end, _LABEL_WINDOW, labels, label_starts)
            if found is None:
                continue
            raw, s, e = found
            formatted = _format_amount(raw)
            if formatted is None:
                continue
            total_debit_candidates.append(_Candidate(
                formatted, _SCORE_TOTAL_DEBIT_DIRECT, "label:total_debit", s,
            ))

    # Currency-prefixed fallback -- `amount` only, and only useful when no
    # label text survived at all. Never allowed to steal a value already
    # claimed by a fee or total-debit label.
    claimed_spans = [e.span for e in fee_entries]
    for lbl, key in ((lbl, (lbl.start, lbl.end)) for lbl in labels if lbl.kind == KIND_TOTAL_DEBIT):
        block = block_values.get(key)
        if block is not None:
            claimed_spans.append((block[1], block[2]))

    for cm in _CURRENCY_RE.finditer(text):
        window_end = min(cm.end() + _CURRENCY_WINDOW, len(text))
        m = _MONEY_RE.search(text, cm.end(), window_end)
        if not m:
            continue
        if _looks_like_account_fragment(text, m.start(), m.end()):
            continue
        if not _is_plausible_amount(m.group(0)):
            continue
        if _span_inside_any((m.start(), m.end()), claimed_spans):
            continue
        formatted = _format_amount(m.group(0))
        if formatted is None:
            continue
        amount_candidates.append(_Candidate(
            formatted, _SCORE_CURRENCY, "currency_prefix:" + cm.group(0).upper(),
            m.start(),
        ))

    return _Gathered(amount_candidates, fee_entries, total_debit_candidates)


def _best_of(candidates: List[_Candidate]) -> Optional[_Candidate]:
    """Highest score wins; ties go to whichever occurs earliest in the
    document."""
    if not candidates:
        return None
    best_by_value: Dict[str, _Candidate] = {}
    for c in candidates:
        prior = best_by_value.get(c.value)
        if prior is None or c.score > prior.score:
            best_by_value[c.value] = c
    return min(best_by_value.values(), key=lambda c: (-c.score, c.start))


def _confidence(score: float, ocr_used: bool) -> float:
    confidence = score / 100.0
    if ocr_used:
        confidence *= _OCR_CONFIDENCE_PENALTY
    return max(0.0, min(confidence, 0.99))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_amount(text: str, ocr_used: bool = True) -> FieldResult:
    """Extract the transaction/transfer amount from IBG receipt text.

    Returns `FieldResult.missing()` rather than a guess when nothing
    trustworthy is found.
    """
    if not text:
        return FieldResult.missing("empty_text")

    gathered = _gather(text)
    candidates_out = [
        (c.value, round(c.score, 2))
        for c in sorted(gathered.amount_candidates, key=lambda c: (-c.score, c.start))
    ]
    winner = _best_of(gathered.amount_candidates)
    if winner is None:
        return FieldResult.missing("no_candidates")

    return FieldResult(
        value=winner.value,
        confidence=_confidence(winner.score, ocr_used),
        source=winner.source,
        candidates=candidates_out,
    )


def extract_fee(text: str, ocr_used: bool = True) -> FieldResult:
    """Extract the total fees/charges from IBG receipt text.

    When the receipt restates its own total ("Total Charges", "Total Fee
    Charges") that single number is used as-is. Otherwise every distinct
    fee-shaped value found (Service Charge, Transaction Fee, GST/SST,
    Commission, ...) is summed -- "total fees/charges", not just the first
    one found. A fee is real money but is never returned by
    `extract_amount`.
    """
    if not text:
        return FieldResult.missing("empty_text")

    gathered = _gather(text)
    entries = gathered.fee_entries
    if not entries:
        return FieldResult.missing("no_candidates")

    candidates_out = [
        (e.value, round(e.score, 2))
        for e in sorted(entries, key=lambda e: (-e.score, e.span[0]))
    ]

    total_flavored = [e for e in entries if e.is_total_flavored]
    if total_flavored:
        winner = max(total_flavored, key=lambda e: (e.score, -e.span[0]))
        return FieldResult(
            value=winner.value,
            confidence=_confidence(winner.score, ocr_used),
            source=winner.source,
            candidates=candidates_out,
        )

    # Sum every distinct component (dedup by the money token's own span --
    # the same physical value can otherwise be reachable through more than
    # one label spelling glued onto one line, e.g. "GST Amount-Service
    # Charge" matching both "GST Amount" and "Service Charge").
    unique_by_span: Dict[Tuple[int, int], _FeeEntry] = {}
    for e in entries:
        unique_by_span[e.span] = e

    total = Decimal("0")
    best_score = 0.0
    for e in unique_by_span.values():
        total += Decimal(e.value)
        best_score = max(best_score, e.score)
    total = total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    return FieldResult(
        value=str(total),
        confidence=_confidence(best_score, ocr_used),
        source="fee:sum_of_components",
        candidates=candidates_out,
    )


def extract_total_debit(text: str, ocr_used: bool = True) -> FieldResult:
    """Extract the total debit (amount + fees) from IBG receipt text, i.e.
    what actually left the account. Only some receipts print this as a
    distinct, restated number ("Total Debit Amount", "Total Amount to
    Debit", "Total Debit", "Total Amount(MYR)"); most do not, and this
    correctly returns missing rather than inventing one by adding
    `extract_amount` and `extract_fee` together.
    """
    if not text:
        return FieldResult.missing("empty_text")

    gathered = _gather(text)
    candidates_out = [
        (c.value, round(c.score, 2))
        for c in sorted(gathered.total_debit_candidates, key=lambda c: (-c.score, c.start))
    ]
    winner = _best_of(gathered.total_debit_candidates)
    if winner is None:
        return FieldResult.missing("no_candidates")

    return FieldResult(
        value=winner.value,
        confidence=_confidence(winner.score, ocr_used),
        source=winner.source,
        candidates=candidates_out,
    )
