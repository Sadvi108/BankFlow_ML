"""Transaction-date extraction for IBG (interbank transfer) receipts.

Owns exactly one field: the date the transfer happened. See
`app/ibg/contract.py` for the `FieldResult` contract every field module
returns, and `tests/fixtures/ibg_corpus.py` for the ground-truth receipts
this module is scored against.

Two problems this module solves that the legacy engine
(`app/ultimate_patterns_v3.py::extract_date`) does not:

1. Normalization. The legacy engine returns whatever raw substring matched
   -- "DD Mon YYYY", "DD/MM/YYYY", and "YYYY-MM-DD" all coexist today,
   depending on which receipt produced the match -- which is unsortable and
   uncomparable once it lands in the history CSV / Supabase. Every value
   returned here is ISO ``YYYY-MM-DD``, always, or ``None``.

2. The print stamp. These receipts are browser-printed to PDF, so every
   page footer repeats a render timestamp: a date immediately followed by a
   comma and a bare ``HH:MM`` -- no seconds -- next to the app name and a
   URL. There are also non-comma "administrative" stamps -- "Printed
   Date/Time", "Printed On", "Generated on", "Date of Export", "Computer
   generated" -- that are equally never the answer even though they are
   shaped like a genuine timestamp (they carry seconds). Neither shape ever
   wins, even when it is the *only* other date on the page, appears
   *first* in the text, or repeats *more often* than the real date --
   frequency is not used as a signal anywhere below.

Label ranking (highest first):
    1. A date introduced by a transaction-detail label: "Transaction date",
       "Date and Time", "Value Date", "Transfer Date", "Payment Date",
       "Debit Date", "Today", "Creation Date". Labels are scanned in
       document order and the *first* one (in text order) that yields a
       real date wins -- these receipts always print the transaction-detail
       block before any later approval/activity log that happens to reuse
       the same column header.
    2. A date introduced by a secondary label: "Advice date", "Created
       Date & Time", an approval/authorisation timestamp, a statement
       period. Only consulted when tier 1 finds nothing anywhere in the
       document -- these labels sit right next to a rank-1 label on some
       receipts (S24: "Advice sending date" vs "Value date"), and rank 1
       must always win when both are present.
    3. Any other date-shaped token in the document (earliest wins), as
       long as it is not print-stamp / administrative-stamp shaped.
Stamp-shaped dates are excluded at every tier -- they never win, even when
nothing else is on the page. In that case the correct answer is
`FieldResult.missing()`, not the stamp.

Date order (day-first vs month-first): Malaysian portals are day-first
without exception; Citibank's CitiDirect is the one outlier and prints US
month-first (see `app/ibg/bank_registry.py`). Order of precedence for an
ambiguous all-numeric date:
    a. An explicit per-label format hint, e.g. "Value Date (mm/dd/yyyy)".
    b. The caller-supplied `bank_key`, resolved via
       `bank_registry.date_order_for`.
    c. A bank brand name/alias detected directly in the document text
       (the coordinator that is meant to resolve and pass `bank_key` does
       not exist yet, so this module falls back to doing that lookup
       itself when `bank_key` is not given).
    d. Same-document inference: if any all-numeric date in the document
       has a first component >12 the document is day-first; if a second
       component >12 it is month-first. A corroborating unambiguous
       month-name date sharing the same day/month/year elsewhere on the
       page can also decide it. Only used when unambiguous.
    e. Default: day-first.

Stdlib only (`re`, `datetime`) -- no third-party date parser, no new
dependency in requirements.txt.
"""
import re
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

from app.ibg.contract import FieldResult
from app.ibg.bank_registry import ALL_ENTITIES, date_order_for

__all__ = ["extract_transaction_date"]


# --------------------------------------------------------------------------
# Month-name lookup, used by the "DD Month YYYY" / "Month DD, YYYY" forms.
# --------------------------------------------------------------------------
_MONTHS: Dict[str, int] = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# --------------------------------------------------------------------------
# Label ranking. Rank 1 is the answer. Rank 2 is only consulted when no
# rank-1 label anywhere in the document yields a usable date.
#
# "Date/Time" and bare "Date" are deliberately NOT rank-1 patterns: they
# would also match inside "Printed Date/Time", one of the never-answer
# administrative stamps below, and there is no corpus layout that needs the
# bare form once "Date and Time", "Transaction Date, Time" etc. are covered
# by the more specific patterns already present.
# --------------------------------------------------------------------------
_RANK1_LABEL_PATTERNS = (
    r"Transaction\s*Date",
    r"Date\s+and\s+Time",
    r"Value\s*Date",
    r"Transfer\s*Date",
    r"Payment\s*Date",
    r"Debit\s*Date",
    r"Creation\s*Date",
    r"\bToday\b",
)
_RANK2_LABEL_PATTERNS = (
    r"Advice\s*Date",
    r"Created\s*Date\s*(?:&|and)?\s*Time",
    r"Approv(?:e|al|ed)\s*(?:Date|Time|Timestamp)?",
    r"Authoris(?:e|ation)\s*(?:Date|Time)?",
    r"Authoriz(?:e|ation)\s*(?:Date|Time)?",
    r"Statement\s*(?:Period|Date)",
)

_RANK1_RE = re.compile(r"\b(?:" + "|".join(_RANK1_LABEL_PATTERNS) + r")\b", re.IGNORECASE)
_RANK2_RE = re.compile(r"\b(?:" + "|".join(_RANK2_LABEL_PATTERNS) + r")\b", re.IGNORECASE)

# Labels that introduce a render/administrative stamp -- never the answer,
# regardless of whether the date next to them happens to be comma-shaped
# print-stamp or a plain "date seconds" timestamp. Reused as the idea behind
# `app/ultimate_patterns_v3.py::_print_stamp_re`, generalized to the
# non-comma administrative variants this corpus also carries ("Printed
# Date/Time", "Generated on", ...).
_NEVER_LABEL_RE = re.compile(
    r"\b(?:"
    r"Printed\s*Date\s*/?\s*Time"
    r"|Printed\s*On"
    r"|Generated\s*on"
    r"|Date\s+of\s+Export"
    r"|Computer\s+generated"
    r")\b",
    re.IGNORECASE,
)
# How far back from a date candidate to look for one of the labels above.
_NEVER_LABEL_LOOKBACK = 40

# A parenthetical format hint directly after a label, e.g.
# "Value Date (dd-mm-yyyy)" or "Value Date (dd/mm/yyyy)". All-letters, so it
# can never itself be mistaken for a date value by the patterns below.
_FORMAT_HINT_RE = re.compile(
    r"\A\s*\(([A-Za-z]{1,4}[/\-. ][A-Za-z]{1,4}[/\-. ][A-Za-z]{1,4})\)"
)

# How far past a label to look for its value. Generous enough to cross the
# column-scrambled layouts in the corpus, where the label block and the
# value block are printed separately and can be a full block of labels
# apart (S11, S23).
_LABEL_WINDOW = 350
# When a window is sliced mid-digit-run, extend forward (up to this many
# extra characters) until landing on a non-digit, so a truncated token like
# "2026" clipped to "20" can never masquerade as a real 2-digit year -- see
# `_extend_window`.
_WINDOW_TRUNCATION_GUARD = 20

# --------------------------------------------------------------------------
# Date-shaped token patterns. Each captures exactly 3 groups in the literal
# left-to-right order they appear in the source text -- NOT day/month/year;
# that assignment happens afterwards, via `field_order`.
# --------------------------------------------------------------------------
# YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD (dotted ISO, e.g. MUFG's "2026.08.03").
# Unambiguous order regardless of `field_order` -- a 4-digit leading group
# can only be the year.
_ISO_RE = re.compile(r"\b(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})\b")
_NUMERIC_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
# DD Month YYYY, also accepting hyphens between every field
# ("15-Jul-2026", "07-Aug-2026") as well as the plain spaced form
# ("06 Aug 2026, 17:47 GMT+08:00").
_DAY_MONTHNAME_YEAR_RE = re.compile(r"\b(\d{1,2})[\s-]+([A-Za-z]{3,9})[\s-]+(\d{2,4})\b")
_MONTHNAME_DAY_YEAR_RE = re.compile(r"\b([A-Za-z]{3,9})[\s-]+(\d{1,2}),?\s*(\d{2,4})\b")

# Malaysian receipts: day-first is the default absent any stronger signal.
_FIELD_ORDER_DEFAULT = ("d", "m", "y")
_MONTH_FIRST = ("m", "d", "y")

# A browser print-stamp is a date immediately followed by a comma and a
# bare HH:MM -- no seconds. Genuine bank transaction timestamps are audit
# records and virtually always carry seconds; only a render stamp in a
# page footer (produced by a JS "toLocaleString()"-style call at print
# time) typically omits them. The comma is part of the signature too: a
# labelled transaction date-time is usually just "<date> <time>" with no
# comma, so requiring *both* (comma AND no seconds) is what lets this tell
# a print stamp apart from an unlabeled-but-genuine date-time that also
# happens to use a comma separator but does carry seconds.
_PRINT_STAMP_TAIL_RE = re.compile(r"\A\s*,\s*\d{1,2}:\d{2}(?!\s*:\s*\d{2})")


def _resolve_two_digit_year(yy: int, reference_year: int) -> int:
    """Expand a 2-digit year, pivoting +10 years from `reference_year`.

    ``31/10/25`` with `reference_year` 2026 -> 2025, since 2000 + 25 = 2025
    is not more than 10 years past 2026. A 2-digit year that WOULD land
    more than 10 years in the future is assumed to be the previous century
    instead (e.g. yy=37 with reference_year 2026 -> 1937, not 2037).
    """
    pivot = reference_year + 10
    candidate = 2000 + yy
    if candidate > pivot:
        return 1900 + yy
    return candidate


def _to_iso(year: int, month: int, day: int) -> Optional[str]:
    """A real calendar date -> ISO string. Impossible dates -> None.

    This is the single point where invalid dates (``31/02/2025``,
    ``13/13/2025``, month 0, day 32, ...) get rejected -- `datetime.date`
    raises `ValueError` for anything that is not a real day on the
    calendar, and that is treated as "not a candidate at all", never
    coerced into something nearby.
    """
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _extend_window(text: str, start: int, length: int,
                    max_extra: int = _WINDOW_TRUNCATION_GUARD) -> str:
    """Slice `text[start:start+length]` without cutting a digit run in half.

    A fixed-length slice can land mid-token: "...07/08/2026..." truncated
    at exactly "07/08/20" is a perfectly well-formed (but wrong) 2-digit-year
    numeric date, silently pivoting "2026" into "2020". Extending the cut
    forward until it lands on a non-digit (bounded by `max_extra`, so a
    pathological run of digits cannot blow the window out arbitrarily) means
    a truncated token is either read whole or not read at all -- never read
    as a shorter, different-meaning token.
    """
    end = start + length
    n = len(text)
    extra = 0
    while end < n and text[end].isdigit() and extra < max_extra:
        end += 1
        extra += 1
    return text[start:end]


def _parse_format_hint(text: str, pos: int,
                        default_order: Tuple[str, str, str],
                        ) -> Tuple[str, str, str]:
    """Field order implied by a label's parenthetical hint, if any.

    ``"Value Date (dd-mm-yyyy)"`` and ``"Value Date (dd/mm/yyyy)"`` both
    yield `default_order` unchanged; a ``"(mm/dd/yyyy)"`` hint overrides it
    to month-first regardless of `default_order`. Falls back to
    `default_order` when there is no parenthetical, or its contents are not
    a clean permutation of d/m/y.
    """
    m = _FORMAT_HINT_RE.match(text[pos:pos + 20])
    if not m:
        return default_order
    tokens = [tok for tok in re.split(r"[/\-. ]", m.group(1)) if tok]
    letters = tuple(tok[0].lower() for tok in tokens)
    if len(letters) == 3 and sorted(letters) == ["d", "m", "y"]:
        return letters
    return default_order


def _assign_fields(g1: str, g2: str, g3: str,
                    field_order: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """Map three positionally-captured groups onto (day, month, year)."""
    slots = dict(zip(field_order, (g1, g2, g3)))
    return slots["d"], slots["m"], slots["y"]


def _normalize_numeric(g1: str, g2: str, g3: str, reference_year: int,
                        field_order: Tuple[str, str, str] = _FIELD_ORDER_DEFAULT,
                        ) -> Optional[str]:
    """Normalize a ``DD/MM/YYYY``-shaped (or hint-reordered) numeric token."""
    day_s, month_s, year_s = _assign_fields(g1, g2, g3, field_order)
    try:
        day, month = int(day_s), int(month_s)
    except ValueError:
        return None
    year = (_resolve_two_digit_year(int(year_s), reference_year)
            if len(year_s) == 2 else int(year_s))
    return _to_iso(year, month, day)


def _normalize_iso(g1: str, g2: str, g3: str) -> Optional[str]:
    """Normalize a ``YYYY-MM-DD``/``YYYY/MM/DD``/``YYYY.MM.DD`` token
    (unambiguous order -- the leading 4-digit group can only be the year)."""
    try:
        return _to_iso(int(g1), int(g2), int(g3))
    except ValueError:
        return None


def _normalize_day_monthname_year(day_s: str, month_name: str, year_s: str,
                                   reference_year: int) -> Optional[str]:
    """Normalize a ``DD Month YYYY`` / ``Month DD, YYYY`` token."""
    month = _MONTHS.get(month_name.lower())
    if month is None:
        return None
    year = (_resolve_two_digit_year(int(year_s), reference_year)
            if len(year_s) == 2 else int(year_s))
    try:
        day = int(day_s)
    except ValueError:
        return None
    return _to_iso(year, month, day)


def _preceded_by_never_label(text: str, match_start: int) -> bool:
    """True when a "never the answer" administrative-stamp label
    (Printed Date/Time, Printed On, Generated on, Date of Export, Computer
    generated) sits directly before this date, within `_NEVER_LABEL_LOOKBACK`
    characters. These stamps do not always take the comma+bare-HH:MM shape
    `_PRINT_STAMP_TAIL_RE` looks for (some carry seconds), so they need to be
    excluded by their label instead.
    """
    start = max(0, match_start - _NEVER_LABEL_LOOKBACK)
    return bool(_NEVER_LABEL_RE.search(text[start:match_start]))


def _iter_date_candidates(text: str, reference_year: int,
                           field_order: Tuple[str, str, str],
                           ) -> List[Tuple[int, str, bool]]:
    """Every date-shaped token in `text`: (start_pos, iso_value, is_stamp).

    Sorted left-to-right by position. A token that does not form a real
    calendar date is dropped entirely -- it is never a candidate, not even
    a low-ranked one. The "never" label lookback (`_preceded_by_never_label`)
    only sees labels that are themselves inside `text`, which is sufficient
    here: label windows built by `_find_labelled_date` start right after
    their own rank-1/rank-2 label, and the tier-3/whole-document scan passes
    the entire document, where an administrative-stamp label always
    precedes its date within the same text.
    """
    seen_spans = set()
    hits = []  # type: List[Tuple[int, str, bool]]

    def _consider(m, iso):
        if iso is None:
            return
        span = (m.start(), m.end())
        if span in seen_spans:
            return
        seen_spans.add(span)
        is_stamp = bool(_PRINT_STAMP_TAIL_RE.match(text[m.end():m.end() + 20]))
        if not is_stamp:
            is_stamp = _preceded_by_never_label(text, m.start())
        hits.append((m.start(), iso, is_stamp))

    for m in _ISO_RE.finditer(text):
        _consider(m, _normalize_iso(*m.groups()))
    for m in _DAY_MONTHNAME_YEAR_RE.finditer(text):
        day_s, month_name, year_s = m.groups()
        _consider(m, _normalize_day_monthname_year(day_s, month_name, year_s, reference_year))
    for m in _MONTHNAME_DAY_YEAR_RE.finditer(text):
        month_name, day_s, year_s = m.groups()
        _consider(m, _normalize_day_monthname_year(day_s, month_name, year_s, reference_year))
    for m in _NUMERIC_RE.finditer(text):
        g1, g2, g3 = m.groups()
        _consider(m, _normalize_numeric(g1, g2, g3, reference_year, field_order))

    hits.sort(key=lambda h: h[0])
    return hits


def _find_labelled_date(text: str, label_re, reference_year: int,
                         default_order: Tuple[str, str, str],
                         ) -> Optional[Tuple[str, int, str]]:
    """First label (document order) whose window yields a real,
    non-stamp date.

    Returns (iso_value, label_start_pos, label_slug), or None if no label
    from `label_re` has a usable date near it anywhere in `text`.
    """
    for lm in label_re.finditer(text):
        field_order = _parse_format_hint(text, lm.end(), default_order)
        window = _extend_window(text, lm.end(), _LABEL_WINDOW)
        for _, iso, is_stamp in _iter_date_candidates(window, reference_year, field_order):
            if not is_stamp:
                return iso, lm.start(), lm.group(0).strip().lower()
    return None


def _rank_candidates(hits: List[Tuple[int, str, bool]]) -> List[Tuple[str, float]]:
    """Collapse (start, iso, is_stamp) hits into a deduped, best-first
    (value, score) list for `FieldResult.candidates`.
    """
    best = {}  # type: Dict[str, float]
    for _, iso, is_stamp in hits:
        score = 1.0 if is_stamp else 30.0
        if score > best.get(iso, -1.0):
            best[iso] = score
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


def _promote(iso: str, score: float,
             existing: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Put the winning value first, at its tier score; keep the rest after."""
    rest = [pair for pair in existing if pair[0] != iso]
    return [(iso, score)] + rest


# --------------------------------------------------------------------------
# Date order (day-first vs month-first) resolution.
# --------------------------------------------------------------------------

def _detect_bank_key(text: str) -> Optional[str]:
    """Best-effort bank identification straight from the receipt text.

    The coordinator that is meant to classify the bank and pass `bank_key`
    in does not exist yet, so when the caller does not supply one this module
    falls back to doing the same lookup itself, using the shared
    `bank_registry` alias/name data (never a corpus literal). Rails
    (DuitNow, JomPAY, RENTAS, ...) are skipped -- they ride on top of a
    bank and are never the issuing bank. The longest matching alias wins,
    so a more specific brand ("CIMB Clicks") is preferred over a shorter,
    more generic one ("CIMB") when both are present.
    """
    if not text:
        return None
    upper = text.upper()
    best_key = None  # type: Optional[str]
    best_len = 0
    for entity in ALL_ENTITIES:
        if entity.tier == "rail":
            continue
        candidates = [entity.name.upper()] + [a.upper() for a in entity.aliases]
        for cand in candidates:
            if cand and len(cand) > best_len and cand in upper:
                best_key = entity.key
                best_len = len(cand)
    return best_key


def _collect_unambiguous_month_dates(text: str, reference_year: int) -> Set[Tuple[int, int, int]]:
    """(day, month, year) triples read off unambiguous month-name dates."""
    found = set()  # type: Set[Tuple[int, int, int]]

    def _add(day_s, month_name, year_s):
        month = _MONTHS.get(month_name.lower())
        if month is None:
            return
        try:
            day = int(day_s)
        except ValueError:
            return
        year = (_resolve_two_digit_year(int(year_s), reference_year)
                if len(year_s) == 2 else int(year_s))
        if _to_iso(year, month, day) is not None:
            found.add((day, month, year))

    for m in _DAY_MONTHNAME_YEAR_RE.finditer(text):
        _add(*m.groups())
    for m in _MONTHNAME_DAY_YEAR_RE.finditer(text):
        month_name, day_s, year_s = m.groups()
        _add(day_s, month_name, year_s)
    return found


def _infer_field_order(text: str, reference_year: int) -> Optional[Tuple[str, str, str]]:
    """Same-document inference of numeric-date field order, or None.

    Two independent signals, either sufficient on its own when unambiguous:

    1. A component that cannot be a month (>12) pins which position is the
       day. If some numeric date's first component is >12, the document
       must be day-first; if some numeric date's second component is >12,
       the document must be month-first. Contradictory evidence (both seen)
       is not used -- an unresolved mix is not a signal.
    2. An unambiguous month-name date sharing its (day, month, year) with
       one -- and only one -- of the two possible readings of some numeric
       date confirms that reading for the whole document.

    Only overrides the caller's default when the result is unambiguous.
    """
    day_first_evidence = False
    month_first_evidence = False
    numeric_hits = []  # type: List[Tuple[int, int, int]]
    for m in _NUMERIC_RE.finditer(text):
        g1, g2, g3 = m.groups()
        try:
            n1, n2 = int(g1), int(g2)
        except ValueError:
            continue
        if n1 > 12:
            day_first_evidence = True
        if n2 > 12:
            month_first_evidence = True
        year = (_resolve_two_digit_year(int(g3), reference_year)
                if len(g3) == 2 else int(g3))
        numeric_hits.append((n1, n2, year))

    if day_first_evidence and not month_first_evidence:
        return _FIELD_ORDER_DEFAULT
    if month_first_evidence and not day_first_evidence:
        return _MONTH_FIRST

    month_dates = _collect_unambiguous_month_dates(text, reference_year)
    if not month_dates or not numeric_hits:
        return None

    day_first_confirmed = False
    month_first_confirmed = False
    for n1, n2, year in numeric_hits:
        if (n1, n2, year) in month_dates:
            day_first_confirmed = True
        if (n2, n1, year) in month_dates:
            month_first_confirmed = True

    if day_first_confirmed and not month_first_confirmed:
        return _FIELD_ORDER_DEFAULT
    if month_first_confirmed and not day_first_confirmed:
        return _MONTH_FIRST
    return None


def _resolve_field_order(text: str, bank_key: Optional[str],
                          reference_year: int) -> Tuple[str, str, str]:
    """The default day/month/year order to use absent a per-label hint.

    Precedence: an explicit `bank_key` from the caller > a bank brand
    detected in the text itself > same-document inference > day-first.
    """
    if bank_key is not None:
        return date_order_for(bank_key)

    detected = _detect_bank_key(text)
    if detected is not None:
        return date_order_for(detected)

    inferred = _infer_field_order(text, reference_year)
    if inferred is not None:
        return inferred

    return _FIELD_ORDER_DEFAULT


def extract_transaction_date(text: str, ocr_used: bool = True,
                              bank_key: Optional[str] = None) -> FieldResult:
    """Extract the IBG transaction date from receipt `text`.

    `FieldResult.value` is always either an ISO ``YYYY-MM-DD`` string or
    `None` -- never a raw substring, never a print/administrative stamp, and
    never an impossible calendar date.

    `bank_key` is an optional hint (see `app/ibg/bank_registry.py`) used to
    resolve ambiguous all-numeric dates such as Citibank's month-first
    ``08/05/2026``. It is `None` by default -- the coordinator responsible
    for classifying the bank and supplying this does not exist yet -- in
    which case this module makes its own best-effort guess from the text
    itself before falling back to same-document inference and finally to
    the day-first default that holds for every other Malaysian portal here.
    """
    if not text:
        return FieldResult.missing(source="missing:empty_text")

    reference_year = date.today().year
    penalty = 0.05 if ocr_used else 0.0
    default_order = _resolve_field_order(text, bank_key, reference_year)

    # Computed once, up front: every date-shaped token in the whole
    # document. Used both as the tier-3 fallback and to populate
    # `candidates` for debugging regardless of which tier ultimately wins.
    all_hits = _iter_date_candidates(text, reference_year, default_order)
    doc_candidates = _rank_candidates(all_hits)

    # Tier 1: transaction-detail labels. Earliest label in the document
    # that has a real date near it wins -- this is what "prefer the
    # transaction-detail block" means in practice on these receipts.
    primary = _find_labelled_date(text, _RANK1_RE, reference_year, default_order)
    if primary is not None:
        iso, _, slug = primary
        return FieldResult(
            value=iso,
            confidence=round(0.95 - penalty, 3),
            source="label:" + slug,
            candidates=_promote(iso, 100.0, doc_candidates),
        )

    # Tier 2: secondary labels (advice date, created date & time,
    # approval/authorisation timestamps, statement period). Only consulted
    # when no rank-1 label anywhere in the document produced a usable date.
    secondary = _find_labelled_date(text, _RANK2_RE, reference_year, default_order)
    if secondary is not None:
        iso, _, slug = secondary
        return FieldResult(
            value=iso,
            confidence=round(0.75 - penalty, 3),
            source="label_secondary:" + slug,
            candidates=_promote(iso, 50.0, doc_candidates),
        )

    # Tier 3: no label at all anywhere in the document. Fall back to the
    # earliest non-stamp date on the page (never "most frequent" --
    # frequency is actively misleading on these receipts).
    non_stamp = [iso for _, iso, is_stamp in all_hits if not is_stamp]
    if non_stamp:
        return FieldResult(
            value=non_stamp[0],
            confidence=round(0.6 - penalty, 3),
            source="fallback:unlabeled_date",
            candidates=doc_candidates,
        )

    if all_hits:
        # Every date-shaped token on the page was stamp-shaped (print or
        # administrative). No unlabeled date beats a labelled one, and a
        # stamp never wins outright -- the honest answer is "not found",
        # not the stamp.
        return FieldResult(
            value=None,
            confidence=0.0,
            source="missing:only_print_stamp",
            candidates=doc_candidates,
        )

    return FieldResult.missing(source="missing:no_date_candidates")
