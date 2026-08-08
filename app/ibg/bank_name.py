"""Issuing-bank and beneficiary-bank extraction for IBG receipts.

An interbank transfer names two banks and they are never the same bank:

  * **Issuing bank** (`extract_bank_name`) -- whose portal produced the
    receipt.
  * **Beneficiary / receiving bank** (`extract_beneficiary_bank`) -- where
    the money landed.

Both functions consume the shared static registry in `app.ibg.bank_registry`
(BIC map, aliases, Islamic-subsidiary entities, payment-rail exclusion list)
rather than hand-rolling a second one.

Signal hierarchy for `extract_bank_name`
-----------------------------------------
The issuing bank is almost always identifiable from *where in the document*
a bank name sits, not from which bank is mentioned most. In this corpus the
issuer's own brand shows up in one of four positions, and the beneficiary's
name never does:

1. **Header** -- the first non-blank line of the document (portal name,
   dual-brand logo bar like "AmBank    AmBank Islamic", or the plain bank
   name, e.g. "UOB", "standard chartered", "BANK OF CHINA").
2. **Footer** -- the last non-blank line (copyright/legal-entity line,
   portal branding, or a portal URL host).
3. **Portal/product branding anywhere** in the document -- a curated subset
   of the registry's aliases name a bank's *online-banking product*
   ("Maybank2u", "M2E", "CIMB Clicks", "PBenterprise", "connectFirst",
   "BizChannel@CIMB", "GCMS Plus", ...) rather than just the bank itself.
   Nobody prints the counterparty's login-portal name on a receipt, so a hit
   here is direction-unambiguous regardless of what else is named in the
   body.
4. **"<Bank> Reference No" adjacency** -- some portals label their own
   transaction-reference field with their own name right in it (OCBC's
   "OCBC Reference No."). A single word immediately before "Reference" that
   resolves to a registered bank alias is trusted the same way.

Header/footer/portal/reference-adjacency hits are collectively "tier 1":
whichever entity wins there is returned outright, at high confidence,
without even looking at what else the document says. Header/footer zones
are capped by length (150 / 120 chars) so a column-scrambled or
backslash-joined document that has stuffed hundreds of characters -- label,
value, and all -- onto "the first/last line" can't smuggle a beneficiary
mention through as if it were header/footer branding (this is exactly what
happens on S4: naive "first N lines" would have caught "Recipient Bank...
RHB BANK" sitting inside one giant OCR-joined line).

Before ranking tier-1 hits (and again before the tier-2 fallback below), any
entity whose only appearance is *as the value under a beneficiary-side
label* -- "Beneficiary/Receiving/Recipient/Payee/Credit ... Bank" -- is
excluded from issuer candidacy outright, by identity (its family key, see
below), not merely down-weighted. That is what keeps a document that
happens to also print its beneficiary's brand in some prominent spot from
being misread.

Islamic subsidiaries (RHB Islamic Bank, AmBank Islamic, Hong Leong Islamic
Bank, ...) collapse to their conventional parent for *issuer* purposes only.
Malaysian banks print a dual-brand logo bar ("AmBank    AmBank Islamic",
"HongLeong Bank    HongLeong Islamic Bank") on every document regardless of
whether the transaction is Islamic-banking; that bar is not a statement
about this specific transaction, and no sample in this corpus expects an
Islamic subsidiary as `bank_name`. `extract_beneficiary_bank` does *not*
collapse -- there the Islamic/conventional distinction is exactly the
information the receipt is recording (S1: the beneficiary is specifically
RHB Islamic Bank, not RHB Bank).

Tier 2, used only when no tier-1 evidence exists at all: a bare bank-name or
BIC mention anywhere in the body, still filtered through the same
beneficiary-value exclusion. If nothing survives that filter, the honest
answer is `FieldResult.missing()`, not a guess (S4: the only bank named
anywhere is sitting under "Recipient Bank", so there is nothing left to
guess from).

Signal hierarchy for `extract_beneficiary_bank`
-------------------------------------------------
1. **A beneficiary-side BIC label** ("Beneficiary Bank Code", "...Routing
   Code") -> BIC -> registry lookup. The BIC is the sturdiest signal in a
   degraded or page-broken layout.
2. **A beneficiary-side name label** ("Beneficiary/Receiving/Recipient/
   Payee/Credit ... Bank[ Name/Information]") -> a windowed scan forward for
   the first line naming a bank, skipping footer noise (date stamps, URLs,
   page markers, placeholder dashes, portal-branding lines). If the window
   never names a bank by alias but does contain a bare BIC (S25: the
   "Bank Name" field is itself printed as the SWIFT code, not a name), that
   BIC resolves the entity instead.
3. **A generic "Bank Name"/"Bank Code" field**, trusted only when a
   beneficiary-side marker (Payee/Recipient/Beneficiary/Credit, or a bare
   "To" section heading) appears in the ~150 characters before it. This
   covers layouts that never use the word "Beneficiary" at all (S20: "To /
   Payee Name ... / Bank Name / RHB BANK").
4. **Fallback: a bare BIC anywhere in the text.**
5. **Fallback: a bare bank alias**, only when some beneficiary-side label
   exists somewhere in the document.

OCR repair
----------
`_repair_ocr_bank_tokens` fixes a small set of known shape-based OCR
misreads of bank tokens (H5BC -> HSBC, C1MB -> CIMB, ...). It is applied
*only* when `ocr_used=True`. Text lifted from a PDF's own text layer has no
OCR misreads, and running digit/letter repair on clean text would corrupt it
instead of fixing anything.
"""
import re
from typing import Dict, List, Optional, Set, Tuple

from app.ibg.contract import FieldResult
from app.ibg.bank_registry import (
    BankEntity,
    BANKS_ONLY,
    by_key,
    is_rail,
    lookup_by_bic,
)

# ---------------------------------------------------------------------------
# OCR token repair -- gated on ocr_used=True by the caller (extract_* below).
# Each entry is a specific shape-based misread of a real registry alias, not
# a blanket character substitution, so it cannot corrupt unrelated text.
# ---------------------------------------------------------------------------
_OCR_TOKEN_REPAIRS = {
    "H5BC": "HSBC",
    "H58C": "HSBC",
    "MAYBONK": "MAYBANK",
    "MAYB0NK": "MAYBANK",
    "C1MB": "CIMB",
    "CIM8": "CIMB",
    "C1M8": "CIMB",
    "R4B": "RHB",
    "RH8": "RHB",
    "U0B": "UOB",
    "0CBC": "OCBC",
}

_OCR_TOKEN_RE = re.compile(r"\b[A-Z0-9]{3,10}\b", re.IGNORECASE)


def _repair_ocr_bank_tokens(text: str) -> str:
    """Repair known OCR misreads of bank-name tokens.

    Caller must gate this on ocr_used=True. Text from a PDF's own text layer
    has no misreads, and this must not run on it.
    """

    def repl(match: "re.Match") -> str:
        token = match.group(0)
        fixed = _OCR_TOKEN_REPAIRS.get(token.upper())
        return fixed if fixed is not None else token

    return _OCR_TOKEN_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Portal/product-branding aliases: the subset of the registry's aliases that
# name a bank's *online-banking product or domain* rather than just the bank
# itself. Presence of one of these is a strong, direction-unambiguous signal
# of "whose portal produced this document" -- it can never be a beneficiary
# value, since nobody prints the recipient's login portal name on a receipt.
# Classification is data about the registry's own aliases (real Malaysian
# bank-portal product names), not anything drawn from the test corpus.
# ---------------------------------------------------------------------------
_PORTAL_ALIASES = frozenset({
    "MAYBANK2U", "M2U", "M2E", "MAYBANK2E", "MAYBANK2E-ADMIN",
    "MAYBANK2U BIZ", "M2U BIZ",
    "CIMB CLICKS", "CIMBCLICKS", "BIZCHANNEL", "BIZCHANNEL@CIMB",
    "MYBUSINESSCARE", "OCTO",
    "PBEBANK", "PBE BANK", "PB ENTERPRISE", "PBENTERPRISE",
    "RHBNOW", "RHB NOW", "REFLEX", "RHBGROUP", "RHBCAMS",
    "HLBCONNECT", "CONNECTFIRST", "CONNECT FIRST",
    "AMONLINE", "AMACCESS", "AMBANK GROUP",
    "ALLIANCEONLINE", "BIZSMART", "ALLIANCEBIZSMART",
    "AFFINONLINE", "AFFINALWAYS",
    "GO BY BANK ISLAM",
    "IMUAMALAT",
    "IRAKYAT",
    "MYBSN",
    "AGRONET",
    "VELOCITY", "OCBC VELOCITY",
    "BIBPLUS",
    "CITIDIRECT",
    "STRAIGHT2BANK",
    "GCMS PLUS", "GCMS", "COMSUITE",
})


def _is_portal_alias(alias_upper: str) -> bool:
    if alias_upper in _PORTAL_ALIASES:
        return True
    # Portal product names are consistently the ones carrying a digit
    # (M2U, M2E, MAYBANK2U, ...); plain bank names never do.
    return any(ch.isdigit() for ch in alias_upper)


# ---------------------------------------------------------------------------
# Alias index built once from the shared registry: (entity, alias, compiled
# regex, is_portal) for every bank (rails excluded -- BANKS_ONLY already
# drops them).
# ---------------------------------------------------------------------------
def _alias_regex(alias_upper: str) -> "re.Pattern":
    escaped = re.escape(alias_upper).replace(r"\ ", r"\s+")
    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


def _build_alias_index() -> List[Tuple[BankEntity, str, "re.Pattern", bool]]:
    index = []
    for entity in BANKS_ONLY:
        seen = set()
        names = list(entity.aliases) + [entity.name.upper()]
        for alias in names:
            alias_u = alias.strip().upper()
            if not alias_u or alias_u in seen:
                continue
            seen.add(alias_u)
            index.append(
                (entity, alias_u, _alias_regex(alias_u), _is_portal_alias(alias_u))
            )
    return index


_ALIAS_INDEX = _build_alias_index()

# Single-word aliases, for the "<word> Reference" adjacency signal -- a
# document labelling its own reference field with its own name in it
# ("OCBC Reference No.").
_SINGLE_WORD_ALIASES: Dict[str, BankEntity] = {}
for _entity, _alias_u, _pattern, _is_portal in _ALIAS_INDEX:
    if " " not in _alias_u and "@" not in _alias_u:
        _SINGLE_WORD_ALIASES.setdefault(_alias_u, _entity)

_BIC_RE = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")

_REFERENCE_ADJACENT_RE = re.compile(
    r"\b([A-Za-z]{2,20})\s+Reference\b", re.IGNORECASE
)


def _scan_bics(text: str) -> Optional[Tuple[BankEntity, int]]:
    """First BIC-shaped token in `text` that resolves to a real bank."""
    for m in _BIC_RE.finditer(text.upper()):
        entity = lookup_by_bic(m.group(0))
        if entity is not None and not is_rail(entity):
            return entity, m.start()
    return None


# hit record: [count, first_pos, entity, max_alias_len]
_Hit = List
_Hits = Dict[str, _Hit]


def _scan_entities(text: str, portal_only: bool) -> _Hits:
    hits: _Hits = {}
    for entity, alias_u, pattern, is_portal in _ALIAS_INDEX:
        if portal_only and not is_portal:
            continue
        for m in pattern.finditer(text):
            rec = hits.get(entity.key)
            if rec is None:
                hits[entity.key] = [1, m.start(), entity, len(alias_u)]
            else:
                rec[0] += 1
                rec[1] = min(rec[1], m.start())
                rec[3] = max(rec[3], len(alias_u))
    return hits


def _rank(hits: _Hits) -> List[_Hit]:
    # Prefer the more specific alias match first (e.g. "RHB ISLAMIC" over
    # "RHB"), then the most frequently seen entity, then earliest position.
    return sorted(hits.values(), key=lambda r: (-r[3], -r[0], r[1]))


def _family_root(entity: BankEntity) -> BankEntity:
    """Collapse an Islamic subsidiary to its conventional parent.

    Used only for issuer resolution (`extract_bank_name`) -- see module
    docstring for why. `extract_beneficiary_bank` must never call this.
    """
    if entity.tier == "islamic" and entity.parent:
        parent = by_key(entity.parent)
        if parent is not None:
            return parent
    return entity


# ---------------------------------------------------------------------------
# Beneficiary-side label detection, shared by both functions.
# ---------------------------------------------------------------------------
# Matches a beneficiary-side "...Bank" label, but not when it is immediately
# followed by "Code" ("Beneficiary Bank Code" is a BIC label, handled
# separately, and must not be treated as if it were a name label).
_BENEFICIARY_LABEL_RE = re.compile(
    r"(?:BENEFICIARY|RECEIVING|RECIPIENT|PAYEE|CREDIT)(?:'S)?\s*BANK(?!\s*CODE)\b",
    re.IGNORECASE,
)
# Broader: any beneficiary-side "...Bank" label at all, including the BIC-
# label form. Used only to test "does some beneficiary marker exist
# anywhere in this document", never to anchor a value scan.
_BENEFICIARY_LABEL_ANY_RE = re.compile(
    r"(?:BENEFICIARY|RECEIVING|RECIPIENT|PAYEE|CREDIT)(?:'S)?\s*BANK\b",
    re.IGNORECASE,
)
_BENEFICIARY_BIC_LABEL_RE = re.compile(
    r"BENEFICIARY\s*BANK\s*(?:[A-Z]+\s*)?CODE\b", re.IGNORECASE
)

_DATE_STAMP_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}$")
_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
_PAGE_MARKER_RE = re.compile(r"^\d+\s*/\s*\d+$")
_DASH_RE = re.compile(r"^-+$")

_ACRONYMS = frozenset({
    "RHB", "CIMB", "HSBC", "UOB", "OCBC", "PBB", "SCB", "HLB", "AMB", "BSN",
    "BIMB", "KFH", "MBSB", "BNP", "MUFG", "SMBC", "ICBC", "CCB", "PBE",
})


def _classify_line(line_upper: str):
    """Classify one candidate line: (has_match, is_portal_noise, entity)."""
    portal_hit = False
    best = None  # (alias_len, entity)
    for entity, alias_u, pattern, is_portal in _ALIAS_INDEX:
        if not pattern.search(line_upper):
            continue
        if is_portal:
            portal_hit = True
            continue
        if best is None or len(alias_u) > best[0]:
            best = (len(alias_u), entity)
    if portal_hit:
        return True, True, None
    if best is not None:
        return True, False, best[1]
    return False, False, None


def _first_beneficiary_value_line(text: str, start: int, max_chars: int = 450):
    """Scan forward from a beneficiary-bank label for its value line.

    Skips blank lines, date stamps, URLs, page markers, placeholder dashes,
    and portal-branding lines (the issuer's own footer, which is not a
    beneficiary value even though it sits inside this window on S1). Returns
    (raw_line, matched_entity_or_None) for the first line that names a bank,
    or (None, None) if nothing turns up inside the window.
    """
    window = text[start:start + max_chars]
    for raw_line in window.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if (_DATE_STAMP_RE.match(line) or _URL_RE.match(line)
                or _PAGE_MARKER_RE.match(line) or _DASH_RE.match(line)):
            continue
        has_match, is_noise, entity = _classify_line(line.upper())
        if is_noise:
            continue
        if has_match:
            return line, entity
    return None, None


def _first_bic_after(text: str, start: int, max_chars: int = 60):
    window = text[start:start + max_chars]
    m = _BIC_RE.search(window.upper())
    if not m:
        return None
    return m.group(0)


# ---------------------------------------------------------------------------
# Generic "Bank Name"/"Bank Code" fields that never say "Beneficiary" at
# all, trusted only when a counterparty marker (Payee/Recipient/
# Beneficiary/Credit, or a bare "To" section heading) sits in the text just
# before them. Covers layouts like S20 (UOB): "To / Payee Name ... /
# Bank Name / RHB BANK".
# ---------------------------------------------------------------------------
_GENERIC_BANK_FIELD_RE = re.compile(
    r"\bBANK\s*(?:NAME|CODE)\b|\bBANK\s*:", re.IGNORECASE
)
_COUNTERPARTY_MARKER_RE = re.compile(
    r"\b(?:PAYEE|RECIPIENT|BENEFICIARY|CREDIT)\b|^[ \t]*TO[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _generic_labeled_entities(text: str) -> List[Tuple[int, BankEntity]]:
    results: List[Tuple[int, BankEntity]] = []
    for m in _GENERIC_BANK_FIELD_RE.finditer(text):
        backward = text[max(0, m.start() - 150):m.start()]
        if not _COUNTERPARTY_MARKER_RE.search(backward):
            continue
        _, entity = _first_beneficiary_value_line(text, m.end(), max_chars=200)
        if entity is not None:
            results.append((m.start(), entity))
    return results


def _has_beneficiary_marker(text: str) -> bool:
    if _BENEFICIARY_LABEL_ANY_RE.search(text):
        return True
    return bool(_generic_labeled_entities(text))


def _excluded_family_keys(text: str) -> Set[str]:
    """Family-root keys of every bank named as a beneficiary-side value.

    These are excluded from issuer candidacy outright (not down-weighted)
    everywhere in `extract_bank_name` -- tier 1 and tier 2 alike.
    """
    excluded: Set[str] = set()
    for m in _BENEFICIARY_LABEL_RE.finditer(text):
        _, entity = _first_beneficiary_value_line(text, m.end())
        if entity is not None:
            excluded.add(_family_root(entity).key)
            continue
        bic = _first_bic_after(text, m.end())
        if bic:
            entity = lookup_by_bic(bic)
            if entity is not None and not is_rail(entity):
                excluded.add(_family_root(entity).key)
    for m in _BENEFICIARY_BIC_LABEL_RE.finditer(text):
        bic = _first_bic_after(text, m.end())
        if bic:
            entity = lookup_by_bic(bic)
            if entity is not None and not is_rail(entity):
                excluded.add(_family_root(entity).key)
    for _pos, entity in _generic_labeled_entities(text):
        excluded.add(_family_root(entity).key)
    return excluded


# ---------------------------------------------------------------------------
# extract_bank_name -- the issuing bank.
# ---------------------------------------------------------------------------
def extract_bank_name(text: str, ocr_used: bool = True) -> FieldResult:
    """Identify the bank whose portal produced this receipt.

    Never returns the beneficiary/receiving bank, never returns a payment
    rail (DuitNow, GIRO, ...), and never returns "Unknown" or "Generic" as a
    literal string -- an unidentifiable bank is `FieldResult.missing()`.
    """
    if not text or not text.strip():
        return FieldResult.missing("empty_text")

    working = _repair_ocr_bank_tokens(text) if ocr_used else text
    excluded = _excluded_family_keys(working)

    lines = [ln for ln in working.splitlines() if ln.strip()]
    header_line = lines[0] if lines else ""
    footer_line = lines[-1] if lines else ""
    header_pos = working.find(header_line) if header_line else -1
    footer_pos = working.rfind(footer_line) if footer_line else -1

    strong_hits: _Hits = {}

    def _add_strong(entity: BankEntity, pos: int, alias_len: int) -> None:
        root = _family_root(entity)
        if root.key in excluded:
            return
        rec = strong_hits.get(root.key)
        if rec is None:
            strong_hits[root.key] = [1, pos, root, alias_len]
        else:
            rec[0] += 1
            rec[1] = min(rec[1], pos)
            rec[3] = max(rec[3], alias_len)

    # Header: the first non-blank line, capped so a column-scrambled or
    # backslash-joined document can't smuggle a beneficiary mention through
    # under the guise of "the first line" (see module docstring, S4).
    if header_line and len(header_line) <= 150:
        for entity, alias_u, pattern, _is_portal in _ALIAS_INDEX:
            m = pattern.search(header_line)
            if m:
                _add_strong(entity, header_pos + m.start(), len(alias_u))

    # Footer: the last non-blank line, same length cap.
    if footer_line and len(footer_line) <= 120:
        for entity, alias_u, pattern, _is_portal in _ALIAS_INDEX:
            m = pattern.search(footer_line)
            if m:
                _add_strong(entity, footer_pos + m.start(), len(alias_u))

    # Portal/product branding anywhere in the document.
    for entity, alias_u, pattern, is_portal in _ALIAS_INDEX:
        if not is_portal:
            continue
        m = pattern.search(working)
        if m:
            _add_strong(entity, m.start(), len(alias_u))

    # "<Bank> Reference No" adjacency.
    for m in _REFERENCE_ADJACENT_RE.finditer(working):
        token = m.group(1).upper()
        entity = _SINGLE_WORD_ALIASES.get(token)
        if entity is not None:
            _add_strong(entity, m.start(1), len(token))

    if strong_hits:
        ranked = _rank(strong_hits)
        best = ranked[0][2]
        candidates = [(r[2].name, float(r[0])) for r in ranked]
        return FieldResult(
            value=best.name,
            confidence=0.95,
            source="strong_signal:%s" % best.key,
            candidates=candidates,
        )

    # Tier 2: a bare bank-name or BIC mention anywhere in the body, still
    # filtered through the same beneficiary-value exclusion.
    raw_hits = _scan_entities(working, portal_only=False)
    bic_hit = _scan_bics(working)

    weak_hits: _Hits = {}
    for rec in raw_hits.values():
        entity = rec[2]
        root = _family_root(entity)
        if root.key in excluded:
            continue
        cur = weak_hits.get(root.key)
        if cur is None:
            weak_hits[root.key] = [rec[0], rec[1], root, rec[3]]
        else:
            cur[0] += rec[0]
            cur[1] = min(cur[1], rec[1])
            cur[3] = max(cur[3], rec[3])

    if bic_hit is not None:
        entity, pos = bic_hit
        root = _family_root(entity)
        if root.key not in excluded:
            cur = weak_hits.get(root.key)
            if cur is None:
                weak_hits[root.key] = [1, pos, root, len(entity.bic or "")]
            else:
                cur[0] += 1
                cur[1] = min(cur[1], pos)

    if weak_hits:
        ranked = _rank(weak_hits)
        best = ranked[0][2]
        candidates = [(r[2].name, float(r[0])) for r in ranked]
        return FieldResult(
            value=best.name,
            confidence=0.6,
            source="bare_mention:%s" % best.key,
            candidates=candidates,
        )

    return FieldResult.missing("no_bank_signal")


# ---------------------------------------------------------------------------
# extract_beneficiary_bank -- the receiving bank.
# ---------------------------------------------------------------------------
def _normalize_bank_phrase(raw: str) -> str:
    """Title-case a raw bank-name phrase while preserving known acronyms.

    "RHB ISLAMIC BANK" -> "RHB Islamic Bank". Used only as a last-resort
    fallback when a label clearly points at a value but the value does not
    resolve to any entity in the shared registry.
    """
    words = raw.split()
    out = []
    for w in words:
        core = re.sub(r"[^A-Za-z0-9]", "", w)
        if core.upper() in _ACRONYMS:
            out.append(core.upper())
        else:
            out.append(w.capitalize())
    return " ".join(out)


def extract_beneficiary_bank(text: str, ocr_used: bool = True) -> FieldResult:
    """Identify the receiving bank named on this receipt.

    Never returns the issuing bank, never returns a payment rail, never
    returns "Unknown"/"Generic" as a literal string. Unlike
    `extract_bank_name`, an Islamic subsidiary is never collapsed to its
    parent -- the Islamic/conventional distinction is exactly the
    information a beneficiary-bank field is recording.
    """
    if not text or not text.strip():
        return FieldResult.missing("empty_text")

    working = _repair_ocr_bank_tokens(text) if ocr_used else text

    code_entity = None
    code_match = _BENEFICIARY_BIC_LABEL_RE.search(working)
    if code_match:
        raw_bic = _first_bic_after(working, code_match.end())
        if raw_bic:
            code_entity = lookup_by_bic(raw_bic)

    name_entity = None
    name_line = None
    name_match = _BENEFICIARY_LABEL_RE.search(working)
    if name_match:
        name_line, name_entity = _first_beneficiary_value_line(
            working, name_match.end()
        )
        if name_entity is None:
            # The value may itself be printed as a bare SWIFT code rather
            # than a name (S25: "Bank Name RHBAMYKLXXX").
            window_bic = _first_bic_after(working, name_match.end(), max_chars=450)
            if window_bic:
                candidate = lookup_by_bic(window_bic)
                if candidate is not None and not is_rail(candidate):
                    name_entity = candidate

    if code_entity is None and name_entity is None:
        # No "Beneficiary/Receiving/.../Bank" label at all -- try a generic
        # "Bank Name"/"Bank Code" field guarded by a counterparty marker
        # nearby (S20: "To / Payee Name ... / Bank Name / RHB BANK").
        generic_hits = _generic_labeled_entities(working)
        if generic_hits:
            name_entity = generic_hits[0][1]

    candidates: List[Tuple[str, float]] = []
    if code_entity is not None:
        agree = name_entity is not None and name_entity.key == code_entity.key
        candidates.append((code_entity.name, 0.97 if agree else 0.9))
    if name_entity is not None and (
        code_entity is None or name_entity.key != code_entity.key
    ):
        candidates.append((name_entity.name, 0.85))
    elif name_entity is None and name_line and code_entity is None:
        candidates.append((_normalize_bank_phrase(name_line), 0.5))

    if candidates:
        candidates.sort(key=lambda c: -c[1])
        best_value, best_conf = candidates[0]
        if code_entity is not None and name_entity is not None and (
            code_entity.key == name_entity.key
        ):
            source = "label_code+name:%s" % code_entity.key
        elif code_entity is not None:
            source = "label_code:%s" % code_entity.key
        elif name_entity is not None:
            source = "label_name:%s" % name_entity.key
        else:
            source = "label_name:raw"
        return FieldResult(
            value=best_value, confidence=best_conf, source=source,
            candidates=candidates,
        )

    # Fallback 1: a bare BIC anywhere in the text.
    bic_hit = _scan_bics(working)
    if bic_hit is not None:
        entity, _pos = bic_hit
        return FieldResult(
            value=entity.name,
            confidence=0.7,
            source="bic_scan:%s" % entity.key,
            candidates=[(entity.name, 0.7)],
        )

    # Fallback 2: a bare bank alias, only trusted once some beneficiary-side
    # label exists somewhere in the document (otherwise there is no basis
    # for calling any particular mention "the beneficiary").
    if _has_beneficiary_marker(working):
        hits = _scan_entities(working, portal_only=False)
        if hits:
            ranked = _rank(hits)
            entity = ranked[0][2]
            candidates = [(r[2].name, float(r[0])) for r in ranked]
            return FieldResult(
                value=entity.name,
                confidence=0.55,
                source="alias_near_label:%s" % entity.key,
                candidates=candidates,
            )

    return FieldResult.missing("no_beneficiary_signal")
