"""Conservative, locally trained OCR repair for field labels, never values.

The learned dictionary contains unambiguous spelling variants of known label
words. Repairs require a complete label made of known vocabulary; arbitrary
prose, names and identifiers are not spell-corrected. No remote inference or
new runtime dependency is needed.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("label_repair_model.json")
MODEL_VERSION = "ocr-labels-v1"

LABEL_WORDS = frozenset((
    "reference ref transaction txn trx recipient customer payment advice bank "
    "service paynet duitnow channel batch import back office instruction utr "
    "uetr rpp business message simplified entry ocbc igtb scb mufg group "
    "number no id end to your other details invoice inv receipt document doc "
    "remittance remitter beneficiary information info remark debit description "
    "amount amt transfer gross net total currency myr rm in charges charge fee "
    "fees tax gst sst commission processing handling levy stamp duty sms fund "
    "date time value execution creation created approval approved authorised "
    "authorized printed print sending statement period and account from name "
    "payer payee ordering applicant credit source destination mode type on of "
    "status reason residency resident identification recipient's recipients"
).split())

# A closed label vocabulary prevents a plausible-looking spelling correction
# from opening a field in ordinary prose or in an organisation's name.
_ANCHORS = frozenset((
    "reference ref amount amt date account beneficiary recipient customer "
    "payer payee charges charge fee fees total transaction payment invoice receipt "
    "remittance debit credit name number details status"
).split())
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SEPARATOR_RE = re.compile(r"[:：﹕]")


@lru_cache(maxsize=1)
def _aliases():
    try:
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        if model.get("version") != MODEL_VERSION:
            return {}
        return model["aliases"]
    except (OSError, ValueError, KeyError):
        # Missing optional model must not take receipt processing offline.
        return {}


def _case_like(word, replacement):
    if word.lower() == replacement.lower():
        return word
    if word.isupper():
        return replacement.upper()
    if word[:1].isupper():
        return replacement.capitalize()
    return replacement


def _repair_inline_references(text, aliases):
    """Recover flattened label columns using the reference parser's boundaries.

    A tentative spelling correction is committed only inside a recognized
    reference label. A colon after that line's label marks value text, which
    must remain untouched (numeric time separators are not field separators).
    """
    from app.ibg.reference_id import _find_labels
    parts, changes, position, cursor = [], [], 0, 0
    for match in _WORD_RE.finditer(text):
        prefix = text[cursor:match.start()]
        replacement = _case_like(match[0], aliases.get(match[0].lower(), match[0]))
        parts.extend((prefix, replacement))
        position += len(prefix)
        if replacement != match[0]:
            changes.append((position, position + len(replacement), match[0]))
        position += len(replacement)
        cursor = match.end()
    parts.append(text[cursor:])
    if not changes:
        return text, []
    candidate = "".join(parts)
    hits = _find_labels(candidate)
    repairs = []
    for start, end, original in reversed(changes):
        hit = next((h for h in hits if h.start <= start and end <= h.end), None)
        line_start = candidate.rfind("\n", 0, start) + 1
        before = candidate[line_start:start]
        before = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?", "", before)
        if (hit is None or _SEPARATOR_RE.search(before)
                or candidate[hit.end:hit.end + 1] in {"-", ".", ","}):
            candidate = candidate[:start] + original + candidate[end:]
        else:
            repairs.append({"line": candidate.count("\n", 0, start) + 1,
                            "original": original, "corrected": candidate[start:end]})
    return candidate, list(reversed(repairs))


def repair_labels(text, ocr_used=True):
    """Return (text for extraction, repairs) while retaining all value text.

    Unicode separator/spacing repair also applies to digital PDFs. Learned
    spelling repair is restricted to OCR, since digital text is authoritative.
    """
    aliases = _aliases() if ocr_used else {}
    output, repairs = [], []
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        separator = _SEPARATOR_RE.search(line)
        end = separator.start() if separator else len(line.rstrip("\r\n"))
        if separator and not line[:end].strip():
            output.append(line[:end] + ":" + line[separator.end():])
            continue
        if not separator:
            # Many portals omit the colon: "Reference ID 123456789".
            # Stop at the first value token and retain its entire suffix.
            for token in _WORD_RE.finditer(line[:end]):
                word = aliases.get(token[0].lower(), token[0].lower())
                if word not in LABEL_WORDS and word not in {"s", "1", "2"}:
                    end = token.start()
                    break
        label = line[:end]
        if len(label) > 80:
            output.append(line)
            continue
        candidate = label.replace("\u00a0", " ").replace("\u2009", " ")
        candidate = candidate.replace("\u200b", "").replace("\ufeff", "")
        candidate = _WORD_RE.sub(
            lambda m: _case_like(m[0], aliases.get(m[0].lower(), m[0])),
            candidate,
        )
        if not separator and end < len(line.rstrip("\r\n")) and candidate != label:
            # A no-colon label/value row needs a separated, digit-bearing
            # value. "Arnount Trading" and "Reforence-0091" may instead be
            # customer-entered names/identifiers and must remain verbatim.
            if not label[-1:].isspace() or not re.search(r"\d", line[end:]):
                output.append(line)
                continue
        words = [w.lower() for w in _WORD_RE.findall(candidate)]
        known = all(w in LABEL_WORDS or w in {"s", "1", "2"} for w in words)
        # Whitelist punctuation too: an identifier must not be treated as a
        # label just because it happens to contain a word such as PAYMENT.
        furniture = re.sub(r"[A-Za-z0-9\s.'’`()/&*,-]", "", candidate)
        if not words or not known or furniture or not (_ANCHORS & set(words)):
            output.append(line)
            continue
        tail = line[end:]
        if separator:
            tail = ":" + line[separator.end():]
        repaired = candidate + tail
        output.append(repaired)
        if repaired != line:
            repairs.append({"line": line_number, "original": label.strip(),
                            "corrected": candidate.strip()})
    repaired = "".join(output)
    if aliases:
        repaired, inline_repairs = _repair_inline_references(repaired, aliases)
        repairs.extend(inline_repairs)
    return repaired, repairs
