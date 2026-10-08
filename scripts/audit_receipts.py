#!/usr/bin/env python
"""Run the extractor over a folder of real receipts and report what it gets wrong.

There are no hand-written labels for these files, so this cannot measure
accuracy. It measures the two things that need no ground truth:

  * coverage  -- did the field come back at all?
  * sanity    -- is the value the *shape* the field should be?

A reference that is prose, an amount that is an account number, a payer that is
actually a form label: all detectable without knowing the right answer. Those
are the bugs worth fixing, and they are exactly the class the owner reported.

    .venv/bin/python scripts/audit_receipts.py Receipts/
    .venv/bin/python scripts/audit_receipts.py Receipts/ --show-suspects
"""
from __future__ import print_function

import os
import re
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ibg.extractor import extract_ibg_fields  # noqa: E402

SCALARS = ("bank_name", "reference_id", "transaction_date", "amount",
           "payer", "beneficiary", "beneficiary_bank", "fee", "total_debit")

# A value shaped wrong for its field. Ground truth is not needed to see these.
_LABELISH = re.compile(
    r"^(?:transaction|transfer|payment|beneficiary|recipient|payee|payer|"
    r"account|bank|debit|credit|value|applicant|ordering|customer|source|"
    r"destination|instruction|additional|reference|service|product|status|"
    r"charges?|fee|amount|currency|date|details?|mode|type|name|number|no|"
    r"total|remarks?|purpose)[\s.:/'-]*$", re.IGNORECASE)
_MONEYISH = re.compile(r"\b(?:RM|MYR)\s*[\d,]+\.\d{2}", re.IGNORECASE)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONEY_OUT = re.compile(r"^\d+\.\d{2}$")


def read_text(path):
    """Embedded text first, then OCR. Mirrors what the server does."""
    from simple_server import _pdf_to_text, ocr_pipeline
    lower = path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg")):
        try:
            import cv2
            image = cv2.imread(path)
            result = ocr_pipeline.extract_text_with_confidence(image)
            return result.get("text", ""), "ocr"
        except Exception:
            return "", "none"
    try:
        result = _pdf_to_text(path)
        return result.get("text", ""), result.get("source", "none")
    except Exception:
        pass
    return "", "none"


def suspects_for(field, value):
    """Reasons `value` looks wrong for `field`. Empty list = looks fine."""
    if value is None:
        return []
    out = []
    v = str(value).strip()
    if _LABELISH.match(v):
        out.append("looks like a form label")
    if field in ("reference_id",) and _MONEYISH.search(v):
        out.append("contains a currency amount")
    if field in ("reference_id", "payer", "beneficiary") and len(v) > 60:
        out.append("implausibly long (%d chars)" % len(v))
    if field == "transaction_date" and not _ISO_DATE.match(v):
        out.append("not ISO yyyy-mm-dd")
    if field in ("amount", "fee", "total_debit") and not _MONEY_OUT.match(v):
        out.append("not a bare 2-decimal amount")
    if field == "reference_id" and v.islower():
        out.append("all lowercase (prose?)")
    if field in ("payer", "beneficiary") and not re.search(r"[A-Za-z]{3}", v):
        out.append("no word characters")
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show-suspects" in sys.argv
    root = args[0] if args else "Receipts"
    cache_path = (Path(root).resolve().parent / "logs" / "receipt_audit_cache.json"
                  if "--cache" in sys.argv else None)
    cache = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    files = []
    for base, _dirs, names in os.walk(root):
        for n in sorted(names):
            if n.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                files.append(os.path.join(base, n))
    if not files:
        print("No receipts under %s" % root)
        return 1

    found = Counter()
    sources = Counter()
    no_text = []
    suspect_rows = []
    per_bank = defaultdict(lambda: Counter())
    missing_rows = []
    total = len(files)

    for i, path in enumerate(files, 1):
        sys.stderr.write("\r  %d/%d" % (i, total))
        sys.stderr.flush()
        stat = os.stat(path)
        key = "%s:%s:%s" % (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
        if key in cache:
            text, how = cache[key]
        else:
            text, how = read_text(path)
            if cache_path:
                cache[key] = [text, how]
                cache_path.parent.mkdir(exist_ok=True)
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
        sources[how] += 1
        if len(text.strip()) < 40:
            no_text.append(path)
            continue
        try:
            res = extract_ibg_fields(text, ocr_used=how not in (
                "embedded", "embedded_partial", "pdfplumber", "pdf_text_extraction"))
        except Exception as exc:                       # noqa: BLE001
            suspect_rows.append((path, "EXTRACTOR", "raised %s" % type(exc).__name__, ""))
            continue

        bank = res["bank_name"]["value"] or "(unidentified)"
        per_bank[bank]["files"] += 1
        for f in SCALARS:
            v = res[f]["value"]
            if v is not None:
                found[f] += 1
                per_bank[bank][f] += 1
            for reason in suspects_for(f, v):
                suspect_rows.append((path, f, reason, v))
        found["_refs"] += res["reference_count"]
        if res["needs_review"]:
            found["_review"] += 1
        absent = [f for f in ("reference_id", "transaction_date", "amount")
                  if res[f]["value"] is None]
        if absent:
            missing_rows.append((os.path.basename(path), absent))

    sys.stderr.write("\r" + " " * 24 + "\r")
    usable = total - len(no_text)

    print("=" * 74)
    print("RECEIPT AUDIT  --  %d files, %d yielded text" % (total, usable))
    print("=" * 74)
    print("text source: " + ", ".join("%s=%d" % kv for kv in sources.most_common()))
    if no_text:
        print("\nNO TEXT EXTRACTED (%d) -- these fail before any field is read:" % len(no_text))
        for p in no_text[:12]:
            print("   %s" % os.path.basename(p)[:70])
        if len(no_text) > 12:
            print("   ... and %d more" % (len(no_text) - 12))

    print("\nFIELD COVERAGE (of %d readable receipts)" % usable)
    for f in SCALARS:
        pct = 100.0 * found[f] / usable if usable else 0
        bar = "#" * int(pct / 4)
        print("  %-18s %4d  %5.1f%%  %s" % (f, found[f], pct, bar))
    print("\n  references captured : %d (avg %.1f per receipt)"
          % (found["_refs"], found["_refs"] / float(usable or 1)))
    print("  flagged for review  : %d (%.0f%%)"
          % (found["_review"], 100.0 * found["_review"] / (usable or 1)))

    print("\nPER-BANK (files | ref | date | amount | payer)")
    for bank, c in sorted(per_bank.items(), key=lambda kv: -kv[1]["files"]):
        print("  %-26s %4d | %3d | %3d | %3d | %3d"
              % (bank[:25], c["files"], c["reference_id"],
                 c["transaction_date"], c["amount"], c["payer"]))

    print("\nSUSPECT VALUES: %d" % len(suspect_rows))
    by_reason = Counter(r for _p, _f, r, _v in suspect_rows)
    for reason, n in by_reason.most_common():
        print("  %-34s %d" % (reason, n))
    if show and suspect_rows:
        print("\n  detail:")
        for p, f, reason, v in suspect_rows[:60]:
            print("   %-34s %-16s %-28s %r"
                  % (os.path.basename(p)[:33], f, reason[:27], str(v)[:34]))
    elif suspect_rows:
        print("\n  (re-run with --show-suspects for the per-file detail)")
    if "--show-missing" in sys.argv:
        print("\nMISSING CORE FIELDS (absence may be intentional):")
        for name, fields in missing_rows:
            print("  %s: %s" % (name, ", ".join(fields)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
