#!/usr/bin/env python
"""Score the IBG extractors against the ground-truth corpus.

One number, honestly derived, that says how good extraction currently is.

    .venv/bin/python scripts/score_corpus.py            # scalar fields
    .venv/bin/python scripts/score_corpus.py --refs     # + multi-reference
    .venv/bin/python scripts/score_corpus.py --legacy   # the live V3 engine too

A cell is `ok` only when the extractor returns exactly the corpus expectation.
`TRAP` means it returned a value the corpus explicitly lists as a known false
positive for that field -- worse than a plain miss, because a trap value is
plausible enough to be trusted downstream.
"""
from __future__ import print_function

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.ibg_corpus import CORPUS  # noqa: E402


def _call(fn, sample, **kw):
    """Call an extractor, tolerating a module that hasn't grown the kwarg yet."""
    if fn is None:
        return "-"
    try:
        return fn(sample["text"], ocr_used=sample["ocr_used"], **kw).value
    except TypeError:
        try:
            return fn(sample["text"], ocr_used=sample["ocr_used"]).value
        except Exception as exc:                      # noqa: BLE001
            return "ERR:%s" % type(exc).__name__
    except Exception as exc:                          # noqa: BLE001
        return "ERR:%s" % type(exc).__name__


def _load(module_path, func_name):
    """Import one extractor, returning None if it isn't written yet."""
    try:
        module = __import__(module_path, fromlist=[func_name])
        return getattr(module, func_name, None)
    except Exception:                                 # noqa: BLE001
        return None


def score_fields(fields, verbose=True):
    """Score scalar fields. Returns (passed, total, failures)."""
    totals = dict((name, [0, 0]) for name, _ in fields)
    failures = []
    rows = []

    for sample in CORPUS:
        row = [sample["id"][:26]]
        for name, fn in fields:
            expected = sample["expected"].get(name)
            traps = [str(t) for t in sample.get("traps", {}).get(name, [])]
            got = _call(fn, sample)
            ok = str(got) == str(expected)
            is_trap = got is not None and str(got) in traps
            totals[name][1] += 1
            if ok:
                totals[name][0] += 1
            else:
                failures.append((sample["id"], name, expected, got, is_trap))
            row.append("ok" if ok else ("TRAP" if is_trap else "X"))
        rows.append(row)

    if verbose:
        head = "%-27s" + " %-6s" * len(fields)
        print(head % tuple(["sample"] + [n[:6] for n, _ in fields]))
        print("-" * (27 + 7 * len(fields)))
        for row in rows:
            print(head % tuple(row))
        print("-" * (27 + 7 * len(fields)))
        print(head % tuple(
            ["PASS RATE"] +
            ["%d/%d" % (totals[n][0], totals[n][1]) for n, _ in fields]))

    passed = sum(totals[n][0] for n, _ in fields)
    total = sum(totals[n][1] for n, _ in fields)
    return passed, total, failures


def score_references(verbose=True):
    """Score multi-reference capture as a set of (value, role) pairs."""
    extract_references = _load("app.ibg.reference_id", "extract_references")
    if extract_references is None:
        print("\nextract_references() not implemented yet -- skipping.")
        return 0, 0, []

    exact = 0
    total_expected = 0
    total_found = 0
    misses = []

    print("\n%-27s %-9s %-9s %s" % ("sample", "expected", "found", "verdict"))
    print("-" * 62)
    for sample in CORPUS:
        want = set(
            (str(v), str(r)) for v, r in sample["expected"].get("references", [])
        )
        try:
            refs = extract_references(sample["text"], ocr_used=sample["ocr_used"])
            got = set((str(r.value), str(r.role)) for r in refs)
        except Exception as exc:                      # noqa: BLE001
            print("%-27s %-9d %-9s ERR:%s"
                  % (sample["id"][:26], len(want), "-", type(exc).__name__))
            misses.append((sample["id"], want, set()))
            total_expected += len(want)
            continue

        total_expected += len(want)
        total_found += len(want & got)
        verdict = "ok" if want == got else "partial" if (want & got) else "MISS"
        if want != got:
            misses.append((sample["id"], want, got))
        else:
            exact += 1
        print("%-27s %-9d %-9d %s"
              % (sample["id"][:26], len(want), len(got), verdict))

    print("-" * 62)
    print("exact-set samples : %d/%d" % (exact, len(CORPUS)))
    print("references found  : %d/%d" % (total_found, total_expected))

    if verbose and misses:
        print("\nPer-sample differences:")
        for sample_id, want, got in misses:
            for item in sorted(want - got):
                print("  %-27s MISSING  %s" % (sample_id[:26], item))
            for item in sorted(got - want):
                print("  %-27s EXTRA    %s" % (sample_id[:26], item))

    return total_found, total_expected, misses


def main():
    args = sys.argv[1:]

    fields = [
        ("reference_id", _load("app.ibg.reference_id", "extract_reference_id")),
        ("bank_name", _load("app.ibg.bank_name", "extract_bank_name")),
        ("transaction_date",
         _load("app.ibg.transaction_date", "extract_transaction_date")),
        ("amount", _load("app.ibg.amount", "extract_amount")),
    ]
    # These only exist once the amount developer has split them out.
    for name, func in (("fee", "extract_fee"),
                       ("total_debit", "extract_total_debit")):
        fn = _load("app.ibg.amount", func)
        if fn is not None:
            fields.append((name, fn))

    print("=" * 62)
    print("app/ibg/ extractors vs corpus (%d samples)" % len(CORPUS))
    print("=" * 62)
    passed, total, failures = score_fields(fields)
    print("\nOVERALL: %d/%d field checks (%.0f%%)"
          % (passed, total, 100.0 * passed / total if total else 0))

    if failures:
        print("\nFailures:")
        for sample_id, field, expected, got, is_trap in failures:
            print("  %-27s %-17s expected=%-14r got=%-14r%s"
                  % (sample_id[:26], field, expected, got,
                     "  <-- TRAP" if is_trap else ""))

    if "--refs" in args:
        score_references()

    if "--legacy" in args:
        from app.ultimate_patterns_v3 import extract_all_fields_v3
        print("\n" + "=" * 62)
        print("legacy extract_all_fields_v3 (the live production path)")
        print("=" * 62)
        print("%-27s %-12s %-12s %s" % ("sample", "amount", "expected", "conf"))
        print("-" * 62)
        wrong = 0
        for sample in CORPUS:
            res = extract_all_fields_v3(sample["text"],
                                        ocr_used=sample["ocr_used"])
            expected = sample["expected"].get("amount")
            got = res.get("amount")
            if str(got) != str(expected):
                wrong += 1
            print("%-27s %-12s %-12s %s%s"
                  % (sample["id"][:26], got, expected, res.get("confidence"),
                     "" if str(got) == str(expected) else "   <-- WRONG"))
        print("-" * 62)
        print("legacy amount wrong on %d/%d samples" % (wrong, len(CORPUS)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
