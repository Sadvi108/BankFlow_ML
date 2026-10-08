#!/usr/bin/env python3
"""Reproducible labeled extraction benchmark, including Git branch snapshots.

    python scripts/benchmark_extraction.py --branches --output docs/receipt_benchmark.json

Exports only app source to temporary directories to compare branch code in
isolated Python processes. Real customer receipts are never exported/uploaded.
Reports exact core-field accuracy on development and holdout layouts, then
deterministic synthetic OCR-label damage. These are bounded benchmark results,
not a claim of accuracy on arbitrary customer documents.
"""
import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("reference_id", "bank_name", "transaction_date", "amount")
MUTATIONS = (
    ("reference_ocr", "Reference", "Reforence"),
    ("transaction_ocr", "Transaction", "Transactlon"),
    ("amount_ocr", "Amount", "Arnount"),
    ("date_ocr", "Date", "Oate"),
)


def evaluate(source_dir):
    sys.path.insert(0, str(ROOT))
    from tests.fixtures.ibg_corpus import CORPUS
    from tests.fixtures.ibg_holdout import HOLDOUT
    if source_dir != ROOT:
        # Fixtures import current role constants; discard those app modules
        # before importing the branch so dependencies come from its snapshot.
        for name in list(sys.modules):
            if name == "app" or name.startswith("app."):
                del sys.modules[name]
    sys.path.insert(0, str(source_dir))
    try:
        from app.ibg.extractor import extract_ibg_fields
        def extract(text, ocr_used):
            result = extract_ibg_fields(text, ocr_used=ocr_used)
            return {field: result[field]["value"] for field in FIELDS}
        engine = "ibg"
    except ModuleNotFoundError as exc:
        if exc.name not in {"app.ibg", "app.ibg.extractor"}:
            raise
        from app.ultimate_patterns_v3 import extract_all_fields_v3
        def extract(text, ocr_used):
            result = extract_all_fields_v3(text)
            return dict(reference_id=result.get("transaction_id"),
                        bank_name=result.get("bank_name"),
                        transaction_date=result.get("date"), amount=result.get("amount"))
        engine = "legacy_v3"

    def score(samples, mutation=None):
        passed, total, failures = 0, 0, []
        for sample in samples:
            text = sample["text"]
            if mutation:
                text = mutation(text)
            result = extract(text, True if mutation else sample["ocr_used"])
            for field in FIELDS:
                if field not in sample["expected"]:
                    continue
                total += 1
                if result[field] == sample["expected"][field]:
                    passed += 1
                else:
                    failures.append({"sample": sample["id"], "field": field})
        return {"passed": passed, "total": total,
                "accuracy_percent": round(100 * passed / total, 2),
                "failure_count": len(failures), "failures": failures[:10]}

    stress = {}
    for name, word, damaged in MUTATIONS:
        stress[name] = score(CORPUS + HOLDOUT, lambda text, w=word, d=damaged:
                             re.sub(w, d, text, flags=re.IGNORECASE))
    # Damage only label colons. Time stamps, URLs, identifiers and values keep
    # their original punctuation, matching the production integrity contract.
    stress["fullwidth_label_colon"] = score(CORPUS + HOLDOUT, lambda text:
        re.sub(r"(?m)^([A-Za-z][A-Za-z0-9 .()/’`'&*,-]{0,75}):", r"\1：", text))
    passed = sum(s["passed"] for s in stress.values())
    total = sum(s["total"] for s in stress.values())
    return {"engine": engine, "training_layouts": score(CORPUS),
            "holdout_layouts": score(HOLDOUT), "ocr_stress": stress,
            "stress_overall": {"passed": passed, "total": total,
                               "accuracy_percent": round(100 * passed / total, 2)}}


def snapshot_score(ref):
    with tempfile.TemporaryDirectory(prefix="bankflow-benchmark-") as directory:
        archive = subprocess.check_output(["git", "archive", ref, "app"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            for member in files.getmembers():
                if member.isfile():
                    destination = Path(directory) / member.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(files.extractfile(member).read())
        raw = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
                                       "--source-dir", directory], cwd=directory)
        return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT)
    parser.add_argument("--branches", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"current": evaluate(args.source_dir)}
    if args.branches:
        refs = subprocess.check_output(["git", "for-each-ref", "--format=%(refname)",
                                        "refs/remotes/origin"], cwd=ROOT, text=True).splitlines()
        report["branches"] = {}
        for ref in refs:
            ref = ref.removeprefix("refs/remotes/")
            if ref == "origin/HEAD":
                continue
            report["branches"][ref] = snapshot_score(ref)["current"]
            print("Compared", ref, file=sys.stderr)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
        print("Saved benchmark to", args.output)
        for name, result in {"current": report["current"], **report.get("branches", {})}.items():
            print(name, "clean", result["training_layouts"]["passed"] + result["holdout_layouts"]["passed"],
                  "stress", result["stress_overall"])
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
