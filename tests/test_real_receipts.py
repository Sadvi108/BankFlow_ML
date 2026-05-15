"""End-to-end accuracy eval over every PDF/image in Receipts/.

Iterates files in `Receipts/`, runs the full pipeline (_pdf_to_text -> patterns ->
layout -> Gemini fallback -> merger), and compares the extracted reference ID
against `tests/golden_results.json`. Prints per-field totals so we can see whether
the change pushes accuracy from 96% toward 100%.

Run:
    python tests/test_real_receipts.py
    python tests/test_real_receipts.py --no-llm     # disable Gemini
    python tests/test_real_receipts.py --limit 10   # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.enhanced_ocr_pipeline import EnhancedOCRPipeline, is_text_garbage
from app.ultimate_patterns_v3 import (
    extract_all_fields_v3,
    validate_id,
    validate_amount,
    validate_date,
)
from app.layout_aware_extractor import layout_extractor
from app.result_merger import merge as merge_results


def _load_golden(path: Path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results", []) if isinstance(data, dict) else data
    return {r["filename"]: r for r in rows if isinstance(r, dict) and r.get("filename")}


def _ocr_file(pipeline: EnhancedOCRPipeline, path: Path):
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png"}:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return {"text": "", "tokens": [], "confidence": 0.0}
        return pipeline.extract_text_with_confidence(img)

    # PDF
    try:
        import fitz
        doc = fitz.open(str(path))
        parts = [p.get_text("text") for p in doc]
        doc.close()
        embedded = " ".join(p for p in parts if p).strip()
        if embedded and not is_text_garbage(embedded):
            return {"text": embedded, "tokens": [], "confidence": 0.7}
    except Exception:
        pass
    try:
        res = pipeline.process_file(str(path))
        conf = res.get("confidence", 0)
        if conf > 1:
            conf = conf / 100.0
        return {"text": res.get("text", ""), "tokens": res.get("tokens", []), "confidence": conf}
    except Exception:
        return {"text": "", "tokens": [], "confidence": 0.0}


def _maybe_llm(path: Path, needs_llm: bool, enabled: bool):
    if not (needs_llm and enabled):
        return None
    from app.gemini_extractor import gemini_extractor
    if not gemini_extractor.available:
        return None
    data = path.read_bytes()
    if path.suffix.lower() == ".pdf":
        r = gemini_extractor.extract_from_pdf_bytes(data)
    else:
        r = gemini_extractor.extract_from_image_bytes(data)
    if not r.get("success"):
        return None
    return {
        "bank_name": r.get("bank_name") or "Unknown",
        "transaction_id": r.get("transaction_id"),
        "date": r.get("date"),
        "amount": r.get("amount"),
    }


def run(receipts_dir: Path, golden: dict, llm_enabled: bool, limit: int | None):
    pipeline = EnhancedOCRPipeline()
    files = sorted(
        p for p in receipts_dir.iterdir()
        if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}
    )
    if limit:
        files = files[:limit]

    totals = {"id": 0, "amount": 0, "date": 0, "bank": 0}
    seen = 0
    matched = 0
    started = time.time()

    for path in files:
        seen += 1
        ocr = _ocr_file(pipeline, path)
        pattern = extract_all_fields_v3(ocr.get("text", ""))
        layout = layout_extractor.extract(ocr, pattern.get("bank_name"))
        local_id = layout.get("reference_id") if layout.get("success") else pattern.get("transaction_id")
        local_bank = (
            layout.get("bank_name")
            if layout.get("success") and layout.get("bank_name") not in (None, "Unknown")
            else pattern.get("bank_name", "Unknown")
        )
        local = {
            "bank_name": local_bank,
            "transaction_id": local_id,
            "date": pattern.get("date"),
            "amount": pattern.get("amount"),
        }
        needs_llm = (
            not validate_id(local_id)
            or not validate_amount(local["amount"])
            or not validate_date(local["date"])
            or local_bank in (None, "Unknown")
        )
        llm = _maybe_llm(path, needs_llm, llm_enabled)
        merged = merge_results(local, llm)

        if validate_id(merged["transaction_id"]):
            totals["id"] += 1
        if validate_amount(merged["amount"]):
            totals["amount"] += 1
        if validate_date(merged["date"]):
            totals["date"] += 1
        if merged["bank_name"] and merged["bank_name"] != "Unknown":
            totals["bank"] += 1

        golden_row = golden.get(path.name)
        if golden_row:
            expected = (golden_row.get("expected") or "").strip().upper().replace(" ", "")
            actual = (merged["transaction_id"] or "").strip().upper().replace(" ", "")
            if expected and actual == expected:
                matched += 1

        status = "OK " if validate_id(merged["transaction_id"]) else "MISS"
        print(f"  [{status}] {path.name[:60]:<60}  id={merged['transaction_id']}  bank={merged['bank_name']}  method={merged['method']}")

    elapsed = time.time() - started
    print()
    print("=" * 80)
    print(f"  Files:      {seen}")
    print(f"  Bank:       {totals['bank']}/{seen}  ({100*totals['bank']/max(seen,1):.1f}%)")
    print(f"  Ref ID:     {totals['id']}/{seen}  ({100*totals['id']/max(seen,1):.1f}%)")
    print(f"  Amount:     {totals['amount']}/{seen}  ({100*totals['amount']/max(seen,1):.1f}%)")
    print(f"  Date:       {totals['date']}/{seen}  ({100*totals['date']/max(seen,1):.1f}%)")
    if golden:
        print(f"  vs golden:  {matched}/{sum(1 for f in files if f.name in golden)} exact ID matches")
    print(f"  LLM:        {'enabled' if llm_enabled else 'disabled'}")
    print(f"  Elapsed:    {elapsed:.1f}s")
    print("=" * 80)
    return totals, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="Disable Gemini fallback")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N files")
    ap.add_argument("--receipts", default=str(ROOT / "Receipts"))
    ap.add_argument("--golden", default=str(ROOT / "tests" / "golden_results.json"))
    args = ap.parse_args()

    receipts = Path(args.receipts)
    if not receipts.exists():
        print(f"Receipts dir not found: {receipts}")
        sys.exit(1)
    golden = _load_golden(Path(args.golden))
    run(receipts, golden, llm_enabled=not args.no_llm, limit=args.limit)


if __name__ == "__main__":
    main()
