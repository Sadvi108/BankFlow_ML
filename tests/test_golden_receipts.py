import os
import re
import json
from pathlib import Path
import requests

RECEIPTS_DIR = Path("Receipts")
API_URL = "http://localhost:8081/extract"

def read_pdf_text(path):
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            parts = []
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
            text = " ".join(parts)
    except Exception:
        pass
    if not text or len(text.strip()) < 20:
        try:
            import fitz
            with fitz.open(path) as doc:
                parts = []
                for page in doc:
                    parts.append(page.get_text("text"))
                text = " ".join(parts)
        except Exception:
            pass
    return text or ""

def extract_expected_ref(text):
    t = text.upper()
    pats = [
        r'TRANSACTION\s+REFERENCE\s+NO\.?\s*[:\-]?\s*([A-Z0-9]{8,30})',
        r'REFERENCE\s+NO\.?\s*[:\-]?\s*([A-Z0-9]{8,30})',
        r'REF(?:ERENCE)?\s*(?:NO|ID)\.?\s*[:\-]?\s*([A-Z0-9]{8,30})',
        r'CUSTOMER\s*REF\s*NO\.?\s*[:\-]?\s*([A-Z0-9]{6,30})',
        r'RECIPIENT\s*REFERENCE\s*[:\-]?\s*([A-Z0-9]{6,30})'
    ]
    for p in pats:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            return m.group(1).strip().upper()
    return None

def normalize_id(s):
    if not s:
        return None
    return s.strip().upper().replace(" ", "").replace("'", "")

def test_one(filepath):
    raw_text = read_pdf_text(filepath)
    expected = extract_expected_ref(raw_text)
    files = None
    try:
        with open(filepath, "rb") as f:
            files = {"file": (filepath.name, f)}
            resp = requests.post(API_URL, files=files, timeout=120)
    except Exception as e:
        return {"filename": filepath.name, "success": False, "error": str(e), "expected": expected, "actual": None}
    if resp.status_code != 200:
        return {"filename": filepath.name, "success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}", "expected": expected, "actual": None}
    data = resp.json()
    if not data.get("success"):
        return {"filename": filepath.name, "success": False, "error": data.get("detail") or data.get("error") or "unknown", "expected": expected, "actual": None}
    payload = data.get("data") or data.get("extraction") or {}
    actual = payload.get("reference_number") or payload.get("transaction_id")
    ok = normalize_id(actual) == normalize_id(expected) if expected else bool(actual)
    return {
        "filename": filepath.name,
        "success": ok,
        "expected": expected,
        "actual": actual,
        "bank": payload.get("bank_name"),
        "all_ids": payload.get("all_ids", [])
    }

def main():
    results = []
    files = [p for p in sorted(RECEIPTS_DIR.glob("*.*")) if p.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png"]]
    passed = 0
    failed = 0
    for fp in files:
        r = test_one(fp)
        results.append(r)
        if r["success"]:
            passed += 1
            print(f"PASS: {fp.name} -> {r.get('actual')}")
        else:
            failed += 1
            print(f"FAIL: {fp.name} -> expected={r.get('expected')} actual={r.get('actual')} error={r.get('error')}")
    summary = {
        "total": len(files),
        "passed": passed,
        "failed": failed,
        "accuracy": (passed / len(files) * 100) if files else 0,
        "results": results
    }
    out = "golden_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out}")

if __name__ == "__main__":
    main()
