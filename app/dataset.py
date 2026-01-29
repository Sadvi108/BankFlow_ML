from pathlib import Path
from typing import Dict, List, Optional
import json


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_PATH = DATASET_DIR / "annotations.jsonl"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not ANNOTATIONS_PATH.exists():
        ANNOTATIONS_PATH.write_text("")


def append_annotation(entry: Dict) -> None:
    ensure_dirs()
    with ANNOTATIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_annotations() -> List[Dict]:
    ensure_dirs()
    rows: List[Dict] = []
    with ANNOTATIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def update_annotation(item_id: str, updates: Dict) -> bool:
    """Update an annotation by id in-place (rewrite JSONL)."""
    ensure_dirs()
    rows = read_annotations()
    changed = False
    for r in rows:
        if r.get("id") == item_id:
            r.update(updates)
            changed = True
            break
    if changed:
        with ANNOTATIONS_PATH.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return changed


def summary() -> Dict:
    rows = read_annotations()
    total = len(rows)
    banks = {}
    for r in rows:
        b = (r.get("bank", {}) or {}).get("name")
        if b:
            banks[b] = banks.get(b, 0) + 1
    return {"total": total, "per_bank": banks}