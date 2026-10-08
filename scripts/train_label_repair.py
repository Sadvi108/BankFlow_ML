#!/usr/bin/env python3
"""Train a deterministic OCR label vocabulary from labeled training receipts.

Only label words are exported, never receipt identifiers or customer data.
Single-edit OCR augmentations teach a small lookup model; conflicting aliases
are rejected rather than assigned the most frequent label. Holdout receipts
are never read by this trainer. Reproduce: python scripts/train_label_repair.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ibg.label_repair import LABEL_WORDS, MODEL_PATH, MODEL_VERSION
from tests.fixtures.ibg_corpus import CORPUS

CONFUSIONS = {
    # m -> ni was observed in "Custonier" on a local scanned advice.
    "m": ("rn", "ni"), "n": ("ri",), "i": ("l", "1", "!"),
    "l": ("i", "1"), "o": ("0",), "e": ("c", "o"),
    "c": ("e",), "d": ("o", "cl"), "t": ("l", "7"), "s": ("5",),
}


def variants(word):
    # Short words are too easy to confuse with legitimate acronyms/values.
    if len(word) < 4 or not word.isalpha():
        return set()
    found = set()
    for i, char in enumerate(word):
        for replacement in CONFUSIONS.get(char, ()):
            found.add(word[:i] + replacement + word[i + 1:])
        if len(word) >= 6:
            found.add(word[:i] + word[i + 1:])
            found.add(word[:i] + char + word[i:])
            if i + 1 < len(word):
                found.add(word[:i] + word[i + 1] + char + word[i + 2:])
    return found - {word}


def train():
    # Train only words supported by this extractor and occurring in its
    # labeled development corpus. Customer values are not model features.
    training_text = "\n".join(s["text"].lower() for s in CORPUS)
    vocabulary = sorted(w for w in LABEL_WORDS if w.isalpha()
                        and w in training_text and len(w) >= 4)
    candidates = defaultdict(set)
    for word in vocabulary:
        for alias in variants(word):
            if alias not in LABEL_WORDS:
                candidates[alias].add(word)
    aliases = {alias: next(iter(targets)) for alias, targets in candidates.items()
               if len(targets) == 1}
    return {
        "version": MODEL_VERSION,
        "training": {"receipts": len(CORPUS), "canonical_words": len(vocabulary),
                     "augmented_examples": sum(len(v) for v in candidates.values()),
                     "ambiguous_variants_rejected": sum(len(v) > 1 for v in candidates.values()),
                     "method": "unique supervised OCR spelling aliases"},
        "aliases": dict(sorted(aliases.items())),
    }


if __name__ == "__main__":
    model = train()
    MODEL_PATH.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(json.dumps(model["training"], sort_keys=True))
    print("Saved", len(model["aliases"]), "aliases to", MODEL_PATH)
