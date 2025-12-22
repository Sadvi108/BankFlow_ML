import re
from typing import Tuple

try:
    import joblib
    import os
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False


BANK_KEYWORDS = {
    "Maybank": [r"\bMaybank\b", r"\bMBB\b"],
    "CIMB": [r"\bCIMB\b"],
    "Public Bank": [r"\bPublic Bank\b", r"\bPBE\b"],
    "RHB": [r"\bRHB\b"],
    "Hong Leong": [r"\bHong\s+Leong\b", r"\bHLB\b"],
    "AmBank": [r"\bAmBank\b"],
    "Bank Islam": [r"\bBank\s+Islam\b"],
    "BSN": [r"\bBSN\b", r"\bBank\s+Simpanan\s+Nasional\b"],
    "Affin Bank": [r"\bAffin\s+Bank\b", r"\bAFFIN\b"],
    "Citibank": [r"\bCitibank\b", r"\bCiti\b"],
    "HSBC": [r"\bHSBC\b"],
    "UOB": [r"\bUOB\b", r"\bUnited\s+Overseas\s+Bank\b"],
    "Standard Chartered": [r"\bStandard\s+Chartered\b", r"\bSCB\b"],
    "DuitNow": [r"\bDuitNow\b"],
}


_model = None
_vectorizer = None

def _load_model():
    global _model, _vectorizer
    if not MODEL_AVAILABLE:
        return False
    try:
        base = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(base, "bank_text_clf.pkl")
        vec_path = os.path.join(base, "bank_text_vectorizer.pkl")
        if os.path.exists(model_path) and os.path.exists(vec_path):
            _model = joblib.load(model_path)
            _vectorizer = joblib.load(vec_path)
            return True
    except Exception:
        pass
    return False


def bank_from_text(text: str) -> Tuple[str, float]:
    """Classify bank.

    Prefers trained text model if available; otherwise falls back to keyword matching.
    Returns (bank_name, confidence in [0,1]).
    """
    if not text:
        return "unknown", 0.0

    # Try model-based classification
    if _model is None or _vectorizer is None:
        _load_model()
    if _model is not None and _vectorizer is not None:
        try:
            X = _vectorizer.transform([text])
            proba = _model.predict_proba(X)[0]
            classes = list(_model.classes_)
            idx = int(proba.argmax())
            bank = classes[idx]
            conf = float(proba[idx])
            return bank, conf
        except Exception:
            pass

    # Fallback: keyword-based
    text_lower = text
    scores = {}
    for bank, patterns in BANK_KEYWORDS.items():
        hits = 0
        for pat in patterns:
            if re.search(pat, text_lower, flags=re.IGNORECASE):
                hits += 1
        scores[bank] = hits

    if not any(scores.values()):
        return "unknown", 0.0

    bank = max(scores.items(), key=lambda kv: kv[1])[0]
    best_hits = scores[bank]
    confidence = min(1.0, best_hits / (len(BANK_KEYWORDS.get(bank, [])) or 1))
    return bank, confidence