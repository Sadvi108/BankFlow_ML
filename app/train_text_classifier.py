"""Train a simple bank classifier from OCR text using TF-IDF + LogisticRegression.

Reads dataset annotations from data/dataset/annotations.jsonl.
If ground_truth.bank_name exists, uses it as label; otherwise uses predicted bank.
Saves model and vectorizer to app/models/.
"""
from pathlib import Path
import json
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_ANN = BASE_DIR / "data" / "dataset" / "annotations.jsonl"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_samples() -> List[Dict]:
    samples: List[Dict] = []
    if not DATASET_ANN.exists():
        return samples
    with DATASET_ANN.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                samples.append(row)
            except Exception:
                continue
    return samples


def main():
    rows = load_samples()
    if not rows:
        print("No dataset annotations found at", DATASET_ANN)
        return

    texts: List[str] = []
    labels: List[str] = []
    for r in rows:
        text = r.get("ocr_text") or ""
        gt = (r.get("ground_truth") or {}).get("bank_name")
        pred = (r.get("bank") or {}).get("name") or "unknown"
        label = gt or pred
        if not text.strip():
            continue
        texts.append(text)
        labels.append(label)

    if len(texts) < 5:
        print("Not enough samples to train a model (need >=5). Got:", len(texts))
        return

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=1)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save components separately so runtime can load vectorizer and model
    tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    joblib.dump(clf, MODELS_DIR / "bank_text_clf.pkl")
    joblib.dump(tfidf, MODELS_DIR / "bank_text_vectorizer.pkl")
    print("Saved model to:", MODELS_DIR)


if __name__ == "__main__":
    main()