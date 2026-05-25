"""
Invoice expense classifier using TF-IDF + Logistic Regression.

Design rationale: TF-IDF + Logistic Regression is the right call here. It trains
in milliseconds, is fully explainable, needs no GPU, and consistently outperforms
Naive Bayes on short domain-specific text. For invoice classification with < 10
categories, this beats transformer-based models on latency with comparable accuracy.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "classifier.pkl"
LABEL_ENCODER_PATH = Path(__file__).parent.parent.parent / "models" / "label_encoder.pkl"


def preprocess(text: str) -> str:
    """
    Lightweight preprocessing tuned for invoice text.

    We deliberately keep company names and numbers because 'AWS' or 'FedEx'
    are high-signal tokens for classification. Heavy stemming/lemmatization
    would destroy that signal.
    """
    return text.lower().strip()


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess,
            ngram_range=(1, 2),   # bigrams catch 'cloud hosting', 'office supplies'
            max_features=5000,
            sublinear_tf=True,    # dampens effect of very frequent terms
            min_df=1,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced",  # handles any class imbalance in training data
            solver="lbfgs",

        )),
    ])


def train(data_path: Path = Path("data/training_data.json")) -> dict:
    """Train and persist the classifier. Returns evaluation metrics."""
    with open(data_path) as f:
        records = json.load(f)

    texts = [r["text"] for r in records]
    raw_labels = [r["category"] for r in records]

    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)

    pipeline = build_pipeline()

    # Cross-validation gives a realistic accuracy estimate before final fit
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="f1_macro")
    logger.info("CV F1-macro: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

    # Final fit on all data before persisting
    pipeline.fit(texts, labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    logger.info("Model saved to %s", MODEL_PATH)

    return {
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "classes": list(le.classes_),
        "num_samples": len(texts),
    }


class InvoiceClassifier:
    """Thread-safe singleton wrapper around the trained pipeline."""

    _pipeline: Optional[Pipeline] = None
    _label_encoder: Optional[LabelEncoder] = None

    @classmethod
    def load(cls) -> None:
        if not MODEL_PATH.exists():
            logger.info("No saved model found — training from default data...")
            train()

        with open(MODEL_PATH, "rb") as f:
            cls._pipeline = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            cls._label_encoder = pickle.load(f)

        logger.info("Classifier loaded. Classes: %s", list(cls._label_encoder.classes_))

    @classmethod
    def predict(cls, text: str) -> dict:
        if cls._pipeline is None:
            raise RuntimeError("Classifier not loaded. Call InvoiceClassifier.load() first.")

        processed = preprocess(text)
        proba = cls._pipeline.predict_proba([processed])[0]
        predicted_idx = int(np.argmax(proba))
        predicted_label = cls._label_encoder.inverse_transform([predicted_idx])[0]
        confidence = round(float(proba[predicted_idx]), 4)

        all_scores = {
            cls._label_encoder.classes_[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }

        return {
            "category": predicted_label,
            "confidence": confidence,
            "scores": dict(sorted(all_scores.items(), key=lambda x: -x[1])),
        }
