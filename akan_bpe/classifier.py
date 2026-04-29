"""ML-based classifier for Akan text domain detection."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DOMAIN_ASR = "asr"
DOMAIN_TTS = "tts"


def extract_features(text: str) -> dict[str, float]:
    """Extract statistical features from text for classification."""
    if not text:
        return {
            "avg_word_len": 0.0,
            "word_count": 0,
            "char_count": 0,
            "punct_ratio": 0.0,
            "upper_ratio": 0.0,
            "digit_ratio": 0.0,
            "quote_count": 0,
            "question_marks": 0,
            "exclamation_marks": 0,
        }

    words = text.split()
    word_count = len(words)
    char_count = len(text)

    return {
        "avg_word_len": sum(len(w) for w in words) / word_count if word_count else 0.0,
        "word_count": word_count,
        "char_count": char_count,
        "punct_ratio": sum(1 for c in text if c in ".,;:!?-()[]{}") / char_count
        if char_count
        else 0.0,
        "upper_ratio": sum(1 for c in text if c.isupper()) / char_count if char_count else 0.0,
        "digit_ratio": sum(1 for c in text if c.isdigit()) / char_count if char_count else 0.0,
        "quote_count": text.count('"') + text.count("'") + text.count("''"),
        "question_marks": text.count("?"),
        "exclamation_marks": text.count("!"),
    }


def load_training_data(asr_path: str, tts_path: str) -> tuple[list[str], list[int]]:
    """Load and combine ASR and TTS training data."""
    texts = []
    labels = []

    with open(asr_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text") or data.get("transcription", "")
            if text.strip():
                texts.append(text)
                labels.append(0)

    with open(tts_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            if text.strip():
                texts.append(text)
                labels.append(1)

    return texts, labels


def train_classifier(
    texts: list[str],
    labels: list[int],
    model_type: str = "logreg",
    vectorizer_type: str = "tfidf",
) -> Pipeline:
    """Train a classifier pipeline."""
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
        )
    else:
        raise ValueError(f"Unknown vectorizer: {vectorizer_type}")

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", model)])
    pipeline.fit(texts, labels)
    return pipeline


def save_classifier(pipeline: Pipeline, path: Path) -> None:
    """Save trained classifier to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)


def load_classifier(path: Path) -> Pipeline:
    """Load trained classifier from file."""
    with open(path, "rb") as f:
        return pickle.load(f)


class MLClassifierRouter:
    """ML-based router using trained classifier."""

    def __init__(self, classifier_path: str | None = None):
        self.classifier = None
        self.classifier_path = classifier_path
        if classifier_path and Path(classifier_path).exists():
            self.classifier = load_classifier(Path(classifier_path))

    def train(self, asr_path: str, tts_path: str, output_path: str) -> dict[str, Any]:
        """Train the classifier and save to file."""
        texts, labels = load_training_data(asr_path, tts_path)
        self.classifier = train_classifier(texts, labels)
        save_classifier(self.classifier, Path(output_path))

        train_accuracy = self.classifier.score(texts, labels)
        return {
            "asr_samples": sum(1 for l in labels if l == 0),
            "tts_samples": sum(1 for l in labels if l == 1),
            "total_samples": len(labels),
            "train_accuracy": train_accuracy,
            "output_path": output_path,
        }

    def predict(self, text: str) -> tuple[str, float]:
        """Predict domain for text. Returns (domain, confidence)."""
        if self.classifier is None:
            raise ValueError("Classifier not trained or loaded")

        pred = self.classifier.predict([text])[0]
        proba = self.classifier.predict_proba([text])[0]

        domain = DOMAIN_ASR if pred == 0 else DOMAIN_TTS
        confidence = float(max(proba))

        return domain, confidence

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Predict domains for batch of texts."""
        if self.classifier is None:
            raise ValueError("Classifier not trained or loaded")

        predictions = self.classifier.predict(texts)
        probabilities = self.classifier.predict_proba(texts)

        results = []
        for pred, proba in zip(predictions, probabilities):
            domain = DOMAIN_ASR if pred == 0 else DOMAIN_TTS
            confidence = float(max(proba))
            results.append((domain, confidence))

        return results
