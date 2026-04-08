"""Lightweight stream classifier for dual-stream tokenization.

Classifies input text as "robust" (ASR-optimized) or "logic" (TTS-optimized).
Uses a trained logistic regression classifier when a model file is available,
falling back to a regex heuristic for zero-dependency environments.

Designed for CPU-efficient inference on edge devices.
"""

import re
import pickle
from pathlib import Path
from typing import ClassVar


class WAXALRouter:
    """Stream classifier for dual-stream tokenization.

    Routes input to the appropriate tokenizer core. Loads a trained
    classifier from disk when available; falls back to a regex heuristic
    if the model file is not found.

    Args:
        language: Target language (e.g. 'akan').
        model_dir: Directory containing trained router .pkl files.
                   Defaults to 'models/router/' relative to cwd.

    Example:
        >>> router = WAXALRouter(language="akan")
        >>> router.classify("uhm chale me dwo")
        'robust'
        >>> router.classify("The formal declaration states...")
        'logic'
    """

    _FALLBACK_MARKERS: ClassVar[list[str]] = [
        r"\buhm\b",
        r"\berr\b",
        r"\bchale\b",
        r"\bnaa\b",
        r"\beh\b",
        r"\buna\b",
    ]

    def __init__(self, language: str = "akan", model_dir: str | Path = "models/router/"):
        self.language = language
        self._classifier = None
        self._compiled_markers = [
            re.compile(m, re.IGNORECASE) for m in self._FALLBACK_MARKERS
        ]

        model_path = Path(model_dir) / f"{language}_router.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self._classifier = pickle.load(f)
        else:
            print(
                f"[WAXALRouter] No classifier found at {model_path}. "
                "Using regex heuristic fallback. "
                "Run scripts/train_router.py to train a classifier."
            )

    @property
    def using_classifier(self) -> bool:
        """True if a trained classifier is loaded."""
        return self._classifier is not None

    def classify(self, text: str) -> str:
        """Classify text as 'robust' (ASR) or 'logic' (TTS).

        Args:
            text: Input text to classify.

        Returns:
            'robust' for ASR-optimized stream, 'logic' for TTS-optimized stream.
        """
        if self._classifier is not None:
            label = self._classifier.predict([text])[0]
            return "robust" if label == 0 else "logic"

        return self._classify_heuristic(text)

    def _classify_heuristic(self, text: str) -> str:
        """Regex-based fallback classifier."""
        text_lower = text.lower()
        has_markers = any(p.search(text_lower) for p in self._compiled_markers)
        is_short = len(text.split()) < 5
        return "robust" if (has_markers or is_short) else "logic"
