"""Lightweight heuristic router for stream classification.

Classifies input text as "robust" (ASR-optimized) or "logic" (TTS-optimized)
based on linguistic markers. Designed for CPU-efficient inference on edge devices.
"""

import re
from typing import ClassVar


class WAXALRouter:
    """Regex-based heuristic classifier for dual-stream tokenization.

    Routes input to the appropriate tokenizer core based on:
    - Presence of conversational markers (fillers, code-switching)
    - Text length (short queries -> robust core)

    Attributes:
        markers: List of regex patterns for conversational markers.

    Example:
        >>> router = WAXALRouter()
        >>> router.classify("uhm chale me dwo")
        'robust'
        >>> router.classify("The formal declaration states...")
        'logic'
    """

    markers: ClassVar[list[str]] = [
        r"\buhm\b",
        r"\berr\b",
        r"\bchale\b",
        r"\bnaa\b",
        r"\beh\b",
        r"\buna\b",
    ]

    def __init__(self, language: str = "akan"):
        """Initialize router for specific African language.

        Args:
            language: Target language for marker optimization.
        """
        self.language = language
        self._compiled_markers = [re.compile(m, re.IGNORECASE) for m in self.markers]

    def classify(self, text: str) -> str:
        """Classify text stream type.

        Args:
            text: Input text to classify.

        Returns:
            "robust" for ASR-optimized stream, "logic" for TTS-optimized stream.
        """
        text_lower = text.lower()

        has_markers = any(pattern.search(text_lower) for pattern in self._compiled_markers)
        is_short = len(text.split()) < 5

        if has_markers or is_short:
            return "robust"
        return "logic"
