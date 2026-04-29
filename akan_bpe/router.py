"""Router and mux utilities for Akan-BPE tokenizer selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from akan_bpe.classifier import MLClassifierRouter
from akan_bpe.tokenizers import load_tokenizer


@dataclass(frozen=True)
class TokenizerConfig:
    name: str
    path: str
    domain: str


@dataclass
class RoutingDecision:
    selected_tokenizer: str
    confidence: float
    domain: str
    reason: str


class AkanBPERouter:
    def __init__(
        self,
        asr_tokenizer_path: str,
        tts_tokenizer_path: str,
        mixed_tokenizer_path: str | None = None,
        use_ml_classifier: bool = False,
        classifier_path: str | None = None,
    ):
        self.configs = [
            TokenizerConfig(name="asr", path=asr_tokenizer_path, domain="asr"),
            TokenizerConfig(name="tts", path=tts_tokenizer_path, domain="tts"),
        ]
        if mixed_tokenizer_path:
            self.configs.append(
                TokenizerConfig(name="mixed", path=mixed_tokenizer_path, domain="mixed")
            )
        self.tokenizers = {}
        for config in self.configs:
            self.tokenizers[config.name] = load_tokenizer(config.path)

        self.use_ml_classifier = use_ml_classifier
        self.ml_classifier = None
        if use_ml_classifier and classifier_path:
            self.ml_classifier = MLClassifierRouter(classifier_path)

    def detect_domain(self, text: str) -> tuple[str, float]:
        """Detect if text is ASR (conversational) or TTS (formal).

        Returns (domain, confidence).

        Heuristics:
        - ASR/realistic speech: shorter words, no punctuation, more common particles
        - TTS/formal text: more punctuation, complex structure, quotations
        """
        words = text.split()
        if not words:
            return "mixed", 0.5

        avg_word_len = sum(len(w) for w in words) / len(words)

        punctuation_marks = sum(1 for c in text if c in ".,;:!?-()[]{}")
        punctuation_ratio = punctuation_marks / len(text) if text else 0

        has_punct = punctuation_ratio > 0.02
        has_unicode = any(ord(c) > 127 for c in text)
        has_formal_punct = any(c in text for c in ';:!"\'\'""')
        long_words = avg_word_len > 5.0

        if long_words and (has_punct or has_formal_punct):
            return "tts", 0.75
        elif has_formal_punct:
            return "tts", 0.7
        elif has_unicode and not has_punct and not has_formal_punct:
            return "asr", 0.65
        elif avg_word_len < 3.0:
            return "asr", 0.6
        elif long_words:
            return "tts", 0.6
        else:
            return "asr", 0.55

    def route(self, text: str) -> RoutingDecision:
        """Route a single text to the appropriate tokenizer."""
        if self.use_ml_classifier and self.ml_classifier:
            domain, confidence = self.ml_classifier.predict(text)
            reason = f"ml_classifier (confidence: {confidence:.2f})"
        else:
            domain, confidence = self.detect_domain(text)
            if domain == "asr":
                reason = f"heuristic (confidence: {confidence:.2f})"
            elif domain == "tts":
                reason = f"heuristic (confidence: {confidence:.2f})"
            else:
                reason = "ambiguous text, using mixed tokenizer"

        if domain == "asr":
            selected = "asr"
        elif domain == "tts":
            selected = "tts"
        else:
            selected = "mixed"

        return RoutingDecision(
            selected_tokenizer=selected,
            confidence=confidence,
            domain=domain,
            reason=reason,
        )

    def tokenize(self, text: str) -> tuple[list[int], RoutingDecision]:
        """Tokenize text using the routed tokenizer."""
        decision = self.route(text)
        tokenizer = self.tokenizers[decision.selected_tokenizer]
        result = tokenizer.encode(text)
        tokens = result if isinstance(result, list) else result.ids
        return tokens, decision

    def tokenize_with_mux(
        self, text: str, asr_weight: float = 0.5
    ) -> tuple[list[int], RoutingDecision | dict[str, object]]:
        """Tokenize using multiplexing strategy."""
        if "mixed" not in self.tokenizers:
            return self.tokenize(text)

        mixed_tokenizer = self.tokenizers["mixed"]
        mixed_result = mixed_tokenizer.encode(text)
        tokens = mixed_result if isinstance(mixed_result, list) else mixed_result.ids

        return tokens, {
            "strategy": "mux",
            "asr_weight": asr_weight,
            "tokenizer": "mixed",
        }

    def close(self) -> None:
        """Close all tokenizer resources."""
        self.tokenizers.clear()


def load_router_config(path: str) -> dict[str, object]:
    """Load router configuration from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_router_config(path: Path, config: dict[str, object]) -> None:
    """Save router configuration as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
