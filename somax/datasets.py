"""Dataset loading and normalization helpers for SOMAX."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CANONICAL_TEXT_KEYS = ("text", "transcription")


@dataclass(frozen=True)
class TextSample:
    """Normalized text sample used throughout the project."""

    id: str
    text: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text, "source": self.source}


def extract_text(payload: dict[str, object]) -> str:
    """Return the first supported text field from a payload."""
    for key in CANONICAL_TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_jsonl_samples(path: Path) -> list[TextSample]:
    """Load normalized text samples from a JSONL file."""
    samples: list[TextSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            text = extract_text(payload)
            if not text:
                continue
            sample_id = str(payload.get("id") or f"{path.stem}_{index}")
            source = str(payload.get("source") or path.stem)
            samples.append(TextSample(id=sample_id, text=text, source=source))
    return samples


def samples_to_texts(samples: Iterable[TextSample]) -> list[str]:
    """Convert normalized samples into plain text strings."""
    return [sample.text for sample in samples]
