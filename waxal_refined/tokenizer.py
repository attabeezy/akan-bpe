"""Dual-core tokenizer for African languages.

Manages two tokenizer streams:
- Robust core (ASR-optimized): Handles noisy, conversational input
- Logic core (TTS-optimized): Handles formal, semantic-rich input

The router dynamically selects the appropriate stream based on input characteristics.
"""

from pathlib import Path
from typing import Literal

from transformers import PreTrainedTokenizerFast

from waxal_refined.router import WAXALRouter


StreamType = Literal["robust", "logic"]


class DualCoreTokenizer:
    """Dual-stream tokenizer with dynamic routing.

    Loads two tokenizer models and routes input based on linguistic
    characteristics detected by WAXALRouter.

    Attributes:
        router: Stream classification router.
        robust_core: ASR-optimized tokenizer.
        logic_core: TTS-optimized tokenizer.

    Example:
        >>> tokenizer = DualCoreTokenizer(
        ...     asr_path="models/tokenizers/akan_asr.json",
        ...     tts_path="models/tokenizers/akan_tts.json"
        ... )
        >>> tokens = tokenizer.encode("The formal text goes here")
    """

    def __init__(self, asr_path: str | Path, tts_path: str | Path, language: str = "akan"):
        """Initialize dual-core tokenizer.

        Args:
            asr_path: Path to ASR tokenizer JSON file.
            tts_path: Path to TTS tokenizer JSON file.
            language: Target language for routing.
        """
        self.router = WAXALRouter(language=language)

        asr_file = Path(asr_path)
        tts_file = Path(tts_path)

        if not asr_file.exists():
            raise FileNotFoundError(f"ASR tokenizer not found: {asr_file}")
        if not tts_file.exists():
            raise FileNotFoundError(f"TTS tokenizer not found: {tts_file}")

        self.robust_core = PreTrainedTokenizerFast(tokenizer_file=str(asr_file))
        self.logic_core = PreTrainedTokenizerFast(tokenizer_file=str(tts_file))

    def classify(self, text: str) -> StreamType:
        """Determine stream type without encoding.

        Args:
            text: Input text.

        Returns:
            Stream classification ("robust" or "logic").
        """
        return self.router.classify(text)

    def encode(self, text: str) -> list[int]:
        """Encode text using appropriate stream.

        Args:
            text: Input text to encode.

        Returns:
            List of token IDs.
        """
        stream = self.classify(text)
        if stream == "robust":
            return self.robust_core.encode(text)
        return self.logic_core.encode(text)

    def decode(self, tokens: list[int], stream: StreamType = "logic") -> str:
        """Decode tokens back to text.

        Args:
            tokens: Token IDs to decode.
            stream: Which core to use for decoding.

        Returns:
            Decoded text string.
        """
        if stream == "robust":
            return self.robust_core.decode(tokens)
        return self.logic_core.decode(tokens)
