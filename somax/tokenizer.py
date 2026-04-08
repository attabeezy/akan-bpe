"""Dual-core tokenizer for African languages.

Both streams share a single unified BPE vocabulary trained on combined
ASR+TTS data, ensuring compatible token IDs and embedding alignment when
fine-tuning the same Llama base model. The router selects which stream's
fine-tuned weights to use at inference time, not which vocabulary.
"""

import json
from pathlib import Path
from typing import Literal

from transformers import PreTrainedTokenizerFast

from somax.router import WAXALRouter


StreamType = Literal["robust", "logic"]


class DualCoreTokenizer:
    """Dual-stream tokenizer with dynamic routing over a shared vocabulary.

    Loads a single unified BPE tokenizer trained on combined ASR+TTS data.
    The WAXALRouter selects the stream at encode time; both streams tokenize
    with the same vocabulary so token IDs are compatible across model variants.

    Args:
        tokenizer_path: Path to the unified tokenizer JSON file.
        language: Target language for routing (e.g. 'akan').
        router_model_dir: Directory containing trained router .pkl files.

    Example:
        >>> tokenizer = DualCoreTokenizer(
        ...     tokenizer_path="models/tokenizers/akan/unified_tokenizer.json",
        ... )
        >>> tokenizer.classify("uhm chale me dwo o")
        'robust'
        >>> tokenizer.encode("The formal text goes here")
        [...]
    """

    def __init__(
        self,
        tokenizer_path: str | Path,
        language: str = "akan",
        router_model_dir: str | Path = "models/router/",
    ):
        tokenizer_file = Path(tokenizer_path)
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_file}")

        self.router = WAXALRouter(language=language, model_dir=router_model_dir)
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))

        stats_path = tokenizer_file.parent / "stream_token_stats.json"
        self.stream_token_stats: dict = {}
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                self.stream_token_stats = json.load(f)

    def classify(self, text: str) -> StreamType:
        """Determine stream type without encoding.

        Args:
            text: Input text.

        Returns:
            'robust' (ASR) or 'logic' (TTS).
        """
        return self.router.classify(text)

    def encode(self, text: str) -> list[int]:
        """Encode text using the unified vocabulary.

        The stream classification influences which fine-tuned model weights
        are used at inference, not which tokenizer is called.

        Args:
            text: Input text to encode.

        Returns:
            List of token IDs.
        """
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: Token IDs to decode.

        Returns:
            Decoded text string.
        """
        return self._tokenizer.decode(tokens)

    def encode_with_stream(self, text: str) -> tuple[list[int], StreamType]:
        """Encode text and return both token IDs and the detected stream.

        Useful for routing the encoded tokens to the correct model variant
        at inference time.

        Args:
            text: Input text to encode.

        Returns:
            Tuple of (token_ids, stream_type).
        """
        stream = self.classify(text)
        return self._tokenizer.encode(text), stream
