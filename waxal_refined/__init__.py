"""WAXAL-Refined: Dual-stream tokenizer for African languages.

Eliminates the Tokenization Tax by routing input through optimized
tokenizers based on linguistic characteristics.

Key components:
- WAXALRouter: Heuristic classifier for stream selection
- DualCoreTokenizer: Manages ASR and TTS tokenization streams
"""

from waxal_refined.router import WAXALRouter
from waxal_refined.tokenizer import DualCoreTokenizer

__all__ = ["WAXALRouter", "DualCoreTokenizer"]
__version__ = "0.1.0"
