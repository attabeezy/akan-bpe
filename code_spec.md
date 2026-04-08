### `code_spec.md`

```markdown
# Coding Specification: somax
**Status:** Implementation Phase (April 2026)
**Hardware:** Cloud (Colab T4) -> Edge (Dell Latitude 7400)

## 1. Directory Structure
```text
somax/
├── data/                  # WAXAL subsets (Akan, Yoruba, Swahili)
├── scripts/               # Research Pipeline
│   ├── download.py        # Dataset downloader
│   ├── train_bpe.py       # Unified vocab generation
│   ├── train_lora.py      # Staged Embedding Training (Variant D logic)
│   ├── train_router.py    # Router classifier training
│   └── export_gguf.py     # Llama.cpp quantization
├── somax/         # Edge Python Library
│   ├── router.py          # TF-IDF or regex-based heuristic classifier
│   └── tokenizer.py       # Dual-Core Stream Manager (Unified Vocabulary)
├── benchmark_fertility.py # Token fertility auditing script
└── benchmark_inference.py # Local latency/TPS/Memory auditing script
```

## 2. Phase 1: Vocabulary & LoRA (Cloud)
### 2.1 Staged Training (Variant D)
```python
# logic for train_lora.py
def train_variant_d(model, datasets):
    # Stage 1: Anchor on Formal Logic (TTS)
    trainer_tts = Trainer(model, datasets['tts'], lr=2e-4)
    trainer_tts.train()
    
    # Stage 2: Adapt to Conversational Noise (ASR)
    trainer_asr = Trainer(model, datasets['asr'], lr=1e-4)
    trainer_asr.train()
    
    # Stage 3: Refine Final Reasoning (TTS)
    trainer_refine = Trainer(model, datasets['tts'], lr=5e-5)
    trainer_refine.train()
```

## 3. Phase 2: Edge Layer (Local)
### 3.1 Dynamic Router (`router.py`)
```python
import re

class WAXALRouter:
    """Lightweight heuristic fallback for the Dell Latitude 7400 CPU."""
    _FALLBACK_MARKERS = [
        r'\buhm\b', r'\berr\b', r'\bchale\b', r'\bnaa\b', r'\beh\b', r'\buna\b'
    ]
        
    def classify(self, text: str) -> str:
        text_lower = text.lower()
        if any(re.search(m, text_lower) for m in self._FALLBACK_MARKERS) or len(text.split()) < 5:
            return "robust" # ASR-optimized stream
        return "logic"      # TTS-optimized stream
```

### 3.2 Dual-Core Tokenizer (`tokenizer.py`)
```python
class DualCoreTokenizer:
    """Uses a unified vocabulary for both streams."""
    def __init__(self, tokenizer_path, language="akan"):
        self.router = WAXALRouter(language=language)
        # Both streams share the same tokenizer file
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    def encode(self, text):
        # Router determines the stream, which influences model weight selection later
        stream = self.router.classify(text)
        return self._tokenizer.encode(text)
```

## 4. Hardware Benchmark (`benchmark_inference.py`)
```python
from llama_cpp import Llama
import psutil

def run_local_audit(gguf_path, prompt):
    # Target: Dell Latitude 7400 i7/i5 (8GB RAM)
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=4)
    
    # Measure: TPS and Memory
    # ... performance logging logic ...
    return {"TPS": tps, "MB": memory_usage}
```

## 5. Dependency Manifest
* **Cloud:** `transformers`, `peft`, `bitsandbytes`, `datasets`
* **Edge:** `llama-cpp-python`, `psutil`
```