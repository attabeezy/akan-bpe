### `code_spec.md`

```markdown
# Coding Specification: WAXAL-Dual-Core
**Status:** Implementation Phase (April 2026)
**Hardware:** Cloud (Colab T4) -> Edge (Dell Latitude 7400)

## 1. Directory Structure
```text
WAXAL-Dual-Core/
├── data/                  # WAXAL subsets (Akan, Yoruba, Swahili)
├── scripts/               # Research Pipeline
│   ├── 01_train_bpe.py    # Vocab generation (ASR vs TTS)
│   ├── 02_train_lora.py   # Staged Embedding Training (Variant E logic)
│   └── 03_export_gguf.py  # Llama.cpp quantization
├── waxal_refined/         # Edge Python Library
│   ├── router.py          # Regex-based heuristic classifier
│   └── tokenizer.py       # Dual-Core Stream Manager
└── benchmark_edge.py      # Local latency/fertility auditing script
```

## 2. Phase 1: Vocabulary & LoRA (Cloud)
### 2.1 Staged Training (Variant E)
```python
# logic for 02_train_lora.py
def train_variant_e(model, datasets):
    # Stage 1: Anchor on Formal Logic
    trainer_tts = Trainer(model, datasets['tts'], lr=2e-4)
    trainer_tts.train()
    
    # Stage 2: Adapt to Conversational Noise
    trainer_asr = Trainer(model, datasets['asr'], lr=1e-4)
    trainer_asr.train()
    
    # Stage 3: Refine Final Reasoning
    trainer_refine = Trainer(model, datasets['tts'], lr=5e-5)
    trainer_refine.train()
```

## 3. Phase 2: Edge Layer (Local)
### 3.1 Dynamic Router (`router.py`)
```python
import re

class WAXALRouter:
    """Lightweight heuristic for the Dell Latitude 7400 CPU."""
    def __init__(self):
        self.markers = [r'\buhm\b', r'\berr\b', r'\bchale\b', r'\bnaa\b']
        
    def classify(self, text: str) -> str:
        text_lower = text.lower()
        if any(re.search(m, text_lower) for m in self.markers) or len(text.split()) < 5:
            return "robust" # ASR-optimized stream
        return "logic"      # TTS-optimized stream
```

### 3.2 Dual-Core Tokenizer (`tokenizer.py`)
```python
class DualCoreTokenizer:
    def __init__(self, asr_path, tts_path):
        self.router = WAXALRouter()
        self.robust_core = PreTrainedTokenizerFast(tokenizer_file=asr_path)
        self.logic_core = PreTrainedTokenizerFast(tokenizer_file=tts_path)

    def encode(self, text):
        stream = self.router.classify(text)
        return self.robust_core.encode(text) if stream == "robust" else self.logic_core.encode(text)
```

## 4. Hardware Benchmark (`benchmark_edge.py`)
```python
from llama_cpp import Llama
import psutil

def run_local_audit(gguf_path, prompt):
    # Target: Dell Latitude 7400 i7/i5 (8GB RAM)
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=4)
    
    # Measure: Token Fertility (F)
    words = len(prompt.split())
    tokens = len(llm.tokenize(prompt.encode('utf-8')))
    fertility = tokens / words
    
    # Measure: Memory & TPS
    # ... performance logging logic ...
    return {"F": fertility}
```

## 5. Dependency Manifest
* **Cloud:** `transformers`, `peft`, `bitsandbytes`, `datasets`
* **Edge:** `llama-cpp-python`, `psutil`
```