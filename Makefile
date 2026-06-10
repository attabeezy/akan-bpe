.PHONY: setup sync download train_asr train_tts train_mixed benchmark lint test

BASELINES ?= xlm-roberta-base bert-base-multilingual-cased google/mt5-base
VOCAB_SIZE ?= 8000
EXPERIMENT_ID ?= tokenizer_fertility_experiment_001

setup:
	uv pip install -e ".[dev]"

sync:
	uv pip install -e ".[dev]" --upgrade

download:
	python scripts/download.py --output-dir data

train_asr:
	python scripts/train_bpe.py --inputs data/aka_asr_train.jsonl --output models/asr_tokenizer.json --name asr --vocab-size $(VOCAB_SIZE)

train_tts:
	python scripts/train_bpe.py --inputs data/pristine_twi_train.jsonl --output models/tts_tokenizer.json --name tts --vocab-size $(VOCAB_SIZE)

train_mixed:
	python scripts/train_bpe.py --inputs data/aka_asr_train.jsonl data/pristine_twi_train.jsonl --output models/mixed_tokenizer.json --name mixed --vocab-size $(VOCAB_SIZE) --balance

benchmark:
	python scripts/benchmark_fertility.py \
		--experiment-id $(EXPERIMENT_ID) \
		--baselines $(BASELINES) \
		--asr-tokenizer models/asr_tokenizer.json \
		--tts-tokenizer models/tts_tokenizer.json \
		--mixed-tokenizer models/mixed_tokenizer.json \
		--asr-test-file data/aka_asr_test.jsonl \
		--tts-test-file data/pristine_twi_test.jsonl \
		--output results/$(EXPERIMENT_ID).json

lint:
	ruff check .
	black --check .
	mypy akan_bpe/

test:
	pytest tests/ -v
