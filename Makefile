.PHONY: setup download train_asr train_tts train_mixed benchmark lint test

CONTROL_TOKENIZER ?= gpt2
VOCAB_SIZE ?= 8000
EXPERIMENT_ID ?= tokenizer_fertility_experiment_001

setup:
	pip install -e ".[dev]"

download:
	python scripts/download.py --output-dir data

train_asr:
	python scripts/train_bpe.py --inputs data/aka_asr_train.jsonl --output models/asr_tokenizer.json --name asr --vocab-size $(VOCAB_SIZE)

train_tts:
	python scripts/train_bpe.py --inputs data/pristine_twi_train.jsonl --output models/tts_tokenizer.json --name tts --vocab-size $(VOCAB_SIZE)

train_mixed:
	python scripts/train_bpe.py --inputs data/aka_asr_train.jsonl data/pristine_twi_train.jsonl --output models/mixed_tokenizer.json --name mixed --vocab-size $(VOCAB_SIZE)

benchmark:
	python scripts/benchmark_fertility.py \
		--experiment-id $(EXPERIMENT_ID) \
		--control-tokenizer $(CONTROL_TOKENIZER) \
		--asr-tokenizer models/asr_tokenizer.json \
		--tts-tokenizer models/tts_tokenizer.json \
		--mixed-tokenizer models/mixed_tokenizer.json \
		--asr-test-file data/aka_asr_test.jsonl \
		--tts-test-file data/pristine_twi_test.jsonl \
		--output results/$(EXPERIMENT_ID).json

lint:
	ruff check .
	black --check .
	mypy somax/

test:
	pytest tests/ -v
