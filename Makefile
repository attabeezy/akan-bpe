.PHONY: setup download train_bpe train_lora export_gguf benchmark clean lint test

setup:
	pip install -e ".[dev,train]"

download:
	python scripts/download.py

train_bpe:
	python scripts/train_bpe.py

train_lora:
	python scripts/train_lora.py

export_gguf:
	python scripts/export_gguf.py

benchmark:
	@echo "Usage examples:"
	@echo "  python benchmark_fertility.py --tokenizer meta-llama/Llama-3.2-1B --test-file data/akan/twi_tts_test.jsonl"
	@echo "  python benchmark_inference.py --model models/gguf/variant_D_Q4.gguf --test-file data/akan/twi_tts_test.jsonl"
	python benchmark_fertility.py -h

lint:
	ruff check .
	black --check .
	mypy somax/

test:
	@if [ -d "tests" ]; then pytest tests/; else echo "No tests/ directory found."; fi

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf data/*.pt data/*.bin data/*.gguf
	find . -type d -name "__pycache__" -exec rm -rf {} +
