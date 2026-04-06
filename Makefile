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
	python benchmark.py

lint:
	ruff check .
	black --check .
	mypy waxal_refined/

test:
	pytest tests/

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf data/*.pt data/*.bin data/*.gguf
	find . -type d -name "__pycache__" -exec rm -rf {} +