.PHONY: help setup index query eval format lint type test

help:
	@echo "Makefile for RAG Demo"
	@echo ""
	@echo "Usage:"
	@echo "  make setup   - Create venv and install all dependencies"
	@echo "  make index   - Build the FAISS index"
	@echo "  make query   - Run a sample query"
	@echo "  make eval    - Run the evaluation script"
	@echo "  make format  - Run black and ruff formatters"
	@echo "  make lint    - Run ruff linter"
	@echo "  make type    - Run mypy type checker"
	@echo "  make test    - Run pytest"

setup:
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment in .venv with uv..."; \
		uv venv .venv; \
	fi
	uv pip install --python .venv/bin/python uv
	.venv/bin/uv pip install -e '.[dev]'

index:
	.venv/bin/uv run python -m apps.cli.build_index

query:
	.venv/bin/uv run python -m apps.cli.query "What is ColBERT?"

eval:
	.venv/bin/uv run python -m eval.run_evaluation

api:
	.venv/bin/uv run python -m apps.api

docker-build:
	docker build -t rag-voyage-demo:latest .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env rag-voyage-demo:latest

format:
	.venv/bin/uv run black . && .venv/bin/uv run ruff format .

lint:
	.venv/bin/uv run ruff check .

type:
	.venv/bin/uv run mypy .

test:
	.venv/bin/uv run pytest
