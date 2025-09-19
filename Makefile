.PHONY: help index query eval format lint type test setup

help:
	@echo "Targets: setup, index, query, eval, format, lint, type, test"

setup:
	@echo "Using uv to sync dependencies" && uv sync --frozen || uv sync

index:
	uv run python build_index.py

query:
	uv run python query.py

# Placeholder; Phase 1 will create eval runner
eval:
	@echo "Eval runner not implemented yet. Will be added in Phase 1."

format:
	uv run black .

lint:
	uv run ruff check .

type:
	uv run mypy .

test:
	uv run pytest -q
