.PHONY: help index query eval format lint type test setup

help:
	@echo "Targets: setup, index, query, eval, format, lint, type, test"

setup:
	@echo "Using uv to sync dependencies" && uv sync --frozen || uv sync

index:
	./.venv/bin/python build_index.py

query:
	./.venv/bin/python query.py

# Placeholder; Phase 1 will create eval runner
eval:
	@echo "Eval runner not implemented yet. Will be added in Phase 1."

format:
	./.venv/bin/black .

lint:
	./.venv/bin/ruff check .

type:
	./.venv/bin/mypy .

test:
	./.venv/bin/pytest -q
