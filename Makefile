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
	@echo "  make tf-init - Terraform init for Databricks"
	@echo "  make tf-plan - Terraform plan"
	@echo "  make tf-apply- Terraform apply"
	@echo "  make tf-destroy- Terraform destroy"

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

# Terraform (Databricks)
TF_DIR=infra/terraform

tf-init:
	cd $(TF_DIR) && terraform init

tf-plan:
	cd $(TF_DIR) && terraform plan -var databricks_host=$$DATABRICKS_HOST -var databricks_token=$$DATABRICKS_TOKEN -var repo_url=$$(git config --get remote.origin.url) -var repo_branch=$$(git rev-parse --abbrev-ref HEAD)

tf-apply:
	cd $(TF_DIR) && terraform apply -auto-approve -var databricks_host=$$DATABRICKS_HOST -var databricks_token=$$DATABRICKS_TOKEN -var repo_url=$$(git config --get remote.origin.url) -var repo_branch=$$(git rev-parse --abbrev-ref HEAD)

tf-destroy:
	cd $(TF_DIR) && terraform destroy -auto-approve -var databricks_host=$$DATABRICKS_HOST -var databricks_token=$$DATABRICKS_TOKEN -var repo_url=$$(git config --get remote.origin.url) -var repo_branch=$$(git rev-parse --abbrev-ref HEAD)

format:
	.venv/bin/uv run black . && .venv/bin/uv run ruff format .

lint:
	.venv/bin/uv run ruff check .

type:
	.venv/bin/uv run mypy .

test:
	.venv/bin/uv run pytest
