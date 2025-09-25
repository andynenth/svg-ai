.PHONY: help setup install test clean run-server benchmark convert batch dev docs

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)SVG AI Converter - Available Commands:$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-15s$(RESET) %s\n", $$1, $$2}'

setup: ## Initial project setup
	@echo "$(CYAN)Setting up project...$(RESET)"
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	@echo "$(GREEN)✓ Setup complete! Activate with: source venv/bin/activate$(RESET)"

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	pytest tests/ -v

test-fast: ## Run fast tests only
	pytest tests/ -v -m "not slow"

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=. --cov-report=html --cov-report=term

clean: ## Clean generated files
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov .coverage
	rm -rf results/*.json results/*.md
	rm -rf temp/*
	rm -rf cache/*
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleaned generated files$(RESET)"

dataset: ## Create test dataset
	python scripts/create_full_dataset.py

convert: ## Convert a single PNG (usage: make convert FILE=image.png)
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make convert FILE=path/to/image.png"; \
	else \
		python convert.py $(FILE); \
	fi

batch: ## Batch convert directory (usage: make batch DIR=path/to/dir)
	@if [ -z "$(DIR)" ]; then \
		echo "Usage: make batch DIR=path/to/directory"; \
	else \
		python batch_convert.py $(DIR) --parallel 4; \
	fi

benchmark: ## Run benchmark on test dataset
	python benchmark.py --test-dir data/logos --report

benchmark-quick: ## Run quick benchmark
	python benchmark.py --quick --test-dir data/logos

server: ## Start web server
	python web_server.py --host 127.0.0.1 --port 8000

server-dev: ## Start web server in development mode
	uvicorn web_server:app --reload --host 127.0.0.1 --port 8000

format: ## Format code with black
	black . --line-length 100

lint: ## Lint code with flake8
	flake8 . --max-line-length=100 --ignore=E501,W503

docker-build: ## Build Docker image
	docker build -t svg-converter .

docker-run: ## Run Docker container
	docker run -p 8000:8000 svg-converter

docs: ## Generate documentation
	@echo "$(CYAN)Generating documentation...$(RESET)"
	python -m pydoc -w converters.vtracer_converter
	python -m pydoc -w utils.quality_metrics
	python -m pydoc -w utils.cache
	@echo "$(GREEN)✓ Documentation generated$(RESET)"

dev: ## Run in development mode with file watching
	@echo "$(CYAN)Starting development mode...$(RESET)"
	@echo "Watching for changes..."
	@while true; do \
		clear; \
		python test_vtracer.py; \
		echo ""; \
		echo "$(GREEN)Waiting for changes... (Ctrl+C to exit)$(RESET)"; \
		sleep 2; \
	done

all: setup dataset test benchmark ## Run full setup and testing

# Default target
.DEFAULT_GOAL := help