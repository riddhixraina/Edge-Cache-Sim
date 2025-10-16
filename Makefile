# Makefile for CDN Cache Simulator

.PHONY: help install test run dev build clean docker-up docker-down docker-build docker-test lint format

# Default target
help:
	@echo "CDN Cache Simulator - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     Install dependencies"
	@echo "  test        Run test suite with coverage"
	@echo "  test-fast   Run tests without coverage"
	@echo "  lint        Run linting (pylint, mypy)"
	@echo "  format      Format code with black"
	@echo ""
	@echo "Simulation:"
	@echo "  run         Run basic simulation"
	@echo "  run-sweep   Run cache size sweep"
	@echo "  benchmark   Run performance benchmarks"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker images"
	@echo "  docker-up       Start services"
	@echo "  docker-down     Stop services"
	@echo "  docker-dev      Start development environment"
	@echo "  docker-test     Run tests in Docker"
	@echo ""
	@echo "Utilities:"
	@echo "  clean       Clean temporary files"
	@echo "  setup       Initial project setup"

# Development Commands
install:
	pip install -r requirements.txt

test:
	python test_runner.py

test-fast:
	python test_runner.py fast

lint:
	pylint src/ tests/ --rcfile=pyproject.toml
	mypy src/ --config-file=pyproject.toml

format:
	black src/ tests/ cli.py streamlit_app.py --config=pyproject.toml

# Simulation Commands
run:
	python cli.py run --policy LRU --nodes 8 --cache-size 100 --requests 10000

run-sweep:
	python cli.py sweep --param cache_size --values 10,50,100,500 --policy LRU --requests 10000

benchmark:
	python cli.py benchmark --requests 1000,10000,100000 --nodes 2,4,8,16 --policies LRU,LFU,TTL

# Docker Commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d simulator

docker-down:
	docker-compose down

docker-dev:
	docker-compose --profile dev up -d dev

docker-test:
	docker-compose --profile test up test

docker-benchmark:
	docker-compose --profile benchmark up benchmark

docker-http:
	docker-compose --profile http up -d simulator origin-server

# Utility Commands
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

setup: install
	mkdir -p results/csv results/plots results/gifs results/reports
	mkdir -p data/traces data/objects data/real_workloads
	@echo "Project setup complete!"

# Development workflow
dev-setup: setup
	@echo "Setting up development environment..."
	pip install -r requirements.txt
	pre-commit install || echo "pre-commit not available"

# CI/CD Commands
ci-test: install test lint
	@echo "CI tests passed!"

ci-build: docker-build docker-test
	@echo "CI build completed!"

# Documentation
docs:
	@echo "Generating documentation..."
	python -c "import src; help(src)" > docs/api_reference.txt
	@echo "Documentation generated in docs/"

# Release preparation
release-check: test lint format
	@echo "Release checks passed!"

# Quick development cycle
quick-test:
	python test_runner.py fast
	python cli.py run --policy LRU --nodes 2 --cache-size 10 --requests 1000

# Performance testing
perf-test:
	python cli.py benchmark --requests 10000,50000,100000 --nodes 4,8,16 --policies LRU,LFU

# Full test suite
full-test: test lint format docker-test
	@echo "Full test suite completed!"

# Help for specific targets
help-docker:
	@echo "Docker Commands:"
	@echo "  docker-build     Build all Docker images"
	@echo "  docker-up        Start production dashboard"
	@echo "  docker-dev       Start development environment"
	@echo "  docker-test      Run tests in Docker"
	@echo "  docker-benchmark Run benchmarks in Docker"
	@echo "  docker-http      Start with HTTP server"

help-simulation:
	@echo "Simulation Commands:"
	@echo "  run              Run basic simulation"
	@echo "  run-sweep        Run parameter sweep"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  perf-test        Run performance tests"
	@echo "  quick-test       Run quick validation test"
