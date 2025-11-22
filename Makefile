.PHONY: install test clean help build upload-pypi

# Variables
PYTHON := python3
PIP := pip3


help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean up cache and temp files"
	@echo "  make build         - Build"
	@echo "  make upload-pypi   - Update pypi package"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true

build:
	python3 -m build

upload-pypi:
	python3 -m pip install --upgrade twine
	python3 -m twine upload dist/*
