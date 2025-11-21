# Makefile for Python package management

.PHONY: clean build test upload-test upload install help

help:
	@echo "Available commands:"
	@echo "  make build        - Build distribution packages"
	@echo "  make check        - Check package with twine"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make upload-test  - Upload to Test PyPI"
	@echo "  make upload       - Upload to PyPI"
	@echo "  make install      - Install package in editable mode"
	@echo "  make test-install - Test install from Test PyPI"

build:
	python -m build

check:
	twine check dist/*

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

install:
	pip install -e .

test-install:
	pip install --index-url https://test.pypi.org/simple/ tieredann

