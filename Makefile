.PHONY: quickstart test clean install dev lint format

# Quick validation of RESONTINEX Fusion System
quickstart:
	@echo "🚀 Running RESONTINEX Fusion Quickstart..."
	@python -m fusion_ops.benchmark \
		--scenarios-dir examples/quickstart \
		--iterations 1 \
		--output quickstart_results.json \
		--verbose
	@echo "✅ Quickstart validation complete"
	@echo "📊 Results saved to quickstart_results.json"

# Install package in development mode
install:
	pip install -e .[dev]

# Run development setup
dev: install
	pre-commit install
	@echo "✅ Development environment ready"

# Run all tests
test:
	python -m pytest tests/ -v

# Run linting and type checking
lint:
	ruff check .
	mypy fusion_ops/
	bandit -r fusion_ops/ -f json -o bandit-report.json || true

# Format code
format:
	black .
	ruff check . --fix
	isort .

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Verify clean clone (cross-platform)
verify-clean:
ifeq ($(OS),Windows_NT)
	@powershell -ExecutionPolicy Bypass -File scripts/verify-clean-clone.ps1
else
	@bash scripts/verify-clean-clone.sh
endif

# Run full CI pipeline locally
ci: clean install lint test quickstart
	@echo "🎉 Local CI pipeline completed successfully"