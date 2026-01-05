# Makefile for Sphinx documentation
#
# Uses `uv run` which automatically manages the virtual environment.
# No manual venv setup required. Works on Windows, macOS, and Linux.

.DEFAULT_GOAL := help

.PHONY: help docs-html docs-clean docs-live docs-publish check-uv

# Detect OS for cross-platform compatibility
ifeq ($(OS),Windows_NT)
    RM_CMD = if exist _build rmdir /s /q _build
    CHECK_UV = @where uv >nul 2>&1 || (echo. && echo uv is not installed. Install from: https://docs.astral.sh/uv/getting-started/installation/ && exit /b 1)
else
    RM_CMD = rm -rf _build
    CHECK_UV = @command -v uv >/dev/null 2>&1 || { echo ""; echo "❌ uv is not installed. Install from: https://docs.astral.sh/uv/getting-started/installation/"; echo ""; exit 1; }
endif

# Help target
help: ## Show this help message
	@echo ""
	@echo "Documentation Build System"
	@echo "=========================="
	@echo ""
	@echo "Available targets:"
	@echo "  make docs-html      Build HTML documentation"
	@echo "  make docs-live      Start live-reload server"
	@echo "  make docs-publish   Build for publication (fail on warnings)"
	@echo "  make docs-clean     Clean built documentation"
	@echo ""

# Check if uv is installed
check-uv:
	$(CHECK_UV)

docs-html: check-uv
	@echo "Building HTML documentation..."
	cd .. && uv run --group docs sphinx-build -b html docs docs/_build/html

docs-publish: check-uv
	@echo "Building HTML documentation for publication (fail on warnings)..."
	cd .. && uv run --group docs sphinx-build --fail-on-warning -b html docs docs/_build/html

docs-clean:
	@echo "Cleaning built documentation..."
	$(RM_CMD)

docs-live: check-uv
	@echo "Starting live-reload server (sphinx-autobuild)..."
	cd .. && uv run --group docs sphinx-autobuild --port 8001 docs docs/_build/html
