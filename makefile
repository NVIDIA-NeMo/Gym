# Makefile targets for Sphinx documentation (all targets prefixed with 'docs-')
# Adapted for NeMo Gym project structure with pure uv dependency management

.PHONY: docs-html docs-clean docs-live docs-env docs-publish \
        docs-html-draft docs-live-draft docs-publish-draft

# Usage:
#   make docs-html                     # Build docs with no special tag
#   make docs-html DOCS_ENV=draft      # Build docs with draft tag
#   make docs-live DOCS_ENV=draft      # Live server with draft tag
#   make docs-publish                  # Production build (fails on warnings)

DOCS_ENV ?=

# Detect OS for cross-platform compatibility
ifeq ($(OS),Windows_NT)
    RM_CMD = if exist docs\_build rmdir /s /q docs\_build
    DOCS_DIR = docs
    BUILD_DIR = docs\_build\html
else
    RM_CMD = cd docs && rm -rf _build
    DOCS_DIR = docs
    BUILD_DIR = docs/_build/html
endif

# Main documentation targets using uv run

docs-html:
	@echo "Building HTML documentation..."
	uv run --group docs sphinx-build -b html $(if $(DOCS_ENV),-t $(DOCS_ENV)) $(DOCS_DIR) $(BUILD_DIR)

docs-publish:
	@echo "Building HTML documentation for publication (fail on warnings)..."
	uv run --group docs sphinx-build --fail-on-warning --builder html $(if $(DOCS_ENV),-t $(DOCS_ENV)) $(DOCS_DIR) $(BUILD_DIR)

docs-clean:
	@echo "Cleaning built documentation..."
	$(RM_CMD)

docs-live:
	@echo "Starting live-reload server (sphinx-autobuild)..."
	uv run --group docs sphinx-autobuild $(if $(DOCS_ENV),-t $(DOCS_ENV)) $(DOCS_DIR) $(BUILD_DIR)

docs-env:
	@echo "Syncing documentation dependencies..."
	uv sync --group docs
	@echo "Documentation dependencies synced!"
	@echo "You can now run 'make docs-html' or 'make docs-live'"

# Build shortcuts for draft mode

docs-html-draft:
	$(MAKE) docs-html DOCS_ENV=draft

docs-publish-draft:
	$(MAKE) docs-publish DOCS_ENV=draft

docs-live-draft:
	$(MAKE) docs-live DOCS_ENV=draft

# Additional convenience targets

docs-help:
	@echo "Available documentation targets:"
	@echo "  docs-env        - Sync documentation dependencies with uv"
	@echo "  docs-html       - Build HTML documentation"
	@echo "  docs-live       - Start live-reload server for development"
	@echo "  docs-publish    - Build documentation with strict error checking"
	@echo "  docs-clean      - Clean built documentation"
	@echo ""
	@echo "Draft mode shortcuts:"
	@echo "  docs-html-draft     - Build with 'draft' tag"
	@echo "  docs-live-draft     - Live server with 'draft' tag"
	@echo "  docs-publish-draft  - Publish build with 'draft' tag"
	@echo ""
	@echo "Usage examples:"
	@echo "  make docs-env                    # Sync dependencies first (recommended)"
	@echo "  make docs-html                   # Basic build"
	@echo "  make docs-html DOCS_ENV=draft    # Build with draft tag"
	@echo "  make docs-live                   # Live server"
	@echo "  make docs-publish                # Production build (fails on warnings)"
	@echo ""
	@echo "Note: Uses 'uv run' with dependencies from pyproject.toml"