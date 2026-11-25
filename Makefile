# NeMo Gym — top-level convenience targets.
# Fern docs targets live at the root so contributors don't have to remember
# the exact `cd fern && npx … fern-api` invocations. CI workflows under
# `.github/workflows/fern-docs-*.yml` are the source of truth for the
# published pipeline; these targets just mirror the local-developer entry
# points.
#
# Use `make docs-login` when Fern commands require authentication.

FERN_DIR := fern
PUBLISH_WORKFLOW := Publish Fern Docs

.DEFAULT_GOAL := help

.PHONY: help \
        docs docs-check docs-preview docs-publish docs-login docs-login-remote docs-generate-library

help:
	@echo ""
	@echo "NeMo Gym top-level Make targets"
	@echo "==============================="
	@echo ""
	@echo "Fern docs:"
	@echo "  make docs-login             Authenticate the Fern CLI"
	@echo "  make docs                   Generate library reference and start Fern dev server"
	@echo "  make docs-check             Validate Fern docs config ('fern check' via npm run check)"
	@echo "  make docs-preview           Build a shared preview URL on *.docs.buildwithfern.com (needs DOCS_FERN_TOKEN)"
	@echo "  make docs-publish           Trigger the 'Publish Fern Docs' workflow on origin/main"
	@echo "  make docs-generate-library  Regenerate the autodoc library reference under fern/product-docs/"
	@echo ""
	@echo "Some Fern commands require an account with the needed organization access."
	@echo ""

# ---------------------------------------------------------------------------
# Fern targets
# ---------------------------------------------------------------------------

# Authenticate using the Fern CLI's supported browser or device-code flow.
docs-login:
	npx -y fern-api@latest login $(LOGIN_FLAGS)

# Same as docs-login, but uses Fern's device-code flow instead of opening a
# browser locally — needed on headless remote machines (SSH dev boxes, etc.)
# where the OAuth callback can't reach a browser. Prints a URL + short code
# you complete on your laptop.
docs-login-remote: LOGIN_FLAGS := --device-code
docs-login-remote: docs-login

# Local-only preview. `fern docs md generate` populates fern/product-docs/ from
# the nemo_gym package source (declared under `libraries:` in fern/docs.yml);
# `fern docs dev` then serves the site on localhost:3000. Re-run `make docs`
# only when the library source changes — for prose-only iteration,
# `cd fern && npx -y fern-api@latest docs dev` alone is enough after the
# first generate.
docs: docs-generate-library
	cd $(FERN_DIR) && npx -y fern-api@latest docs dev

docs-check:
	cd $(FERN_DIR) && npm run check

# Wraps the autodoc generate step and points contributors to the public
# maintainer route when their account lacks required access.
docs-generate-library:
	@cd $(FERN_DIR) && npx -y fern-api@latest docs md generate || { \
		status=$$?; \
		echo ""; \
		echo "✗ 'fern docs md generate' failed (exit $$status)."; \
		echo ""; \
		echo "If the error indicates that your account lacks access, request"; \
		echo "help through the public maintainer issue form:"; \
		echo ""; \
		echo "    https://github.com/NVIDIA-NeMo/Gym/issues/new/choose"; \
		echo ""; \
		echo "For prose-only preview, use 'cd fern && npm run dev'."; \
		echo ""; \
		exit $$status; \
	}

# Shared preview hosted at <repo-slug>.docs.buildwithfern.com — useful for
# sharing a work-in-progress link before merge. Requires DOCS_FERN_TOKEN in
# the environment (org secret of the same name is wired into CI).
docs-preview:
	cd $(FERN_DIR) && npx -y fern-api@latest generate --docs --preview

# Trigger the Publish Fern Docs workflow on origin/main via workflow_dispatch.
# Alternative: tag a release with `git tag docs/v0.3.0 && git push origin docs/v0.3.0`
# — the workflow also fires on `docs/v*` tag pushes.
docs-publish:
	gh workflow run "$(PUBLISH_WORKFLOW)" --ref main
