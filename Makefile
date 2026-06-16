# NeMo Gym — top-level convenience targets.
# Fern docs targets live at the root so contributors don't have to remember
# the exact `cd fern && npx … fern-api` invocations. CI workflows under
# `.github/workflows/fern-docs-*.yml` are the source of truth for the
# published pipeline; these targets just mirror the local-developer entry
# points.
#
# First time on this machine? Run `make docs-login` before `make docs`.
# Without dashboard provisioning, `fern docs md generate` fails with
# `HTTP 403: User does not belong to organization`.

FERN_DIR := fern
PUBLISH_WORKFLOW := Publish Fern Docs

.DEFAULT_GOAL := help

.PHONY: help \
        docs docs-check docs-preview docs-publish docs-login docs-generate-library

help:
	@echo ""
	@echo "NeMo Gym top-level Make targets"
	@echo "==============================="
	@echo ""
	@echo "Fern docs:"
	@echo "  make docs-login             FIRST-TIME SETUP — provision Fern account + CLI auth"
	@echo "  make docs                   Generate library reference and start Fern dev server"
	@echo "  make docs-check             Validate Fern docs config ('fern check' via npm run check)"
	@echo "  make docs-preview           Build a shared preview URL on *.docs.buildwithfern.com (needs DOCS_FERN_TOKEN)"
	@echo "  make docs-publish           Trigger the 'Publish Fern Docs' workflow on origin/main"
	@echo "  make docs-generate-library  Regenerate the autodoc library reference under fern/product-docs/"
	@echo ""
	@echo "First time? Run 'make docs-login' before 'make docs' or your"
	@echo "autodoc step will fail with HTTP 403."
	@echo ""

# ---------------------------------------------------------------------------
# Fern targets
# ---------------------------------------------------------------------------

# First-time auth setup. Fern's CLI requires that the user already exist in
# Fern's user DB *before* `fern login` can complete; signing in to the
# dashboard is what creates that user record. Skipping step 1 below leaves
# the CLI login prompt with no working path forward (you can't get past the
# email entry), and skipping it before `fern docs md generate` returns:
#
#     HTTP 403: User does not belong to organization
#
# This target blocks on a confirmation prompt so step 1 isn't accidentally
# skipped. (Tracked upstream at Fern; ping #fern-cli on Slack if it changes.)
docs-login:
	@echo ""
	@echo "Fern auth — one-time setup per machine"
	@echo "======================================="
	@echo ""
	@echo "  Step 1 (REQUIRED FIRST): open"
	@echo ""
	@echo "      https://dashboard.buildwithfern.com/login"
	@echo ""
	@echo "  in a browser and sign in with your @nvidia.com email. Use the"
	@echo "  email / magic-link flow — NOT the Google SSO button — because"
	@echo "  Google sign-in does not always provision the account record"
	@echo "  that the CLI later needs."
	@echo ""
	@echo "  External contributors: any email works (Fern provisions a"
	@echo "  personal account); ask in #fern to be added to the 'nvidia'"
	@echo "  org if you want to run library autodoc generation."
	@echo ""
	@echo "  Step 2: confirm the 'nvidia' organization shows in the"
	@echo "  dashboard sidebar. If it doesn't, you signed in with the wrong"
	@echo "  account — sign out and retry step 1."
	@echo ""
	@echo "  Step 3: this target will run 'fern login' next; complete the"
	@echo "  browser flow with the SAME email you used in step 1."
	@echo ""
	@echo "---"
	@printf "Have you completed step 1 (dashboard sign-in)? [yes/N]: "
	@read confirm; case "$$confirm" in \
		yes|YES|y|Y) ;; \
		*) echo ""; echo "Bailing. Open the dashboard URL above, sign in, then re-run 'make docs-login'."; exit 1 ;; \
	esac
	@echo ""
	npx -y fern-api@latest login

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

# Wraps the autodoc generate step. If it fails (the most common cause is
# the HTTP 403 from missing dashboard provisioning), print the recovery
# pointer so contributors aren't left guessing.
docs-generate-library:
	@cd $(FERN_DIR) && npx -y fern-api@latest docs md generate || { \
		status=$$?; \
		echo ""; \
		echo "✗ 'fern docs md generate' failed (exit $$status)."; \
		echo ""; \
		echo "If the error mentions 'HTTP 403: User does not belong to"; \
		echo "organization', your CLI auth is missing the dashboard"; \
		echo "provisioning step. Fix:"; \
		echo ""; \
		echo "    make docs-login"; \
		echo ""; \
		echo "Then re-run: make docs"; \
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
