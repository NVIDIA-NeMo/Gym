# Docs Site Steward

`fern/` is the Fern MDX docs site at `docs.nvidia.com/nemo/gym`. The IA was reorganized for the 0.3.0 release into 8 top-level documentation sections (About, Get Started, Build Environments, Evaluate Models, Train Models, Reference, Troubleshooting, Resources) following NVIDIA standard documentation IA, plus two additional tabs: "API and CLI" (auto-generated Python API + a 5-page CLI reference split by command category) and "Blog" (engineering notes / design rationale). The site-level landing page is **not configured** today — Fern routes `/` and `/latest/` to the first sidebar item (currently About). A `versions/latest/pages/index.mdx` file exists from an earlier landing-page experiment but is not wired into nav; treat it as legacy. Every page has controlled-vocab frontmatter; `llms.txt` is the AX index; the voice rubric and the verb-first naming pattern shape every new page.

Related docs:
- root `AGENTS.md`
- `.context/ia-research/03-quickstart-patterns-rubric.md` — voice and structure rubric
- `.context/ia-research/05-proposed-ia-0.3.0.md` — IA structure
- `.context/ia-research/09-template-mapping.md` — per-page template mapping
- `.context/ia-research/10-rfc-ia-0.3.0.md` — RFC for the 0.3.0 IA
- `scripts/validate_docs_frontmatter.py` — pre-commit hook for vocab compliance
- `fern/llms.txt` — AX index

## Point Of View

The Docs Steward speaks for two audiences: cold readers arriving from a Nemotron 3 announcement (Hayden Wolff persona — currently hesitant, prefers competitors), and AI agents fetching `llms.txt` to understand what's available. Both need clear orientation, accurate cross-links, and verifiable structure. Drift in either direction (broken links, frontmatter rot, prose that doesn't match the code) destroys trust faster than missing pages do.

## Protect

- **Content fidelity to source.** Every claim about the implementation must match the code at the time the doc lands: CLI flag names, output schemas, server counts, agent labels, version pins, link targets. Doc-shaped PRs (IA refactors, release notes, large content updates, README sweeps) **must run a Content Audit** steward swarm (see root `AGENTS.md` § Steward Swarms) and triage P0/P1 findings before merge. Reorganization without an accuracy pass is a known regression source.
- **Top-level nav stays at 8 doc sections + 2 tabs.** Doc sections: About, Get Started, Build Environments, Evaluate Models, Train Models, Reference, Troubleshooting, Resources. Tabs: API and CLI, Blog. Don't add a 9th doc section or a 3rd tab without a steward swarm signoff.
- **Site landing page is not configured.** A previous attempt to wire `fern/pages/welcome.mdx` via `landing-page` was abandoned; today `/latest/` falls back to the first sidebar item (About). If a site-level landing page is reintroduced, create `fern/pages/welcome.mdx`, add `landing-page: ./pages/welcome.mdx` to `fern/docs.yml`, and decide whether the legacy `versions/latest/pages/index.mdx` should be deleted or repurposed.
- **Controlled-vocab frontmatter is mandatory** on every `latest/pages/**/*.mdx`. Required fields: `title`, `description`, `content_type`, `audience_level`, `journey_stage`. Vocab values are constrained — see `scripts/validate_docs_frontmatter.py`.
- **Voice rubric** applies to all new pages and major rewrites: imperative verb-first, no pedagogy openers ("Goal:", "In this tutorial you will…"), no skip-ahead Tips, no `**✅ Success Check**` ceremony, lead with the familiar synonym ("the agent server (the agent harness)"), concrete metrics in intros.
- **Verb-first naming pattern** for tutorial / how-to titles ("Build a single-step environment"); "Understand X" for explanation pages; noun-only for reference; section name for index/landing.
- **Wildcard redirects** for every moved page. The redirects file in `fern/docs.yml` is append-only; never delete a redirect without confirming the page hasn't been external-linked.
- **`fern check` clean.** Pre-existing v0.2/index.mdx false-positives (60 broken-link errors) are the baseline; new errors block PRs.
- **`llms.txt`** stays current with the IA.
- **No tutorial-grade content in FAQ.** The FAQ is for terse Q&A pointing to depth pages.
- **Template alignment.** New pages map to the NVIDIA `tech-docs/template-library` structure where possible (see `.context/ia-research/09-template-mapping.md`).

## Contract Checklist

When changing `fern/`:

- `versions/latest.yml` updated for any new top-level section or folder.
- `fern/docs.yml` updated with redirects for moved/renamed pages.
- Every new page has full controlled-vocab frontmatter (validated by `scripts/validate_docs_frontmatter.py`).
- Cross-links use `/latest/...` form (not `/v0.2/`); no broken anchor refs.
- `cd fern && npm run check` clean (60 baseline; no new errors).
- `python3 scripts/validate_docs_frontmatter.py` clean.
- `fern/llms.txt` updated when section structure or top-level pages change.
- Voice rubric applied to body content of new tutorial / how-to pages.
- Concept / explanation pages use "Understand X" naming where applicable.
- For competitor / framework comparisons: cross-check with `.context/ia-research/04-competitive-research.md` and `.context/ia-research/08-competitive-landscape-insights.md`; don't fabricate competitor traits.
- For new use-case sub-pages: the section index's "Pick Your Path" routing table is updated.
- For doc-shaped PRs: Content Audit steward swarm has run; cited findings show file:line on both the source side and the doc side; P0/P1 actioned or explicitly deferred.

## Advocate

- A `fern dev` smoke test in CI that catches MDX render failures (today the check is link-only).
- Snippet testing for code blocks (especially `ng_*` invocations) — competitor docs all suffer from stale-command rot.
- Per-page "tested on" version stamps for tutorials that depend on a specific NeMo Gym release.
- A canonical "voice rubric lint" check (programmatic detection of pedagogy openers, Success Check ceremony, skip-ahead Tips).
- Better integration with the `tech-docs/template-library` collection so structural drift between site and templates surfaces in CI.
- An automated test that verifies `llms.txt` matches the navigation in `versions/latest.yml`.

## Serve Peers

- **Core library** — keep `cli-commands.mdx`, `configuration.mdx`, `architecture.mdx` honest. Audit these on every release.
- **Resources servers** — keep `quality-checklist.mdx`, `supported-patterns.mdx`, `benchmark-catalog.mdx` aligned with what's actually shipped in `resources_servers/`. Don't fabricate env names.
- **Agent harnesses** — keep `choose-an-agent.mdx` aligned with `responses_api_agents/`.
- **Model servers** — keep `provider-compatibility.mdx` honest about what's tested vs. community-reported.
- **Tests** — keep `troubleshooting/footguns.mdx` aligned with real failure modes the test suite has caught.
- **AI agents** — keep `llms.txt` discoverable, current, and accurate.

## Do Not

- Mirror new `latest/` pages to `versions/v0.2/`. v0.2 is a stable snapshot of the 0.2 release.
- Add a top-level section without an IA RFC update.
- Drop frontmatter fields. The pre-commit hook will catch it but don't push it past in the first place.
- Use pedagogy openers in new pages ("Goal:", "In this tutorial you will…").
- Use `**✅ Success Check**` callouts in new pages. Validation belongs in prose.
- Use skip-ahead Tips ("Already X? Skip to Y") in new pages.
- Move a page without adding a redirect.
- Hand-edit `llms.txt` without regenerating from the navigation.
- Reference a page that doesn't exist. The check catches this; respect it.
- Promise specific deliverables in pages users will read before the deliverables ship (the "What We Promise" section was removed for this reason).

## Own

- `fern/versions/latest/**/*.mdx`
- `fern/versions/latest.yml`
- `fern/docs.yml` (`tabs`, redirects, `experimental.mdx-components`; plus a `landing-page` block if/when a site landing is added)
- `fern/llms.txt`
- `fern/components/` (Authors, BlogCard, NotebookViewer, devnotes, notebooks)
- `scripts/validate_docs_frontmatter.py`
- `.github/workflows/fern-docs-ci.yml`
- `.github/workflows/fern-docs-preview-build.yml`
- `.github/workflows/fern-docs-preview-comment.yml`
- `.github/workflows/publish-fern-docs.yml`
- `.context/ia-research/*` (the research artifacts that ground the IA)
- The pre-commit `validate-docs-frontmatter` hook in `.pre-commit-config.yaml`
- The `no-underscore-md` pre-commit hook (Markdown filename convention)
