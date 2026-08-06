# BrowserGym Web Resources Server

This stateful server owns the live Playwright context used by WebArena,
VisualWebArena, and the BrowserGym-backed WebVoyager profile. It deliberately
keeps benchmark-specific launch and evaluation logic behind one HTTP contract:

```text
seed_session -> observe -> step* -> evaluate -> close
```

WebArena and VisualWebArena use their BrowserGym task validators in the live
session. WebVoyager uses `browsergym/openended`; its final screenshots are
judged by the separate `webvoyager_judge` resource server.

## Runtime setup

Install the server environment and Chromium:

```bash
uv sync --project resources_servers/browsergym_web
uv run --project resources_servers/browsergym_web playwright install chromium
```

BrowserGym 0.14.3 pins Playwright 1.44.0, which in turn declares a greenlet
release that predates Python 3.13. This component overrides greenlet to 3.1.1,
the first compatible line with CPython 3.13 wheels. The override is declared in
both `pyproject.toml` (for `uv sync`) and `overrides.txt` (for Gym's isolated
component installer); the BrowserGym and Playwright versions remain pinned.

The `data/` files are five schema and data-validation fixtures, not benchmark
scores. End-to-end results require the official site stack and evaluator.

BrowserGym expects the official site-stack URLs in environment variables. For
WebArena these are `WA_SHOPPING`, `WA_SHOPPING_ADMIN`, `WA_REDDIT`,
`WA_GITLAB`, `WA_WIKIPEDIA`, `WA_MAP`, and `WA_HOMEPAGE`. VisualWebArena uses
`VWA_SHOPPING`, `VWA_REDDIT`, `VWA_WIKIPEDIA`, `VWA_CLASSIFIEDS`,
`VWA_CLASSIFIEDS_RESET_TOKEN`, and `VWA_HOMEPAGE`.

For VisualWebArena, a successful request to the `VWA_HOMEPAGE` root is not a
sufficient readiness check. The homepage service must expose every input image
referenced by the prepared task population. Validate those URLs before a long
run; a missing task image is reported as a masked `benchmark_precondition`, not
as a model failure or a retryable capacity error.

VisualWebArena 0.0.15 also hard-codes `gpt-4-1106-preview` for fuzzy and
unachievable-answer evaluator calls. Set `OPENAI_BASE_URL` and
`OPENAI_API_KEY` for the judge endpoint, then set
`visualwebarena_evaluator_model` to an available OpenAI-compatible model. The
adapter remaps only the model argument; upstream evaluator prompts and score
parsing remain unchanged. Leave the setting null to retain exact upstream
behavior.

## Isolation and scheduling boundary

The default `site_pool_mode: unmanaged` retains the single-session behavior.
For scheduler-annotated datasets, `site_pool_mode: local_locks` allows shared
`read_only`/`session_only` leases and takes exclusive leases for every other
`mutation_class`. Cross-site tasks acquire every `site_locks` entry atomically,
so writers on different sites can proceed concurrently without same-site
overlap. BrowserGym calls themselves remain on one thread-affine Playwright
executor because BrowserGym 0.14.x owns a process-global Sync API object.

Neither mode resets or clones a mutable website. Local locks coordinate tasks
within one resource-server process only; parallel processes require an
external lock/replica manager. A fresh browser context is not a site reset, and
state-changing suites must still use the benchmark's official reset procedure.

Step `execution_ok` reports whether the browser action executed; evaluator
score is returned separately as `benchmark_reward`. Browser/evaluator
infrastructure failures are surfaced for masking rather than converted into a
benchmark score of zero.

The HTTP error envelope includes `error_kind` and `retryable`. Capacity,
session-loss, and session-conflict failures remain retryable. Invalid tasks and
deterministic BrowserGym reset preconditions return a non-retryable error so
the agent does not spend its rollout retry budget on an unchanged deployment.
