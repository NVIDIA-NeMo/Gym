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
The component also pins `datasets==5.0.1`: the otherwise valid resolver choice
of `datasets==2.14.4` with current `pyarrow` releases fails at evaluator import
because that older datasets release still uses the removed
`pyarrow.PyExtensionType` API.

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

WebArena and VisualWebArena also hard-code `gpt-4-1106-preview` for fuzzy and
unachievable-answer evaluator calls. Set `evaluator_base_url`, put the judge
credential in the environment variable named by `evaluator_api_key_env`
(default `OPENAI_API_KEY`), and set `webarena_evaluator_model` and/or
`visualwebarena_evaluator_model` to an available OpenAI-compatible model. The
adapter changes only the model argument; upstream prompts, generation options,
score parsers, and reward composition remain unchanged. The effective judge
model is included in `verifier_version`. A task whose retained metadata
declares a model-backed evaluator is rejected before browser startup when its
evaluator model is absent. This becomes a masked configuration error rather
than a benchmark failure.

`libvisualwebarena==0.0.15` constructs an OpenAI client when its evaluator
module is imported, even for a rule-only task. For a task whose metadata proves
that no model evaluator is reachable, the adapter supplies an explicit
non-secret placeholder key only to satisfy that import. It never substitutes a
placeholder for a fuzzy or unachievable-answer judge call.

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
score is returned separately as `benchmark_reward`. BrowserGym already catches
action mapping/execution exceptions and reports them as `last_action_error`.
An exception escaping `Env.step()` therefore remains an evaluator/runtime
failure and is surfaced for masking rather than converted into a policy action
failure or benchmark score of zero.

Formal `evaluate()` output uses the native evaluator reward attached to the
terminal observation (`score_semantics=terminal_native_evaluator_reward`). The
largest per-step reward is retained as `best_observed_reward` metadata for
diagnostics, but does not silently replace the official final score. RL callers
can still consume every step's `benchmark_reward`.

The HTTP error envelope includes `error_kind` and `retryable`. Capacity,
session-loss, and session-conflict failures remain retryable. Invalid tasks and
deterministic BrowserGym reset preconditions return a non-retryable error so
the agent does not spend its rollout retry budget on an unchanged deployment.

When `record_video: true`, BrowserGym writes video files under the session
artifact directory. The `close` response is emitted after browser shutdown has
flushed those files and includes the `session_id` plus typed
`recording_artifacts` references. Empty in-progress files are not reported.
