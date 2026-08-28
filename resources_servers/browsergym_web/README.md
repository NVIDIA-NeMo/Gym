# BrowserGym Web Resources Server

This stateful server owns the BrowserGym/Playwright context used by the legacy
WebVoyager profile. It exposes the shared Gym web lifecycle:

```text
seed_session -> observe -> step* -> evaluate -> close
```

WebVoyager uses `browsergym/openended`. Browser execution produces screenshot
evidence, while the separate `webvoyager_judge` resource server owns final
answer-and-screenshot scoring.

## Runtime setup

Install the component environment and Chromium:

```bash
uv sync --project resources_servers/browsergym_web
uv run --project resources_servers/browsergym_web playwright install chromium
```

The component pins `browsergym-core==0.14.3`. BrowserGym pins Playwright 1.44,
whose declared greenlet release predates Python 3.13; `overrides.txt` selects
greenlet 3.1.1, which provides compatible CPython 3.13 wheels without changing
the BrowserGym or Playwright API versions.

The committed `data/` files are five schema-validation fixtures, not benchmark
scores. Use `benchmarks/webvoyager/config.yaml` for the legacy benchmark route.

## Isolation and failure semantics

Each live BrowserGym session owns one thread-affine Playwright executor. The
thread-local compatibility shim prevents BrowserGym 0.14.x from sharing its
process-global synchronous Playwright object across independent session
threads. A blocked reset therefore does not serialize unrelated sessions.

Step `execution_ok` reports browser action execution separately from benchmark
reward. Exceptions escaping `Env.step()` are treated as runtime/evaluator
failures rather than correctable policy actions. When `record_video: true`,
finalized non-empty videos are indexed under the session artifact directory and
returned after browser shutdown has flushed them.

The default `site_pool_mode: unmanaged` tracks ownership without claiming that
public websites are isolated. `local_locks` is available to coordinate task
metadata within one process, but it is not a cross-process deployment lock.
