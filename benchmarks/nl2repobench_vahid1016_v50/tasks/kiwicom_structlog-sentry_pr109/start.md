## structlog-sentry Project Introduction and Goals

The `structlog-sentry` project provides a small, focused Python library that integrates [structlog](https://www.structlog.org/) with the [Sentry SDK](https://sentry.io/for/python/). Its single public component, `SentryProcessor`, is a structlog processor which inspects logging events flowing through structlog and forwards them to Sentry: highly severe events are reported as Sentry events (optionally with exception traceback data derived from `sys.exc_info()`), and lower-severity events are recorded as Sentry breadcrumbs. The processor also supports tagging, contextual data, skipping specific log calls, and per-processor logger blacklists.

The goal of the reconstruction is to produce a fully importable, installable package named `structlog_sentry` containing `SentryProcessor` as documented, together with standard packaging metadata (pyproject-based, using the `poetry-core` build backend, which is already available in the environment).

## Natural Language Instructions (Prompt)

1. Create a Python package directory `structlog_sentry` with an `__init__.py` that exposes the single public class `SentryProcessor` (no other public names are required, but private helpers must exist as needed for the class's behavior).
2. Add a `py.typed` marker file inside `structlog_sentry` so consumers can use static type checking; declare the package as typed via packaging metadata if applicable.
3. Add a `pyproject.toml` at the project root that builds with the `poetry.core.masonry.api` backend (build backend `poetry-core`), declares the project name `structlog-sentry`, a plausible stable version (e.g. `2.0.1`), description "Sentry integration for structlog", MIT license, and runtime dependencies on `packaging`, `sentry-sdk`, and `structlog` with permissive version constraints, and includes the `py.typed` file in the built distribution.
4. Also provide a `README.md` describing installation with pip and basic usage with structlog and sentry-sdk, mirroring the documented feature set (processors must be configured after `structlog.stdlib.add_log_level`).
5. Implement `SentryProcessor.__init__` with the documented parameters and defaults: `level` (breadcrumb threshold, default `logging.INFO`), `event_level` (event threshold, default `logging.WARNING`), `active` (enable/disable flag, default `True`), `as_context` (send event_dict as Sentry context, default `True`), `tag_keys` (list of keys, the literal string `"__all__"`, or `None`), `ignore_loggers` (iterable of logger-name patterns), `verbose` (annotate event_dict with actions taken, default `False`), and `hub` (optional `sentry_sdk.Hub` instance, default `None` meaning use `Hub.current`).
6. Implement a module-level helper that converts an arbitrary `exc_info` value into a 3-tuple: a `BaseException` instance becomes `(type(exc), exc, exc.__traceback__)`; a tuple is returned as-is; a truthy value falls back to `sys.exc_info()`; otherwise the value is returned unchanged.
7. On each processor call, the processor must: pop the `sentry_skip` key (defaulting to `False`) from the event_dict; skip all Sentry work when inactive, skipped, or the logger name matches an ignored logger pattern (using `fnmatch`, combining Sentry SDK's `_IGNORED_LOGGERS` set with the processor's own ignored patterns); otherwise convert the string log level to a numeric `logging` level and report an event if `level >= event_level` and a breadcrumb if `level >= level`, and always return the (possibly modified) event_dict.
8. Event reporting builds a Sentry event: if a real `exc_info` is present, use `sentry_sdk.utils.event_from_exception` with the client's local-variable option (key `include_local_variables`, or `with_locals` on old SDKs, detected by comparing `sentry_sdk.VERSION` with version `"1.17.0"` via `packaging.version`), sets the event's `message`, `level`, and (if present) `logger` from the event_dict, attaches `contexts.structlog` with a copy of the original event_dict when `as_context` is true, and sets `tags` from `tag_keys` (`"__all__"` copies all original key/value pairs; a list selects those keys that are present). The event is sent via `hub.capture_event` wrapped in `sentry_sdk.utils.capture_internal_exceptions`; if a Sentry event id is returned, store it in the event_dict under `"sentry_id"`; in verbose mode record status.
9. Breadcrumb reporting constructs a breadcrumb dict of type `log` with level, category (logger name), message, timestamp, and empty data, plus a hint containing the event dict as `log_record`, and adds it via `hub.add_breadcrumb`, again wrapped in `capture_internal_exceptions`.
10. Logger names are resolved in order: the `logger` key in the event_dict, the `name` attribute of a `_record` entry (stdlib log record), or the `name` attribute of the wrapped logger itself.
11. In verbose mode, ignored loggers annotate the event_dict with `"sentry": "ignored"` during filtering, and if no other status was set the default annotation is `"sentry": "skipped"`; successful sends set `"sentry": "sent"`.
12. Store the original (unmodified) event_dict copy per call so context/tags reflect what the caller passed, and keep the returned event_dict otherwise untouched apart from documented additions.
13. Ensure the package imports cleanly under Python 3.11 with the installed `sentry-sdk`, `structlog`, and `packaging` versions.

## Environment Configuration

- Task constraint: implement the target package from scratch using this specification only. Do not recover or copy an existing target implementation from installed packages, vendored copies, wheel caches, or upstream repositories, and do not seek hidden tests or manipulate test collection/grading. Ordinary use of unrelated installed dependencies and writing your own tests are allowed. This is a reconstruction task.
- The workspace initially contains only `start.md`; all source, packaging metadata, and documentation must be authored by the implementation agent.
- Internet is unavailable. All third-party dependencies needed are already installed in the task image; do not attempt to install anything.
- Observed Python: 3.11.14.
- Observed installed distributions relevant to this task: `sentry-sdk==2.48.0`, `structlog==25.5.0`, `packaging==25.0`, and build tooling `poetry-core==2.5.0`, `setuptools==79.0.1`, `wheel==0.45.1`, `pip==24.0`. Test tooling (pytest, pytest-cov, pytest-mock, pytest-timeout, coverage, flake8 and their transitive dependencies) is also present.
- Because `poetry-core` is installed, the package may be installed with e.g. `pip install -e .` using a `[build-system]` requiring `poetry-core>=1.0.0` with backend `poetry.core.masonry.api` (or the project may be structured so the `structlog_sentry` directory is importable from the workspace root).

## structlog-sentry Project Architecture

```
structlog-sentry/
├── README.md
├── pyproject.toml
├── structlog_sentry/
│   ├── __init__.py      # SentryProcessor and private helpers
│   └── py.typed         # PEP 561 marker (empty file)
└── start.md             # this specification (already present)
```

Notes:
- The entire library lives in `structlog_sentry/__init__.py`; no submodules are required.
- `py.typed` is a zero-byte marker file that must be recreated and included in the package distribution.
- `pyproject.toml` must be a valid PEP 517 project definition; the original project used Poetry metadata under `[tool.poetry]` with `include` listing `structlog_sentry/py.typed`.

## API Usage Guide

### P