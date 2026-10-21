16. Implement the interactive telemetry subpackage `telemetry/interactive/`: its `__init__.py` re-exports `ping`, `opt_in`, `capture_session` from `flows`; if the import fails, module-level `__getattr__` raises an ImportError mentioning installing with `pip install tabpfn_common_utils[interactive]`. `flows.py` implements: `capture_session(enabled=True)` capturing a `SessionEvent`; `ping(enabled=True)` which, for each of the frequencies daily/weekly/monthly (1/7/30 days), checks state keys `last_pinged_at_{frequency}` (plus the legacy `last_pinged_at` for daily backward compatibility), skips when the last ping is within the delta, otherwise captures a `PingEvent(frequency=...)` and records the current UTC time. A helper `_trigger_prompts(delta_days, max_prompts)` gates on the remote config key `prompt_user`, records/uses `install_date`, honors a `prompt_delta_hours` (default 24) initial quiet period, counts usages (`nr_usages`, compared against config `prompt_nr_usages`, default 5), checks the count of `nr_prompts` against the maximum, and compares against `last_prompted_at`. `_build_prompts()` builds the `NewsletterPrompt` and `IdentityPrompt` prompt specs. `opt_in(enabled=True, delta_days=30, max_prompts=2)` is a no-op unless the runtime kernel is "jupyter" or "ipython" and `_trigger_prompts` passes; it then runs each prompt with `trigger()`/`ask()`/`on_done(result)` in order and finally updates `last_prompted_at` and increments `nr_prompts`. The `prompts/` subpackage provides `base.py` with the `Outcome` literal type ("accepted", "declined", "dismissed", "timeout"), frozen dataclasses `PromptResult(outcome, data)` and `PromptSpec(kind, trigger, ask, on_done)`, `render_html(title, body, hint)` displaying an HTML block via `IPython.display` (HTML-escaping the inputs), and `parse_input(input_prompt, parser, max_retries=3, on_retry_message=...)` — a blocking loop mapping user input through a parser callback returning `(outcome, data)` where `outcome is None` marks invalid input (retried up to `max_retries`, then treated as "dismissed"); a `KeyboardInterrupt` yields "dismissed". `prompts/newsletter.py` and `prompts/identity.py` implement the two prompt classes with `build()` static methods returning PromptSpec instances; newsletter prompts ask for an email (validated by an email regex, empty input declined), post to `<TABPFN_API_URL or https://tabpfn-server-wjedmz7r5a-ez.a.run.app>/newsletter/subscribe` with `json={"email": lowercase email}` and a 5-second timeout, and persist `email` in state; identity prompts ask a yes/no consent question and, on acceptance, generate and persist a random UUID `user_id` in state. Both prompts only trigger when the corresponding state property is unset.
17. Include the PEP 561 marker file `src/tabpfn_common_utils/py.typed` (empty file), a `README.md` describing the package features and usage as a general utility package for TabPFN, and include the Apache 2.0 license reference in the README.

## Environment Configuration

- Python 3.11.14; the interpreter available in the task environment is the target interpreter for this reconstruction.
- The workspace contains only this `start.md`. The agent must create the full project source tree, `pyproject.toml`, README, and any supporting metadata needed to make `tabpfn_common_utils` importable and the distribution installable.
- Internet access is unavailable. Third-party dependencies declared by the project are already installed in the task environment, including posthog==6.9.3, platformdirs==4.5.1, filelock==3.25.2, requests==2.32.5, numpy==2.4.3, pandas==2.3.3, scikit-learn==1.8.0, typing_extensions==4.15.0, plus pytest/pytest-cov/pytest-timeout, ruff, pyright, hatchling, wheel and setuptools. The original `tabpfn-common-utils` distribution has been removed from the environment.
- Task constraint: implement the target library purely from this specification. Do not recover, copy, or decompile an existing implementation of `tabpfn-common-utils` from installed packages, vendored copies, wheel caches, or any upstream repository. Do not attempt to obtain hidden tests or manipulate test collection or grading. Writing your own tests and using third-party packages for their normal purposes is permitted. This is a reconstruction task.
- The `py.typed` marker, `README.md`, and packaging metadata (version 0.2.8) must be authored by the agent; no source files exist in the workspace.

## TabPFN Common Utils Project Architecture

```
.
├── README.md
├── pyproject.toml
└── src/
    └── tabpfn_common_utils/
        ├── __init__.py
        ├── py.typed
        ├── expense_estimation.py
        ├── regression_pred_result.py
        ├── utils.py
        └── telemetry/
            ├── __init__.py
            ├── core/
            │   ├── __init__.py
            │   ├── config.py
            │   ├── decorators.py
            │   ├── events.py
            │   ├── runtime.py
            │   ├── service.py
            │   └── state.py
            └── interactive/
                ├── __init__.py
                ├── flows.py
                └── prompts/
                    ├── __init__.py
                    ├── base.py
                    ├── identity.py
                    └── newsletter.py
```

`pyproject.toml` uses `hatchling` as the build backend with the wheel target listing `src/tabpfn_common_utils` and force-including `src/tabpfn_common_utils/py.typed` at `tabpfn_common_utils/py.typed` inside the wheel. A `prompts/__init__.py` must exist (it may be minimal since prompts are imported as a subpackage).

## API Usage Guide

### `tabpfn_common_utils.__init__`
- `__version__: str` — distribution version string (e.g. `"0.2.8"` once installed, `"unknown"` if metadata is unavailable).

### `tabpfn_common_utils.expense_estimation`
- `VERTEX_GPU_FACTOR: float` — `1e-11`.
- `estimate_duration(num_rows: int, num_features: int, task: Literal["classification","regression"], tabpfn_config: dict = {}, duration_factor: float = VERTEX_GPU_FACTOR, latency_offset: float = 0.0) -> float` — returns `round(base_duration * duration_factor + latency_offset, 3)`. `n_estimators` comes from `tabpfn_config.get("n_estimators", 4 if task == "classification" else 8)`.

```python
from tabpfn_common_utils.expense_estimation import estimate_duration
d = estimate_duration(1000, 20, "regression")
```

### `tabpfn_common_utils.regression_pred_result`
- `RegressionPredictResult(res: Dict[str, Any])` — requires keys `mean`, `median`, `mode`; all keys starting with `quantile_` are collected into `self.quantiles`. Raises `ValueError` if `mean` is neither `np.ndarray` nor `list`. Raises `AssertionError` if values have mixed types. Property `val_type` returns `np.ndarray` or `list`.
- `RegressionPredictResult.to_basic_representation(res) -> Dict[str, List]` (static) — dict of plain lists; arrays converted with `.tolist()`.
- `RegressionPredictResult.from_basic_representation(basic_repr) -> Dict[str, np.ndarray]` (static) — converts the same keys back via `np.array` (quantile keys re-detected by the `quantile_` prefix).

```python
r = RegressionPredictResult({"mean": [1.0], "median": [1.0], "mode": [1.0], "quantile_0.5": [1.0]})
basic = RegressionPredictResult.to_basic_representation(r)
```

### `tabpfn_common_utils.utils`
- `serialize_to_csv_formatted_bytes(data) -> bytes` — accepts `pd.DataFrame`, `pd.Series`, `np.n