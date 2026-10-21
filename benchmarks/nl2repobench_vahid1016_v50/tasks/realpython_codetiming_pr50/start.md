## codetiming Project Introduction and Goals

`codetiming` is a small, dependency-free Python library providing a flexible, customizable timer. The central object, `Timer`, can be used in three interchangeable ways: as an explicit class (`.start()`/`.stop()`), as a context manager (`with Timer(...)`), or as a decorator (`@Timer(...)`). Timers can report elapsed time through a configurable text template and logger, and named timers accumulate their timings in a shared registry (`Timer.timers`) that supports descriptive statistics (count, total, min, max, mean, median, stdev).

The project goals are:

- Provide `codetiming.Timer` and `codetiming.TimerError` as the public API.
- Keep the package pure standard-library Python with full type annotations.
- Package the project with flit (pyproject.toml only, no setup.py) and ship a `py.typed` marker so type checkers can use the annotations.

## Natural Language Instructions (Prompt)

1. Create an importable Python package named `codetiming` with three source modules: `__init__.py`, `_timer.py`, and `_timers.py`, plus a `py.typed` marker file so the package is recognized as typed.
2. `codetiming.__init__` must expose exactly the public API `Timer` and `TimerError` (imported from `codetiming._timer`), declare them via `__all__`, and define the module-level version string `__version__ = "1.4.0"`. Include a module docstring briefly describing the three usage modes (class, context manager, decorator).
3. Implement `codetiming._timers.Timers`, a subclass of `collections.UserDict` mapping timer names (str) to accumulated time (float):
   - Its `__init__` accepts arbitrary positional and keyword arguments passed to the superclass, and additionally initializes a private per-name list of individual timings backed by `collections.defaultdict(list)`.
   - `add(name, value)` appends the value to the private timing list and accumulates it in the underlying data dict (assumed zero on first add).
   - `clear()` clears both the accumulated data and the timing lists.
   - `__setitem__(name, value)` must raise a `TypeError` with a message stating that the class does not support item assignment and directing the user to `.add()`; other mapping operations (get, delete via `del`, etc.) may continue to work through `UserDict`.
   - `apply(func, name)` applies `func` to the list of timings recorded under `name` and returns the result; it raises `KeyError(name)` if `name` has never been timed.
   - Provide the statistics helpers `count(name)` (len of timings), `total(name)` (sum of timings), `min(name)`, `max(name)`, `mean(name)`, and `median(name)` — each of min/max/mean/median treating an empty timing list as `[0]` via `values or [0]` — and `stdev(name)`, which returns `statistics.stdev` of the timings if at least two exist, `math.nan` for a single timing, and raises `KeyError` for an unknown name.
4. Implement `codetiming._timer.TimerError`, a plain subclass of `Exception` used to report misuse of the timer, and `Timer`, a `@dataclass` that also inherits from `contextlib.ContextDecorator` (giving it decorator and context-manager behavior).
5. `Timer` fields, in order:
   - `timers: ClassVar[Timers]` — a single shared class-level `Timers` registry created once (all named timers report here).
   - `name: Optional[str] = None` — optional timer name; when set, elapsed time is accumulated in the shared registry under this name.
   - `text: Union[str, Callable[[float], str]] = "Elapsed time: {:0.4f} seconds"` — the reporting template.
   - `initial_text: Union[bool, str] = False` — optional text logged when the timer starts (a custom string, or `True` for a default message).
   - `logger: Optional[Callable[[str], None]] = print` — the reporting function; set to `None` to disable reporting.
   - Internal fields excluded from both `init` and `repr`: `_start_time` (initially `None`) and `last` (a float, initially `math.nan`, holding the most recent elapsed time). Because `last` follows fields with defaults, it must be declared after all init fields.
6. `Timer.start()`:
   - Raises `TimerError("Timer is running. Use .stop() to stop it")` if the timer is already running.
   - If both a logger and `initial_text` are set, logs the initial text: a custom string value is `.format(name=self.name)`-formatted; `True` with a name logs `"Timer {name} started"`; `True` without a name logs `"Timer started"`. A falsy `initial_text` logs nothing.
   - Records the current `time.perf_counter()` as the start time.
7. `Timer.stop() -> float`:
   - Raises `TimerError("Timer is not running. Use .start() to start it")` if the timer is not running.
   - Computes the elapsed time, stores it in `last`, clears the start time, logs the report (if a logger is set), accumulates under `self.name` in the shared registry (if named), and returns the elapsed time. Stopping an unnamed timer does not touch the registry.
   - Reporting: if `text` is callable, it is called with the elapsed seconds and its string result is passed to the logger. Otherwise, `text.format(...)` is invoked with the elapsed seconds as the first positional argument plus the keyword attributes `name` (the timer name), `milliseconds` (elapsed × 1000), `seconds` (elapsed), and `minutes` (elapsed / 60), so templates such as `"{name}: {minutes:.1f} minutes"` or `"{milliseconds:.0f} ms"` work in addition to the default `"{}"`-style placeholder.
8. `Timer.__enter__()` calls `start()` and returns the timer itself; `Timer.__exit__(*exc_info)` calls `stop()` unconditionally (even if an exception propagated), takes and ignores the exception info, and returns nothing. Because `Timer` inherits `ContextDecorator`, the instance can also be used as a decorator applied to functions; the wrapped call starts the timer, runs the function, and stops the timer, returning the function's result via the `ContextDecorator` machinery.
9. The `dataclass` must remain a dataclass: `Timer(name="x")` constructs via `__init__`, and instances compare equal iff all init fields match (equality generated by `@dataclass`).
10. All public classes, methods, and modules must carry docstrings (the project enforces 100% docstring coverage with interrogate in CI).
11. Packaging: create a `pyproject.toml` using the `flit_core >=3.2,<4` build backend with project name `codetiming`, author metadata (Geir Arne Hjelle / Real Python), `readme = "README.md"`, dynamic version and description drawn from docstrings/`__version__`, `requires-python = ">=3.7"`, no runtime dependencies, MIT license classifier, `Typing :: Typed` classifier, and the keyword/URL/classifier metadata mirroring the published project. The library must be installable with `pip install .` in the prepared environment.
12. Do not add any runtime dependency on third-party libraries; the package uses only the standard library.
13. Write a brief `README.md` describing the three usage styles, the constructor arguments (`name`, `text`, `initial_text`, `logger`), template attributes (`name`, `milliseconds`, `seconds`, `minutes`), callable `text`, capturing elapsed time via `stop()`/`.last`, and named-timer statistics.

## Environment Configuration

- Python 3.11.14 is the interpreter; the environment is offline (no internet).
- The project workspace initially contains only `start.md`. All package source, packaging metadata, and documentation must be written from scratch.
- Build tooling available in the environment: `flit_core==3.12.0`, `flit==3.12.0`, `pip==24.0`, `setuptools==79.0.1`, `wheel==0.45.1`, plus dev tools (`black==