## DNRY-Config Project Introduction and Goals

DNRY-Config is a small, multi-source configuration library for Python. Its goal is to simplify configuration loading and overriding: an application registers any number of configuration sources (YAML files, command-line arguments, environment variables, in-memory dictionaries, or arbitrary callables), and the library loads and deep-merges them in order into a single namespaced configuration object. Later sources override earlier ones on key collisions; lists are concatenated; nested dictionaries merge recursively.

Configuration values are accessed through a `ConfigSection` object using colon-separated keys (for example `conf.get("app:message")` accesses the `message` key inside the `app` mapping). The package defines abstract interfaces (`IConfigSection`, `IConfigSource`, `IConfigFactory`) so users can supply their own section or source implementations, and allows overriding the merge behavior and section class on the factory itself.

## Natural Language Instructions (Prompt)

1. Create an importable Python package `dnry.config` with the following subpackages: `dnry.config.arg`, `dnry.config.delegate`, `dnry.config.environ`, `dnry.config.in_memory`, and `dnry.config.yaml`.
2. Define the abstract interfaces in `dnry/config/types.py`:
   - `IConfigSection` (ABC) with abstract methods `get(key: str)`, `get_section(key: str) -> Union["IConfigSection", None]`, and `has(key: str) -> bool`.
   - `IConfigSource` (ABC) with abstract method `load(fact: "IConfigFactory", conf: IConfigSection) -> dict`.
   - `IConfigFactory` (ABC) with abstract methods `add_source(source: IConfigSource) -> None`, `build() -> IConfigSection`, and `add_configuration(conf: IConfigSection)`.
3. Implement `ConfigSection` in `dnry/config/section.py`. Its constructor takes a single `root: dict` and assigns it directly into `self.__dict__` so that the section's internal dictionary is the caller's dictionary. Key lookups walk the dictionary splitting the key on `:`. `get` returns the value or `None` if any path segment is missing; `has` returns a boolean for presence of the full path; `get_section` returns a new `ConfigSection` wrapping the value at the key if that value is a `dict`, otherwise `None`.
4. Implement helper functions in `dnry/config/helpers.py`:
   - `merge(a: dict, b: dict) -> dict` — a deep merge where values from `b` take precedence; if both values at a key are lists, concatenate them (left then right); if both values are dicts, merge recursively; keys present only in `a` are kept. It must not mutate either input and returns a fresh dict.
   - `explode(items: Dict[str, Union[str, int, bool, list]], delimiter: str = ':') -> dict` — converts a flat map with delimited keys into a nested map (e.g. `{"a:b": "v"}` becomes `{"a": {"b": "v"}}`). `None` values are dropped; if a non-final key exists but does not hold a dict, the remainder of that key is skipped.
5. Implement `ConfigFactory` in `dnry/config/factory.py`:
   - Constructor `__init__(self, sources: List[IConfigSource] = None)` initializes empty internal lists; `sources` may be `None` or a list of sources.
   - Class attributes `Section = ConfigSection` and `merge = merge` allow users to override the section class (which must accept a single dict in `__init__`) and the merge function.
   - `add_source(source)` appends a source; `add_configuration(conf)` appends an already-built `IConfigSection`.
   - `build()` iterates sources in order, merging each source's `load` result into an accumulated context; each `load` call receives the factory as `fact` and a `ConfigFactory.Section(context)` instance built from the current accumulated context as `conf`. After all sources, explicit configurations added via `add_configuration` are merged in, using `vars(conf)` on each. The final result is `ConfigFactory.Section(context)` and is returned.
6. Implement `YamlSource` in `dnry/config/yaml/source.py` (exported from `dnry.config.yaml`):
   - Constructor `__init__(self, path: Union[str, List[str]], required: bool = True, **kwargs)`. A single string path is wrapped into a one-element list. The YAML loader defaults to `yaml.SafeLoader` and can be overridden via the `loader` keyword.
   - `load` picks the first path in the list that exists on disk. If none exists and `required` is `True`, raise `RuntimeError` with a message listing the candidate paths joined by commas, prefixed with "Configuration Error: None of these paths could be found: ". If none exists and `required` is `False`, return an empty dict. Otherwise open the existing path and return `yaml.load(f, Loader=self.__loader) or dict()` — an empty/None document yields an empty dict.
7. Implement `InMemorySource` in `dnry/config/in_memory/source.py` (exported from `dnry.config.in_memory`): constructor takes a `data: dict`, and `load` simply returns that dictionary.
8. Implement `EnvironmentSource` in `dnry/config/environ/source.py` (exported from `dnry.config.environ`):
   - Constructor `__init__(self, prefix: str = None)`; a `None`/falsy prefix becomes `""`.
   - `load` scans `os.environ` for variables whose name starts with the prefix, strips the prefix, left-strips underscores from the remainder, then explodes the result into a nested map using `"_"` as the delimiter — so `ENV_SECTION_KEY="hello"` (with prefix `ENV_`) is readable as `conf.get("SECTION:KEY")`.
9. Implement `ArgumentSource` in `dnry/config/arg/source.py` (exported from `dnry.config.arg`):
   - Constructor `__init__(self, argument_parser: ArgumentParser, argv: List[str] = None)` stores the parser and optional argv list (passed to `parse_args`, so `None` means system `sys.argv`).
   - `load` calls `self.__ap.parse_args(self.__argv)`, converts the namespace to a dict with `vars`, and explodes it with the default `:` delimiter so dotted argument destinations containing colons become nested sections.
10. Implement `DelegateSource` in `dnry/config/delegate/source.py` (exported from `dnry.config.delegate`): constructor takes a callable `Callable[[IConfigFactory], Union[dict, None]]`; `load` invokes it with the factory and section arguments and returns the result or `{}` if it is falsy.
11. Top-level package `dnry.config` must re-export, from its `__init__.py`: `ConfigFactory` (from `factory`), `ConfigSection` (from `section`), and `IConfigSection`, `IConfigSource`, `IConfigFactory` (from `types`).
12. Provide packaging metadata compatible with the observed project: a `setup.py` declaring the distribution name `dnry_config`, version `0.1.4` (or later 0.x), requires `PyYAML` as the only install dependency, lists all six packages above, uses a Markdown long description from `README.md`, and declares the MIT license. Provide a `requirements.txt` listing PyYAML and a `README.md` describing the library and showing the quick-start usage (instantiate the factory with a `YamlSource`, call `build()`, read values with colon-separated keys). Ensure `pip install .` works offline from the workspace.
13. All modules must be importable on Python 3.11 with only PyYAML as a third-party runtime dependency. Do not use deprecated YAML 1.x `yaml.load` without a Loader argument.

## Environment Configuration

- Python 3.11.14 is the interpreter (seen as `python3`/`pip` in the task image).
- Installed third-party distributions available offline: PyYAML==6.0.3 (the only runtime dependency), plus pytest==9.0.2 and pytest-timeout==2.4.0 for writing your own tests, and setuptools==79.0.1 / wheel==0.45.1 for building/installing the package. Other installed tools (flake8 7.3.0, click 8.3.1, etc.)