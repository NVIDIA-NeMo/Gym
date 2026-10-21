## sphinx-pyproject Project Introduction and Goals

sphinx-pyproject is a small Python library that lets Sphinx users move part of their documentation build configuration out of `conf.py` and into a standard `pyproject.toml` file. Its main component, the `SphinxConfig` class, reads a `pyproject.toml` file, extracts a small subset of PEP 621 project metadata (`name`, `version`, `description`, and the author list), merges that metadata with the free-form `[tool.sphinx-pyproject]` table, and (optionally) injects the resulting values into the global namespace of the Sphinx `conf.py` file. A `collections.abc.Mapping` interface gives access to the free-form table's keys and values so they can be used programmatically.

The library ships two metadata parsers: a PEP 621 parser for the `[project]` table and a Poetry parser for the `[tool.poetry]` table, both selectable at runtime via a `style` keyword argument. The project uses `dom-toml` and `domdf-python-tools` as runtime dependencies. The goal of this task is to reconstruct the entire importable `sphinx_pyproject` package, plus the necessary packaging metadata so that it is installed and importable in the prepared environment.

## Natural Language Instructions (Prompt)

1. Implement the `sphinx_pyproject` package as a small, importable Python package whose public API consists of `SphinxConfig`, `ProjectParser`, and `PoetryProjectParser`.

2. `SphinxConfig` must implement the `typing.Mapping[str, Any]` interface, and expose, additionally, the attributes of the free-form `tool.sphinx-pyproject` table (for example, `config.project` or `config.version`) via `__getattr__`. It must also be constructible without arguments, using `"../pyproject.toml"` as the default `pyproject_file` argument.

3. The constructor signature must be `SphinxConfig(pyproject_file, globalns=None, style="pep621")`. `globalns` must be keyword-only; `style` must be keyword-only. `pyproject_file` must accept any path-like object via `PathLike`.

4. If the given file does not contain the requested metadata, the constructor must raise `BadConfigError` (imported from `dom_toml.parser`). Each error must clearly state the cause of the error, including for absent or dynamic PEP 621 keys.

5. If the file cannot be parsed as TOML, the constructor must raise a `TOMLDecodeError`. Similarly, raise `KeyError` if a config value requested from the mapping is absent.

6. The `SphinxConfig` object must expose the parsed PEP 621-style configuration values as attributes and as mapping keys. So, in addition to ``config["name"]``, the user must also be able to write ``config.name``.

7. All Sphinx configuration values that are *not* valid Python identifiers (e.g. ``author``) must fall back to the values from the free-form table when accessed as attributes. Valid Python identifiers must prefer the value from the free-form table over the PEP 621 value, when both exist.

8. `style` must accept `"pep621"` or `"poetry"`. An unknown style must raise a `ValueError` listing the supported styles.

9. The ``SphinxConfig`` object must lazily create the attribute mapping once, and cache it in the `__dict__` of the instance, so that later calls do not redo all the parsing work.

10. The `ProjectParser` class must subclass `dom_toml.parser.AbstractConfigParser` and implement `parse_name`, `parse_version`, `parse_description`, `parse_author` and wrap these in the `parse()` method. The `keys` property must return `["name", "version", "description", "author"]`.

11. `ProjectParser.get_namespace` must return the `project` table from the TOML file, or raise `BadConfigError` if it is absent. `PoetryProjectParser.get_namespace` must return the `tool.poetry` table, raising a similar error if absent.

12. The `parse_name` method must assert the value is a string (per PEP 621) and return it with underscores replaced by hyphens. `parse_version` must assert the value is a string or an integer and return it as a string. `parse_description` must assert the value is a string and return it unchanged.

13. `parse_author`, in `ProjectParser`, must accept a list of dictionaries (`name` and optionally `email`), collect all non-`None` names, reject any name containing a comma (raising `BadConfigError` with the author's index in the message), reject an empty final list, and join the remaining names into a single string using `domdf_python_tools.words.word_join`, which joins multiple names in an English-style list ("A, B and C"). The `SphinxConfig` object must then expose this joined string as the value of ``config.author``.

14. `ProjectParser.parse` must re-write the ``authors`` key (or ``maintainers`` key, as a fallback) as the ``author`` key before delegating the rest of the work to the parent class's `parse` method. The `set_defaults` argument has no effect here.

15. `PoetryProjectParser` must subclass `ProjectParser` and override `get_namespace` and `parse_author`. Its `parse_author` must accept poetry-style authors (strings of the form ``"Name <email>"``), extract the name and email by regex, convert them to the PEP 621 dictionary style, and then delegate to `ProjectParser.parse_author`.

16. Both parsers must operate on a dictionary of TOML-parsed data (types from `TOML_TYPES`), and any configuration parsing errors must raise the appropriate errors (e.g. `BadConfigError`) with informative messages.

17. A module-level dictionary `project_parser_styles` must map the string names `"pep621"` and `"poetry"` to the respective parser classes. `SphinxConfig` uses this mapping internally; unknown styling must raise an error listing the supported styles.

18. The TOML file must be loaded with `dom_toml.load`, using the `TomlPureDecoder` from `dom_toml.decoder`.

19. The package must also expose `__version__` (as a plain string, e.g. `"0.2.0"`), `__author__`, `__copyright__`, `__license__`, and `__email__` module-level strings, following the docstring and metadata in the upstream project.

20. `ProjectParser` and `PoetryProjectParser` must themselves be parseable and their class-level `__doc__`s should be usable in autodoc-style documentation. Their docstrings should note the required keys and behavior of each parse method.

21. The package must include a `py.typed` marker file to indicate that the package supports type checking, and the build system must be configured to include it in the distribution.

22. Create a `pyproject.toml` for the reconstructed project which declares the project name (`sphinx-pyproject`), version (`0.2.0`), description, authors, license (MIT), keywords, and runtime dependencies (`dom-toml>=0.3.0` and `domdf-python-tools>=2.7.0`, matching the project's `requirements.txt`). Also provide a `README.rst`-style short description ("Move some of your Sphinx configuration into pyproject.toml") as the package's summary text.

23. The library must import cleanly on Python 3.11 using only installed dependencies (`dom_toml`, `domdf_python_tools`, and the standard library). Do not depend on Sphinx itself for imports; the library only manipulates dictionaries and namespaces.

## Environment Configuration

- Python: 3.11.14
- The task Python environment already has all third-party dependencies installed, including `dom-toml==2.1.0` and `domdf-python-tools==3.10.0` (plus `pyproject-parser`, `packaging`, `typing_extensions`, etc.). Testing tools such as `pytest` are available for writing your own tests.
- Internet access is unavailable. Do not attempt to fe