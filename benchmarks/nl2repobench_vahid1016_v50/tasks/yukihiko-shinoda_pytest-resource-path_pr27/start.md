## pytest-resource-path Project Introduction and Goals

`pytest-resource-path` is a small pytest plugin that provides two fixtures, `resource_path` and `resource_path_root`, giving tests uniform, absolute `pathlib.Path` access to test resources stored in an isolated directory tree. Instead of placing data files next to test code, resources live under a dedicated resource-root directory (default `tests/testresources/`), mirroring the package/module structure of the requesting test, with an optional directory per test-function name. Directory names for the tests root and the resources root are configurable via `pytest.ini` ini options. The project is packaged with setuptools via `pyproject.toml`, registered as a pytest-11 entry point plugin, typed (`py.typed`), and versioned at 1.4.1.

## Natural Language Instructions (Prompt)

1. Create an importable Python package named `pytest_resource_path` implementing a pytest plugin, plus the packaging metadata (`pyproject.toml`, `LICENSE`, `README.md`, `MANIFEST.in`) needed to build and install it as a distribution named `pytest-resource-path` at version 1.4.1.
2. `pyproject.toml` must use the setuptools build backend, declare `dependencies = ["pytest>=3.5.0"]`, `requires-python = ">=3.7"`, set the readme to `README.md`, include the MIT `LICENSE` file, package data `py.typed` for package `pytest_resource_path`, `zip-safe = false`, and register the pytest-11 entry point `resource-path = "pytest_resource_path"` so the plugin auto-loads once installed.
3. The package `__init__.py` must re-export public names from its submodules via star-imports, aggregate an `__all__` list from each submodule's `__all__`, and define `__version__ = "1.4.1"`.
4. Implement an `exceptions` module defining `Error` (subclass of `Exception`) and `LogicError` (subclass of `Error`), both exported via the module's `__all__`.
5. Implement a `path_factory` module exposing a `PathFactory` class with class methods: `create_path_as_same_as_file_name(item)` returning the file's path with its suffix stripped (`path.parent / path.stem`), and `create_absolute_path_by_function(item)` returning the resolved absolute `Path` of the source file in which `item` (a function or method) is defined. A private helper resolves the module of `item` (`sys.modules[item.__module__]`), reads its `__file__`, and raises `ValueError("File not found.")` if that is falsy.
6. Implement an `absolute_path_factory` module exposing `AbsolutePathFactory(path_target)` storing `path_target`. Method `create_by_function(item)` derives an absolute path from the item's module file via `PathFactory` then delegates to `create_by_path`. `create_by_path(path)` scans `path.parts` from the beginning, records the index of the last part equal to `str(self.path_target)`, and returns the ancestor directory of `path` at relative depth `index - index_tests - 1` below the top (i.e., `path.parents[index - index_tests - 1]`), effectively yielding the directory whose last component's name matches `path_target` when `path_target` is a relative path such as `Path("tests")`. If the scan cannot find the target name, raise `LogicError` with a diagnostic message including the path, the target string, and the indices.
7. Implement a `path_to_resource_factory` module exposing `PathToResourceFactory(path_tests=None, path_test_resources=None)`. Class constants `DIRECTORY_NAME_TESTS_DEAFAULT = "tests"` and `DIRECTORY_NAME_TEST_RESOURCES_DEFAULT = "testresources"` (note the misspelled first constant is part of the public API). Missing arguments default to `Path` of those constant strings. The constructor stores an `AbsolutePathFactory` over the initialized tests path as `absolute_path_factory` and the resources path as `path_test_resources`.
8. `PathToResourceFactory.create(item)` returns a `Path` computed as: the resources-root directory, joined with the item's file path relative to the tests root directory (with the file suffix stripped, so directories mirror package/module structure), joined with the item's `__name__` (a directory per test function). Class names are ignored.
9. `PathToResourceFactory.create_path_to_resource_root(path)` accepts a path (e.g., the pytest request path), runs it through `AbsolutePathFactory.create_by_path` to find the tests root, and returns the tests root joined with `path_test_resources`. This backs the `resource_path_root` fixture.
10. Implement a `pytest_backward_compatibility` module with `get_path(request)` returning `request.path`, falling back to `Path(str(request.fspath))` on `AttributeError` for pytest older than 7.0.0.
11. Implement the plugin module `pytest_resource_path/pytest_resource_path.py` defining ini keys `resource-path.directory-name-tests` and `resource-path.directory-name-test-resources` together with defaults matching the class constants above, a `pytest_addoption(parser)` hook registering both ini options (with those human-readable help strings and defaults), a helper `create_path_to_resource_factory(request)` that reads both ini values as `Path` and constructs a `PathToResourceFactory`, a function-scoped `resource_path` fixture returning `create(...)` applied to `request.function` (an absolute `Path` to the per-test resource directory), and a package-scoped `resource_path_root` fixture returning the resolved resource-root `Path` (defaults to `<tests root>/testresources`). Export all six public names in `__all__`.
12. Include an empty `pytest_resource_path/py.typed` marker file and ensure it ships in the installed package.
13. Write a `README.md` describing features, installation, usage of both fixtures (including the `Path(f'{resource_path}.txt')` idiom to omit the per-method directory), the expected directory layouts, and the ini customization of directory names. Create a `LICENSE` file with the MIT license text (copyright Yukihiko Shinoda). `MANIFEST.in` may contain `recursive-include docs *`.
14. Ensure the package imports cleanly with only pytest and the standard library; do not depend on `invoke` or other dev-group tools at runtime. A `tasks.py` invoke task file is optional and not required for grading.
15. Constraints: reconstruct strictly from this specification. Do not recover, copy, or decompile the original target implementation from installed packages, wheel caches, vendored copies, or upstream repositories; do not seek hidden tests or manipulate test collection or grading. Ordinary use of installed third-party dependencies and writing your own tests are allowed. This is a reconstruction task.

## Environment Configuration

- Python: 3.11.14 (observed task interpreter).
- Installed distributions relevant to the task: `pytest==9.0.2`, `iniconfig==2.3.0`, `pluggy==1.6.0`, `packaging==25.0`, `setuptools==79.0.1`, `wheel==0.45.1`, `pip==24.0`. Other installed distributions (flake8, mypy-adjacent tooling, pipx, pytest-timeout, etc.) are unrelated tooling and not runtime requirements.
- The project workspace starts with only `start.md`. No internet access is available; the target distribution has been removed and must be rebuilt from this specification, not copied from any installed package or cache.
- The package must be installable (e.g., `pip install -e .` or `pip install .`) using the setuptools build backend, and must register the `pytest11` entry point so the fixtures become available after installation.

## pytest-resource-path Project Architecture

```plaintext
pytest-resource-path/
+-- MANIFEST.in
+-- README.md
+-- LICENSE
+-- pyproject.toml
+-- pytest_resource_path/
|   +-- __init__.py
|   +-- absolute_path_factory.py
|   +-- excep