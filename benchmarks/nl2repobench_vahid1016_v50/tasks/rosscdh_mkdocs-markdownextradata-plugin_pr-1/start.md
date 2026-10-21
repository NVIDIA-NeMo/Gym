## mkdocs-markdownextradata-plugin Project Introduction and Goals

This project reconstructs `mkdocs-markdownextradata-plugin`, a small MkDocs plugin that injects the variables defined in a site's `mkdocs.yml` `extra:` hash — plus optional data loaded from external YAML/JSON "data" directories — into each page's markdown (and page title) before MkDocs renders it to HTML. The plugin renders markdown through a Jinja2 environment, so `{{ customer.name }}` style placeholders in `*.md` files resolve against the merged configuration and extra data.

Goals:

- Provide a `MarkdownExtraDataPlugin` class implementing the MkDocs `BasePlugin` hooks `on_config`, `on_pre_build`, and `on_page_markdown`.
- Read plugin options `data` (folder or comma-separated list of folders with YAML/JSON data files) and `jinja_options` (arbitrary Jinja2 `Environment` keyword arguments), falling back to `_data` folders next to `mkdocs.yml` and inside the docs dir when `data` is unset.
- Package the plugin as an installable distribution exposing the `mkdocs.plugins` entry point `markdownextradata`.

## Natural Language Instructions (Prompt)

1. Create a Python package directory named `markdownextradata` with an empty `__init__.py` (no imports, exports, or side effects; the plugin lives only in `plugin.py`).
2. In `markdownextradata/plugin.py`, define a module-level logger via `logging.getLogger(__name__)` and a module-level constant `CONFIG_KEYS` equal to the list `["site_name", "site_author", "site_url", "repo_url", "repo_name"]`.
3. Define a module-level `str_type` variable: on Python 3 it is simply `str` (the plugin retains a guarded fallback referencing `mkdocs.utils.string_types` only for legacy Python 2, which never executes here).
4. Define the public class `MarkdownExtraDataPlugin` subclassing `mkdocs.plugins.BasePlugin`, with a class attribute `JINJA_OPTIONS` equal to the string `"jinja_options"`.
5. Give the class a `config_scheme` tuple of exactly two entries: `("data", mkdocs.config.config_options.Type(str_type, default=None))` and `(MarkdownExtraDataPlugin.JINJA_OPTIONS, mkdocs.config.config_options.Type(dict, default={}))`.
6. Implement a private helper (name starting with double underscore, conceptually `__add_data__`) taking `(config, namespace, data)`: it splits the namespace string on `os.sep`, prepends the literal key `"extra"`, walks/creates nested dicts under `config`, and assigns the loaded `data` at the final key. Intermediate segments that are absent become empty dicts.
7. Implement `on_config(self, mkdocsConfig, **kwargs)`: reads `self.config[self.JINJA_OPTIONS]`, creates `self.env` as a `jinja2.Environment` with `undefined=jinja2.DebugUndefined` plus the user-supplied Jinja options, and stores the full MkDocs config on `self.mkdocsConfig`.
8. Implement `on_pre_build(self, config, **kwargs)`: resolves the data source directories. If the `data` option is a string, split it on commas and prefix each relative entry with the directory containing `config["config_file_path"]`. If `data` is empty/None, look for `_data` subfolders under both the directory of `config["config_file_path"]` and `config["docs_dir"]`, adding only those that exist. If no folders resolve, return without action.
9. For each existing data folder, recursively glob `**/*.yaml`, `**/*.yml`, and `**/*.json` (chained). For each file, compute the namespace as the file's path relative to the data folder with the extension stripped (path separators become namespace separators), then load YAML files with `yaml.load(..., Loader=yaml.FullLoader)` from the raw bytes and JSON files with `json.loads` of the raw bytes, injecting the result under the computed namespace in `config` via the helper above.
10. Implement `on_page_markdown(self, markdown, page, **kwargs)`: apply template substitution to `page.title` and assign the result back to `page.title`, then return the substituted `markdown` as the hook result.
11. Implement `apply_template(self, template_string)`: build a Jinja2 template from `self.env.from_string(template_string)` and render it with a context formed by merging the entire `self.mkdocsConfig` dict with `self.mkdocsConfig.get("extra")` (extra keys take precedence). If rendering raises `jinja2.exceptions.TemplateSyntaxError`, print an explanatory message to stdout hinting that YAML files may not comply with Python variable naming conventions and that variables can be accessed via the `extra` dictionary, then re-raise the error.
12. Provide `setup.py` reproducing the original distribution metadata: name `mkdocs-markdownextradata-plugin`, version `0.2.5`, description and long description read from `README.md` as `text/markdown`, keywords, project URL, author and email, MIT license, `python_requires='>=3.6'`, `install_requires=['mkdocs', 'pyyaml']`, appropriate trove classifiers, `find_packages(exclude=['*.tests'])`, and the `mkdocs.plugins` entry point mapping `markdownextradata` to `markdownextradata.plugin:MarkdownExtraDataPlugin`. Include the README content (installation, `extra` variables usage, external data files with `_data` fallback and namespace nesting, `extra['...']` access for names that are not valid Python identifiers, and `jinja_options` configuration) in the project.
13. Ensure the package is importable as `import markdownextradata.plugin` (with `markdownextradata` itself importable as an empty namespace) and that installing the project registers the plugin entry point so `plugins: - markdownextradata` works in `mkdocs.yml`.

## Environment Configuration

- Python 3.11.14 is the interpreter. No internet access is available.
- The project workspace initially contains only `start.md`. The target distribution `mkdocs-markdownextradata-plugin` has been removed from the environment; its source must be reconstructed solely from this specification.
- Third-party dependencies observed as installed: `mkdocs==1.6.1`, `Jinja2==3.1.6`, `PyYAML==6.0.3`, `MarkupSafe==3.0.3`, `mergedeep`, `pyyaml_env_tag`, `setuptools==79.0.1`, `wheel==0.45.1`, `pytest==9.0.2`, among other unrelated packages. These satisfy the plugin's runtime requirements (`mkdocs`, `pyyaml`, `jinja2`).
- Task constraint: implement the target library from this specification only. Do not recover or copy an existing target implementation from installed packages, vendored copies, wheel caches, site-packages metadata (e.g. dist-info entry points of the removed package), or upstream repositories. Do not seek hidden tests or manipulate test collection or grading. Ordinary use of unrelated installed dependencies and writing your own tests are allowed. This is a reconstruction task.
- After writing the sources, install the package (e.g. `pip install -e .` or `pip install .`) so that both the import and the `mkdocs.plugins` entry point resolve.

## mkdocs-markdownextradata-plugin Project Architecture

```
mkdocs-markdownextradata-plugin/
├── setup.py
├── README.md
└── markdownextradata/
    ├── __init__.py      # empty module
    └── plugin.py         # MarkdownExtraDataPlugin, CONFIG_KEYS, str_type, log
```

- `markdownextradata/__init__.py` contains no statements; importing the package must succeed without side effects.
- `markdownextradata/plugin.py` holds all logic; it imports `os`, `sys`, `json`, `yaml`, `mkdocs`, `logging`, `jinja2`, `pathlib.Path`, and `itertools.chain` at module level.
- `setup.py` uses `setuptools.setup` with `find_packages` and declares the `mkdocs.plugins` entry point. No other packaging files (pyproject.toml, requirements files, tests) existed in the s