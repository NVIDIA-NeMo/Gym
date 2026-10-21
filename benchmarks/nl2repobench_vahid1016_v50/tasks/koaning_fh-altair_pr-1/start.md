## fh-altair Project Introduction and Goals

`fh-altair` is a tiny helper library by Vincent D. Warmerdam that makes it easy to render Altair (Vega-Lite) charts inside FastHTML applications. Altair charts are normally rendered in a web page by calling the `vegaEmbed` JavaScript library with a JSON chart specification. FastHTML, meanwhile, builds HTML pages out of Python "FT" components. `fh-altair` bridges the two worlds:

- It provides `altair_headers`, a list of FastHTML header components (`Script`/`Style` FT objects) that load the Vega, Vega-Lite, and Vega-Embed JavaScript bundles from the jsDelivr CDN, plus a small stylesheet making `.vega-embed` containers full-width and fixing the positioning of the embedded details/summary action menu.
- It provides `altair2fasthtml(chart, vega_options=..., full_width=...)`, a function that serializes an Altair chart to JSON, gives the chart a unique container id, and returns a `Div` FastHTML FT component containing an inline `Script` that calls `vegaEmbed('#<id>', <spec>, <options>)`.
- It keeps a backwards-compatible, deliberately misspelled alias `altair2fasthml` that prints a deprecation-style warning to stdout and delegates to the correctly spelled function.

## Natural Language Instructions (Prompt)

1. Create a Python package named `fh_altair` with a single module `fh_altair/__init__.py` exposing the full public API: `altair_headers`, `altair2fasthtml`, and the misspelled legacy alias `altair2fasthml`.
2. Import `json` and `uuid.uuid4` from the standard library, and import `Div`, `Script`, and `Style` from `fasthtml.common`.
3. Define `altair_headers` as a module-level list of four FastHTML header components in this order:
   - `Script(src="https://cdn.jsdelivr.net/npm/vega@5")`
   - `Script(src="https://cdn.jsdelivr.net/npm/vega-lite@5")`
   - `Script(src="https://cdn.jsdelivr.net/npm/vega-embed@6")`
   - `Style(...)` whose CSS text sets `.vega-embed { width: 100%; display: flex; }` and sets `position: relative` on `.vega-embed details` and `.vega-embed details summary`.
   These headers are meant to be passed as the `hdrs` argument to `fast_app` so the required JavaScript and CSS are always loaded on every page.
4. Implement `altair2fasthtml(chart, vega_options={"actions": False}, full_width=False)`. It converts an Altair chart into a FastHTML FT component:
   - If `full_width` is true, first call `chart.properties(width="container")` so the chart takes the full width of its parent container, reassigning the result to `chart`.
   - Serialize the chart with `chart.to_json()` to produce the Vega-Lite JSON spec string.
   - Generate a unique container id of the form `uniq-<uuid4()>`, e.g. `uniq-` followed by a UUID4 hex string.
   - Return `Div(Script(...), id=chart_id)` where the script text is `vegaEmbed('#<chart_id>', <jsonstr>, <json.dumps(vega_options)>);` — i.e. the chart id prefixed with `#`, followed by the JSON spec and the JSON-encoded options dict, all interpolated into the JS call.
   - Document the parameters (`chart`, `vega_options`, `full_width`) and the return type (a FastHTML FT component containing the target div and embedding script) in a docstring.
5. `vega_options` is passed straight through to the `vegaEmbed` JavaScript library, so arbitrary Vega-Embed options are supported, e.g. `{"renderer": "svg"}` to render SVG instead of canvas, or `{"actions": True}` to enable the default action links (they are disabled by the default `{"actions": False}`).
6. Implement the legacy alias `altair2fasthml(chart)` (note the missing `t` in "html"): it prints a message to stdout explaining that `altair2fasthml` is misspelled and will be deprecated in favour of `altair2fasthtml` in a later release, then returns `altair2fasthtml(chart)`.
7. Provide packaging metadata so the project is pip-installable: a `setup.py` that uses `setuptools.setup` with `find_packages()`, name `fh-altair`, version `0.1.0`, author `Vincent D. Warmerdam`, a short description, the README text as the long description with `text/markdown` content type, `install_requires=["altair", "python-fasthtml"]`, and an extras group `dev` containing `pytest` and `pandas`. Read the long description from `README.md` located next to `setup.py` using `pathlib.Path`.
8. Provide a `README.md` at the project root documenting the installation (`python -m pip install fh-altair`), the header setup via `altair_headers` in `fast_app(hdrs=...)`, the basic `altair2fasthtml(chart)` usage returning a `Div`, the custom `vega_options` behavior (SVG rendering, enabling actions), and the `full_width=True` responsive-width behavior.
9. The package must be importable as `import fh_altair` and all three public names must be importable directly from the package root, e.g. `from fh_altair import altair2fasthtml`.

## Environment Configuration

- Python 3.11.14; no internet access.
- The workspace starts with only this `start.md`. All source, packaging files, and documentation must be written from scratch.
- This is a reconstruction task: implement the target library from this specification. Do not recover or copy an existing target implementation from installed packages, vendored copies, wheel caches, or upstream repositories, and do not seek hidden tests or manipulate test collection/grading.
- The following relevant third-party distributions are already installed (no pins invented; these are the observed versions): `altair==6.0.0`, `python-fasthtml==0.13.1` (with `fastcore==1.12.31`, `starlette==1.0.0`, etc.), `pandas==3.0.1`, `numpy==2.4.3`, `pytest==9.0.2`.
- Ordinary use of these installed dependencies during implementation and testing is allowed; writing your own tests is encouraged.

## fh-altair Project Architecture

```
.
├── README.md
├── setup.py
└── fh_altair/
    └── __init__.py
```

- `fh_altair/__init__.py` contains the entire runtime implementation: the `altair_headers` list and the two conversion functions.
- `setup.py` provides setuptools-based packaging; `find_packages()` picks up `fh_altair`.
- `README.md` doubles as the PyPI long description and is read at build time by `setup.py`.

## API Usage Guide

### `fh_altair.altair_headers`

List of FastHTML FT components (`fasthtml.common.Script` and `Style` instances) to be passed as the `hdrs` of a FastHTML app. Contains three CDN script tags (vega@5, vega-lite@5, vega-embed@6 from jsDelivr) and one `Style` with the `.vega-embed` CSS rules (width 100%, flex display, `position: relative` for the details/summary elements).

```python
from fh_altair import altair_headers
from fasthtml.common import fast_app
app, rt = fast_app(hdrs=altair_headers)
```

### `fh_altair.altair2fasthtml`

**Signature:** `altair2fasthtml(chart, vega_options={"actions": False}, full_width=False)`

- `chart`: an Altair chart object (e.g. `alt.Chart`); must support `.to_json()` and, when `full_width` is true, `.properties(width="container")`.
- `vega_options`: dict serializable via `json.dumps`, forwarded as the options argument of the `vegaEmbed` JS call. Default `{"actions": False}` hides the Vega action menu.
- `full_width`: if `True`, the chart's width property is set to `"container"` so the rendered chart fills the parent's width.
- **Returns:** a `fasthtml.common.Div` FT component whose `id` is `"uniq-" + str(uuid4())` and which contains a single inline `Script` child whose JS text is `vegaEmbed('#<id>