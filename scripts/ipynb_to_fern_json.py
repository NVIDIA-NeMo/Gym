# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert Jupyter notebooks (.ipynb) to Fern NotebookViewer JSON+TS format.

Reads notebook JSON and outputs a minimal format with cells array:
  { "cells": [
      { "type": "markdown", "source": "...", "source_html": "<rendered>" },
      { "type": "code", "source": "...", "language": "python",
        "source_html": "<pygments-html>",
        "outputs": [{ "type": "text"|"image", "data": "...", "format"?: "plain"|"html" }]
      }
  ] }

Markdown cells are pre-rendered to HTML via markdown-it-py (CommonMark + tables +
strikethrough). Code cells are pre-rendered with Pygments (friendly style, inline
styles via noclasses=True so no CSS class dependency).

Writes both <name>.json (canonical data) and <name>.ts (default-export wrapper that
MDX imports — Fern's bundler doesn't follow JSON imports cleanly).

Mirrors NVIDIA-NeMo/DataDesigner/fern/scripts/ipynb-to-fern-json.py for consistency.

Usage:
  python3 scripts/ipynb_to_fern_json.py input.ipynb -o output.json
  python3 scripts/ipynb_to_fern_json.py input.ipynb -o fern/components/notebooks/foo.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    from markdown_it import MarkdownIt
    HAS_MARKDOWN_IT = True
except ImportError:
    HAS_MARKDOWN_IT = False

try:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name
    from pygments.util import ClassNotFound
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False

# CommonMark-compliant markdown renderer with table + strikethrough +
# raw-HTML support. Used to pre-render markdown cell sources to HTML at
# build time so NotebookViewer doesn't have to ship a JS markdown parser.
_MD = (
    MarkdownIt("commonmark", {"html": True, "linkify": False, "breaks": False})
    .enable("table")
    .enable("strikethrough")
    if HAS_MARKDOWN_IT
    else None
)

COLAB_BADGE_RE = re.compile(
    r"colab\.research\.google\.com/(?:assets/colab-badge\.svg|github/)",
    re.IGNORECASE,
)


def _join_source(source: list | str) -> str:
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def get_language(metadata: dict) -> str:
    info = metadata.get("kernelspec", {}) or {}
    lang = info.get("language", "python")
    return "python" if lang == "python3" else lang


def highlight_code(source: str, language: str) -> str | None:
    if not HAS_PYGMENTS:
        return None
    try:
        lexer = get_lexer_by_name(language, stripall=True)
    except ClassNotFound:
        return None
    formatter = HtmlFormatter(noclasses=True, style="friendly", nowrap=True)
    return highlight(source, lexer, formatter)


def is_colab_badge_cell(cell: dict) -> bool:
    """True if the cell is a redundant Colab badge cell. NotebookViewer renders its own banner."""
    if cell.get("cell_type") != "markdown":
        return False
    src = _join_source(cell.get("source", []))
    return bool(COLAB_BADGE_RE.search(src))


def extract_outputs(outputs: list) -> list[dict]:
    result: list[dict] = []
    for out in outputs:
        out_type = out.get("output_type", "")
        if out_type == "stream":
            text = _join_source(out.get("text", []))
            if text.strip():
                result.append({"type": "text", "data": text.rstrip("\n"), "format": "plain"})
        elif out_type in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                result.append({"type": "image", "data": b64})
            elif "text/html" in data:
                html = data["text/html"]
                if isinstance(html, list):
                    html = "".join(html)
                if html.strip():
                    result.append({"type": "text", "data": html, "format": "html"})
            elif "text/plain" in data:
                text = data["text/plain"]
                if isinstance(text, list):
                    text = "".join(text)
                if text.strip():
                    result.append({"type": "text", "data": text.rstrip("\n"), "format": "plain"})
        elif out_type == "error":
            tb = "\n".join(out.get("traceback", []))
            if tb:
                result.append({"type": "text", "data": tb, "format": "plain"})
    return result


def convert_cell(cell: dict, default_language: str) -> dict:
    cell_type = cell.get("cell_type", "code")
    source = _join_source(cell.get("source", [])).rstrip("\n")
    result: dict = {"type": cell_type, "source": source}
    if cell_type == "code":
        result["language"] = default_language
        source_html = highlight_code(source, default_language)
        if source_html:
            result["source_html"] = source_html
        raw_outputs = cell.get("outputs", [])
        if raw_outputs:
            extracted = extract_outputs(raw_outputs)
            if extracted:
                result["outputs"] = extracted
    elif cell_type == "markdown" and source:
        # Pre-render markdown to HTML at build time. NotebookViewer renders
        # this directly, side-stepping the JS-side markdown parser that
        # doesn't handle blockquotes, fenced code, tables, or nested lists.
        if _MD is not None:
            result["source_html"] = _MD.render(source)
    return result


def convert_notebook(ipynb_path: Path) -> tuple[dict, int]:
    """Convert a .ipynb file to Fern format. Returns (data, n_skipped_colab_cells)."""
    with open(ipynb_path, encoding="utf-8") as f:
        nb = json.load(f)
    metadata = nb.get("metadata", {})
    default_language = get_language(metadata)
    raw_cells = nb.get("cells", [])
    skipped = 0
    cells = []
    for cell in raw_cells:
        if is_colab_badge_cell(cell):
            skipped += 1
            continue
        # Skip empty markdown cells
        if cell.get("cell_type") == "markdown" and not _join_source(cell.get("source", [])).strip():
            continue
        cells.append(convert_cell(cell, default_language))
    return {"cells": cells}, skipped


def write_ts_export(data: dict, ts_path: Path) -> None:
    """Write a .ts file that exports the notebook data inline (MDX imports the .ts, not the .json)."""
    cells_json = json.dumps(data["cells"], indent=2, ensure_ascii=False)
    ts_path.write_text(
        f"/** Auto-generated by scripts/ipynb_to_fern_json.py — do not edit */\n"
        f"export default {{ cells: {cells_json} }};\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    if not argv or "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    input_path = Path(argv[0])
    output_path: Path | None = None
    if "-o" in argv:
        idx = argv.index("-o")
        if idx + 1 < len(argv):
            output_path = Path(argv[idx + 1])
    elif len(argv) >= 2:
        # Back-compat: positional output path
        output_path = Path(argv[1])

    if not output_path:
        output_path = input_path.with_suffix(".json")
    if output_path.suffix not in (".json", ".ts"):
        output_path = output_path.with_suffix(".json")

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        return 1

    if not HAS_MARKDOWN_IT:
        print(
            "Warning: markdown-it-py not installed. Markdown cells will fall back to "
            "the JS-side renderer (which has known gaps). Install with: pip install markdown-it-py",
            file=sys.stderr,
        )
    if not HAS_PYGMENTS:
        print(
            "Warning: Pygments not installed. Code cells will not have syntax highlighting. "
            "Install with: pip install pygments",
            file=sys.stderr,
        )

    data, skipped = convert_notebook(input_path)

    # Normalize: if output ends in .json, write JSON + .ts; if output ends in .ts, write .ts only
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {output_path}")
        ts_path = output_path.with_suffix(".ts")
        write_ts_export(data, ts_path)
        print(f"Wrote {ts_path}")
    else:  # .ts
        write_ts_export(data, output_path)
        print(f"Wrote {output_path}")

    n_md = sum(1 for c in data["cells"] if c["type"] == "markdown")
    n_code = sum(1 for c in data["cells"] if c["type"] == "code")
    print(f"  {n_md} markdown + {n_code} code cells")
    if skipped:
        print(f"  (skipped {skipped} colab-badge cell{'s' if skipped != 1 else ''})")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
