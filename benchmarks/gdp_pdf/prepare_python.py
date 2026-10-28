# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the GDP.pdf rows for python.yaml: the benchmark rows, each offering the resources server's `python` tool."""

from pathlib import Path

from benchmarks.gdp_pdf.prepare import DATA_DIR, _read_jsonl, _write_jsonl
from benchmarks.gdp_pdf.prepare import prepare as _prepare


OUTPUT_FPATH = DATA_DIR / "gdp_pdf_python_benchmark.jsonl"

# The tool POST /python serves, as the resources server describes it (a test keeps the two equal).
PYTHON_TOOL = {
    "type": "function",
    "name": "python",
    "description": (
        "Run Python in a persistent IPython kernel; the task's source PDF is document.pdf in its working directory."
    ),
    "parameters": {
        "properties": {"code": {"description": "The Python code to run.", "title": "Code", "type": "string"}},
        "required": ["code"],
        "title": "PythonToolRequest",
        "type": "object",
    },
    "strict": None,
}


def prepare() -> Path:
    rows = []
    for row in _read_jsonl(_prepare()):
        row["responses_create_params"]["tools"] = [PYTHON_TOOL]
        rows.append(row)
    _write_jsonl(OUTPUT_FPATH, rows)
    print(f"Wrote benchmark JSONL to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
