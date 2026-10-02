# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare both published Safe-Child-LLM age splits from pinned XLSX sources."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "safe_child_llm_benchmark.jsonl"

UPSTREAM_REPOSITORY = "The-Responsible-AI-Initiative/Safe_Child_LLM_Evaluation"
UPSTREAM_REVISION = "f69a651ff5c992c6d423b6a129ade8bf674fb63b"  # pragma: allowlist secret
RAW_BASE_URL = f"https://raw.githubusercontent.com/{UPSTREAM_REPOSITORY}/{UPSTREAM_REVISION}/assets"
SOURCES = {
    "6-12": "6_12_ChildSafeLLM.xlsx",
    "13-17": "13_17_ChildSafeLLM.xlsx",
}
EXPECTED_ROWS_PER_SPLIT = 100
_XML_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _download(filename: str) -> bytes:
    with urllib.request.urlopen(f"{RAW_BASE_URL}/{filename}", timeout=60) as response:
        return response.read()


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(node.text or "" for node in item.findall(".//main:t", _XML_NS)) for item in root]


def _cell_value(cell: ElementTree.Element, strings: list[str]) -> str:
    if cell.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//main:t", _XML_NS))
    value = cell.find("main:v", _XML_NS)
    if value is None or value.text is None:
        return ""
    if cell.get("t") == "s":
        return strings[int(value.text)]
    return value.text


def _column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    result = 0
    for character in letters:
        result = result * 26 + ord(character.upper()) - ord("A") + 1
    return result - 1


def _read_first_sheet(content: bytes) -> list[dict[str, str]]:
    with zipfile.ZipFile(BytesIO(content)) as archive:
        strings = _shared_strings(archive)
        root = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row_node in root.findall(".//main:sheetData/main:row", _XML_NS):
        values: dict[int, str] = {}
        for cell in row_node.findall("main:c", _XML_NS):
            values[_column_index(cell.get("r", "A1"))] = _cell_value(cell, strings)
        if values:
            rows.append([values.get(index, "") for index in range(max(values) + 1)])

    headers = [value.strip() for value in rows[0]]
    return [dict(zip(headers, row, strict=False)) for row in rows[1:] if any(row)]


def _render(sources: dict[str, bytes]) -> str:
    output: list[str] = []
    seen: set[str] = set()
    for age_group, content in sources.items():
        rows = _read_first_sheet(content)
        if len(rows) != EXPECTED_ROWS_PER_SPLIT:
            raise ValueError(f"Safe-Child-LLM split {age_group} has {len(rows)} rows; expected 100")
        for row in rows:
            prompt = row["query"].strip()
            source_index = int(float(row["Index"]))
            identifier = f"safe-child-{age_group}-{source_index:03d}"
            if identifier in seen:
                raise ValueError(f"duplicate Safe-Child-LLM id {identifier}")
            seen.add(identifier)
            output.append(
                json.dumps(
                    {
                        # Paper section 3.3: temperature 0 and a fixed maximum token budget; the pinned
                        # kidsafellm/utils/gpt_api.py wrappers use max_tokens=1024 (and temperature 0.2,
                        # which contradicts the paper; the paper value is used here).
                        "responses_create_params": {
                            "input": [{"role": "user", "content": prompt}],
                            "temperature": 0.0,
                            "max_output_tokens": 1024,
                        },
                        "safe_child_id": identifier,
                        "age_group": age_group,
                        "category": row["category"].strip(),
                        "source": row["source"].strip(),
                        "prompt": prompt,
                        "upstream_revision": UPSTREAM_REVISION,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return "".join(output)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare() -> Path:
    sources = {age_group: _download(filename) for age_group, filename in SOURCES.items()}
    _atomic_write(OUTPUT_FPATH, _render(sources))
    print(f"Wrote {len(SOURCES) * EXPECTED_ROWS_PER_SPLIT} Safe-Child-LLM rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
