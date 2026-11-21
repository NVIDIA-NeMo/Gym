# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A BIRD-Interact database's column meanings, as the descriptions Auto Ontology's catalog holds for its columns.

auto-ontology-eval builds its BIRD-Interact catalog with the release's column documentation
(``scripts/bird_interact/seed_bird_interact.py`` and ``ontology_sql_eval/ingestion/enrich_graph.py`` of
https://github.com/NVIDIA/auto-ontology-eval at e727e4d6e1e198355099de4803590bb3681793a4, Apache-2.0). This module
writes the same descriptions, so a Gym catalog built with them matches theirs:

- A column's description is its entry in ``<db>_column_meaning_base.json``. A JSONB column's entry holds the column's
  meaning and one meaning per field; each field becomes a ``path: meaning`` line.
- A field's line ends with the values the field holds in the sample rows of ``<db>_schema.txt``, since the documented
  example can differ in form from what is stored (``1.005`` documented, ``'100.50%'`` stored).
- A name in the meanings file is matched to the catalog's without regard to case. A column whose table the catalog
  does not have is matched by its name alone, when exactly one catalog table has it.

Only the standard library is used: the catalog image's build runs it with Auto Ontology's Python.
"""

import json
import re
from pathlib import Path
from typing import Any


# The line before a table's sample rows; one Full schema quotes SQL words, so its heading is quoted too.
SAMPLE_ROWS_HEADINGS = ("First 3 rows:", '"First" 3 rows:')
_TABLE_NAME = re.compile(r'TABLE\s+"?([^"\s(]+)"?\s*\(')
# The tokens of a Python repr of a JSON object: punctuation, a quoted string, a number, None, True or False.
_PUNCTUATION = frozenset("{}[]:,")
_STRING = re.compile(r"""'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*\"""")
_ATOM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|None|True|False")


def sample_cells(schema_text: str) -> dict[str, dict[str, list[str]]]:
    """Each table's sample-row cells by column, from a ``<db>_schema.txt``; names are lower case.

    The rows are a fixed-width table under the heading: the column names, a rule of dashes whose runs give each
    column's span, then the rows, up to a blank line or ``...``. The last column runs to the end of its line.
    """
    tables: dict[str, dict[str, list[str]]] = {}
    lines = schema_text.splitlines()
    table = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if match := _TABLE_NAME.search(line):
            table = match.group(1).lower()
        if line.strip() not in SAMPLE_ROWS_HEADINGS or table is None or index + 2 >= len(lines):
            index += 1
            continue
        spans = [match.span() for match in re.finditer(r"-+", lines[index + 2])]
        names = [lines[index + 1][start:end].strip().lower() for start, end in spans]
        cells: dict[str, list[str]] = {name: [] for name in names}
        index += 3
        while index < len(lines) and lines[index].strip() not in ("", "..."):
            for position, (name, (start, end)) in enumerate(zip(names, spans)):
                cell = lines[index][start : None if position == len(spans) - 1 else end].strip()
                if cell:
                    cells[name].append(cell)
            index += 1
        tables[table] = cells
    return tables


def _tokens(text: str):
    """(kind, text, end) for each token of ``text``, up to the first that is none of the repr's tokens."""
    position = 0
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            return
        if text[position] in _PUNCTUATION:
            yield "punctuation", text[position], position + 1
            position += 1
            continue
        match = _STRING.match(text, position) or _ATOM.match(text, position)
        if match is None:
            return
        yield ("string" if match.re is _STRING else "atom"), match.group(), match.end()
        position = match.end()


def stored_fields(cell: str) -> dict[str, str]:
    """``path: value`` for each scalar field of a sample cell that holds a JSON object's Python repr.

    The schema file cuts long rows, so the repr may end anywhere: a field the cut reaches is left out, and so is
    everything after a token that is not part of a repr. A nested field's path joins its keys with dots. Lists are
    skipped. A value keeps its literal form, so ``'100.50%'`` shows that the field holds text.
    """
    fields: dict[str, str] = {}
    keys: list[str | None] = []  # The key of each open object, None for the outermost.
    key = None
    reading_key = False
    lists = 0  # Open lists: everything inside one is skipped.
    for kind, token, end in _tokens(cell):
        if lists:
            lists += {"[": 1, "]": -1}.get(token, 0) if kind == "punctuation" else 0
            continue
        if kind == "punctuation":
            if token == "{":
                keys.append(key)
                key, reading_key = None, True
            elif token == "}":
                if keys:
                    keys.pop()
                key, reading_key = None, False
            elif token == "[":
                lists = 1
            elif token in ",:":
                # A comma opens the next key, a colon its value; a stray "]" changes nothing.
                reading_key = token == ","
        elif reading_key and kind == "string":
            key = token[1:-1]
        elif key is not None:
            if end < len(cell):
                fields[".".join([name for name in keys if name is not None] + [key])] = token
            key = None
    return fields


def description(meaning: Any, cells: list[str]) -> str:
    """A column's description from its meaning entry and its sample cells."""
    if not isinstance(meaning, dict):
        return str(meaning)
    stored: dict[str, list[str]] = {}
    for cell in cells:
        for path, value in stored_fields(cell).items():
            values = stored.setdefault(path, [])
            if value != "None" and value not in values:
                values.append(value)
    lines = [str(meaning.get("column_meaning", "")).strip()]

    def add_fields(fields: dict[str, Any], prefix: str) -> None:
        for name, value in fields.items():
            if isinstance(value, dict):
                add_fields(value, f"{prefix}{name}.")
                continue
            line = f"{prefix}{name}: {value}"
            if values := stored.get(f"{prefix}{name}"):
                line += f" Stored values in sample rows: {', '.join(values)}."
            lines.append(line)

    add_fields(meaning.get("fields_meaning") or {}, "")
    return "\n".join(line for line in lines if line)


def column_descriptions(database: str, database_dir: Path) -> list[tuple[str, str, str]]:
    """(table, column, description) for each entry of the database's column meanings, in the file's order.

    ``database_dir`` holds ``<db>/<db>_column_meaning_base.json`` and ``<db>/<db>_schema.txt``. A database without
    column meanings has none.
    """
    meanings_file = database_dir / database / f"{database}_column_meaning_base.json"
    if not meanings_file.is_file():
        return []
    schema_file = meanings_file.with_name(f"{database}_schema.txt")
    cells = sample_cells(schema_file.read_text()) if schema_file.is_file() else {}
    descriptions = []
    for key, meaning in json.loads(meanings_file.read_text()).items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _, table, column = parts
        column_cells = cells.get(table.lower(), {}).get(column.lower(), []) if isinstance(meaning, dict) else []
        descriptions.append((table, column, description(meaning, column_cells)))
    return descriptions


def catalog_column(table: str, column: str, catalog: list[tuple[str, str]]) -> tuple[str, str] | None:
    """The catalog's (table, column) that a meanings entry names, or None.

    Names match without regard to case. A table the catalog has is never matched to another table's column; a table
    it lacks is matched by the column's name, when exactly one catalog table has that column.
    """
    exact = [
        entry
        for entry in catalog
        if (entry[0].casefold(), entry[1].casefold()) == (table.casefold(), column.casefold())
    ]
    if exact:
        return exact[0] if len(exact) == 1 else None
    if any(name.casefold() == table.casefold() for name, _ in catalog):
        return None
    by_column = [entry for entry in catalog if entry[1].casefold() == column.casefold()]
    return by_column[0] if len(by_column) == 1 else None
