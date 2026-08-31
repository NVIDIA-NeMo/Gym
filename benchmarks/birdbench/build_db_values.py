# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Maintainer-only script: schema + example-row selection for BIRD dev questions.

BM25 search over each database's distinct column values uses ``bm25s`` (pure-Python,
numpy/scipy-backed, no JVM) -- each database's index is built in-memory and used
immediately, with nothing persisted to disk ahead of time.

Per database, two sources of "interesting" column values feed into row selection:
- A baseline sample (``_sample_table_values``): a couple of distinct values per column,
  independent of any question, computed once per database.
- Per-question BM25 hits (``_relevant_hits_for_question``): column values whose text
  substring-matches the question well.

Rather than printing isolated per-column example values as ``-- example: [...]`` comments,
we look up one real row per interesting (table, column, value) -- keeping every shown row
internally consistent -- and return it as an ``INSERT`` statement, ready to slot into a
schema dump the way ``prepare.py`` already renders one.

Not part of ``prepare()`` and not run automatically -- ``prepare_with_bm25.py`` is what
normally drives this module. Run directly only to produce the standalone
``data/db_values.json`` artifact, e.g. for inspection.

Usage::

    pip install bm25s nltk
    python -m benchmarks.birdbench.build_db_values
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path
from sqlite3 import Cursor
from typing import Any, Dict, List, Tuple

from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "db_values.json"

_VALUE_MAX_LEN = 40
_SAMPLE_VALUES_PER_COLUMN = 2  # baseline sample size per column, independent of any question
_NGRAM_MAX_N = 8
_TOP_K_RETRIEVE = 10
_TOP_K_KEEP = 20
_SCORE_THRESHOLD = 0.85
_MAX_ROWS_PER_TABLE = 10  # matches prepare.py's INSERT-chain truncation cap


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------------------
# Schema introspection
# --------------------------------------------------------------------------------------


def _table_names(cur: Cursor) -> List[str]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [name for (name,) in cur.fetchall() if name != "sqlite_sequence"]


def _column_names(cur: Cursor, table_name: str) -> List[str]:
    cur.execute(f"PRAGMA table_info('{table_name}')")
    return [row[1] for row in cur.fetchall()]


def _create_table_statements(cur: Cursor, table_names: List[str]) -> Dict[str, str]:
    """The database's own literal ``CREATE TABLE`` text -- no need to reconstruct DDL."""
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL;")
    return {name: sql for name, sql in cur.fetchall() if name in table_names}


# --------------------------------------------------------------------------------------
# Value sources: baseline per-column sample + per-question BM25 retrieval
# --------------------------------------------------------------------------------------


def _collect_column_values(cur: Cursor, table_names: List[str]) -> List[Dict[str, str]]:
    """Distinct, non-numeric, short string values from every column -- the BM25 corpus."""
    corpus: List[Dict[str, str]] = []
    for table_name in table_names:
        for column_name in _column_names(cur, table_name):
            try:
                cur.execute(f'SELECT DISTINCT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL;')
            except sqlite3.OperationalError:
                continue
            for (value,) in cur.fetchall():
                if not isinstance(value, str) or _is_number(value):
                    continue
                if 0 < len(value) <= _VALUE_MAX_LEN:
                    corpus.append({"table": table_name, "column": column_name, "contents": value})
    return corpus


def _sample_table_values(cur: Cursor, table_names: List[str]) -> Dict[Tuple[str, str], List[Any]]:
    """A couple of distinct values per column, independent of any question.

    Computed once per database, not re-run per question.
    """
    sampled: Dict[Tuple[str, str], List[Any]] = {}
    for table_name in table_names:
        for column_name in _column_names(cur, table_name):
            cur.execute(
                f"""
                SELECT "{column_name}" FROM (
                    SELECT DISTINCT "{column_name}" FROM "{table_name}"
                    WHERE "{column_name}" IS NOT NULL AND "{column_name}" != ''
                ) LIMIT {_SAMPLE_VALUES_PER_COLUMN};
                """
            )
            values = [row[0] for row in cur.fetchall()]
            # Truncate long strings so an oversized sampled value doesn't itself balloon
            # the shown row/prompt.
            values = [v[:_VALUE_MAX_LEN] if isinstance(v, str) else v for v in values]
            if values:
                sampled[(table_name, column_name)] = values
    return sampled


def _obtain_n_grams(text: str, max_n: int) -> List[str]:
    """Word n-grams via nltk's tokenizer -- unlike a bare word-regex, nltk keeps punctuation
    as its own token, which matters here: BIRD's evidence hints are full of literal
    single-character values ("bond_type = '#'"), and a word-only tokenizer would silently
    drop the very tokens retrieval needs to find them.
    """
    import nltk
    from nltk import ngrams
    from nltk.tokenize import word_tokenize

    # nltk >=3.8.2 needs the newer "punkt_tab" resource; older nltk (e.g. pinned to stay
    # compatible with an older Python) only knows the classic "punkt" resource, and can raise
    # a raw OSError (not the usual LookupError) when asked to look up "punkt_tab" at all.
    # Probe both defensively rather than hard-requiring one -- a failed *check* here shouldn't
    # block tokenization if a compatible resource is already installed.
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            break
        except Exception:
            try:
                nltk.download(resource, quiet=True)
                break
            except Exception:
                continue

    tokens = word_tokenize(text)
    return [" ".join(gram) for n in range(1, max_n + 1) for gram in ngrams(tokens, n)]


def _substring_match_percentage(query: str, target: str) -> float:
    """What fraction of ``query`` is covered by its longest substring found in ``target``."""
    query, target = query.lower(), target.lower()
    best = 0
    for i in range(len(query)):
        for j in range(i + 1, len(query) + 1):
            if query[i:j] in target:
                best = max(best, j - i)
    return best / len(query) if query else 0.0


def _build_retriever(corpus: List[Dict[str, str]]):
    import bm25s

    corpus_tokens = bm25s.tokenize([c["contents"] for c in corpus], stopwords=None, show_progress=False)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens, show_progress=False)
    return retriever


def _retrieve_hits_for_queries(retriever, queries: List[str]) -> Dict[str, List[Dict[str, str]]]:
    import bm25s

    unique_queries = list(dict.fromkeys(queries))
    if not unique_queries:
        return {}

    k = min(_TOP_K_RETRIEVE, len(retriever.corpus))
    query_tokens = bm25s.tokenize(unique_queries, stopwords=None, show_progress=False)
    results, _scores = retriever.retrieve(query_tokens, k=k, show_progress=False)

    query_to_hits: Dict[str, List[Dict[str, str]]] = {}
    for row_idx, query in enumerate(unique_queries):
        hits = [results[row_idx, col] for col in range(results.shape[1])]
        seen = set()
        deduped = []
        for hit in hits:
            key = (hit["table"], hit["column"], hit["contents"])
            if key not in seen:
                seen.add(key)
                deduped.append(hit)
        query_to_hits[query] = deduped
    return query_to_hits


def _relevant_hits_for_question(retriever, question: str) -> List[Dict[str, str]]:
    """BM25 hits whose text substring-matches ``question`` well, deduped and capped."""
    queries = _obtain_n_grams(question, _NGRAM_MAX_N) + [question]
    query_to_hits = _retrieve_hits_for_queries(retriever, queries)

    candidates: List[Dict[str, str]] = []
    seen = set()
    for query in queries:
        for hit in query_to_hits.get(query, []):
            key = (hit["table"], hit["column"], hit["contents"])
            if key not in seen:
                seen.add(key)
                candidates.append(hit)

    scored = []
    for idx, hit in enumerate(candidates):
        score = _substring_match_percentage(hit["contents"], question)
        if score > _SCORE_THRESHOLD:
            scored.append((score, len(hit["contents"]), idx, hit))
    scored.sort(key=lambda s: s[:3], reverse=True)
    return [hit for *_rest, hit in scored[:_TOP_K_KEEP]]


# --------------------------------------------------------------------------------------
# Row selection: turn "interesting" (table, column, value) pairs into real INSERT rows
# --------------------------------------------------------------------------------------


def _select_example_rows(
    cur: Cursor,
    table_names: List[str],
    column_names_by_table: Dict[str, List[str]],
    sampled_values: Dict[Tuple[str, str], List[Any]],
    relevant_hits: List[Dict[str, str]],
) -> Dict[str, List[str]]:
    """One real row per interesting (table, column, value), rendered as an ``INSERT`` statement.

    Keeps a row's other columns consistent with each other, unlike showing isolated per-column
    example values. Quoting is done by SQLite's own ``quote()`` SQL function (same approach as
    ``prepare.py``'s ``_iterdump_no_fk_check``) rather than reimplemented in Python, so every
    column type (``NULL``, integer, real, blob, text) is rendered correctly.
    """
    from sqlite3.dump import _quote_name

    interesting_by_table: Dict[str, List[Tuple[str, Any]]] = {t: [] for t in table_names}
    for (table_name, column_name), values in sampled_values.items():
        interesting_by_table[table_name].extend((column_name, v) for v in values)
    for hit in relevant_hits:
        interesting_by_table[hit["table"]].append((hit["column"], hit["contents"]))

    statements_by_table: Dict[str, List[str]] = {}
    for table_name, column_value_pairs in interesting_by_table.items():
        insert_expr = "'INSERT INTO {0} VALUES(' || {1} || ')'".format(
            _quote_name(table_name),
            " || ',' || ".join(f"quote({_quote_name(c)})" for c in column_names_by_table[table_name]),
        )

        seen = set()
        statements: List[str] = []
        for column_name, value in column_value_pairs:
            if len(statements) >= _MAX_ROWS_PER_TABLE:
                break
            try:
                cur.execute(f'SELECT {insert_expr} FROM "{table_name}" WHERE "{column_name}" = ? LIMIT 1;', (value,))
            except sqlite3.OperationalError:
                continue
            row = cur.fetchone()
            if row is not None and row[0] not in seen:
                seen.add(row[0])
                statements.append(row[0] + ";")
        if statements:
            statements_by_table[table_name] = statements
    return statements_by_table


def build_sql_context(create_statements: Dict[str, str], insert_statements: List[str]) -> str:
    """Schema dump + selected rows, rendered the way ``prepare.py``'s iterdump-based dump is."""
    # sqlite_master.sql text has no trailing ";" -- add one so consecutive CREATE TABLEs don't
    # run together into a single broken statement.
    create_lines = [f"{sql};" for sql in create_statements.values()]
    lines = ["BEGIN TRANSACTION;", *create_lines, *insert_statements, "COMMIT;"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Alternate rendering: per-column YAML schema (data type, description, sampled values)
# --------------------------------------------------------------------------------------

_MAX_VALUES_PER_COLUMN = 6  # matches extract_values.py's (deleted) per-column example cap


def _column_types(cur: Cursor, table_name: str) -> Dict[str, str]:
    """{column_name: declared SQLite type}, e.g. "TEXT", "INTEGER", "REAL"."""
    cur.execute(f"PRAGMA table_info('{table_name}')")
    return {row[1]: row[2] for row in cur.fetchall()}


def _read_description_rows(description_dir: Path, table_name: str):
    """Yields ``(original_column_name, column_name, column_description, value_description)``
    from BIRD's ``database_description/<table>.csv``, one tuple per row with a non-blank
    ``original_column_name`` (the real, queryable SQLite identifier -- used elsewhere to align
    CSV rows to actual columns).
    """
    csv_path = description_dir / f"{table_name}.csv"
    if not csv_path.exists():
        return

    import csv as csv_module

    # BIRD's CSVs aren't all clean UTF-8 -- replace undecodable bytes rather than crash.
    with open(csv_path, encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv_module.DictReader(f):
            original_column_name = (row.get("original_column_name") or "").strip()
            if not original_column_name:
                continue
            yield (
                original_column_name,
                _clean_text(row.get("column_name") or ""),
                _clean_text(row.get("column_description") or ""),
                _clean_text(row.get("value_description") or ""),
            )


def _column_descriptions(description_dir: Path, table_name: str) -> Dict[str, str]:
    """{original_column_name: description}: the CSV's ``column_name`` (a cleaned/renamed
    label BIRD provides), falling back to ``column_description`` (a full sentence) when
    ``column_name`` is blank, which it is for a large fraction of rows.
    """
    return {
        original: column_name or column_description
        for original, column_name, column_description, _value_description in _read_description_rows(
            description_dir, table_name
        )
    }


_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _clean_text(text: str) -> str:
    """Strip C0/C1 control characters and collapse whitespace/newlines to single spaces.

    BIRD's CSVs occasionally contain mojibake control characters (e.g. U+0095) where a
    bullet-point separator clearly was intended -- pure noise in a model-facing prompt, not
    a case of guessing at data meaning, so safe to clean up rather than pass through verbatim.
    """
    return " ".join(_CONTROL_CHARS_PATTERN.sub(" ", text).split())


def _ensure_trailing_punctuation(text: str) -> str:
    text = text.rstrip()
    if text and text[-1] not in ".,":
        text += "."
    return text


def _append_value_description(description: str, value_description: str) -> str:
    if not value_description:
        return description
    if not description:
        return value_description
    return f"{_ensure_trailing_punctuation(description)} {value_description}"


def _column_descriptions_with_value_hints(description_dir: Path, table_name: str) -> Dict[str, str]:
    """Like ``_column_descriptions``, with BIRD's ``value_description`` (e.g. what a coded
    value like ``'+'`` means) appended when present, ensuring the base description ends with
    "." or "," first so the two read as separate clauses rather than running together.
    """
    result: Dict[str, str] = {}
    for original, column_name, column_description, value_description in _read_description_rows(
        description_dir, table_name
    ):
        result[original] = _append_value_description(column_name or column_description, value_description)
    return result


def _primary_key_columns(cur: Cursor, table_name: str) -> set:
    cur.execute(f"PRAGMA table_info('{table_name}')")
    return {row[1] for row in cur.fetchall() if row[5]}  # row[5] = pk (0 if not, else its position in the PK)


def _all_foreign_keys(cur: Cursor, table_names: List[str]) -> List[str]:
    """["table.from_col = ref_table.to_col", ...] across every table in this database."""
    foreign_keys: List[str] = []
    for table_name in table_names:
        cur.execute(f"PRAGMA foreign_key_list('{table_name}')")
        for row in cur.fetchall():
            ref_table, from_col, to_col = row[2], row[3], row[4]
            foreign_keys.append(f"{table_name}.{from_col} = {ref_table}.{to_col}")
    return foreign_keys


def _select_column_values(
    sampled_values: Dict[Tuple[str, str], List[Any]],
    relevant_hits: List[Dict[str, str]],
) -> Dict[Tuple[str, str], List[Any]]:
    """Per-column values for the YAML view: relevant hits first (question-specific, so they
    should survive the cap), then baseline samples filling any remaining room, deduped, capped
    at ``_MAX_VALUES_PER_COLUMN`` -- matches extract_values.py's (deleted) ``obtain_db_details``
    value-merging order.
    """
    values_by_column: Dict[Tuple[str, str], List[Any]] = {}
    for hit in relevant_hits:
        key = (hit["table"], hit["column"])
        bucket = values_by_column.setdefault(key, [])
        if hit["contents"] not in bucket:
            bucket.append(hit["contents"])
    for key, values in sampled_values.items():
        bucket = values_by_column.setdefault(key, [])
        for value in values:
            if value not in bucket:
                bucket.append(value)
    return {key: values[:_MAX_VALUES_PER_COLUMN] for key, values in values_by_column.items()}


def build_yaml_context(
    table_names: List[str],
    column_types_by_table: Dict[str, Dict[str, str]],
    descriptions_by_table: Dict[str, Dict[str, str]],
    values_by_column: Dict[Tuple[str, str], List[Any]],
) -> str:
    """Per-table, per-column schema: name, data type, description, sampled values."""
    import yaml

    doc = []
    for table_name in table_names:
        columns = [
            {
                column_name: {
                    "data_type": data_type,
                    "description": descriptions_by_table.get(table_name, {}).get(column_name, ""),
                    "sampled_values": values_by_column.get((table_name, column_name), []),
                }
            }
            for column_name, data_type in column_types_by_table[table_name].items()
        ]
        doc.append({table_name: columns})
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def _render_scalar(value: Any) -> str:
    """Emit ``value`` unquoted unless quoting is genuinely necessary (embedded newline, empty
    string, or leading/trailing whitespace that plain rendering would silently drop) -- see
    ``build_yaml_context_v3``'s docstring for why this doesn't just defer to a YAML dumper.
    """
    if not isinstance(value, str):
        return str(value)
    if value == "" or "\n" in value or value != value.strip():
        return json.dumps(value)
    return value


def build_yaml_context_v3(
    table_names: List[str],
    column_types_by_table: Dict[str, Dict[str, str]],
    primary_keys_by_table: Dict[str, set],
    descriptions_by_table: Dict[str, Dict[str, str]],
    values_by_column: Dict[Tuple[str, str], List[Any]],
    foreign_keys: List[str],
) -> str:
    """Like ``build_yaml_context`` (v1), plus a primary-key marker per column and a trailing
    "#### Foreign key" section:

        #### Tables
        - column_name:
            data_type: TEXT (primary key)
            description: ...
            values:
            - value1
            - value2

        #### Foreign key
        - table1.column1 = table2.column2

    "(primary key)" is appended to ``data_type`` only for actual primary-key columns --
    matches how SQL DDL itself writes it (``TYPE PRIMARY KEY``, right after the type) -- rather
    than a separate ``is primary: true/false`` field that would otherwise say "false" on every
    non-PK column, needlessly inflating every table's token count.

    The "#### Tables"/"#### Foreign key" headers keep the foreign-key list visually and
    structurally separate from the table list, rather than appearing as just another
    same-level list entry that could be mistaken for a table.

    Hand-formatted rather than rendered via ``yaml.safe_dump``, for the same reason as
    ``build_yaml_context_v2``: a real YAML dumper would quote-and-escape any scalar
    containing ``": "`` (common in these descriptions, e.g. "commonsense evidence: ..."),
    which is unnecessary since nothing downstream parses this text as YAML. Quoting here is
    minimal -- only when a value would otherwise be ambiguous or malformed on the page (an
    embedded newline, an empty value, or stray leading/trailing whitespace).
    """
    lines: List[str] = ["#### Tables"]
    for table_name in table_names:
        lines.append(f"- {table_name}:")
        pk_columns = primary_keys_by_table.get(table_name, set())
        for column_name, data_type in column_types_by_table[table_name].items():
            description = descriptions_by_table.get(table_name, {}).get(column_name, "")
            values = values_by_column.get((table_name, column_name), [])
            type_display = f"{data_type} (primary key)" if column_name in pk_columns else data_type
            lines.append(f"  - {column_name}:")
            lines.append(f"      data_type: {type_display}")
            lines.append(f"      description: {_render_scalar(description)}")
            lines.append("      values:")
            lines.extend(f"      - {_render_scalar(v)}" for v in values)
    if foreign_keys:
        lines.append("")
        lines.append("#### Foreign key")
        lines.extend(f"- {fk}" for fk in foreign_keys)
    return "\n".join(lines)


def _format_column_entry_compact(
    data_type: str, values: List[Any], is_primary_key: bool, is_composite_key: bool, description: str
) -> str:
    """"data_type, (example: [...]), is primary key, description" -- "(example: ...)" is
    omitted when there are no sampled values. The primary-key marker distinguishes a
    single-column key (that column's values are genuinely unique) from a column that's only
    one member of a multi-column key (unique as a tuple with its co-members, not on its own).

    Renders the example list via ``json.dumps`` (double-quoted elements) rather than Python's
    ``str()`` (single-quoted) -- this only matters because of the deliberately-unquoted,
    not-strictly-YAML rendering in ``build_yaml_context_v2``: with no outer quoting to escape
    against, whichever quote style the list uses is what actually shows up in the prompt.
    """
    parts = [data_type]
    if values:
        parts.append(f"(example: {json.dumps(values)})")
    if is_primary_key:
        parts.append("is part of a composite primary key" if is_composite_key else "is primary key")
    if description:
        parts.append(description)
    return ", ".join(parts)


def build_yaml_context_v2(
    table_names: List[str],
    column_types_by_table: Dict[str, Dict[str, str]],
    primary_keys_by_table: Dict[str, set],
    descriptions_by_table: Dict[str, Dict[str, str]],
    values_by_column: Dict[Tuple[str, str], List[Any]],
    foreign_keys: List[str],
) -> str:
    """Per-table, per-column compact schema, plus a trailing ``foreign_keys`` block.

    Each column is one ``column_name: data_type, (example: [...]), is primary key,
    description`` line instead of a nested mapping -- avoids repeating "data_type:"/
    "description:" keys on every column, which cost tokens for no benefit once the column
    count gets large.

    Deliberately hand-formatted (YAML-*styled*, not YAML-*validated*) rather than rendered
    via ``yaml.safe_dump``: the column line inherently contains ``": "`` inside "(example:
    ...)", which is ambiguous for a plain YAML scalar, so a real YAML dumper is forced to wrap
    the whole line in quotes and escape any quote characters inside it -- e.g. every example
    value's quote mark doubled (``''Sarratore''``). Nothing downstream parses this text as
    YAML (the model just reads it as text), so that safety isn't worth the readability cost.
    """
    lines: List[str] = []
    for table_name in table_names:
        lines.append(f"- {table_name}:")
        pk_columns = primary_keys_by_table.get(table_name, set())
        is_composite_key = len(pk_columns) > 1
        for column_name, data_type in column_types_by_table[table_name].items():
            entry = _format_column_entry_compact(
                data_type,
                values_by_column.get((table_name, column_name), []),
                column_name in pk_columns,
                is_composite_key,
                descriptions_by_table.get(table_name, {}).get(column_name, ""),
            )
            lines.append(f"  - {column_name}: {entry}")
    if foreign_keys:
        lines.append("- foreign_keys:")
        lines.extend(f"  - {fk}" for fk in foreign_keys)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


class DbHandle:
    """Everything needed to answer questions against one database, computed once."""

    def __init__(self, cur: Cursor, description_dir: Path):
        self.cur = cur
        self.table_names = _table_names(cur)
        self.create_statements = _create_table_statements(cur, self.table_names)
        self.column_names_by_table = {t: _column_names(cur, t) for t in self.table_names}
        self.column_types_by_table = {t: _column_types(cur, t) for t in self.table_names}
        self.descriptions_by_table = {t: _column_descriptions(description_dir, t) for t in self.table_names}
        self.descriptions_with_hints_by_table = {
            t: _column_descriptions_with_value_hints(description_dir, t) for t in self.table_names
        }
        self.primary_keys_by_table = {t: _primary_key_columns(cur, t) for t in self.table_names}
        self.foreign_keys = _all_foreign_keys(cur, self.table_names)
        self.sampled_values = _sample_table_values(cur, self.table_names)
        corpus = _collect_column_values(cur, self.table_names)
        self.retriever = _build_retriever(corpus) if corpus else None

    def relevant_hits_for_question(self, question: str) -> List[Dict[str, str]]:
        return _relevant_hits_for_question(self.retriever, question) if self.retriever else []

    def insert_statements_for_question(self, question: str) -> List[str]:
        relevant_hits = self.relevant_hits_for_question(question)
        statements_by_table = _select_example_rows(
            self.cur, self.table_names, self.column_names_by_table, self.sampled_values, relevant_hits
        )
        return [statement for statements in statements_by_table.values() for statement in statements]

    def yaml_context_for_question(self, question: str) -> str:
        relevant_hits = self.relevant_hits_for_question(question)
        values_by_column = _select_column_values(self.sampled_values, relevant_hits)
        return build_yaml_context(
            self.table_names, self.column_types_by_table, self.descriptions_by_table, values_by_column
        )

    def yaml_v2_context_for_question(self, question: str) -> str:
        relevant_hits = self.relevant_hits_for_question(question)
        values_by_column = _select_column_values(self.sampled_values, relevant_hits)
        return build_yaml_context_v2(
            self.table_names,
            self.column_types_by_table,
            self.primary_keys_by_table,
            self.descriptions_with_hints_by_table,
            values_by_column,
            self.foreign_keys,
        )

    def yaml_v3_context_for_question(self, question: str) -> str:
        relevant_hits = self.relevant_hits_for_question(question)
        values_by_column = _select_column_values(self.sampled_values, relevant_hits)
        return build_yaml_context_v3(
            self.table_names,
            self.column_types_by_table,
            self.primary_keys_by_table,
            self.descriptions_by_table,
            values_by_column,
            self.foreign_keys,
        )


def build_db_values(
    dev_databases_dir: Path, dev_json_path: Path
) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, Any]]]:
    """Returns ``(create_statements_by_db, per_question_rows)``.

    ``create_statements_by_db``: ``{db_id: {table_name: "CREATE TABLE ...;"}}``.
    ``per_question_rows``: one ``{"id", "db_id", "insert_statements"}`` per BIRD dev row, in
    ``dev.json`` order (so ``id`` aligns with ``prepare.py``'s row ids).
    """
    with open(dev_json_path) as f:
        entries = json.load(f)

    db_ids = sorted({entry["db_id"] for entry in entries})
    db_handles: Dict[str, DbHandle] = {}
    create_statements_by_db: Dict[str, Dict[str, str]] = {}
    for db_id in db_ids:
        print(f"Indexing {db_id} ...")
        db_path = dev_databases_dir / db_id / f"{db_id}.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        db_handles[db_id] = DbHandle(conn.cursor(), dev_databases_dir / db_id / "database_description")
        create_statements_by_db[db_id] = db_handles[db_id].create_statements

    per_question_rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        # Matches prepare.py's row["question"]: evidence carries the literal-value hints
        # (e.g. "triple type bonds refers to bond_type = '#'") that retrieval needs to find.
        question = entry["evidence"] + "\n" + entry["question"]
        insert_statements = db_handles[entry["db_id"]].insert_statements_for_question(question)
        per_question_rows.append({"id": i, "db_id": entry["db_id"], "insert_statements": insert_statements})

    return create_statements_by_db, per_question_rows


def main() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH)
    args = parser.parse_args()

    dev_databases_dir = ensure_bird_sql()
    dev_json_path = dev_databases_dir.parent / "dev.json"
    if not dev_json_path.exists():
        raise RuntimeError(f"Expected BIRD dev.json at {dev_json_path}")

    create_statements_by_db, per_question_rows = build_db_values(dev_databases_dir, dev_json_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"create_statements_by_db": create_statements_by_db, "rows": per_question_rows}, f, indent=2)
    print(f"Wrote schema + example rows for {len(per_question_rows)} questions to {args.output}")
    return args.output


if __name__ == "__main__":
    main()
