# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema introspection, BM25 retrieval, and ``sql_context`` rendering for ``prepare.py``.

BM25 search over each database's distinct column values uses ``bm25s`` (pure-Python,
numpy/scipy-backed, no JVM) -- each database's index is built in-memory and used
immediately, with nothing persisted to disk ahead of time.

Per database, two sources of "interesting" column values feed into the example values shown
per column:
- A baseline sample (``_sample_table_values``): a couple of distinct values per column,
  independent of any question, computed once per database.
- Per-question BM25 hits (``_relevant_hits_for_question``): column values whose text
  substring-matches the question well.

``sql_context`` itself is a per-table, per-column schema (data type, description, example
values) followed by a "#### Foreign key" section -- see ``build_sql_context``.

Requires ``bm25s`` and ``nltk`` (``pip install bm25s nltk``).
"""

import json
import re
import sqlite3
from pathlib import Path
from sqlite3 import Cursor
from typing import Any, Dict, List, Tuple


_VALUE_MAX_LEN = 40
_SAMPLE_VALUES_PER_COLUMN = 2  # baseline sample size per column, independent of any question
_NGRAM_MAX_N = 8
_TOP_K_RETRIEVE = 10
_TOP_K_KEEP = 20
_SCORE_THRESHOLD = 0.85
_MAX_VALUES_PER_COLUMN = 6


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


def _column_types(cur: Cursor, table_name: str) -> Dict[str, str]:
    """{column_name: declared SQLite type}, e.g. "TEXT", "INTEGER", "REAL"."""
    cur.execute(f"PRAGMA table_info('{table_name}')")
    return {row[1]: row[2] for row in cur.fetchall()}


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


# --------------------------------------------------------------------------------------
# BIRD's column descriptions (database_description/<table>.csv)
# --------------------------------------------------------------------------------------

_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _clean_text(text: str) -> str:
    """Strip C0/C1 control characters and collapse whitespace/newlines to single spaces.

    BIRD's CSVs occasionally contain mojibake control characters (e.g. U+0095) where a
    bullet-point separator clearly was intended -- pure noise in a model-facing prompt, not
    a case of guessing at data meaning, so safe to clean up rather than pass through verbatim.
    """
    return " ".join(_CONTROL_CHARS_PATTERN.sub(" ", text).split())


def _read_description_rows(description_dir: Path, table_name: str):
    """Yields ``(original_column_name, column_name, column_description)`` from BIRD's
    ``database_description/<table>.csv``, one tuple per row with a non-blank
    ``original_column_name`` (the real, queryable SQLite identifier -- used to align CSV rows
    to actual columns).
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
            )


def _column_descriptions(description_dir: Path, table_name: str) -> Dict[str, str]:
    """{original_column_name: description}: the CSV's ``column_name`` (a cleaned/renamed
    label BIRD provides), falling back to ``column_description`` (a full sentence) when
    ``column_name`` is blank, which it is for a large fraction of rows.
    """
    return {
        original: column_name or column_description
        for original, column_name, column_description in _read_description_rows(description_dir, table_name)
    }


# --------------------------------------------------------------------------------------
# Value sources: baseline per-column sample + per-question BM25 retrieval
# --------------------------------------------------------------------------------------


def _collect_column_values(cur: Cursor, table_names: List[str]) -> List[Dict[str, str]]:
    """Distinct, non-numeric, short string values from every column -- the BM25 corpus."""
    corpus: List[Dict[str, str]] = []
    for table_name in table_names:
        for column_name in _column_types(cur, table_name):
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
        for column_name in _column_types(cur, table_name):
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


def _select_column_values(
    sampled_values: Dict[Tuple[str, str], List[Any]],
    relevant_hits: List[Dict[str, str]],
) -> Dict[Tuple[str, str], List[Any]]:
    """Per-column example values: relevant hits first (question-specific, so they should
    survive the cap), then baseline samples filling any remaining room, deduped, capped at
    ``_MAX_VALUES_PER_COLUMN``.
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


# --------------------------------------------------------------------------------------
# sql_context rendering
# --------------------------------------------------------------------------------------


def _render_scalar(value: Any) -> str:
    """Emit ``value`` unquoted unless quoting is genuinely necessary (embedded newline, empty
    string, or leading/trailing whitespace that plain rendering would silently drop) -- see
    ``build_sql_context``'s docstring for why this doesn't just defer to a YAML dumper.
    """
    if not isinstance(value, str):
        return str(value)
    if value == "" or "\n" in value or value != value.strip():
        return json.dumps(value)
    return value


def build_sql_context(
    table_names: List[str],
    column_types_by_table: Dict[str, Dict[str, str]],
    primary_keys_by_table: Dict[str, set],
    descriptions_by_table: Dict[str, Dict[str, str]],
    values_by_column: Dict[Tuple[str, str], List[Any]],
    foreign_keys: List[str],
) -> str:
    """Per-table, per-column schema, plus a trailing "#### Foreign key" section:

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

    Hand-formatted rather than rendered via ``yaml.safe_dump``: a real YAML dumper would
    quote-and-escape any scalar containing ``": "`` (common in these descriptions, e.g.
    "commonsense evidence: ..."), which is unnecessary since nothing downstream parses this
    text as YAML. Quoting here is minimal -- only when a value would otherwise be ambiguous or
    malformed on the page (an embedded newline, an empty value, or stray leading/trailing
    whitespace).
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


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


class DbHandle:
    """Everything needed to answer questions against one database, computed once."""

    def __init__(self, cur: Cursor, description_dir: Path):
        self.cur = cur
        self.table_names = _table_names(cur)
        self.column_types_by_table = {t: _column_types(cur, t) for t in self.table_names}
        self.descriptions_by_table = {t: _column_descriptions(description_dir, t) for t in self.table_names}
        self.primary_keys_by_table = {t: _primary_key_columns(cur, t) for t in self.table_names}
        self.foreign_keys = _all_foreign_keys(cur, self.table_names)
        self.sampled_values = _sample_table_values(cur, self.table_names)
        corpus = _collect_column_values(cur, self.table_names)
        self.retriever = _build_retriever(corpus) if corpus else None

    def relevant_hits_for_question(self, question: str) -> List[Dict[str, str]]:
        return _relevant_hits_for_question(self.retriever, question) if self.retriever else []

    def sql_context_for_question(self, question: str) -> str:
        relevant_hits = self.relevant_hits_for_question(question)
        values_by_column = _select_column_values(self.sampled_values, relevant_hits)
        return build_sql_context(
            self.table_names,
            self.column_types_by_table,
            self.primary_keys_by_table,
            self.descriptions_by_table,
            values_by_column,
            self.foreign_keys,
        )
