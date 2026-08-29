# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Maintainer-only script: pyserini/Lucene value retrieval for BIRD dev questions.

Adapted from OmniSQL's ``process_dataset.py`` (Apache 2.0):
https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/process_dataset.py
License: https://github.com/RUCKBReasoning/OmniSQL/issues/25

Companion to ``build_db_values.py``: same retrieval design (baseline per-column sample +
per-question BM25 substring-match retrieval), reusing its engine-agnostic pieces
(``_table_names``, ``_column_names``, ``_sample_table_values``, ``_obtain_n_grams``,
``_substring_match_percentage``) verbatim -- but this script does the actual search with
real pyserini/Lucene instead of ``bm25s``, so the two can be compared for parity.

Does NOT build prompts, schema dumps, or ``INSERT`` statements -- only a per-question dump
of "interesting" column values (baseline sample + BM25 hits). Row selection and rendering
is ``build_db_values.py``'s ``_select_example_rows``/``build_sql_context``, reused as-is by
``prepare_with_bm25_pyserini.py``, which is what actually turns this dump into
``sql_context`` text. That split is intentional: this script needs pyserini (a JVM), so it's
meant to run once on a machine that has it (e.g. remote), producing
``data/db_values_pyserini.json``; ``prepare_with_bm25_pyserini.py`` then only needs SQLite
to consume that file, on any machine, no pyserini required there.

Usage::

    pip install pyserini nltk
    python build_index.py   # builds the Lucene index this script searches (unchanged, still pyserini-based;
                             # note build_index.sh's documented command line doesn't match build_index.py's
                             # actual __main__, which uses its own hardcoded dataset_info paths -- run
                             # build_index.py directly and pass its index_path_prefix as --db-content-index-path)
    python -m benchmarks.birdbench.extract_values --db-content-index-path <build_index.py's index_path_prefix>
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.birdbench.build_db_values import (
    _NGRAM_MAX_N,
    _SCORE_THRESHOLD,
    _TOP_K_KEEP,
    _TOP_K_RETRIEVE,
    _column_names,
    _obtain_n_grams,
    _sample_table_values,
    _substring_match_percentage,
    _table_names,
)
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "db_values_pyserini.json"


def _retrieve_hits_for_queries(searcher, queries: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Matches extract_values.py's (pre-refactor) ``retrieve_relevant_hits``."""
    unique_queries = list(dict.fromkeys(queries))
    if not unique_queries:
        return {}

    q_ids = [str(idx) for idx in range(len(unique_queries))]
    search_results = searcher.batch_search(unique_queries, q_ids, k=_TOP_K_RETRIEVE, threads=60)

    query_to_hits: Dict[str, List[Dict[str, str]]] = {}
    for query, q_id in zip(unique_queries, q_ids):
        hits = list(dict.fromkeys(hit.raw for hit in search_results[q_id]))
        query_to_hits[query] = [json.loads(hit) for hit in hits]
    return query_to_hits


def _relevant_hits_for_question(searcher, question: str) -> List[Dict[str, str]]:
    """BM25 hits whose text substring-matches ``question`` well, deduped and capped.

    Matches ``build_db_values.py``'s ``_relevant_hits_for_question`` exactly, aside from the
    search engine: pyserini hit ids are ``"table-**-column-**-c_id"`` (set by ``build_index.py``)
    rather than a corpus dict, so hits are normalized to the same ``{table, column, contents}``
    shape before scoring.
    """
    queries = _obtain_n_grams(question, _NGRAM_MAX_N) + [question]
    query_to_hits = _retrieve_hits_for_queries(searcher, queries)

    candidates: List[Dict[str, str]] = []
    seen = set()
    for query in queries:
        for hit in query_to_hits.get(query, []):
            table_name, column_name, _c_id = hit["id"].split("-**-")
            key = (table_name, column_name, hit["contents"])
            if key not in seen:
                seen.add(key)
                candidates.append({"table": table_name, "column": column_name, "contents": hit["contents"]})

    scored = []
    for idx, hit in enumerate(candidates):
        score = _substring_match_percentage(hit["contents"], question)
        if score > _SCORE_THRESHOLD:
            scored.append((score, len(hit["contents"]), idx, hit))
    scored.sort(key=lambda s: s[:3], reverse=True)
    return [hit for *_rest, hit in scored[:_TOP_K_KEEP]]


def extract_values(dev_databases_dir: Path, dev_json_path: Path, db_content_index_path: Path) -> Dict[str, Any]:
    """Returns ``{"sampled_values_by_db": ..., "rows": ...}``, ready for ``json.dump``.

    ``sampled_values_by_db``: ``{db_id: [{"table", "column", "values"}, ...]}``.
    ``rows``: one ``{"id", "db_id", "relevant_hits"}`` per BIRD dev row, in ``dev.json`` order
    (so ``id`` aligns with ``prepare.py``'s and ``build_db_values.py``'s row ids).
    """
    from pyserini.search.lucene import LuceneSearcher

    with open(dev_json_path) as f:
        entries = json.load(f)

    db_ids = sorted({entry["db_id"] for entry in entries})
    searchers = {}
    sampled_values_by_db: Dict[str, Any] = {}
    for db_id in db_ids:
        print(f"Loading {db_id} ...")
        db_path = dev_databases_dir / db_id / f"{db_id}.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()

        table_names = _table_names(cur)
        sampled_values = _sample_table_values(cur, table_names)
        sampled_values_by_db[db_id] = [
            {"table": table_name, "column": column_name, "values": values}
            for (table_name, column_name), values in sampled_values.items()
        ]

        searchers[db_id] = LuceneSearcher(str(db_content_index_path / db_id))

    rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        # Matches prepare.py's row["question"]: evidence carries the literal-value hints
        # (e.g. "triple type bonds refers to bond_type = '#'") that retrieval needs to find.
        question = entry["evidence"] + "\n" + entry["question"]
        relevant_hits = _relevant_hits_for_question(searchers[entry["db_id"]], question)
        rows.append({"id": i, "db_id": entry["db_id"], "relevant_hits": relevant_hits})

    return {"sampled_values_by_db": sampled_values_by_db, "rows": rows}


def main() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-content-index-path",
        type=Path,
        required=True,
        help="Directory of per-db_id Lucene indexes built by build_index.py/build_index.sh.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH)
    args = parser.parse_args()

    dev_databases_dir = ensure_bird_sql()
    dev_json_path = dev_databases_dir.parent / "dev.json"
    if not dev_json_path.exists():
        raise RuntimeError(f"Expected BIRD dev.json at {dev_json_path}")

    output = extract_values(dev_databases_dir, dev_json_path, args.db_content_index_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote sampled + relevant values for {len(output['rows'])} questions to {args.output}")
    return args.output


if __name__ == "__main__":
    main()
