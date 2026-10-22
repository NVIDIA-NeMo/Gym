# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare BIRD benchmark data.

Per-row output schema: ``{question, gt_sql, sql_context, difficulty, db_id, id}``.

``sql_context`` is a per-table, per-column schema listing each column's data type (from
SQLite's own ``PRAGMA table_info``), a human-readable description (from BIRD's
``database_description/<table>.csv``, aligned to real column names via
``original_column_name`` -- see the ``dscp`` argument below for which description fields are
used), and example values, followed by a "#### Foreign key" section (from ``PRAGMA
foreign_key_list``). Example values combine a baseline per-column sample (independent of the
question) with per-question BM25 hits (via ``bm25s`` -- pure Python, no JVM) against the
question text, so the values shown are ones actually likely to be relevant. See
``build_db_values.py`` for the full retrieval and rendering design.

With ``dialect="spark"`` the schema is rendered for Spark SQL instead (``schema.table`` names, backtick-quoted
identifiers, Spark column types); ``gt_sql`` is still the original SQLite gold, which the ``bird_sql`` server
runs on SQLite and compares against the model's Spark SQL result.

Each dialect writes its own file, ``data/birdbench_<dialect>_benchmark.jsonl`` (one row per task, 1534 rows; the
dataset is never repeated -- see ``num_repeats`` in config.yaml).

``split="train"`` prepares BIRD's train split instead (for training data, not evaluation): the "uberset" of BIRD's
``train.json`` with every row ReViSQL re-verified (``bird-verified-train.json``) swapped for its verified version, run
against ``train_databases``. It writes ``data/birdbench_train_<dialect>.jsonl`` and has no ``difficulty`` labels.
The train databases are downloaded on first use (~9 GB, shared cache with the ``bird_sql`` server dir), or point
``train_dir`` at an already-extracted BIRD ``train/`` directory.

Calls ``ensure_bird_sql()`` so the download cache is shared with the ``bird_sql`` resource
server (avoids a duplicate ~1.4 GB download).

Requires ``bm25s`` and ``nltk`` (``uv pip install bm25s nltk``, see benchmarks/birdbench/README.md), unlike a plain schema dump.
"""

import json
import shutil
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.birdbench.build_db_values import DbHandle
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql, ensure_bird_train


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DIALECTS = ("sqlite", "spark")
SPLITS = ("dev", "train")

VERIFIED_TRAIN_URL = "https://raw.githubusercontent.com/uiuc-kang-lab/ReViSQL/main/data/bird-verified-train.json"
# Train databases can be multi-GB each (dev's 11 together are ~1.4 GB); the BM25 corpus scan is capped per column
# there since only a few example values per column are ever shown. Dev keeps the full, unbounded scan.
TRAIN_CORPUS_LIMIT_PER_COLUMN = 2000


def output_fpath_for(dialect: str, split: str = "dev") -> Path:
    """``data/birdbench_<dialect>_benchmark.jsonl`` for dev, ``data/birdbench_train_<dialect>.jsonl`` for train."""
    if split == "dev":
        return DATA_DIR / f"birdbench_{dialect}_benchmark.jsonl"
    return DATA_DIR / f"birdbench_{split}_{dialect}.jsonl"


def build_uberset(train_entries: List[Dict[str, Any]], verified_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """BIRD train rows with each ReViSQL-verified row swapped in for the original it corrects.

    A verified row's ``question_id`` is the 0-indexed position of the row it replaces in ``train.json``. Matching on
    that (rather than on question text) is robust because verification sometimes edits the question itself; the
    original text is then kept in ``original_question`` (``None`` when unchanged). Verified rows are appended after
    the untouched originals, so the result has the same length as ``train_entries``.
    """
    replaced = set()
    mismatched = []
    for v in verified_entries:
        idx = int(v["question_id"])
        original_question = v["original_question"] if v.get("original_question") is not None else v["question"]
        orig = train_entries[idx]
        if orig["db_id"] != v["db_id"] or orig["question"] != original_question:
            mismatched.append(idx)
        replaced.add(idx)
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} verified rows do not match train.json at their question_id: {mismatched[:10]}"
        )
    uberset = [e for i, e in enumerate(train_entries) if i not in replaced] + list(verified_entries)
    assert len(uberset) == len(train_entries), "uberset must be the same size as train.json"
    return uberset


def databases_dir_for(split: str, train_dir: Optional[str] = None) -> Path:
    """The directory holding ``<db_id>/<db_id>.sqlite`` for ``split``, downloading it on first use.

    The questions file sits one level up (``dev.json`` / ``train.json``); see ``load_entries``.
    """
    if split == "dev":
        return ensure_bird_sql()
    if split != "train":
        raise ValueError(f"Unsupported split {split!r}; expected one of {SPLITS}")
    databases_dir = (Path(train_dir) if train_dir else ensure_bird_train()) / "train_databases"
    if not databases_dir.is_dir():
        raise FileNotFoundError(f"Train databases directory not found: {databases_dir}")
    return databases_dir


def prompt_config_fpath_for(dialect: str) -> Path:
    """``prompts/default.yaml`` for sqlite, ``prompts/<dialect>.yaml`` otherwise (as ``eval.sh`` picks it)."""
    return BENCHMARK_DIR / "prompts" / ("default.yaml" if dialect == "sqlite" else f"{dialect}.yaml")


def load_entries(split: str, databases_dir: Path) -> List[Dict[str, Any]]:
    """The raw BIRD questions for ``split``: ``dev.json``, or the train uberset (see ``build_uberset``)."""
    if split == "train":
        return _load_train_entries(databases_dir.parent)
    dev_json_path = databases_dir.parent / "dev.json"
    if not dev_json_path.exists():
        raise RuntimeError(f"Expected BIRD dev.json at {dev_json_path}")
    with open(dev_json_path) as f:
        return json.load(f)


def _load_train_entries(train_dir: Path) -> List[Dict[str, Any]]:
    train_json = train_dir / "train.json"
    if not train_json.exists():
        raise FileNotFoundError(
            f"Expected BIRD train.json at {train_json}; pass train_dir=<extracted BIRD train/ dir> or omit it to download"
        )
    verified_json = DATA_DIR / "bird_verified_train.json"
    if not verified_json.exists():
        print(f"Downloading {VERIFIED_TRAIN_URL} ...")
        req = urllib.request.Request(VERIFIED_TRAIN_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(verified_json, "wb") as f:
            shutil.copyfileobj(resp, f)
    with open(train_json) as f:
        train_entries = json.load(f)
    with open(verified_json) as f:
        verified_entries = json.load(f)
    return build_uberset(train_entries, verified_entries)


def prepare(
    include_evidence: bool = True,
    dscp: str = "name_or_col_dscp",
    dialect: str = "sqlite",
    split: str = "dev",
    train_dir: Optional[str] = None,
) -> Path:
    """Download BIRD dev (or read local train), produce the prepared jsonl for ``split``/``dialect``. Returns its path.

    Args:
        include_evidence: When ``True`` (default), prepend each entry's ``evidence`` field --
            the literal-value hints BIRD provides (e.g. "triple type bonds refers to
            bond_type = '#'") -- to the question, when present. When ``False``, or when an
            entry has no (or a ``None``) ``evidence`` field, the question is used as-is.
        dscp: Comma-separated, ordered list of BIRD description fields to concatenate into each
            column's ``sql_context`` description: ``name`` (column_name), ``col_dscp``
            (column_description), ``val_dscp`` (value_description), or ``name_or_col_dscp``
            (name if non-blank else description, as exactly one of the two, never both --
            the default, matching this benchmark's original pre-``dscp`` behavior). E.g.
            ``"name,col_dscp"`` uses both fields, name first. Blank fields are dropped. See
            ``build_db_values._combine_description``.
        dialect: SQL dialect the ``sql_context`` is rendered for: ``sqlite`` (default) or ``spark``. Each dialect
            writes its own file, see ``output_fpath_for``.
        split: ``dev`` (default; the 1534-task benchmark) or ``train`` (the uberset, see module docstring).
        train_dir: ``split="train"`` only: extracted BIRD ``train/`` directory holding ``train.json`` and
            ``train_databases/``. Default: downloaded by ``ensure_bird_train()``.
    """
    if dialect not in DIALECTS:
        raise ValueError(f"Unsupported dialect {dialect!r}; expected one of {DIALECTS}")
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}; expected one of {SPLITS}")
    output_fpath = output_fpath_for(dialect, split)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    databases_dir = databases_dir_for(split, train_dir)
    entries = load_entries(split, databases_dir)
    corpus_limit = TRAIN_CORPUS_LIMIT_PER_COLUMN if split == "train" else None

    print("Building per-db schemas (types, PK/FK, descriptions, BM25 indexes)...")
    db_ids = sorted({entry["db_id"] for entry in entries})
    db_handles: Dict[str, DbHandle] = {}
    for db_id in db_ids:
        print(f"Indexing {db_id} ...")
        db_path = databases_dir / db_id / f"{db_id}.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        db_handles[db_id] = DbHandle(
            conn.cursor(),
            databases_dir / db_id / "database_description",
            dscp=dscp,
            dialect=dialect,
            db_id=db_id,
            corpus_limit_per_column=corpus_limit,
        )

    count = 0
    with open(output_fpath, "w") as f_out:
        for i, entry in enumerate(entries):
            db_id = entry["db_id"]
            evidence = entry.get("evidence")
            # Evidence carries the literal-value hints (e.g. "triple type bonds refers to
            # bond_type = '#'") that the model needs to disambiguate.
            if include_evidence and evidence:
                question = evidence + "\n" + entry["question"]
            else:
                question = entry["question"]
            assert question, f"Entry {i} (db_id={db_id!r}) has no question"
            row = {
                "question": question,
                "gt_sql": entry["SQL"],
                "sql_context": db_handles[db_id].sql_context_for_question(question),
                # Train labels (ReViSQL's verified rows) carry no BIRD difficulty tag.
                "difficulty": entry.get("difficulty"),
                "db_id": db_id,
                "id": i,
            }
            f_out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} BIRD {split} entries to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    prepare()
