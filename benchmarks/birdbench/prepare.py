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

Each (dialect, split) pair writes its own file, ``data/birdbench_<dialect>_<split>.jsonl`` (dev: one row per task, 1534 rows; the
dataset is never repeated -- see ``num_repeats`` in config.yaml).

``split="train"`` prepares BIRD's train split instead (it is not the evaluation set): the "uberset" of BIRD's
``train.json`` with every row ReViSQL re-verified (``bird-verified-train.json``) swapped for its verified version, run
against ``train_databases``. It has no ``difficulty`` labels.
The train databases are downloaded on first use (~9 GB, shared cache with the ``bird_sql`` server dir), or point
``train_dir`` at an already-extracted BIRD ``train/`` directory.

``version`` picks a release of the dev questions (``VERSIONS``): ``dev_20240627`` (the default, ``dev.json`` in
``dev.zip``), the authors' revised dev of ``2025-11-06``, or the 500-question Mini-Dev of ``2025-07-04`` or of the
``2026-10-28`` update. The newer ones come from the BIRD authors' Hugging Face datasets at a pinned revision, checked
against their SHA-256; all run on ``dev.zip``'s 11 databases. A non-default version writes its own file,
``data/birdbench_<dialect>_<version>.jsonl``, and its rows also carry ``question_id`` and ``bird_version``.

``prompt_format="schema_only"`` (SQLite only) prepares rows for ``prompts/schema_only_sqlite.yaml`` instead:
auto-ontology-eval's schema-only control prompt, whose ``sql_context`` is the database's ``CREATE TABLE`` statements
plus BIRD's column descriptions, with the question and its evidence apart. Its file ends in ``_schema_only``.

Calls ``ensure_bird_sql()`` so the download cache is shared with the ``bird_sql`` resource
server (avoids a duplicate ~1.4 GB download).

Requires ``bm25s`` and ``nltk`` (``uv sync --locked --project benchmarks/birdbench``, see benchmarks/birdbench/README.md), unlike a plain schema dump.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import sqlite3
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.birdbench.build_db_values import DbHandle
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql, ensure_bird_train


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DIALECTS = ("sqlite", "spark", "postgres")
SPLITS = ("dev", "train")
PROMPT_FORMATS = ("default", "schema_only")


@dataclass(frozen=True)
class BirdVersion:
    """A release of BIRD's dev questions. All of them run on ``dev.zip``'s 11 SQLite databases."""

    released: str
    tasks: int
    # Where the questions come from; None for dev_20240627, whose dev.json is inside dev.zip.
    url: Optional[str] = None
    sha256: Optional[str] = None


_HF = "https://huggingface.co/datasets/birdsql"
DEFAULT_VERSION = "dev_20240627"
VERSIONS: Dict[str, BirdVersion] = {
    DEFAULT_VERSION: BirdVersion("2024-06-27", 1534),
    # The authors' quality review of every dev question: 614 of the 1,534 changed (question 182, evidence 378,
    # gold SQL 450, difficulty 86); same ids and databases.
    "dev_20251106": BirdVersion(
        "2025-11-06",
        1534,
        f"{_HF}/bird_sql_dev_20251106/resolve/3c11fb193e5439b338e23677fa0aae11e8b85db9/data/dev_20251106-00000-of-00001.json",
        "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb",
    ),
    # Mini-Dev as corrected on 2025-07-04 (500 SQLite questions, dev question ids), archived by the authors.
    "minidev_20250704": BirdVersion(
        "2025-07-04",
        500,
        f"{_HF}/bird_mini_dev/resolve/5b256b5a9c17478f06ed6912612e9a14d17065c8/archive/bird_mini_dev_2025-07-04/data/"
        "mini_dev_sqlite-00000-of-00001.json",
        "88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2",
    ),
    # Mini-Dev's 2026 update ("Mini-Dev 1024"): 500 questions renumbered 0-499, revised or replaced after review.
    "minidev_20261028": BirdVersion(
        "2026-10-28",
        500,
        f"{_HF}/bird_mini_dev/resolve/5b256b5a9c17478f06ed6912612e9a14d17065c8/data/mini_dev_sqlite.jsonl",
        "926af2a0d1b47e5d0d15a7aa8a34d4721007f4550a881555b6a89496b54a2606",
    ),
}

VERIFIED_TRAIN_URL = "https://raw.githubusercontent.com/uiuc-kang-lab/ReViSQL/main/data/bird-verified-train.json"
# Train databases can be multi-GB each (dev's 11 together are ~1.4 GB); the BM25 corpus scan is capped per column
# there since only a few example values per column are ever shown. Dev keeps the full, unbounded scan.
TRAIN_CORPUS_LIMIT_PER_COLUMN = 2000


def output_fpath_for(
    dialect: str, split: str = "dev", version: str = DEFAULT_VERSION, prompt_format: str = "default"
) -> Path:
    """``data/birdbench_<dialect>_<split or version>[_schema_only].jsonl``: the split for the default version, else the
    version, so a file prepared from one release is never read as another."""
    name = version if split == "dev" and version != DEFAULT_VERSION else split
    suffix = "" if prompt_format == "default" else f"_{prompt_format}"
    return DATA_DIR / f"birdbench_{dialect}_{name}{suffix}.jsonl"


# A table heading of a spark/postgres ``sql_context``: ``- `db_id`.`table`:`` (see build_db_values.render_sql_context).
_QUOTED_TABLE_HEADING = re.compile(r"^- `[^`]+`\.`([^`]+)`:$", re.MULTILINE)


def is_prepared_data_current(fpath: Path) -> bool:
    """Whether a cached prepared file can be reused (``gym eval prepare +use_cached_prepared_benchmarks=true``).

    A Spark file is stale when a table heading in its ``sql_context`` is not a name Spark can hold: the server
    registers tables under ``setup_bird_spark.spark_table_name`` (``\\w`` characters only), so a file prepared before
    that mapping existed shows the model names such as ``air carriers`` that its SQL then cannot resolve.
    """
    fpath = Path(fpath)
    if not fpath.is_file():
        return False
    if "_spark_" not in fpath.name:
        return True
    with fpath.open() as f_in:
        for line in f_in:
            if not line.strip():
                continue
            for table in _QUOTED_TABLE_HEADING.findall(json.loads(line).get("sql_context", "")):
                if not re.fullmatch(
                    r"\w+", table
                ):  # == spark_table_name(table) != table, without importing pyspark deps
                    return False
    return True


def build_uberset(train_entries: List[Dict[str, Any]], verified_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """BIRD train rows with each ReViSQL-verified row swapped in for the original it corrects.

    A verified row's ``question_id`` is the 0-indexed position of the row it replaces in ``train.json``. Matching on
    that (rather than on question text) is robust because verification sometimes edits the question itself; the
    original text is then kept in ``original_question`` (``None`` when unchanged). Verified rows are appended after
    the untouched originals, so the result has the same length as ``train_entries``. Each original gets its own
    position in ``train.json`` as ``question_id``, so every row of the result has one.
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
    originals = [{**e, "question_id": i} for i, e in enumerate(train_entries) if i not in replaced]
    uberset = originals + list(verified_entries)
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
    """``prompts/default_<dialect>.yaml``."""
    return BENCHMARK_DIR / "prompts" / f"default_{dialect}.yaml"


def load_entries(split: str, databases_dir: Path, version: str = DEFAULT_VERSION) -> List[Dict[str, Any]]:
    """The raw BIRD questions for ``split``: ``dev.json`` or another dev ``version``, or the train uberset (see
    ``build_uberset``)."""
    if split == "train":
        return _load_train_entries(databases_dir.parent)
    if version != DEFAULT_VERSION:
        return load_version_entries(version)
    dev_json_path = databases_dir.parent / "dev.json"
    if not dev_json_path.exists():
        raise RuntimeError(f"Expected BIRD dev.json at {dev_json_path}")
    with open(dev_json_path) as f:
        return json.load(f)


def load_version_entries(version: str) -> List[Dict[str, Any]]:
    """A dev release's questions, downloaded once into ``data/versions/`` and checked against its pinned SHA-256."""
    release = VERSIONS[version]
    cached = DATA_DIR / "versions" / f"{version}{Path(release.url).suffix}"
    if not cached.exists() or hashlib.sha256(cached.read_bytes()).hexdigest() != release.sha256:
        print(f"Downloading BIRD {version} from {release.url} ...")
        cached.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(release.url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read()
        if hashlib.sha256(content).hexdigest() != release.sha256:
            raise RuntimeError(f"BIRD {version} from {release.url} does not match its pinned SHA-256")
        cached.write_bytes(content)
    text = cached.read_text(encoding="utf-8")
    entries = json.loads(text) if cached.suffix == ".json" else [json.loads(x) for x in text.splitlines() if x.strip()]
    if len(entries) != release.tasks:
        raise RuntimeError(f"BIRD {version} has {len(entries)} questions, expected {release.tasks}")
    return entries


def sqlite_schema_dump(db_path: Path) -> str:
    """The database's ``CREATE TABLE`` statements in creation order, wrapped as a SQL dump."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.text_factory = lambda b: b.decode(errors="ignore")
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    body = "\n".join(r[0].strip().rstrip(";") + ";" for r in rows)
    return f"BEGIN TRANSACTION;\n{body}\nCOMMIT;"


def column_description_block(description_dir: Path) -> str:
    """BIRD's ``database_description/*.csv`` as text: per table, each column whose description or value description
    says more than its name. Empty when there is nothing to show."""
    if not description_dir.is_dir():
        return ""
    sections = []
    for csv_path in sorted(description_dir.glob("*.csv")):
        lines = []
        # BIRD's CSVs are not uniformly UTF-8 and several carry a BOM.
        with csv_path.open(newline="", encoding="utf-8-sig", errors="ignore") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("original_column_name") or "").strip()
                desc = (row.get("column_description") or "").strip()
                value = (row.get("value_description") or "").strip()
                if not name:
                    continue
                if desc.lower() == name.lower():
                    desc = ""
                if desc or value:
                    lines.append(f"  {name}: " + "; ".join(p for p in (desc, value) if p))
        if lines:
            sections.append(f"Table {csv_path.stem}:\n" + "\n".join(lines))
    return "Column descriptions:\n" + "\n".join(sections) if sections else ""


def schema_only_context(databases_dir: Path, db_id: str) -> str:
    """``sql_context`` for ``prompt_format="schema_only"``: the schema dump, then the column descriptions.

    Adapted from auto-ontology-eval (https://github.com/NVIDIA/auto-ontology-eval, ``ontology_sql_eval/gym/ddl.py``
    at e727e4d6e, Apache-2.0), so its schema-only runs and Gym's send the same prompt.
    """
    dump = sqlite_schema_dump(databases_dir / db_id / f"{db_id}.sqlite")
    block = column_description_block(databases_dir / db_id / "database_description")
    return f"{dump}\n\n{block}" if block else dump


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
    version: str = DEFAULT_VERSION,
    prompt_format: str = "default",
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
        dialect: SQL dialect the ``sql_context`` is rendered for: ``sqlite`` (default), ``spark`` or ``postgres``. Each dialect
            writes its own file, see ``output_fpath_for``.
        split: ``dev`` (default; the 1534-task benchmark) or ``train`` (the uberset, see module docstring).
        train_dir: ``split="train"`` only: extracted BIRD ``train/`` directory holding ``train.json`` and
            ``train_databases/``. Default: downloaded by ``ensure_bird_train()``.
        version: ``split="dev"`` only: which release of the dev questions (``VERSIONS``). Default ``dev_20240627``.
        prompt_format: ``default``, or ``schema_only`` (SQLite only) for ``prompts/schema_only_sqlite.yaml``: the schema
            dump and column descriptions as ``sql_context``, and the question and its evidence (``evidence_section``)
            apart. See the module docstring.
    """
    if dialect not in DIALECTS:
        raise ValueError(f"Unsupported dialect {dialect!r}; expected one of {DIALECTS}")
    if split not in SPLITS:
        raise ValueError(f"Unsupported split {split!r}; expected one of {SPLITS}")
    if version not in VERSIONS:
        raise ValueError(f"Unsupported version {version!r}; expected one of {tuple(VERSIONS)}")
    if split != "dev" and version != DEFAULT_VERSION:
        raise ValueError(f"version selects a release of BIRD dev; it can't be used with split={split!r}")
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"Unsupported prompt_format {prompt_format!r}; expected one of {PROMPT_FORMATS}")
    if prompt_format == "schema_only" and dialect != "sqlite":
        raise ValueError("prompt_format='schema_only' is SQLite only")
    output_fpath = output_fpath_for(dialect, split, version, prompt_format)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    databases_dir = databases_dir_for(split, train_dir)
    entries = load_entries(split, databases_dir, version)
    corpus_limit = TRAIN_CORPUS_LIMIT_PER_COLUMN if split == "train" else None
    db_ids = sorted({entry["db_id"] for entry in entries})

    if prompt_format == "schema_only":
        return _write_schema_only(entries, databases_dir, db_ids, include_evidence, version, output_fpath)

    print("Building per-db schemas (types, PK/FK, descriptions, BM25 indexes)...")
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
                "task_id": str(entry["question_id"]),
                "question": question,
                "gt_sql": entry["SQL"],
                "sql_context": db_handles[db_id].sql_context_for_question(question),
                # Train labels (ReViSQL's verified rows) carry no BIRD difficulty tag.
                "difficulty": entry.get("difficulty"),
                "db_id": db_id,
                "id": i,
            }
            if version != DEFAULT_VERSION:
                row.update(question_id=entry["question_id"], bird_version=version)
            f_out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} BIRD {split} entries to {output_fpath}")
    return output_fpath


def _write_schema_only(
    entries: List[Dict[str, Any]],
    databases_dir: Path,
    db_ids: List[str],
    include_evidence: bool,
    version: str,
    output_fpath: Path,
) -> Path:
    """Rows for ``prompts/schema_only_sqlite.yaml``: every row of a database shares its schema dump."""
    contexts = {db_id: schema_only_context(databases_dir, db_id) for db_id in db_ids}
    with open(output_fpath, "w") as f_out:
        for i, entry in enumerate(entries):
            evidence = (entry.get("evidence") or "").strip() if include_evidence else ""
            row = {
                "question": entry["question"],
                "evidence_section": f"\n\n### Evidence\n{evidence}" if evidence else "",
                "gt_sql": entry["SQL"],
                "sql_context": contexts[entry["db_id"]],
                "difficulty": entry.get("difficulty"),
                "db_id": entry["db_id"],
                "id": i,
                "question_id": entry["question_id"],
                "bird_version": version,
            }
            f_out.write(json.dumps(row) + "\n")
    print(f"Wrote {len(entries)} BIRD {version} schema-only entries to {output_fpath}")
    return output_fpath


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare BIRD data: python -m benchmarks.birdbench.prepare")
    parser.add_argument("--dialect", choices=DIALECTS, default="sqlite")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument(
        "--train-dir", default=None, help="Extracted BIRD train/ dir. Default: downloaded on first use."
    )
    parser.add_argument("--dscp", default="name_or_col_dscp", help="See prepare()'s `dscp` argument")
    parser.add_argument("--no-evidence", action="store_true", help="Do not prepend BIRD's `evidence` to the question")
    parser.add_argument("--version", choices=tuple(VERSIONS), default=DEFAULT_VERSION, help="Release of BIRD dev")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="default")
    args = parser.parse_args(argv)
    prepare(
        include_evidence=not args.no_evidence,
        dscp=args.dscp,
        dialect=args.dialect,
        split=args.split,
        train_dir=args.train_dir,
        version=args.version,
        prompt_format=args.prompt_format,
    )


if __name__ == "__main__":
    main()
