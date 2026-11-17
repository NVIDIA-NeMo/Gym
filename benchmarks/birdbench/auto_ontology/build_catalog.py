# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build Auto Ontology's catalog of the BIRD dev databases, inside the catalog image.

    python build_catalog.py MODELS_ENV

The Dockerfile runs this with Auto Ontology's Python. MODELS_ENV is a ``KEY=VALUE`` file with the model keys
(``DEFAULT_MODELS_API_KEY``, and ``EMBED_*``, ``NON_REASONING_*`` or ``REASONING_*`` settings where they differ from the
defaults). For each database under ``/opt/bird-catalog/databases`` it:

1. ingests the schema into the catalog (``auto_ontology.catalog.ingest_catalog``);
2. writes BIRD's column descriptions, from the database's ``database_description`` CSVs, into the catalog
   (``auto_ontology.dal.datasources.apply_metadata_batch``): a SQLite database carries none, and BIRD ships them apart;
3. embeds the described tables and columns into the data layer's vector store;
4. compiles the semantic layer (``auto_ontology.semantic.compile.run_semantic_compilation``).

The catalog schema comes from Auto Ontology's own Alembic migrations. The build fails when any model call failed after
Auto Ontology's retries, or a database has no vectors: Auto Ontology logs both and goes on, which would leave a catalog
with holes. Last, it writes ``/opt/bird-catalog/.env`` with what an episode needs apart from keys: the source
databases, and the embedding and reranking endpoints and models the catalog was built with.
"""

import csv
import io
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


CATALOG_DIR = Path("/opt/bird-catalog")
DATABASES_DIR = CATALOG_DIR / "databases"
AUTO_ONTOLOGY_DIR = Path("/opt/auto-ontology")

LOG = logging.getLogger("build_catalog")


def read_env_file(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _cell(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def column_descriptions(description_dir: Path) -> list[dict[str, str]]:
    """BIRD's column descriptions of one database, as ``apply_metadata_batch`` column rows.

    BIRD ships one CSV per table, named after it, with ``original_column_name`` (the column's name in the database),
    ``column_description`` and ``value_description``. A column's description is its ``column_description``, then its
    ``value_description`` after a comma; a ``value_description`` of "not useful" is an annotator's note and is left
    out. Some CSVs are Windows-1252 rather than UTF-8.
    """
    rows = []
    for csv_path in sorted(description_dir.glob("*.csv")):
        try:
            text = csv_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = csv_path.read_text(encoding="cp1252")
        # A quoted cell can span lines, as many of BIRD's value descriptions do.
        for record in csv.DictReader(io.StringIO(text, newline="")):
            fields = {(key or "").strip().lstrip("﻿"): value for key, value in record.items()}
            name = _cell(fields.get("original_column_name")) or _cell(fields.get("column_name"))
            if not name:
                continue
            values = _cell(fields.get("value_description"))
            if values.lower() == "not useful":
                values = ""
            description = ", ".join(part for part in (_cell(fields.get("column_description")), values) if part)
            if description:
                rows.append({"table_name": csv_path.stem, "column_name": name, "description": description})
    return rows


class _FailedModelCalls(logging.Handler):
    """Counts the model calls Auto Ontology gave up on, which it logs as errors and goes on."""

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(("auto_ontology.utils.llm_invoke", "auto_ontology.utils.embedding")):
            self.messages.append(record.getMessage()[:300])


def _postgres(*args: str) -> None:
    subprocess.run(["su", "postgres", "-c", " ".join(args)], check=True)


def _start_postgres() -> None:
    pgdata, port = os.environ["PGDATA"], os.environ["POSTGRES_PORT"]
    _postgres("initdb", "-D", pgdata, f"--username={os.environ['POSTGRES_USER']}", "--auth=trust", ">/dev/null")
    _postgres(
        "pg_ctl", "-D", pgdata, "-l", f"{pgdata}/server.log", "-w", "-t", "120",
        "-o", f"'-p {port} -c listen_addresses=localhost'", "start", ">/dev/null",
    )  # fmt: skip
    subprocess.run(
        ["psql", "-h", "localhost", "-p", port, "-U", os.environ["POSTGRES_USER"], "-q", "-c",
         f"CREATE DATABASE {os.environ['POSTGRES_DATABASE']}"],
        check=True,
    )  # fmt: skip


def _query(sql: str) -> str:
    result = subprocess.run(
        ["psql", "-h", "localhost", "-p", os.environ["POSTGRES_PORT"], "-U", os.environ["POSTGRES_USER"],
         "-d", os.environ["POSTGRES_DATABASE"], "-tAc", sql],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return result.stdout.strip()


def vector_count(layer: str, database: str) -> int:
    """Rows of ``database`` in the vector store ``layer``; the store creates a layer's table with its first rows."""
    if _query(f"SELECT to_regclass('vdb.{layer}') IS NOT NULL") != "t":
        return 0
    return int(_query(f"SELECT count(*) FROM vdb.{layer} WHERE database_name = '{database}'") or 0)


def build(databases: list[Path]) -> None:
    """Ingest, describe, embed and compile each database. Auto Ontology must be importable and configured."""
    from auto_ontology.catalog import ingest_catalog
    from auto_ontology.connectors.registry import create_connector
    from auto_ontology.dal.datasources import apply_metadata_batch, fetch_tables_and_columns_by_node_ids
    from auto_ontology.semantic.compile import run_semantic_compilation
    from auto_ontology.utils import get_embed_params
    from auto_ontology.utils.embedding import batch_embed
    from auto_ontology.utils.embedding_rows import CatalogEmbeddingRowsOp
    from auto_ontology.vdb import get_data_vdb
    from auto_ontology.vdb.records import write_embedded_rows

    for database in databases:
        name = database.parent.name
        LOG.info("Ingesting %s", name)
        connector = create_connector(f"sqlite:///{database}")
        try:
            tables, _ = ingest_catalog(connector)
        finally:
            connector.close()
        descriptions = column_descriptions(database.parent / "database_description")
        apply_metadata_batch(name, [], descriptions)
        LOG.info("Described %d columns of %s", len(descriptions), name)
        # Embed the catalog as described, not as ingested.
        table_ids = [str(value) for value in tables["id"].dropna()]
        tables, columns, _ = fetch_tables_and_columns_by_node_ids(table_ids)
        embedded = batch_embed(CatalogEmbeddingRowsOp(database_name=name)((tables, columns)), get_embed_params())
        store = get_data_vdb(database_name=name, reset=True)
        written = write_embedded_rows(store, embedded) if len(embedded) else 0
        LOG.info("Embedded %d catalog rows of %s", written, name)
        LOG.info("Compiling the semantic layer of %s", name)
        run_semantic_compilation(name)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    databases = sorted(DATABASES_DIR.glob("*/*.sqlite"))
    if not databases:
        raise SystemExit(f"No BIRD databases under {DATABASES_DIR}")
    connections = ",".join(f"sqlite:///{database}" for database in databases)
    # The keys, then this image's catalog and databases over anything the file says about them.
    image = {key: value for key, value in os.environ.items() if key.startswith("POSTGRES_")}
    os.environ.update(read_env_file(Path(argv[0])))
    os.environ.update(image)
    os.environ["CONNECTION_STRINGS"] = connections

    _start_postgres()
    subprocess.run(
        [str(AUTO_ONTOLOGY_DIR / ".venv/bin/alembic"), "upgrade", "head"], cwd=AUTO_ONTOLOGY_DIR, check=True
    )
    failed = _FailedModelCalls()
    logging.getLogger().addHandler(failed)
    build(databases)

    problems = [f"a model call failed: {message}" for message in failed.messages[:5]]
    for database in databases:
        name = database.parent.name
        for layer in ("data_objects_layer", "semantic_layer"):
            if vector_count(layer, name) == 0:
                problems.append(f"{name} has no vectors in vdb.{layer}")
    if problems:
        raise SystemExit("The catalog is incomplete; check the keys and rebuild:\n- " + "\n- ".join(problems))

    from auto_ontology.utils.model_config import resolve

    settings = [f"CONNECTION_STRINGS={connections}"] + [
        f"{role}_{field}={resolve(role, field)}" for role in ("EMBED", "RERANK") for field in ("ENDPOINT", "MODEL")
    ]
    (CATALOG_DIR / ".env").write_text("\n".join(settings) + "\n")
    _postgres("pg_ctl", "-D", os.environ["PGDATA"], "-w", "stop", ">/dev/null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
