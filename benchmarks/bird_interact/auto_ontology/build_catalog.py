# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build Auto Ontology's catalog of a BIRD-Interact release, inside the catalog image.

    python build_catalog.py MODELS_ENV

The Dockerfile runs this with Auto Ontology's Python, in the release's database image. MODELS_ENV is a ``KEY=VALUE``
file with the model keys (``DEFAULT_MODELS_API_KEY``, and ``EMBED_*``, ``NON_REASONING_*`` or ``REASONING_*`` settings
where they differ from the defaults). It starts the release's PostgreSQL, creates the catalog database there with Auto
Ontology's own Alembic migrations, and for each of the release's databases:

1. ingests the schema into the catalog (``auto_ontology.catalog.ingest_catalog``);
2. embeds the tables and columns into the data layer's vector store;
3. compiles the semantic layer (``auto_ontology.semantic.compile.run_semantic_compilation``).

The catalog holds what BIRD-Interact's c-Interact shows an agent: the schema. It holds no column meanings, which
c-Interact does not show, and no knowledge base, part of which each task hides; the agent gets a task's visible
knowledge with the task. auto-ontology-eval's catalog of BIRD-Interact is built the same way.

The build fails when any model call failed after Auto Ontology's retries, or a database has no vectors: Auto Ontology
logs both and goes on, which would leave a catalog with holes. Last, it writes ``/opt/bird-catalog/.env`` with what an
episode needs apart from keys: the source databases, on the port each episode starts PostgreSQL on, and the embedding
and reranking endpoints and models the catalog was built with.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path


CATALOG_DIR = Path("/opt/bird-catalog")
DUMPS_DIR = Path("/docker-entrypoint-initdb.d/postgre_table_dumps")
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


def release_databases() -> list[str]:
    """The release's databases: one dump folder, ``<db>_template``, for each, restored as ``<db>`` and as itself."""
    return sorted(path.name.removesuffix("_template") for path in DUMPS_DIR.glob("*_template") if path.is_dir())


def source(database: str, port: str) -> str:
    """The connection string of a source database; trust authentication needs no password."""
    return f"postgresql://{os.environ['POSTGRES_USER']}@localhost:{port}/{database}"


class _FailedModelCalls(logging.Handler):
    """Counts the model calls Auto Ontology gave up on, which it logs as errors and goes on."""

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(("auto_ontology.utils.llm_invoke", "auto_ontology.utils.embedding")):
            self.messages.append(record.getMessage()[:300])


# Compiling the semantic layer of the release's 18 databases holds more than PostgreSQL's default of 100 connections
# with the Auto Ontology version Gym pins ("too many clients"); its connectors are cached, so the cause is elsewhere.
SERVER_OPTIONS = "-c listen_addresses=localhost -c max_connections=500"


def _pg_ctl(*args: str) -> None:
    subprocess.run(["gosu", "postgres", "pg_ctl", "-D", os.environ["PGDATA"], "-w", *args], check=True)


def _query(sql: str, database: str | None = None) -> str:
    result = subprocess.run(
        ["psql", "-h", "localhost", "-p", os.environ["POSTGRES_PORT"], "-U", os.environ["POSTGRES_USER"],
         "-d", database or os.environ["POSTGRES_DATABASE"], "-tAc", sql],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return result.stdout.strip()


def vector_count(layer: str, database: str) -> int:
    """Rows of ``database`` in the vector store ``layer``; the store creates a layer's table with its first rows."""
    if _query(f"SELECT to_regclass('vdb.{layer}') IS NOT NULL") != "t":
        return 0
    return int(_query(f"SELECT count(*) FROM vdb.{layer} WHERE database_name = '{database}'") or 0)


def build(databases: list[str]) -> None:
    """Ingest, embed and compile each database. Auto Ontology must be importable and configured."""
    from auto_ontology.catalog import ingest_catalog
    from auto_ontology.connectors.registry import create_connector
    from auto_ontology.dal.datasources import fetch_tables_and_columns_by_node_ids
    from auto_ontology.semantic.compile import run_semantic_compilation
    from auto_ontology.utils import get_embed_params
    from auto_ontology.utils.embedding import batch_embed
    from auto_ontology.utils.embedding_rows import CatalogEmbeddingRowsOp
    from auto_ontology.vdb import get_data_vdb
    from auto_ontology.vdb.records import write_embedded_rows

    for database in databases:
        LOG.info("Ingesting %s", database)
        connector = create_connector(source(database, os.environ["POSTGRES_PORT"]))
        try:
            tables, _ = ingest_catalog(connector)
        finally:
            connector.close()
        # Embed the catalog's rows, as Auto Ontology's own ingest does.
        table_ids = [str(value) for value in tables["id"].dropna()]
        tables, columns, _ = fetch_tables_and_columns_by_node_ids(table_ids)
        embedded = batch_embed(CatalogEmbeddingRowsOp(database_name=database)((tables, columns)), get_embed_params())
        store = get_data_vdb(database_name=database, reset=True)
        written = write_embedded_rows(store, embedded) if len(embedded) else 0
        LOG.info("Embedded %d catalog rows of %s", written, database)
        LOG.info("Compiling the semantic layer of %s", database)
        run_semantic_compilation(database)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    databases = release_databases()
    if not databases:
        raise SystemExit(f"No BIRD-Interact databases under {DUMPS_DIR}")
    # The keys, then this image's catalog and databases over anything the file says about them.
    image = {key: value for key, value in os.environ.items() if key.startswith("POSTGRES_")}
    os.environ.update(read_env_file(Path(argv[0])))
    os.environ.update(image)
    os.environ["CONNECTION_STRINGS"] = ",".join(
        source(database, os.environ["POSTGRES_PORT"]) for database in databases
    )

    _pg_ctl(
        "-l",
        "/tmp/catalog-build.log",
        "-t",
        "120",
        "-o",
        f"-p {os.environ['POSTGRES_PORT']} {SERVER_OPTIONS}",
        "start",
    )
    _query(f"CREATE DATABASE {os.environ['POSTGRES_DATABASE']}", database="postgres")
    subprocess.run(
        [str(AUTO_ONTOLOGY_DIR / ".venv/bin/alembic"), "upgrade", "head"], cwd=AUTO_ONTOLOGY_DIR, check=True
    )
    failed = _FailedModelCalls()
    logging.getLogger().addHandler(failed)
    build(databases)

    problems = [f"a model call failed: {message}" for message in failed.messages[:5]]
    for database in databases:
        for layer in ("data_objects_layer", "semantic_layer"):
            if vector_count(layer, database) == 0:
                problems.append(f"{database} has no vectors in vdb.{layer}")
    if problems:
        raise SystemExit("The catalog is incomplete; check the keys and rebuild:\n- " + "\n- ".join(problems))

    from auto_ontology.utils.model_config import resolve

    # Each episode starts PostgreSQL on its own port; the episode runner fills in ${POSTGRES_PORT}.
    connections = ",".join(source(database, "${POSTGRES_PORT}") for database in databases)
    settings = [f"CONNECTION_STRINGS={connections}"] + [
        f"{role}_{field}={resolve(role, field)}" for role in ("EMBED", "RERANK") for field in ("ENDPOINT", "MODEL")
    ]
    (CATALOG_DIR / ".env").write_text("\n".join(settings) + "\n")
    _pg_ctl("-m", "fast", "stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
