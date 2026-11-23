# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download BEAVER's MySQL dumps and load them into a MySQL server.

BEAVER ships its databases as mysqldump files in ``beaver_db.zip`` on the gated Hugging Face dataset
``beaverbench/beaver-table`` (accept its terms on the dataset page, then ``hf auth login``). The zip is pinned to
one revision, and each dump is checked against its SHA-256, so every run loads the same data.

``ensure_beaver_mysql`` loads each requested ``db_id`` into a MySQL database of that name, records it in
``_beaver_meta.loaded`` with the dump's hash and table count, and creates the read-only user the grader connects as.
It only drops or replaces databases it recorded itself: a database of the same name that it didn't load is an error,
so pointing it at a shared server can't overwrite someone else's data. ``check_beaver_mysql`` is the grader's startup
check: every ``db_id`` is loaded from the expected dump and the grader's user can see all of its tables.

Questions and loader adapted from auto-ontology-eval's BEAVER seeder
(https://github.com/NVIDIA/auto-ontology-eval, ``scripts/seed/seed_beaverbench.py`` at e727e4d6e, Apache-2.0).
"""

import hashlib
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib.parse import unquote, urlparse


logger = logging.getLogger(__name__)

HF_TABLE_REPO = "beaverbench/beaver-table"
HF_TABLE_REVISION = "86f2c4ec45990ac30677dfdfec9826f28171000a"
HF_DUMP_FILE = "beaver_db.zip"

# dw_real questions run against the dw database, as in BEAVER's own evaluator.
DOMAIN_DB = {"dw": "dw", "dw_real": "dw", "nova": "nova", "neutron": "neutron"}


@dataclass(frozen=True)
class BeaverDump:
    member: str  # path inside beaver_db.zip
    sha256: str
    tables: int


# SHA-256 and table count of each dump in beaver_db.zip at HF_TABLE_REVISION (the zip itself: 818ffd77b7ff...).
# nova.sql is about 2 GB; dw and neutron are 93 MB and 6 MB.
DUMPS: Dict[str, BeaverDump] = {
    "dw": BeaverDump("beaver_db/dw.sql", "bf2535fd86bce88748bd8b5a9dd65691e08b82fc43a352d6f17defc00ade2ad0", 97),
    "nova": BeaverDump("beaver_db/nova.sql", "0c237eb9810c0d9d7b65e61e848516afab24409f604de0806c0d64f48b28d9fe", 109),
    "neutron": BeaverDump(
        "beaver_db/neutron.sql", "10ae308e1ff9c30e0547aad3b421a85d5770cd6b84885e9d17bd173018b366a3", 175
    ),
}

META_DB = "_beaver_meta"


def parse_mysql_dsn(dsn: str) -> Dict[str, object]:
    """``mysql://user:password@host:port[/database]`` as ``mysql.connector.connect`` keyword arguments.

    The database part is optional: the grader picks the database per query.
    """
    parsed = urlparse(dsn)
    if parsed.scheme != "mysql":
        raise ValueError(f"expected a mysql:// DSN, got scheme {parsed.scheme!r}")
    params: Dict[str, object] = {
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
    }
    database = unquote(parsed.path.lstrip("/"))
    if database:
        params["database"] = database
    return params


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_beaver_dumps(cache_dir: Path, db_ids: List[str]) -> Dict[str, Path]:
    """Download ``beaver_db.zip`` (once) and extract the dumps for ``db_ids``. Returns ``{db_id: dump path}``.

    Every extracted dump is checked against ``DUMPS``; a mismatch (a partial extract, another revision) is an error.
    """
    cache_dir = Path(cache_dir)
    dumps_dir = cache_dir / "dumps"
    out: Dict[str, Path] = {}
    missing = [db_id for db_id in db_ids if not _dump_is_current(dumps_dir / f"{db_id}.sql", db_id)]
    if missing:
        zip_path = _download_dump_zip(cache_dir)
        dumps_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for db_id in missing:
                target = dumps_dir / f"{db_id}.sql"
                logger.info("Extracting %s to %s", _dump(db_id).member, target)
                partial = target.with_suffix(".sql.partial")
                with zf.open(_dump(db_id).member) as src, open(partial, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                partial.replace(target)
                if not _dump_is_current(target, db_id):
                    raise RuntimeError(f"BEAVER dump {target} does not match the pinned SHA-256 for {db_id!r}")
    for db_id in db_ids:
        out[db_id] = dumps_dir / f"{db_id}.sql"
    return out


def _dump(db_id: str) -> BeaverDump:
    if db_id not in DUMPS:
        raise ValueError(f"unknown BEAVER database {db_id!r}; known: {sorted(DUMPS)}")
    return DUMPS[db_id]


def _dump_is_current(path: Path, db_id: str) -> bool:
    return path.is_file() and sha256_of(path) == _dump(db_id).sha256


def _download_dump_zip(cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    logger.info("Downloading %s from %s@%s ...", HF_DUMP_FILE, HF_TABLE_REPO, HF_TABLE_REVISION[:9])
    return Path(
        hf_hub_download(
            repo_id=HF_TABLE_REPO,
            filename=HF_DUMP_FILE,
            repo_type="dataset",
            revision=HF_TABLE_REVISION,
            local_dir=str(cache_dir),
        )
    )


def dump_statements(dump_path: Path) -> Iterator[str]:
    """The SQL statements of a mysqldump file, in order, without its ``CREATE DATABASE`` and ``USE`` lines.

    mysqldump writes each ``INSERT`` on one line, with newlines inside values escaped, so a statement ends at a line
    that ends with ``;``. Comment lines are skipped; ``/*!...*/;`` lines are statements and are kept.
    """
    current: List[str] = []
    with open(dump_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not current:
                if not stripped or stripped.startswith("--"):
                    continue
                upper = stripped.upper()
                if upper.startswith("CREATE DATABASE") or upper.startswith("USE "):
                    continue
            current.append(line.rstrip("\n"))
            if stripped.endswith(";"):
                yield "\n".join(current)
                current = []
    if current:
        raise ValueError(f"{dump_path} ends inside a statement")


def create_table_statements(dump_path: Path) -> List[str]:
    """The ``CREATE TABLE`` statements of a dump, verbatim, in dump order."""
    return [s for s in dump_statements(dump_path) if s.lstrip().upper().startswith("CREATE TABLE")]


def _connect(dsn: str, **overrides):
    import mysql.connector

    params = parse_mysql_dsn(dsn)
    params.update(overrides)
    return mysql.connector.connect(**params)


def _ensure_meta(cur) -> None:
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{META_DB}`")
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS `{META_DB}`.loaded ("
        "db_id VARCHAR(64) PRIMARY KEY, dump_sha256 CHAR(64) NOT NULL, tables INT NOT NULL, "
        "loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )


def _recorded(cur) -> Dict[str, str]:
    cur.execute(f"SELECT db_id, dump_sha256 FROM `{META_DB}`.loaded")
    return dict(cur.fetchall())


def _existing_databases(cur) -> List[str]:
    cur.execute("SELECT schema_name FROM information_schema.schemata")
    return [row[0] for row in cur.fetchall()]


def load_dump(conn, db_id: str, dump_path: Path) -> int:
    """(Re)create database ``db_id`` from ``dump_path`` and record it. Returns the number of tables created.

    The caller has checked that ``db_id`` is either absent or recorded as loaded by this module.
    """
    cur = conn.cursor()
    cur.execute(f"DELETE FROM `{META_DB}`.loaded WHERE db_id = %s", (db_id,))
    conn.commit()
    cur.execute(f"DROP DATABASE IF EXISTS `{db_id}`")
    cur.execute(f"CREATE DATABASE `{db_id}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci")
    cur.execute(f"USE `{db_id}`")
    tables = 0
    for statement in dump_statements(dump_path):
        cur.execute(statement)
        if statement.lstrip().upper().startswith("CREATE TABLE"):
            tables += 1
        conn.commit()
    cur.execute(
        f"INSERT INTO `{META_DB}`.loaded (db_id, dump_sha256, tables) VALUES (%s, %s, %s)",
        (db_id, _dump(db_id).sha256, tables),
    )
    conn.commit()
    cur.close()
    return tables


def _ensure_reader(cur, admin_user: str, reader_dsn: str, db_ids: List[str]) -> None:
    """Create the grader's user (from ``reader_dsn``) with ``SELECT`` on the BEAVER databases and the meta table only.

    No other privilege: it can't write, create temporary tables, or read files on the server (``FILE``).
    """
    params = parse_mysql_dsn(reader_dsn)
    user, password = str(params["user"]), str(params["password"])
    if not user or user == admin_user:
        logger.warning("BEAVER MySQL: grader DSN uses the loader's user %r; no read-only user is enforced", user)
        return
    cur.execute("CREATE USER IF NOT EXISTS %s@'%%' IDENTIFIED BY %s", (user, password))
    cur.execute("ALTER USER %s@'%%' IDENTIFIED BY %s", (user, password))
    for db in (META_DB, *db_ids):
        cur.execute(f"GRANT SELECT ON `{db}`.* TO %s@'%%'", (user,))


def ensure_beaver_mysql(cache_dir: Path, db_ids: List[str], admin_dsn: str, reader_dsn: str) -> None:
    """Load every ``db_id`` not yet loaded from the pinned dump, then create the grader's read-only user.

    Idempotent. A ``db_id`` recorded with another dump hash is reloaded; a database of that name this module didn't
    record is never touched (error).
    """
    conn = _connect(admin_dsn, autocommit=False)
    try:
        cur = conn.cursor()
        _ensure_meta(cur)
        conn.commit()
        recorded = _recorded(cur)
        existing = set(_existing_databases(cur))
        foreign = sorted(db_id for db_id in db_ids if db_id in existing and db_id not in recorded)
        if foreign:
            raise RuntimeError(
                f"MySQL already has databases {foreign} that this loader didn't create; drop them or use another server"
            )
        stale = [db_id for db_id in db_ids if recorded.get(db_id) != _dump(db_id).sha256]
        if stale:
            dumps = ensure_beaver_dumps(cache_dir, stale)
            for db_id in stale:
                logger.info("Loading BEAVER %s into MySQL ...", db_id)
                tables = load_dump(conn, db_id, dumps[db_id])
                if tables != _dump(db_id).tables:
                    raise RuntimeError(f"BEAVER {db_id}: loaded {tables} tables, expected {_dump(db_id).tables}")
        cur.execute("SELECT CURRENT_USER()")
        admin_user = str(cur.fetchone()[0]).split("@")[0]
        _ensure_reader(cur, admin_user, reader_dsn, db_ids)
        conn.commit()
        cur.close()
    finally:
        conn.close()


def check_beaver_mysql(reader_dsn: str, db_ids: List[str], connection_timeout: Optional[int] = 10) -> None:
    """Raise unless every ``db_id`` is loaded from the pinned dump and the grader's user can see all of its tables.

    Run at startup: a missing database or grant would otherwise score every model query as an empty result.
    """
    conn = _connect(reader_dsn, connection_timeout=connection_timeout)
    try:
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT db_id, dump_sha256, tables FROM `{META_DB}`.loaded")
        except Exception as e:
            raise RuntimeError(
                f"BEAVER MySQL has no {META_DB}.loaded table visible to the grader; set mysql_admin_dsn to load the data"
            ) from e
        loaded = {db_id: (sha, tables) for db_id, sha, tables in cur.fetchall()}
        problems = []
        for db_id in db_ids:
            if db_id not in loaded:
                problems.append(f"{db_id}: not loaded")
                continue
            sha, tables = loaded[db_id]
            if sha != _dump(db_id).sha256:
                problems.append(f"{db_id}: loaded from another dump")
                continue
            cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s", (db_id,))
            visible = cur.fetchone()[0]
            if visible != tables:
                problems.append(f"{db_id}: the grader's user sees {visible} of {tables} tables")
        cur.close()
    finally:
        conn.close()
    if problems:
        raise RuntimeError("BEAVER MySQL is not ready: " + "; ".join(problems))
