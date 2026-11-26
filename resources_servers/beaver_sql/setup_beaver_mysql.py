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
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib.parse import unquote, urlparse


logger = logging.getLogger(__name__)

HF_TABLE_REPO = "beaverbench/beaver-table"
HF_TABLE_REVISION = "86f2c4ec45990ac30677dfdfec9826f28171000a"  # pragma: allowlist secret
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
        "beaver_db/neutron.sql",
        "10ae308e1ff9c30e0547aad3b421a85d5770cd6b84885e9d17bd173018b366a3",  # pragma: allowlist secret
        175,
    ),
}

META_DB = "_beaver_meta"
# The grader reads this one table of META_DB, granted as SHOW GRANTS names it.
_META_TABLE = f"`{META_DB}`.`loaded`"
# ER_NONEXISTING_GRANT: REVOKE of a grant the account doesn't hold.
_NO_SUCH_GRANT = 1141
# The dump hash recorded while a database is being loaded.
_LOADING = "loading"

CASE_INSENSITIVE_NAMES_HELP = (
    "BEAVER needs MySQL with case-insensitive table names: start it with --lower-case-table-names=1 (it can only be "
    "set when the data directory is created). The dumps were made on macOS, where MySQL compares table names "
    "case-insensitively, and many gold queries spell table names or aliases in another case: with case-sensitive "
    "names (Linux's default), 950 of neutron's 1,017 and 15 of dw_real's 121 gold queries fail."
)


def require_case_insensitive_names(cur) -> None:
    """Raise unless the server compares table names case-insensitively, as the macOS MySQL the dumps came from."""
    cur.execute("SELECT @@lower_case_table_names")
    if int(cur.fetchone()[0]) == 0:
        raise RuntimeError(CASE_INSENSITIVE_NAMES_HELP)


# `user:anything@` in the user info, also when the password holds a character urlparse would split the URL at.
_DSN_PASSWORD = re.compile(r"^[^:/?#]*://[^/?#@:]*:[^@]*@")


class MySqlDsnError(Exception):
    """A MySQL DSN the grader refuses. Its message never quotes the DSN. Not a ``ValueError``, so pydantic doesn't wrap
    it, at server startup, in an error that prints the config holding the DSN."""


def parse_mysql_dsn(dsn: str, password: Optional[str] = None) -> Dict[str, object]:
    """``mysql://user@host:port[/database]`` and the user's password as ``mysql.connector.connect`` keyword arguments.

    The password comes apart from the DSN (``mysql_password`` in the server's config, from the environment), and a DSN
    that holds one is refused: a run records its DSN as a setting, and a password must never be in it. The database
    part is optional: the grader picks the database per query.
    """
    holds_password = (
        "the MySQL DSN holds a password: leave it out of the DSN and set mysql_password (mysql_admin_password for "
        "the admin DSN) instead, for example from ${oc.env:BEAVER_MYSQL_PASSWORD}"
    )
    # Checked on the raw string first, and no parsing error is passed on: urlparse's own errors can quote the user
    # info, password included.
    if _DSN_PASSWORD.match(dsn):
        raise MySqlDsnError(holds_password)
    try:
        parsed = urlparse(dsn)
        scheme, netloc, has_password = parsed.scheme, parsed.netloc, parsed.password is not None
        username, hostname, path = parsed.username, parsed.hostname, parsed.path
        extra = parsed.query or parsed.fragment
        port = parsed.port
    except ValueError:
        raise MySqlDsnError("the MySQL DSN is not a valid URL of the form mysql://user@host:port") from None
    if scheme != "mysql" or not netloc:
        raise MySqlDsnError("expected a MySQL DSN of the form mysql://user@host:port")
    if has_password:
        raise MySqlDsnError(holds_password)
    if extra:
        # A query can carry a password too (`?password=...`), and the driver would ignore it here anyway.
        raise MySqlDsnError("the MySQL DSN takes no query or fragment (mysql://user@host:port[/database])")
    params: Dict[str, object] = {
        "user": unquote(username or ""),
        "password": password or "",
        "host": hostname or "127.0.0.1",
        "port": port or 3306,
    }
    database = unquote(path.lstrip("/"))
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
    for db_id in db_ids:
        _dump(db_id)
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


def _connect(dsn: str, password: Optional[str] = None, **overrides):
    import mysql.connector

    params = parse_mysql_dsn(dsn, password)
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


def load_dump(conn, db_id: str, dump_path: Path, owned: bool) -> int:
    """Create database ``db_id`` from ``dump_path`` and record it. Returns the number of tables created.

    ``owned``: the database is recorded as this module's, so it is dropped and created again. Otherwise it is only
    created, which fails if someone else created a database of that name since the caller checked, and recorded once
    created.

    While loading, the record holds a mark instead of the pinned hash: if the load stops part-way, the next start finds
    the mark, owns the database and loads it again. An owned database is marked before it is dropped. A new one is
    marked only after its ``CREATE`` succeeds, so a ``CREATE`` that fails on someone else's database leaves no record
    that would let a later start drop it; a stop between the two leaves an unrecorded database, which the next start
    refuses to touch.
    """
    cur = conn.cursor()
    mark = f"REPLACE INTO `{META_DB}`.loaded (db_id, dump_sha256, tables) VALUES (%s, %s, 0)"
    create = f"CREATE DATABASE `{db_id}` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
    if owned:
        cur.execute(mark, (db_id, _LOADING))
        conn.commit()
        cur.execute(f"DROP DATABASE IF EXISTS `{db_id}`")
        cur.execute(create)
    else:
        cur.execute(create)
        cur.execute(mark, (db_id, _LOADING))
        conn.commit()
    cur.execute(f"USE `{db_id}`")
    tables = 0
    for statement in dump_statements(dump_path):
        cur.execute(statement)
        if statement.lstrip().upper().startswith("CREATE TABLE"):
            tables += 1
        conn.commit()
    if tables != _dump(db_id).tables:
        # Left marked as loading, so the next start loads it again.
        raise RuntimeError(f"BEAVER {db_id}: loaded {tables} tables, expected {_dump(db_id).tables}")
    cur.execute(
        f"UPDATE `{META_DB}`.loaded SET dump_sha256 = %s, tables = %s WHERE db_id = %s",
        (_dump(db_id).sha256, tables, db_id),
    )
    conn.commit()
    cur.close()
    return tables


def _ensure_reader(cur, admin_user: str, reader_dsn: str, reader_password: Optional[str], db_ids: List[str]) -> None:
    """Create the grader's user (from ``reader_dsn``) if it doesn't exist, and grant it ``SELECT`` on the BEAVER databases
    and the meta table.

    An existing user keeps its password (this never resets an account it may not own); the startup check then refuses it
    if it holds anything beyond ``SELECT``, or if ``reader_password`` is wrong.
    """
    import mysql.connector

    params = parse_mysql_dsn(reader_dsn, reader_password)
    user, password = str(params["user"]), str(params["password"])
    if not user or user == admin_user:
        raise RuntimeError("the grader's DSN must use its own read-only user, not the loader's")
    # The host is a parameter too: the driver's C extension leaves a literal '%%' as two characters.
    cur.execute("CREATE USER IF NOT EXISTS %s@%s IDENTIFIED BY %s", (user, "%", password))
    # The meta database is granted by its table: in a database-level grant `_` is a wildcard (unless partial_revokes is
    # on), so `_beaver_meta`.* would reach every database named like it. That grant, which this loader used to make,
    # is revoked; revoking it directly needs no read access to the grant tables, and "no such grant" is the usual case.
    try:
        cur.execute(f"REVOKE SELECT ON `{META_DB}`.* FROM %s@%s", (user, "%"))
    except mysql.connector.Error as e:
        if e.errno != _NO_SUCH_GRANT:
            raise
    cur.execute(f"GRANT SELECT ON {_META_TABLE} TO %s@%s", (user, "%"))
    for db in db_ids:
        cur.execute(f"GRANT SELECT ON `{db}`.* TO %s@%s", (user, "%"))


def ensure_beaver_mysql(
    cache_dir: Path,
    db_ids: List[str],
    admin_dsn: str,
    reader_dsn: str,
    *,
    admin_password: Optional[str] = None,
    reader_password: Optional[str] = None,
) -> None:
    """Load every ``db_id`` not yet loaded from the pinned dump, then create the grader's read-only user.

    Idempotent. A ``db_id`` recorded with another dump hash, or recorded but missing from the server, is reloaded; a
    database of that name this module didn't record is never touched (error).
    """
    # Both DSNs are checked before anything changes, so a DSN holding a password fails here, not after a load.
    parse_mysql_dsn(reader_dsn, reader_password)
    conn = _connect(admin_dsn, admin_password, autocommit=False)
    try:
        cur = conn.cursor()
        require_case_insensitive_names(cur)
        # One loader at a time per server (released when this connection closes).
        cur.execute("SELECT GET_LOCK('beaver_sql_loader', 3600)")
        if cur.fetchone()[0] != 1:
            raise RuntimeError("another BEAVER loader holds the lock on this MySQL server")
        _ensure_meta(cur)
        conn.commit()
        recorded = _recorded(cur)
        existing = set(_existing_databases(cur))
        foreign = sorted(db_id for db_id in db_ids if db_id in existing and db_id not in recorded)
        if foreign:
            raise RuntimeError(
                f"MySQL already has databases {foreign} that this loader didn't create; drop them or use another server"
            )
        # A recorded database that is gone (dropped, or a backup restored without it) is loaded again as a new one: its
        # record goes first, so one that someone else creates meanwhile makes the CREATE fail instead of being dropped.
        for db_id in [db_id for db_id in db_ids if db_id in recorded and db_id not in existing]:
            cur.execute(f"DELETE FROM `{META_DB}`.loaded WHERE db_id = %s", (db_id,))
            del recorded[db_id]
        conn.commit()
        stale = [db_id for db_id in db_ids if recorded.get(db_id) != _dump(db_id).sha256]
        if stale:
            dumps = ensure_beaver_dumps(cache_dir, stale)
            for db_id in stale:
                logger.info("Loading BEAVER %s into MySQL ...", db_id)
                load_dump(conn, db_id, dumps[db_id], owned=db_id in recorded)
        cur.execute("SELECT CURRENT_USER()")
        admin_user = str(cur.fetchone()[0]).split("@")[0]
        _ensure_reader(cur, admin_user, reader_dsn, reader_password, db_ids)
        conn.commit()
        cur.close()
    finally:
        conn.close()


_GRANT = re.compile(r"^GRANT (.+?) ON (\S+) TO ")


def grader_privilege_problems(grants: List[str]) -> List[str]:
    """The grants (``SHOW GRANTS`` lines) that give the grader's user more than reading BEAVER: a role, any privilege
    but ``SELECT`` and ``USAGE``, ``SELECT`` beyond the BEAVER databases and the meta table (``*.*`` would expose
    the ``mysql`` system tables; ``_beaver_meta``.* every database its wildcards match), or grant authority
    (``WITH GRANT OPTION``), which would let it pass its access on."""
    readable = {_META_TABLE, *(f"`{db}`.*" for db in DUMPS)}
    problems = []
    for grant in grants:
        match = _GRANT.match(grant)
        if match is None:
            problems.append(grant)  # a role grant
            continue
        if grant.rstrip().upper().endswith(" WITH GRANT OPTION"):
            problems.append(grant)
            continue
        privileges = {p.strip().upper() for p in match.group(1).split(",")}
        if privileges == {"USAGE"}:
            continue
        if privileges != {"SELECT"} or match.group(2) not in readable:
            problems.append(grant)
    return problems


def check_beaver_mysql(
    reader_dsn: str, db_ids: List[str], connection_timeout: Optional[int] = 10, *, password: Optional[str] = None
) -> None:
    """Raise unless the grader's user holds only ``SELECT`` and can read every table of every ``db_id``, loaded from the
    pinned dump.

    Run at startup: a missing database or grant would otherwise score every model query as an empty result, and a user
    that can write would let a model's query change the data.
    """
    conn = _connect(reader_dsn, password, connection_timeout=connection_timeout)
    try:
        cur = conn.cursor()
        require_case_insensitive_names(cur)
        cur.execute("SHOW GRANTS")
        extra = grader_privilege_problems([g for (g,) in cur.fetchall()])
        # Privileges can also come through roles: an active one, or a mandatory one the user could activate.
        cur.execute("SELECT CURRENT_ROLE(), @@GLOBAL.mandatory_roles")
        current_role, mandatory = cur.fetchone()
        if current_role not in (None, "", "NONE"):
            extra.append(f"active role {current_role}")
        if mandatory:
            extra.append(f"mandatory roles {mandatory}")
        if extra:
            message = "the grader's MySQL user must hold only SELECT (and USAGE); it also has: " + "; ".join(extra)
            if any(g.startswith(f"GRANT SELECT ON `{META_DB}`.* TO ") for g in extra):
                message += (
                    f". Its SELECT on `{META_DB}`.* comes from an earlier version of this loader: start once with "
                    f"mysql_admin_dsn, which moves it to the {_META_TABLE} table"
                )
            raise RuntimeError(message)
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
                problems.append(
                    f"{db_id}: " + ("its load didn't finish" if sha == _LOADING else "loaded from another dump")
                )
                continue
            tables = _dump(db_id).tables
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (db_id,))
            visible = [name for (name,) in cur.fetchall()]
            if len(visible) != tables:
                problem = f"{db_id}: the grader's user sees {len(visible)} of {tables} tables"
                if not visible:
                    # The grader can't tell a missing database from one it has no grant on.
                    problem += " (the database is missing or not granted to it: start with mysql_admin_dsn to load it"
                    problem += " and grant it)"
                problems.append(problem)
                continue
            # Seeing a table isn't reading it (a REFERENCES grant shows it too): read each one.
            unreadable = []
            for name in visible:
                try:
                    cur.execute(f"SELECT * FROM `{db_id}`.`{name}` LIMIT 0")
                    cur.fetchall()
                except Exception:
                    unreadable.append(name)
            if unreadable:
                problems.append(
                    f"{db_id}: the grader's user can't read {len(unreadable)} tables, e.g. {unreadable[:3]}"
                )
        cur.close()
    finally:
        conn.close()
    if problems:
        raise RuntimeError("BEAVER MySQL is not ready: " + "; ".join(problems))
