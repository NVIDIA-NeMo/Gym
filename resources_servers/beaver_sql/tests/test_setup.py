# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import mysql.connector
import pytest

from resources_servers.beaver_sql import setup_beaver_mysql as setup
from resources_servers.beaver_sql.setup_beaver_mysql import (
    BeaverDump,
    MySqlDsnError,
    check_beaver_mysql,
    create_table_statements,
    dump_statements,
    ensure_beaver_dumps,
    ensure_beaver_mysql,
    parse_mysql_dsn,
)


TINY_DUMP = """CREATE DATABASE tiny;
USE tiny;

-- MySQL dump
/*!40101 SET NAMES utf8mb4 */;

--
-- Table structure for table `T`
--

DROP TABLE IF EXISTS `T`;
CREATE TABLE `T` (
  `ID` int DEFAULT NULL,
  `NAME` varchar(16) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
INSERT INTO `T` VALUES (1,'a;b'),(2,'c');
"""


@pytest.fixture
def tiny(tmp_path, monkeypatch) -> Path:
    """A one-table dump registered as db_id ``tiny``, served from a zip that a fake download returns."""
    dump = TINY_DUMP.encode()
    zip_path = tmp_path / "src" / "beaver_db.zip"
    zip_path.parent.mkdir()
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("beaver_db/tiny.sql", dump)
    monkeypatch.setitem(setup.DUMPS, "tiny", BeaverDump("beaver_db/tiny.sql", hashlib.sha256(dump).hexdigest(), 1))
    downloads = []
    monkeypatch.setattr(setup, "_download_dump_zip", lambda cache_dir: downloads.append(cache_dir) or zip_path)
    return downloads


def test_parse_mysql_dsn() -> None:
    assert parse_mysql_dsn("mysql://u%40x@host:3307/db", "p/w") == {
        "user": "u@x",
        "password": "p/w",
        "host": "host",
        "port": 3307,
        "database": "db",
    }
    assert parse_mysql_dsn("mysql://u@host") == {"user": "u", "password": "", "host": "host", "port": 3306}
    with pytest.raises(MySqlDsnError, match="mysql://"):
        parse_mysql_dsn("postgresql://u@h/db")


@pytest.mark.parametrize(
    "dsn",
    [
        "mysql://u:hunter2@h:3306",
        "mysql://u:hunter2%40x@h",
        "mysql://u:@h",
        "mysql://u@x:hunter2@h",
        # Characters a URL can't hold unescaped, which make urlparse misread the password as a host or a fragment.
        "mysql://u:hunter2/x@h:3306",
        "mysql://u:hunter2#x@h",
        "mysql://u:hunter2?x@h",
        # urlparse's own errors would quote these.
        "mysql://u:hunter2／x@h",
        "mysql://u:hunter2@[h",
    ],
)
def test_a_dsn_holding_a_password_is_refused_without_echoing_it(dsn) -> None:
    with pytest.raises(MySqlDsnError, match="holds a password.*mysql_password") as error:
        parse_mysql_dsn(dsn, "other")
    assert "hunter2" not in str(error.value) and error.value.__cause__ is None


@pytest.mark.parametrize(
    "dsn,message",
    [
        ("mysql://u@h:hunter2", "not a valid URL"),
        ("mysql://u@[hunter2", "not a valid URL"),
        ("mysql://u@h／hunter2", "not a valid URL"),
        ("mysql:hunter2@h", "of the form"),
        # A query or fragment could carry a password the driver never sees (and a run would record).
        ("mysql://u@h/?password=hunter2", "no query or fragment"),
        ("mysql://u@h/db#hunter2", "no query or fragment"),
    ],
)
def test_a_malformed_dsn_is_refused_without_echoing_it(dsn, message) -> None:
    with pytest.raises(MySqlDsnError, match=message) as error:
        parse_mysql_dsn(dsn)
    assert "hunter2" not in str(error.value) and error.value.__cause__ is None


def test_dump_statements(tmp_path) -> None:
    path = tmp_path / "tiny.sql"
    path.write_text(TINY_DUMP)
    statements = list(dump_statements(path))
    assert statements[0] == "/*!40101 SET NAMES utf8mb4 */;"
    assert statements[1] == "DROP TABLE IF EXISTS `T`;"
    assert statements[2].startswith("CREATE TABLE `T` (\n") and statements[2].endswith("utf8mb4_0900_ai_ci;")
    assert statements[3] == "INSERT INTO `T` VALUES (1,'a;b'),(2,'c');"
    assert len(statements) == 4
    assert create_table_statements(path) == [statements[2]]
    path.write_text("CREATE TABLE `T` (\n  `ID` int")
    with pytest.raises(ValueError, match="ends inside a statement"):
        list(dump_statements(path))


def test_unknown_database() -> None:
    with pytest.raises(ValueError, match="unknown BEAVER database"):
        ensure_beaver_dumps(Path("/nonexistent"), ["nope"])


def test_ensure_beaver_dumps_extracts_once(tmp_path, tiny) -> None:
    cache = tmp_path / "cache"
    paths = ensure_beaver_dumps(cache, ["tiny"])
    assert paths == {"tiny": cache / "dumps" / "tiny.sql"} and paths["tiny"].read_text() == TINY_DUMP
    assert ensure_beaver_dumps(cache, ["tiny"]) == paths
    assert len(tiny) == 1


def test_ensure_beaver_dumps_rejects_another_dump(tmp_path, tiny, monkeypatch) -> None:
    monkeypatch.setitem(setup.DUMPS, "tiny", BeaverDump("beaver_db/tiny.sql", "0" * 64, 1))
    with pytest.raises(RuntimeError, match="does not match the pinned SHA-256"):
        ensure_beaver_dumps(tmp_path / "cache", ["tiny"])


# --- Loader and startup check against a fake server ------------------------------------------------------------------


class _FakeMySql:
    """Just enough of a MySQL server for the loader: databases, the meta table, users and grants."""

    def __init__(self, lower_case_table_names: int = 1, databases: Optional[List[str]] = None) -> None:
        self.lctn = lower_case_table_names
        self.databases = set(databases or [])
        self.loaded: Dict[str, tuple] = {}
        self.tables: Dict[str, int] = {}
        self.visible: Optional[Dict[str, int]] = None
        self.statements: List[str] = []
        self.users: Dict[str, str] = {}
        self.grants: List[tuple] = []
        # What the grader's SHOW GRANTS returns, and tables it may see but not read.
        self.show_grants: Optional[List[str]] = None
        self.unreadable: set = set()
        self.lock_free = True
        self.current_role, self.mandatory_roles = "NONE", ""
        self.connects: List[Dict[str, Any]] = []

    def connect(self, **params: Any) -> "_FakeConn":
        self.connects.append(params)
        return _FakeConn(self, params)


class _FakeConn:
    def __init__(self, server: _FakeMySql, params: Dict[str, Any]) -> None:
        self.server, self.params = server, params

    def cursor(self) -> "_FakeCur":
        return _FakeCur(self)

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeCur:
    def __init__(self, conn: _FakeConn) -> None:
        self.conn, self.server = conn, conn.server
        self.result: List[tuple] = []
        self.db: Optional[str] = None

    def execute(self, sql: str, params: tuple = ()) -> None:
        s = self.server
        s.statements.append(sql)
        self.result = []
        if sql == "SELECT @@lower_case_table_names":
            self.result = [(s.lctn,)]
        elif sql.startswith("CREATE DATABASE IF NOT EXISTS"):
            s.databases.add(sql.split("`")[1])
        elif sql.startswith("CREATE DATABASE"):
            name = sql.split("`")[1]
            if name in s.databases:
                raise mysql.connector.errors.DatabaseError("database exists", errno=1007)
            s.databases.add(name)
            s.tables[name] = 0
        elif sql.startswith("DROP DATABASE"):
            s.databases.discard(sql.split("`")[1])
        elif sql.startswith("USE"):
            self.db = sql.split("`")[1]
        elif sql.startswith("CREATE TABLE"):
            if self.db is not None:
                s.tables[self.db] += 1
        elif "FROM `_beaver_meta`.loaded" in sql and sql.startswith("SELECT db_id, dump_sha256, tables"):
            if "_beaver_meta" not in s.databases:
                raise mysql.connector.errors.ProgrammingError("no such table", errno=1146)
            self.result = [(k, sha, n) for k, (sha, n) in s.loaded.items()]
        elif sql.startswith("SELECT db_id, dump_sha256 FROM"):
            self.result = [(k, sha) for k, (sha, _) in s.loaded.items()]
        elif sql.startswith("DELETE FROM `_beaver_meta`.loaded"):
            s.loaded.pop(params[0], None)
        elif sql.startswith("INSERT INTO `_beaver_meta`.loaded") or sql.startswith(
            "REPLACE INTO `_beaver_meta`.loaded"
        ):
            s.loaded[params[0]] = (params[1], 0)
        elif sql.startswith("UPDATE `_beaver_meta`.loaded"):
            s.loaded[params[2]] = (params[0], params[1])
        elif sql.startswith("SELECT CURRENT_ROLE()"):
            self.result = [(s.current_role, s.mandatory_roles)]
        elif sql.startswith("SELECT schema_name"):
            self.result = [(d,) for d in sorted(s.databases)]
        elif sql == "SELECT CURRENT_USER()":
            self.result = [("root@%",)]
        elif sql.startswith("CREATE USER"):
            s.users.setdefault(f"{params[0]}@{params[1]}", params[2])
        elif sql.startswith("GRANT SELECT"):
            s.grants.append((sql.split("`")[1], f"{params[0]}@{params[1]}"))
        elif sql.startswith("SELECT GET_LOCK"):
            self.result = [(1 if s.lock_free else 0,)]
        elif sql == "SHOW GRANTS":
            grants = s.show_grants
            if grants is None:
                grants = ["GRANT USAGE ON *.* TO `beaver_reader`@`%`"] + [
                    f"GRANT SELECT ON `{db}`.* TO `beaver_reader`@`%`" for db, _ in s.grants
                ]
            self.result = [(g,) for g in grants]
        elif sql.startswith("SELECT table_name FROM information_schema.tables"):
            counts = s.visible if s.visible is not None else s.tables
            self.result = [(f"t{i}",) for i in range(counts.get(params[0], 0))]
        elif sql.startswith("SELECT * FROM `"):
            if sql.split("`")[3] in s.unreadable:
                raise mysql.connector.errors.ProgrammingError("SELECT command denied", errno=1142)

    def fetchall(self) -> List[tuple]:
        return self.result

    def fetchone(self) -> tuple:
        return self.result[0]

    def close(self) -> None:
        pass


@pytest.fixture
def fake_mysql(monkeypatch):
    def install(**kwargs: Any) -> _FakeMySql:
        server = _FakeMySql(**kwargs)
        monkeypatch.setattr(mysql.connector, "connect", server.connect)
        return server

    return install


ADMIN, READER = "mysql://root@h:3306", "mysql://beaver_reader@h:3306"


def test_load_then_check(tmp_path, tiny, fake_mysql) -> None:
    server = fake_mysql()
    ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER, admin_password="pw", reader_password="rpw")
    assert server.connects[0]["user"] == "root" and server.connects[0]["password"] == "pw"
    assert server.loaded["tiny"] == (setup.DUMPS["tiny"].sha256, 1)
    assert server.users == {"beaver_reader@%": "rpw"}
    assert server.grants == [("_beaver_meta", "beaver_reader@%"), ("tiny", "beaver_reader@%")]
    assert "INSERT INTO `T` VALUES (1,'a;b'),(2,'c');" in server.statements
    # A new database is recorded as loading only once it is created.
    assert server.statements.index("CREATE DATABASE `tiny` CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci") < next(
        i for i, s in enumerate(server.statements) if s.startswith("REPLACE INTO `_beaver_meta`.loaded")
    )
    check_beaver_mysql(READER, ["tiny"], password="rpw")
    assert server.connects[-1]["user"] == "beaver_reader" and server.connects[-1]["password"] == "rpw"
    # Loaded already: a second run loads nothing.
    loads = [s for s in server.statements if s.startswith("CREATE TABLE `T`")]
    ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    assert [s for s in server.statements if s.startswith("CREATE TABLE `T`")] == loads


def test_refuses_case_sensitive_names(tmp_path, tiny, fake_mysql) -> None:
    fake_mysql(lower_case_table_names=0)
    with pytest.raises(RuntimeError, match="lower-case-table-names=1"):
        ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    with pytest.raises(RuntimeError, match="lower-case-table-names=1"):
        check_beaver_mysql(READER, ["tiny"])


def test_never_touches_a_database_it_did_not_load(tmp_path, tiny, fake_mysql) -> None:
    server = fake_mysql(databases=["tiny"])
    with pytest.raises(RuntimeError, match="didn't create"):
        ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    assert not any(s.startswith("DROP DATABASE") for s in server.statements)


def test_reloads_a_stale_dump(tmp_path, tiny, fake_mysql) -> None:
    server = fake_mysql(databases=["tiny"])
    server.loaded["tiny"] = ("old", 1)
    ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    assert server.loaded["tiny"][0] == setup.DUMPS["tiny"].sha256
    assert "DROP DATABASE IF EXISTS `tiny`" in server.statements


def test_wrong_table_count_is_an_error(tmp_path, tiny, fake_mysql, monkeypatch) -> None:
    fake_mysql()
    monkeypatch.setitem(setup.DUMPS, "tiny", BeaverDump("beaver_db/tiny.sql", setup.DUMPS["tiny"].sha256, 2))
    with pytest.raises(RuntimeError, match="loaded 1 tables, expected 2"):
        ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)


def test_reader_same_as_admin_is_refused(tmp_path, tiny, fake_mysql) -> None:
    server = fake_mysql()
    with pytest.raises(RuntimeError, match="its own read-only user"):
        ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, "mysql://root@h:3306")
    assert server.users == {} and server.grants == []


@pytest.mark.parametrize(
    "admin,reader", [(ADMIN, "mysql://beaver_reader:rpw@h:3306"), ("mysql://root:pw@h:3306", READER)]
)
def test_a_password_in_either_dsn_is_refused_before_anything_changes(
    tmp_path, tiny, fake_mysql, admin, reader
) -> None:
    server = fake_mysql()
    with pytest.raises(MySqlDsnError, match="holds a password"):
        ensure_beaver_mysql(tmp_path, ["tiny"], admin, reader)
    assert server.connects == [] and server.statements == []


def test_existing_reader_keeps_its_password(tmp_path, tiny, fake_mysql) -> None:
    server = fake_mysql()
    server.users["beaver_reader@%"] = "theirs"
    ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    assert server.users == {"beaver_reader@%": "theirs"}
    assert not any(s.startswith("ALTER USER") for s in server.statements)


def test_one_loader_at_a_time(tmp_path, tiny, fake_mysql) -> None:
    server = fake_mysql()
    server.lock_free = False
    with pytest.raises(RuntimeError, match="holds the lock"):
        ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    assert not any(s.startswith("CREATE DATABASE `tiny`") for s in server.statements)


def test_a_database_created_by_someone_else_meanwhile_is_not_dropped(tmp_path, tiny, fake_mysql, monkeypatch) -> None:
    # Absent when checked; created by another client while the dump downloads.
    server = fake_mysql()
    real = setup.ensure_beaver_dumps

    def download_then_race(cache_dir, db_ids):
        paths = real(cache_dir, db_ids)
        server.databases.add("tiny")
        return paths

    monkeypatch.setattr(setup, "ensure_beaver_dumps", download_then_race)
    with pytest.raises(mysql.connector.errors.DatabaseError, match="database exists"):
        ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    # The failed CREATE left no record, so the next start sees someone else's database and leaves it alone.
    assert "tiny" not in server.loaded
    monkeypatch.setattr(setup, "ensure_beaver_dumps", real)
    with pytest.raises(RuntimeError, match="didn't create"):
        ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    assert not any(s.startswith("DROP DATABASE") for s in server.statements)


@pytest.mark.parametrize(
    "loaded,visible,message",
    [
        ({}, None, "tiny: not loaded"),
        ({"tiny": ("other", 1)}, None, "tiny: loaded from another dump"),
        ({"tiny": (None, 1)}, {"tiny": 0}, "sees 0 of 1 tables"),
    ],
)
def test_check_names_what_is_missing(tiny, fake_mysql, loaded, visible, message) -> None:
    server = fake_mysql(databases=["_beaver_meta"])
    server.loaded = {k: (v[0] or setup.DUMPS["tiny"].sha256, v[1]) for k, v in loaded.items()}
    server.visible = visible
    with pytest.raises(RuntimeError, match=message):
        check_beaver_mysql(READER, ["tiny"])


def test_check_reads_every_table(tmp_path, tiny, fake_mysql) -> None:
    server = fake_mysql()
    ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    server.unreadable = {"t0"}  # visible (a REFERENCES grant, say) but not readable
    with pytest.raises(RuntimeError, match="can't read 1 tables"):
        check_beaver_mysql(READER, ["tiny"])


@pytest.mark.parametrize(
    "grant",
    [
        "GRANT SELECT, INSERT ON `tiny`.* TO `beaver_reader`@`%`",
        "GRANT ALL PRIVILEGES ON *.* TO `beaver_reader`@`%`",
        "GRANT SELECT ON *.* TO `beaver_reader`@`%`",
        "GRANT SELECT ON `mysql`.* TO `beaver_reader`@`%`",
        "GRANT FILE ON *.* TO `beaver_reader`@`%`",
        "GRANT `admin_role`@`%` TO `beaver_reader`@`%`",
    ],
)
def test_check_refuses_a_grader_that_can_do_more_than_read_beaver(tmp_path, tiny, fake_mysql, grant) -> None:
    server = fake_mysql()
    ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    server.show_grants = ["GRANT USAGE ON *.* TO `beaver_reader`@`%`", grant]
    with pytest.raises(RuntimeError, match="must hold only SELECT"):
        check_beaver_mysql(READER, ["tiny"])


def test_grader_privilege_problems_accepts_select_on_beaver() -> None:
    assert (
        setup.grader_privilege_problems(
            [
                "GRANT USAGE ON *.* TO `beaver_reader`@`%`",
                "GRANT SELECT ON `dw`.* TO `beaver_reader`@`%`",
                "GRANT SELECT ON `_beaver_meta`.* TO `beaver_reader`@`%`",
            ]
        )
        == []
    )


def test_check_without_meta_table(tiny, fake_mysql) -> None:
    fake_mysql()
    with pytest.raises(RuntimeError, match="set mysql_admin_dsn"):
        check_beaver_mysql(READER, ["tiny"])


# DSNs without passwords; the passwords in BEAVER_MYSQL_TEST_ADMIN_PASSWORD and BEAVER_MYSQL_TEST_PASSWORD.
_LIVE_ADMIN = os.environ.get("BEAVER_MYSQL_TEST_ADMIN_DSN")
_LIVE_READER = os.environ.get("BEAVER_MYSQL_TEST_DSN")
_LIVE_ADMIN_PASSWORD = os.environ.get("BEAVER_MYSQL_TEST_ADMIN_PASSWORD")
_LIVE_READER_PASSWORD = os.environ.get("BEAVER_MYSQL_TEST_PASSWORD")


@pytest.mark.skipif(
    not (_LIVE_ADMIN and _LIVE_READER), reason="set BEAVER_MYSQL_TEST_ADMIN_DSN and BEAVER_MYSQL_TEST_DSN"
)
def test_live_load_check_and_read_only(tmp_path, tiny) -> None:
    """On a real MySQL 8 server with --lower-case-table-names=1: load, check, and the grader's user can't write."""
    import asyncio

    from resources_servers.beaver_sql.ex_acc import MySqlRunner

    try:
        ensure_beaver_mysql(
            tmp_path,
            ["tiny"],
            _LIVE_ADMIN,
            _LIVE_READER,
            admin_password=_LIVE_ADMIN_PASSWORD,
            reader_password=_LIVE_READER_PASSWORD,
        )
        check_beaver_mysql(_LIVE_READER, ["tiny"], password=_LIVE_READER_PASSWORD)
        runner = MySqlRunner(_LIVE_READER, password=_LIVE_READER_PASSWORD, timeout_s=2.0)
        # No session settings: the model's SQL sees the server's own value.
        seen = asyncio.run(runner.run("tiny", "SELECT @@session.max_execution_time = @@global.max_execution_time"))
        assert seen.frame.iloc[0, 0] == 1
        rows = asyncio.run(runner.run("tiny", "SELECT name FROM t ORDER BY id"))
        assert rows.status == "ok" and rows.frame.iloc[:, 0].tolist() == ["a;b", "c"]
        for statement in (
            "DROP TABLE T",
            "CREATE TEMPORARY TABLE x (a INT)",
            "SELECT LOAD_FILE('/etc/hosts') IS NULL",
        ):
            outcome = asyncio.run(runner.run("tiny", statement))
            assert outcome.status == "error" or outcome.frame.iloc[0, 0] == 1, statement
        slow = asyncio.run(runner.run("tiny", "SELECT SLEEP(5)"))
        assert slow.status == "timeout"
        # A second statement doesn't run on after the first result: its connection is killed.
        two = asyncio.run(runner.run("tiny", "SELECT 1; DO SLEEP(5)"))
        assert two.status == "ok" and two.frame.iloc[0, 0] == 1
        admin = setup._connect(_LIVE_ADMIN, _LIVE_ADMIN_PASSWORD)
        try:
            cur = admin.cursor()
            cur.execute("SELECT COUNT(*) FROM information_schema.processlist WHERE state = 'User sleep'")
            assert cur.fetchone()[0] == 0
        finally:
            admin.close()
    finally:
        # Leave the server as found: the test database, its record and the grader's grant on it go.
        conn = setup._connect(_LIVE_ADMIN, _LIVE_ADMIN_PASSWORD)
        try:
            cur = conn.cursor()
            cur.execute("DROP DATABASE IF EXISTS `tiny`")
            cur.execute(f"DELETE FROM `{setup.META_DB}`.loaded WHERE db_id = 'tiny'")
            reader = parse_mysql_dsn(_LIVE_READER)["user"]
            cur.execute("SELECT COUNT(*) FROM mysql.db WHERE db = 'tiny' AND user = %s", (reader,))
            if cur.fetchone()[0]:
                cur.execute("REVOKE SELECT ON `tiny`.* FROM %s@%s", (reader, "%"))
            conn.commit()
        finally:
            conn.close()


def test_an_interrupted_load_is_loaded_again(tmp_path, tiny, fake_mysql) -> None:
    # A previous run marked tiny as loading, created it, and stopped part-way.
    server = fake_mysql(databases=["_beaver_meta", "tiny"])
    server.loaded["tiny"] = ("loading", 0)
    with pytest.raises(RuntimeError, match="its load didn't finish"):
        check_beaver_mysql(READER, ["tiny"])
    ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    assert "DROP DATABASE IF EXISTS `tiny`" in server.statements
    assert server.loaded["tiny"] == (setup.DUMPS["tiny"].sha256, 1)


def test_a_short_load_stays_marked_and_fails_the_check(tmp_path, tiny, fake_mysql, monkeypatch) -> None:
    server = fake_mysql()
    monkeypatch.setitem(setup.DUMPS, "tiny", BeaverDump("beaver_db/tiny.sql", setup.DUMPS["tiny"].sha256, 2))
    with pytest.raises(RuntimeError, match="loaded 1 tables, expected 2"):
        ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    assert server.loaded["tiny"][0] == "loading"
    with pytest.raises(RuntimeError, match="its load didn't finish"):
        check_beaver_mysql(READER, ["tiny"])


@pytest.mark.parametrize(
    "role,mandatory,message", [("`admin`@`%`", "", "active role"), ("NONE", "`dba`@`%`", "mandatory")]
)
def test_check_refuses_privileges_through_roles(tmp_path, tiny, fake_mysql, role, mandatory, message) -> None:
    server = fake_mysql()
    ensure_beaver_mysql(tmp_path, ["tiny"], ADMIN, READER)
    server.current_role, server.mandatory_roles = role, mandatory
    with pytest.raises(RuntimeError, match=message):
        check_beaver_mysql(READER, ["tiny"])
