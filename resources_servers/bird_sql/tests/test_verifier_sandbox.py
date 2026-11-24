# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BIRD verifier sandbox: the runner that grades inside it, and the sandbox lifecycle around it."""

import asyncio
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.core import PROVIDER_UNAVAILABLE, VERIFIER_ERROR
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxSpec
from nemo_gym.sandbox.testing import FakeSandboxProvider
from nemo_gym.verification import SandboxVerificationError
from resources_servers.bird_sql.verifier_sandbox import (
    BirdSqlSandboxConfig,
    SqliteGrade,
    grade_in_fresh_sandbox,
    probe_sandbox,
    runner_timeout_s,
)


SERVER_DIR = Path(__file__).resolve().parents[1]
# Runs until the per-query timeout stops it.
RUNAWAY_SQL = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c) SELECT count(*) FROM c"


@pytest.fixture
def database(tmp_path) -> Path:
    path = tmp_path / "TestDB.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    conn.commit()
    conn.close()
    return path


def run_runner(tmp_path: Path, database: Path | str, *, pred_sql: str, gold_sql: str = "SELECT x FROM t", **request):
    """Run sandbox_runner.py as the sandbox does: in a directory holding it, eval_utils.py and request.json.

    ``-S -E`` keeps site-packages and the environment out, so an import outside the standard library fails here as it
    would in a bare image.
    """
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("sandbox_runner.py", "eval_utils.py"):
        shutil.copy(SERVER_DIR / name, work / name)
    request = {
        "database": str(database),
        "gold_sql": gold_sql,
        "pred_sql": pred_sql,
        "timeout_s": 10.0,
        "memory_limit_mib": None,
        "delete_database": False,
        **request,
    }
    (work / "request.json").write_text(json.dumps(request))
    process = subprocess.run(
        [sys.executable, "-S", "-E", "sandbox_runner.py", "request.json", "result.json"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads((work / "result.json").read_text()) if (work / "result.json").exists() else None
    if result is not None:
        assert result.pop("sqlite_version") == sqlite3.sqlite_version
    return process, result


class TestRunner:
    """The runner grades with eval_utils.execute_and_compare, so its verdicts are the Resources Server's old ones."""

    @pytest.mark.parametrize(
        ("pred_sql", "expected"),
        [
            ("SELECT x FROM t ORDER BY x DESC", {"match": True, "error": None}),
            ("SELECT x FROM t WHERE x > 1", {"match": False, "error": None}),
            ("SELECT nope FROM t", {"match": False, "error": "pred_sql_error"}),
            ("DROP TABLE t; SELECT 1", {"match": False, "error": "pred_sql_error"}),
        ],
    )
    def test_grades_the_model_query_against_the_gold_query(self, tmp_path, database, pred_sql, expected):
        process, result = run_runner(tmp_path, database, pred_sql=pred_sql)
        assert process.returncode == 0, process.stderr
        assert result == expected

    def test_a_query_that_writes_fails_and_leaves_the_database_unchanged(self, tmp_path, database):
        _, result = run_runner(tmp_path, database, pred_sql="DROP TABLE t")
        assert result == {"match": False, "error": "pred_sql_error"}
        assert sqlite3.connect(str(database)).execute("SELECT count(*) FROM t").fetchone() == (3,)

    def test_a_relative_database_path_is_read_from_the_runner_directory(self, tmp_path, database):
        (tmp_path / "work").mkdir()
        shutil.copy(database, tmp_path / "work" / "copy.sqlite")
        _, result = run_runner(tmp_path, "copy.sqlite", pred_sql="SELECT x FROM t", delete_database=True)
        assert result == {"match": True, "error": None}
        assert not (tmp_path / "work" / "copy.sqlite").exists()

    def test_running_out_of_memory_comparing_results_is_the_model_query_failing(self, monkeypatch, database):
        monkeypatch.syspath_prepend(str(SERVER_DIR))
        import sandbox_runner

        async def too_large(**_kwargs):
            raise MemoryError

        monkeypatch.setattr(sandbox_runner, "execute_and_compare", too_large)
        request = {"database": str(database), "gold_sql": "SELECT 1", "pred_sql": "SELECT 1", "timeout_s": 1.0}
        result = asyncio.run(sandbox_runner._grade(request))
        assert (result["match"], result["error"]) == (False, "pred_sql_error")

    def test_gold_query_failure_is_reported_as_such(self, tmp_path, database):
        _, result = run_runner(tmp_path, database, pred_sql="SELECT x FROM t", gold_sql="SELECT missing FROM t")
        assert result == {"match": False, "error": "gold_sql_error"}

    def test_runaway_gold_query_is_reported_as_a_gold_timeout(self, tmp_path, database):
        _, result = run_runner(tmp_path, database, pred_sql="SELECT 1", gold_sql=RUNAWAY_SQL, timeout_s=0.5)
        assert result == {"match": False, "error": "gold_sql_timeout"}

    def test_a_truncated_upload_fails_the_runner_and_is_removed(self, tmp_path, database):
        size = database.stat().st_size
        process, result = run_runner(
            tmp_path, database, pred_sql="SELECT x FROM t", delete_database=True, database_bytes=size + 1
        )
        assert process.returncode != 0 and f"expected {size + 1}" in process.stderr
        assert result is None
        assert not database.exists()

    def test_runaway_model_query_is_stopped_at_its_timeout(self, tmp_path, database):
        process, result = run_runner(tmp_path, database, pred_sql=RUNAWAY_SQL, timeout_s=0.5)
        assert process.returncode == 0, process.stderr
        assert result == {"match": False, "error": "pred_sql_timeout"}

    def test_uploaded_copy_is_deleted_only_when_asked(self, tmp_path, database):
        _, result = run_runner(tmp_path, database, pred_sql="SELECT x FROM t", delete_database=True)
        assert result == {"match": True, "error": None}
        assert not database.exists()

    def test_missing_database_fails_the_runner_instead_of_grading(self, tmp_path):
        process, result = run_runner(tmp_path, tmp_path / "absent.sqlite", pred_sql="SELECT 1")
        assert process.returncode != 0
        assert "is missing in the sandbox" in process.stderr
        assert result is None
        assert not (tmp_path / "absent.sqlite").exists()

    @pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is enforced only on Linux, as in the sandbox")
    def test_memory_cap_turns_a_huge_allocation_into_a_failed_model_query(self, tmp_path, database):
        huge = "SELECT hex(zeroblob(100000000))"  # a 200 MB string
        _, uncapped = run_runner(tmp_path / "uncapped", database, pred_sql=huge)
        _, capped = run_runner(tmp_path / "capped", database, pred_sql=huge, memory_limit_mib=256)
        assert uncapped == {"match": False, "error": None}
        assert capped == {"match": False, "error": "pred_sql_error"}


class ScriptedProvider(FakeSandboxProvider):
    """A fake provider that records what the grader sends and answers the runner with ``result_text``."""

    def __init__(
        self,
        *,
        execution: SandboxExecResult | None = None,
        result_text: str = '{"match": true, "error": null}',
        create_error: Exception | None = None,
        upload_error: Exception | None = None,
        hang: asyncio.Event | None = None,
    ) -> None:
        super().__init__()
        self.execution = execution or SandboxExecResult(stdout="", stderr=None, return_code=0)
        self.result_text = result_text
        self.create_error = create_error
        self.upload_error = upload_error
        self.hang = hang
        self.specs: list[SandboxSpec] = []
        self.uploads: list[tuple[Path, str]] = []
        self.commands: list[tuple[str, dict]] = []

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if self.create_error is not None:
            raise self.create_error
        self.specs.append(spec)
        return await super().create(spec)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        if self.upload_error is not None and target_path.endswith(".sqlite"):
            raise self.upload_error
        self.uploads.append((source_path, target_path))

    async def exec(self, handle: SandboxHandle, command: str, **kwargs) -> SandboxExecResult:
        self.commands.append((command, kwargs))
        if self.hang is not None and "sandbox_runner.py" in command:
            await self.hang.wait()
        return self.execution if "sandbox_runner.py" in command else SandboxExecResult("", None, 0)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target_path.write_text(self.result_text)


async def grade(provider: ScriptedProvider, database: Path, **config) -> SqliteGrade:
    return await grade_in_fresh_sandbox(
        provider=provider,
        config=BirdSqlSandboxConfig(**config),
        database=database,
        db_id="TestDB",
        gold_sql="SELECT x FROM t",
        pred_sql="SELECT 1",
        sql_execution_timeout_s=30.0,
        metadata={"sandbox-api": "scripted", "db_id": "TestDB"},
    )


def database_uploads(provider: ScriptedProvider) -> list[tuple[Path, str]]:
    """Uploads after start: the ``SandboxSpec.files`` uploads go through the provider too and are left out."""
    return [(source, target) for source, target in provider.uploads if target.endswith(".sqlite")]


def uploaded_request(provider: ScriptedProvider) -> tuple[str, dict]:
    [spec] = provider.specs
    [request_path] = [path for path in spec.files if path.endswith("/request.json")]
    return request_path.removesuffix("/request.json"), json.loads(spec.files[request_path])


class TestFreshSandbox:
    @pytest.mark.asyncio
    async def test_delivers_the_runner_request_and_database_then_stops(self, database):
        provider = ScriptedProvider()

        result = await grade(provider, database)

        assert result == SqliteGrade(match=True, error=None)
        work_dir, request = uploaded_request(provider)
        assert work_dir.startswith("/tmp/bird_sql/")
        [spec] = provider.specs
        assert spec.files[f"{work_dir}/sandbox_runner.py"] == (SERVER_DIR / "sandbox_runner.py").read_text()
        assert spec.files[f"{work_dir}/eval_utils.py"] == (SERVER_DIR / "eval_utils.py").read_text()
        assert request == {
            "database": "TestDB.sqlite",
            "gold_sql": "SELECT x FROM t",
            "pred_sql": "SELECT 1",
            "timeout_s": 30.0,
            "memory_limit_mib": 3072,
            "delete_database": True,
            "database_bytes": database.stat().st_size,
        }
        assert database_uploads(provider) == [(database, f"{work_dir}/TestDB.sqlite")]
        assert spec.image == "python:3.13.14-slim-trixie"
        assert spec.metadata == {"sandbox-api": "scripted", "db_id": "TestDB"}
        assert spec.ttl_s == 600 + runner_timeout_s(30.0) + 300
        [(command, kwargs)] = provider.commands
        assert command.startswith(f"cd {work_dir} && python3 sandbox_runner.py request.json result.json")
        assert kwargs["timeout_s"] == runner_timeout_s(30.0) == 110.0
        assert provider.closed == ["fake-1"]

    @pytest.mark.asyncio
    async def test_databases_dir_grades_the_image_copy_without_uploading(self, database):
        provider = ScriptedProvider()

        await grade(provider, database, databases_dir="/data/bird")

        _, request = uploaded_request(provider)
        assert request["database"] == "/data/bird/TestDB/TestDB.sqlite"
        assert request["delete_database"] is False and request["database_bytes"] is None
        assert database_uploads(provider) == []

    @pytest.mark.parametrize(
        "execution",
        [
            SandboxExecResult(stdout=None, stderr=None, return_code=-1, error_type="timeout"),
            SandboxExecResult(stdout=None, stderr="Killed", return_code=137),
        ],
        ids=["error_type", "nonzero_exit"],
    )
    @pytest.mark.asyncio
    async def test_a_runner_that_does_not_finish_is_inconclusive(self, database, execution):
        provider = ScriptedProvider(execution=execution)

        with pytest.raises(SandboxVerificationError) as raised:
            await grade(provider, database)

        assert raised.value.failure_kind == VERIFIER_ERROR
        assert provider.closed == ["fake-1"]

    @pytest.mark.parametrize(
        "result_text",
        ["not json", '{"match": "yes", "error": null}', '{"match": false, "error": "made_up"}', "[]"],
    )
    @pytest.mark.asyncio
    async def test_an_invalid_result_is_inconclusive(self, database, result_text):
        provider = ScriptedProvider(result_text=result_text)

        with pytest.raises(SandboxVerificationError) as raised:
            await grade(provider, database)

        assert raised.value.failure_kind == VERIFIER_ERROR
        assert provider.closed == ["fake-1"]

    @pytest.mark.asyncio
    async def test_a_sandbox_that_cannot_start_is_provider_unavailable(self, database):
        provider = ScriptedProvider(create_error=RuntimeError("quota exceeded"))

        with pytest.raises(SandboxVerificationError) as raised:
            await grade(provider, database)

        assert raised.value.failure_kind == PROVIDER_UNAVAILABLE
        assert provider.commands == []

    @pytest.mark.asyncio
    async def test_a_failed_database_upload_stops_the_sandbox_before_any_query(self, database):
        provider = ScriptedProvider(upload_error=OSError("disk full"))

        with pytest.raises(SandboxVerificationError) as raised:
            await grade(provider, database)

        assert raised.value.failure_kind == PROVIDER_UNAVAILABLE
        assert provider.commands == []
        assert provider.closed == ["fake-1"]

    @pytest.mark.asyncio
    async def test_a_cancelled_grade_still_stops_its_sandbox(self, database):
        provider = ScriptedProvider(hang=asyncio.Event())
        task = asyncio.create_task(grade(provider, database))
        while not provider.commands:
            await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert provider.closed == ["fake-1"]


def test_the_memory_cap_must_stay_below_the_sandbox_memory():
    with pytest.raises(ValueError, match="memory_limit_mib"):
        BirdSqlSandboxConfig(resources={"memory_mib": 2048}, memory_limit_mib=2048)
    assert BirdSqlSandboxConfig(resources={"memory_mib": 4096}, memory_limit_mib=3072).memory_limit_mib == 3072


class TestProbe:
    """The startup probe grades a known answer through the real runner on the local provider (tests only)."""

    @staticmethod
    async def probe(tmp_path: Path, database: Path, **config) -> str:
        config = BirdSqlSandboxConfig(
            python=sys.executable, work_dir=str(tmp_path / "work"), memory_limit_mib=None, **config
        )
        return await probe_sandbox(
            provider={"local": {}},
            config=config,
            database=database,
            db_id="TestDB",
            sql_execution_timeout_s=10.0,
            metadata={},
        )

    @pytest.mark.asyncio
    async def test_returns_the_sqlite_version_the_image_grades_with(self, tmp_path, database):
        assert await self.probe(tmp_path, database) == sqlite3.sqlite_version
        assert database.exists(), "only the uploaded copy is deleted"

    @pytest.mark.asyncio
    async def test_a_databases_dir_without_the_database_fails(self, tmp_path, database):
        with pytest.raises(SandboxVerificationError) as raised:
            await self.probe(tmp_path, database, databases_dir=str(tmp_path / "absent"))
        assert raised.value.failure_kind == VERIFIER_ERROR
        assert "is missing in the sandbox" in str(raised.value)
