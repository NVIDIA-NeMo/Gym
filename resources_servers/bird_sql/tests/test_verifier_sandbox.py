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
from resources_servers.bird_sql.verifier_sandbox import (
    BirdSqlSandboxConfig,
    SandboxGradingError,
    SqliteGrade,
    grade_in_fresh_sandbox,
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


def run_runner(tmp_path: Path, database: Path, *, pred_sql: str, gold_sql: str = "SELECT x FROM t", **request):
    """Run sandbox_runner.py as the sandbox does: in a directory holding it, eval_utils.py and request.json."""
    work = tmp_path / "work"
    work.mkdir(parents=True)
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
        [sys.executable, "sandbox_runner.py", "request.json", "result.json"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads((work / "result.json").read_text()) if (work / "result.json").exists() else None
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

    def test_gold_query_failure_is_reported_as_such(self, tmp_path, database):
        _, result = run_runner(tmp_path, database, pred_sql="SELECT x FROM t", gold_sql="SELECT missing FROM t")
        assert result == {"match": False, "error": "gold_sql_error"}

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


async def grade(provider: ScriptedProvider, monkeypatch, database: Path, **config) -> SqliteGrade:
    monkeypatch.setattr("resources_servers.bird_sql.verifier_sandbox.resolve_provider_config", lambda *_a: provider)
    return await grade_in_fresh_sandbox(
        sandbox_provider={"scripted": {}, "default_metadata": {"sandbox-api": "scripted"}},
        global_config_dict={},
        config=BirdSqlSandboxConfig(**config),
        database=database,
        db_id="TestDB",
        gold_sql="SELECT x FROM t",
        pred_sql="SELECT 1",
        sql_execution_timeout_s=30.0,
        metadata={"db_id": "TestDB"},
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
    async def test_delivers_the_runner_request_and_database_then_stops(self, monkeypatch, database):
        provider = ScriptedProvider()

        result = await grade(provider, monkeypatch, database)

        assert result == SqliteGrade(match=True, error=None)
        work_dir, request = uploaded_request(provider)
        assert work_dir.startswith("/tmp/bird_sql/")
        [spec] = provider.specs
        assert spec.files[f"{work_dir}/sandbox_runner.py"] == (SERVER_DIR / "sandbox_runner.py").read_text()
        assert spec.files[f"{work_dir}/eval_utils.py"] == (SERVER_DIR / "eval_utils.py").read_text()
        assert request == {
            "database": f"{work_dir}/TestDB.sqlite",
            "gold_sql": "SELECT x FROM t",
            "pred_sql": "SELECT 1",
            "timeout_s": 30.0,
            "memory_limit_mib": 3072,
            "delete_database": True,
        }
        assert database_uploads(provider) == [(database, f"{work_dir}/TestDB.sqlite")]
        assert spec.image == "python:3.13-slim"
        assert spec.metadata == {"sandbox-api": "scripted", "db_id": "TestDB"}
        assert spec.ttl_s == 600 + runner_timeout_s(30.0) + 300
        [(command, kwargs)] = provider.commands
        assert command.startswith(f"cd {work_dir} && python3 sandbox_runner.py request.json result.json")
        assert kwargs["timeout_s"] == runner_timeout_s(30.0) == 110.0
        assert provider.closed == ["fake-1"]

    @pytest.mark.asyncio
    async def test_databases_dir_grades_the_image_copy_without_uploading(self, monkeypatch, database):
        provider = ScriptedProvider()

        await grade(provider, monkeypatch, database, databases_dir="/data/bird")

        _, request = uploaded_request(provider)
        assert request["database"] == "/data/bird/TestDB/TestDB.sqlite"
        assert request["delete_database"] is False
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
    async def test_a_runner_that_does_not_finish_is_inconclusive(self, monkeypatch, database, execution):
        provider = ScriptedProvider(execution=execution)

        with pytest.raises(SandboxGradingError) as raised:
            await grade(provider, monkeypatch, database)

        assert raised.value.failure_kind == VERIFIER_ERROR
        assert provider.closed == ["fake-1"]

    @pytest.mark.parametrize(
        "result_text",
        ["not json", '{"match": "yes", "error": null}', '{"match": false, "error": "made_up"}', "[]"],
    )
    @pytest.mark.asyncio
    async def test_an_invalid_result_is_inconclusive(self, monkeypatch, database, result_text):
        provider = ScriptedProvider(result_text=result_text)

        with pytest.raises(SandboxGradingError) as raised:
            await grade(provider, monkeypatch, database)

        assert raised.value.failure_kind == VERIFIER_ERROR
        assert provider.closed == ["fake-1"]

    @pytest.mark.asyncio
    async def test_a_sandbox_that_cannot_start_is_provider_unavailable(self, monkeypatch, database):
        provider = ScriptedProvider(create_error=RuntimeError("quota exceeded"))

        with pytest.raises(SandboxGradingError) as raised:
            await grade(provider, monkeypatch, database)

        assert raised.value.failure_kind == PROVIDER_UNAVAILABLE
        assert provider.commands == []

    @pytest.mark.asyncio
    async def test_a_failed_database_upload_stops_the_sandbox_before_any_query(self, monkeypatch, database):
        provider = ScriptedProvider(upload_error=OSError("disk full"))

        with pytest.raises(SandboxGradingError) as raised:
            await grade(provider, monkeypatch, database)

        assert raised.value.failure_kind == PROVIDER_UNAVAILABLE
        assert provider.commands == []
        assert provider.closed == ["fake-1"]

    @pytest.mark.asyncio
    async def test_a_cancelled_grade_still_stops_its_sandbox(self, monkeypatch, database):
        provider = ScriptedProvider(hang=asyncio.Event())
        task = asyncio.create_task(grade(provider, monkeypatch, database))
        while not provider.commands:
            await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert provider.closed == ["fake-1"]
