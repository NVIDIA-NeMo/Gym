# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade a BIRD SQLite answer in a verifier sandbox, so neither the model's SQL nor the gold SQL runs on this server.

Each grade runs ``sandbox_runner.py`` in a fresh sandbox through ``run_verification_sandbox``: the runner, a copy of
``eval_utils.py`` and the request go in as files, and the row's database is uploaded from this server (or found in the
image under ``databases_dir``). The runner and its request are self-contained, with the database path relative to the
runner's directory, so the same files can grade in a sandbox the Environment Server lends once that exists.

A sandbox that cannot start, run the runner or report a valid result raises ``SandboxVerificationError``: the grade is
inconclusive, which the caller masks instead of scoring the answer zero. Nothing falls back to running on the host.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.verification import SandboxVerificationConfig, run_verification_sandbox
from resources_servers.bird_sql.eval_utils import _OUTER_TIMEOUT_BUFFER_S


_HERE = Path(__file__).parent
# The runner imports eval_utils, which uses only the standard library, so any image with Python 3.11+ runs both.
_RUNNER_SOURCES = {name: (_HERE / name).read_text() for name in ("sandbox_runner.py", "eval_utils.py")}
# Interpreter start-up and writing the result, on top of the two queries.
_RUNNER_OVERHEAD_S = 30.0
# The startup probe grades this as both the gold and the model's query: it must match, and returns the version.
PROBE_SQL = "SELECT sqlite_version()"
_ERROR_TAGS = frozenset({"gold_sql_error", "gold_sql_timeout", "pred_sql_error", "pred_sql_timeout"})


class BirdSqlSandboxConfig(SandboxVerificationConfig):
    """The verifier sandbox: one per grade. The image's SQLite decides which SQL functions a query can call."""

    # Same CPython as the server, with SQLite 3.53 (Debian trixie's 3.46 lacks IF(), which model answers use).
    image: str | None = "python:3.13.14-alpine3.24"
    # Absolute directory in the sandbox; each grade uses its own subdirectory for the runner, request and database.
    work_dir: str = "/tmp/bird_sql"
    # Directory in the image, or a read-only provider mount, that holds ``<db_id>/<db_id>.sqlite``. Unset, the row's
    # database is uploaded from this server into each sandbox.
    databases_dir: str | None = None
    # Address-space cap for the runner, below the sandbox's memory: a query that needs more fails as a query instead
    # of the sandbox being killed and the grade lost.
    memory_limit_mib: int | None = Field(default=3072, ge=256)
    # The startup probe refuses an image whose SQLite is older: 3.48 added IF() and IIF() with more than three
    # arguments, which model answers use and the resources server's own Python already had. Null accepts any version.
    min_sqlite_version: str | None = "3.48.0"

    @model_validator(mode="after")
    def _cap_below_sandbox_memory(self) -> "BirdSqlSandboxConfig":
        memory_mib = self.resources.memory_mib
        if self.memory_limit_mib is not None and memory_mib is not None and self.memory_limit_mib >= memory_mib:
            raise ValueError(
                f"sandbox_config.memory_limit_mib ({self.memory_limit_mib}) must be below "
                f"sandbox_config.resources.memory_mib ({memory_mib})"
            )
        return self


@dataclass(frozen=True)
class SqliteGrade:
    """The runner's verdict: ``error`` is ``None`` or a tag of ``eval_utils.execute_and_compare``."""

    match: bool
    error: str | None
    sqlite_version: str | None = None


def runner_timeout_s(sql_execution_timeout_s: float) -> float:
    """Deadline for one runner: two queries, each with eval_utils' backstop, plus start-up."""
    return 2 * (sql_execution_timeout_s + _OUTER_TIMEOUT_BUFFER_S) + _RUNNER_OVERHEAD_S


def parse_grade(text: str) -> SqliteGrade:
    """Read ``result.json``; anything but the runner's own result shape raises ``ValueError``."""
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("match"), bool):
        raise ValueError(f"invalid runner result: {text[:200]!r}")
    error = data.get("error")
    if error is not None and error not in _ERROR_TAGS:
        raise ValueError(f"invalid runner error tag: {error!r}")
    version = data.get("sqlite_version")
    return SqliteGrade(
        match=data["match"] and error is None,
        error=error,
        sqlite_version=version if isinstance(version, str) else None,
    )


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


async def grade_in_fresh_sandbox(
    *,
    provider: Mapping[str, Any],
    config: BirdSqlSandboxConfig,
    database: Path,
    db_id: str,
    gold_sql: str,
    pred_sql: str,
    sql_execution_timeout_s: float,
    metadata: Mapping[str, str],
) -> SqliteGrade:
    """Grade one answer in a new sandbox, stopped on every path.

    ``database`` is this server's copy of the row's database, uploaded unless ``config.databases_dir`` names the
    image's copy. ``provider`` is a resolved provider config and ``metadata`` the sandbox metadata.

    Raises:
        SandboxVerificationError: ``provider_unavailable`` if the sandbox could not start or receive the database,
            ``verifier_error`` if it could not run the runner or report a valid result.
    """
    upload = config.databases_dir is None
    request = {
        # Relative to the runner's directory when uploaded there.
        "database": database.name if upload else f"{config.databases_dir}/{db_id}/{db_id}.sqlite",
        "gold_sql": gold_sql,
        "pred_sql": pred_sql,
        "timeout_s": sql_execution_timeout_s,
        "memory_limit_mib": config.memory_limit_mib,
        "delete_database": upload,
        # Lets the runner tell a truncated upload, an infrastructure fault, from the model's query failing.
        "database_bytes": database.stat().st_size if upload else None,
    }

    async def deliver_database(sandbox: AsyncSandbox, call_dir: str) -> None:
        await sandbox.upload(database, f"{call_dir}/{database.name}")

    return await run_verification_sandbox(
        provider=provider,
        config=config,
        files=_RUNNER_SOURCES,
        request=request,
        exec_timeout_s=runner_timeout_s(sql_execution_timeout_s),
        parse_result=parse_grade,
        metadata=metadata,
        work_dir=config.work_dir,
        setup=deliver_database if upload else None,
    )


async def probe_sandbox(
    *,
    provider: Mapping[str, Any],
    config: BirdSqlSandboxConfig,
    database: Path,
    db_id: str,
    sql_execution_timeout_s: float,
    metadata: Mapping[str, str],
) -> str:
    """Grade one known answer the way every row is graded, and return the image's SQLite version.

    Run once at startup with a small database, so a provider that cannot start sandboxes or deliver the database, an
    image without Python 3.11+, a ``databases_dir`` the image lacks or an image whose SQLite is older than
    ``min_sqlite_version`` fails the server instead of masking or shifting every row.

    Raises:
        SandboxVerificationError: The sandbox could not grade.
        RuntimeError: It graded, but not as expected, or its SQLite is too old.
    """
    grade = await grade_in_fresh_sandbox(
        provider=provider,
        config=config,
        database=database,
        db_id=db_id,
        gold_sql=PROBE_SQL,
        pred_sql=PROBE_SQL,
        sql_execution_timeout_s=sql_execution_timeout_s,
        metadata=metadata,
    )
    if not grade.match or grade.sqlite_version is None:
        raise RuntimeError(f"the probe answer did not match in the verifier sandbox: {grade}")
    minimum = config.min_sqlite_version
    if minimum is not None and _version_tuple(grade.sqlite_version) < _version_tuple(minimum):
        raise RuntimeError(
            f"the verifier image has SQLite {grade.sqlite_version}, below sandbox_config.min_sqlite_version {minimum}"
        )
    return grade.sqlite_version
