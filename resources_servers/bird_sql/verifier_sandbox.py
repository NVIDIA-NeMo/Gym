# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade a BIRD SQLite answer in a verifier sandbox, so neither the model's SQL nor the gold SQL runs on this server.

``grade_in_fresh_sandbox`` starts one sandbox per grade from the configured provider, delivers ``sandbox_runner.py``,
``eval_utils.py`` and the request through ``SandboxSpec.files`` and the row's database by upload (or finds it in the
image under ``databases_dir``), runs the runner and stops the sandbox on every path. ``grade_in_sandbox`` only runs
the runner and reads its result, so a grader given an already started sandbox, such as one the Environment Server
lends for verification, can call it once the same files are in place.

A sandbox that cannot start, run the runner or report a valid result raises ``SandboxGradingError``: the grade is
inconclusive, which the caller masks instead of scoring the answer zero. Nothing falls back to running on the host.
"""

import json
import logging
import shlex
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import Field

from nemo_gym.core import PROVIDER_UNAVAILABLE, VERIFIER_ERROR
from nemo_gym.sandbox import AsyncSandbox, SandboxConfig, read_text, resolve_provider_config


LOG = logging.getLogger(__name__)

_HERE = Path(__file__).parent
RUNNER_FILE = "sandbox_runner.py"
# The runner imports eval_utils, which uses only the standard library, so any image with Python 3.11+ runs both.
_RUNNER_SOURCES = {name: (_HERE / name).read_text() for name in (RUNNER_FILE, "eval_utils.py")}
# Extra time each query may take past its timeout before eval_utils' backstop fires; see execute_sqlite_async.
_QUERY_BACKSTOP_S = 10.0
# Interpreter start-up and reading the result, on top of the two queries.
_RUNNER_OVERHEAD_S = 30.0
# Server-side backstop for a sandbox whose stop never reaches the provider.
_TTL_MARGIN_S = 300.0
_LOG_TAIL_BYTES = 2000


class BirdSqlSandboxConfig(SandboxConfig):
    """The verifier sandbox: one per grade, from ``image``, with no network the provider can cut."""

    image: str = "python:3.13-slim"
    # Interpreter inside the image that runs the runner; it needs the sqlite3 module.
    python: str = "python3"
    # Absolute directory in the sandbox; each grade uses its own subdirectory for the runner, request and database.
    work_dir: str = "/tmp/bird_sql"
    # Directory in the image (or a provider mount) that already holds ``<db_id>/<db_id>.sqlite``. Unset, the row's
    # database is uploaded from this server into each sandbox.
    databases_dir: str | None = None
    # Address-space cap for the runner: a query that needs more fails as a query instead of killing the sandbox.
    memory_limit_mib: int | None = Field(default=3072, ge=256)


class SandboxGradingError(Exception):
    """The verifier sandbox failed, so the answer was not graded. ``failure_kind`` names the shared failure kind."""

    def __init__(self, failure_kind: str, message: str) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


@dataclass(frozen=True)
class SqliteGrade:
    """The runner's verdict: ``error`` is ``None`` or a tag of ``eval_utils.execute_and_compare``."""

    match: bool
    error: str | None


_ERROR_TAGS = frozenset({"gold_sql_error", "gold_sql_timeout", "pred_sql_error", "pred_sql_timeout"})


def runner_timeout_s(sql_execution_timeout_s: float) -> float:
    """Deadline for one runner: two queries, each with eval_utils' backstop, plus start-up."""
    return 2 * (sql_execution_timeout_s + _QUERY_BACKSTOP_S) + _RUNNER_OVERHEAD_S


def parse_grade(text: str) -> SqliteGrade:
    """Read ``result.json``; anything but the runner's own result shape raises ``ValueError``."""
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("match"), bool):
        raise ValueError(f"invalid runner result: {text[:200]!r}")
    error = data.get("error")
    if error is not None and error not in _ERROR_TAGS:
        raise ValueError(f"invalid runner error tag: {error!r}")
    return SqliteGrade(match=data["match"] and error is None, error=error)


async def grade_in_sandbox(sandbox: AsyncSandbox, *, work_dir: str, python: str, timeout_s: float) -> SqliteGrade:
    """Run the runner already delivered to ``work_dir`` with its ``request.json``, and read its result.

    Raises:
        SandboxGradingError: ``verifier_error`` when the command reports an ``error_type`` (no exit code, such as a
            timeout), exits nonzero or leaves no valid result.
    """
    command = (
        f"cd {shlex.quote(work_dir)} && {shlex.quote(python)} {RUNNER_FILE} request.json result.json > runner.log 2>&1"
    )
    execution = await sandbox.exec(command, timeout_s=timeout_s)
    if execution.error_type is not None or execution.return_code != 0:
        raise SandboxGradingError(
            VERIFIER_ERROR,
            f"runner failed (exit={execution.return_code}, error_type={execution.error_type}): "
            f"{(execution.stderr or '')[-_LOG_TAIL_BYTES:]}{await _runner_log_tail(sandbox, work_dir)}",
        )
    text = await read_text(sandbox, path=f"{work_dir}/result.json")
    try:
        return parse_grade(text)
    except ValueError as e:
        raise SandboxGradingError(VERIFIER_ERROR, str(e)) from e


async def grade_in_fresh_sandbox(
    *,
    sandbox_provider: str | Mapping[str, Any],
    global_config_dict: Mapping[str, Any],
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
    image's copy. ``metadata`` adds to the provider's and the config's sandbox metadata.

    Raises:
        SandboxGradingError: ``provider_unavailable`` if the sandbox could not start or receive its files,
            ``verifier_error`` if it could not run the runner or report a valid result.
    """
    work_dir = f"{config.work_dir.rstrip('/')}/{uuid.uuid4().hex}"
    upload = config.databases_dir is None
    sandbox_database = f"{work_dir}/{database.name}" if upload else f"{config.databases_dir}/{db_id}/{db_id}.sqlite"
    request = {
        "database": sandbox_database,
        "gold_sql": gold_sql,
        "pred_sql": pred_sql,
        "timeout_s": sql_execution_timeout_s,
        "memory_limit_mib": config.memory_limit_mib,
        "delete_database": upload,
    }
    files = {f"{work_dir}/{name}": source for name, source in _RUNNER_SOURCES.items()}
    files[f"{work_dir}/request.json"] = json.dumps(request)
    exec_timeout_s = runner_timeout_s(sql_execution_timeout_s)
    spec = config.spec(
        sandbox_provider,
        image=config.image,
        workdir=None,
        metadata=metadata,
        files=files,
        named_configs=global_config_dict,
    )
    if spec.ttl_s is None:
        spec = replace(spec, ttl_s=(config.ready_timeout_s or 600) + exec_timeout_s + _TTL_MARGIN_S)

    async def deliver_database(sandbox: AsyncSandbox) -> None:
        if upload:
            await sandbox.upload(database, sandbox_database)

    sandbox = AsyncSandbox(resolve_provider_config(sandbox_provider, global_config_dict))
    try:
        try:
            # start_with_setup stops the sandbox itself when the files or the database cannot be delivered.
            await sandbox.start_with_setup(spec, deliver_database)
        except ImportError:
            # A missing provider SDK is a deployment error, not an outage: masking it would drop every row.
            raise
        except Exception as e:
            raise SandboxGradingError(PROVIDER_UNAVAILABLE, f"verifier sandbox start failed: {e!r}") from e
        return await grade_in_sandbox(sandbox, work_dir=work_dir, python=config.python, timeout_s=exec_timeout_s)
    except (SandboxGradingError, ImportError):
        raise
    except Exception as e:
        # Request boundary: any other sandbox failure leaves only this grade inconclusive.
        raise SandboxGradingError(VERIFIER_ERROR, f"verifier sandbox failed: {e!r}") from e
    finally:
        try:
            await sandbox.stop()
        except Exception:
            LOG.warning("Failed to stop a BIRD verifier sandbox; the provider reclaims it after ttl_s", exc_info=True)


async def _runner_log_tail(sandbox: AsyncSandbox, work_dir: str) -> str:
    try:
        tail = await sandbox.exec(f"tail -c {_LOG_TAIL_BYTES} {shlex.quote(work_dir)}/runner.log", timeout_s=30)
    except Exception:
        return ""
    return f"\nrunner.log tail:\n{tail.stdout}" if tail.stdout else ""
