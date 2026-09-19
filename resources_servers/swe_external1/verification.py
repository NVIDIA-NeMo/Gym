# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upload trusted task assets and execute the supplied grader, without rewriting it."""

import asyncio
import math
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from resources_servers.swe_external1.task_data import TaskFile, TaskMetadata


@dataclass
class VerificationResult:
    reward: float = 0.0
    evaluation_completed: bool = False
    test_output: str = ""
    solution_output: str = ""
    exit_code: int | None = None
    error: str | None = None


def output_text(result: Any) -> str:
    return ((result.stdout or "") + "\n" + (result.stderr or ""))[-100_000:]


def require_success(result: Any, stage: str) -> None:
    if result.error_type or result.return_code != 0:
        raise RuntimeError(f"{stage} failed: {result.error_type or result.return_code}; {output_text(result)[-2000:]}")


async def upload_files(sandbox: Any, files: list[TaskFile], root: str, timeout_s: int) -> None:
    if root not in ("/tests", "/solution"):
        raise ValueError("unsupported asset root")
    # Reserved paths inside the disposable task sandbox, never host paths.
    result = await sandbox.exec(f"rm -rf -- {root} && mkdir -p -- {root}", timeout_s=timeout_s)
    require_success(result, "asset directory preparation")
    with TemporaryDirectory(prefix="swe-assets-") as directory:
        for index, asset in enumerate(files):
            local = Path(directory) / str(index)
            await asyncio.to_thread(local.write_bytes, asset.decoded())
            remote = str(PurePosixPath(root) / asset.path)
            result = await sandbox.exec(
                f"mkdir -p -- {shlex.quote(str(PurePosixPath(remote).parent))}", timeout_s=timeout_s
            )
            require_success(result, "asset parent preparation")
            await sandbox.upload(local_path=local, remote_path=remote)
            result = await sandbox.exec(f"chmod {asset.mode:o} -- {shlex.quote(remote)}", timeout_s=timeout_s)
            require_success(result, "asset mode restoration")


async def run_verification(
    sandbox: Any, task: TaskMetadata, *, golden: bool = False, timeout_cap_s: int = 3600
) -> VerificationResult:
    result = VerificationResult()
    timeout = min(task.verifier_timeout_s, timeout_cap_s)
    try:
        if golden:
            if "solve.sh" not in {file.path for file in task.solution_files}:
                raise ValueError("golden mode requires solution/solve.sh")
            await upload_files(sandbox, task.solution_files, "/solution", timeout)
            solution = await sandbox.exec(
                "bash /solution/solve.sh",
                cwd=task.workdir,
                timeout_s=min(task.solution_timeout_s, timeout_cap_s),
            )
            result.solution_output = output_text(solution)
            require_success(solution, "golden solution")
        await upload_files(sandbox, task.test_files, "/tests", timeout)
        # A reward from the image, agent, or previous execution must not survive into this run.
        prepared = await sandbox.exec(
            "test ! -L /logs && mkdir -p /logs && rm -rf -- /logs/verifier && mkdir -p /logs/verifier",
            timeout_s=timeout,
        )
        require_success(prepared, "reward directory preparation")
        execution = await sandbox.exec("bash /tests/test.sh", cwd=task.workdir, timeout_s=timeout)
        result.test_output = output_text(execution)
        result.exit_code = execution.return_code
        if execution.error_type:
            raise RuntimeError(f"verifier execution error: {execution.error_type}")
        with TemporaryDirectory(prefix="swe-reward-") as directory:
            reward_path = Path(directory) / "reward.txt"
            await sandbox.download("/logs/verifier/reward.txt", reward_path)
            reward_text = await asyncio.to_thread(reward_path.read_text, encoding="utf-8")
        reward = float(reward_text.strip())
        if not math.isfinite(reward) or reward not in (0.0, 1.0):
            raise ValueError("grader must write a finite binary reward")
        if reward == 1.0 and execution.return_code != 0:
            raise ValueError("passing reward conflicts with nonzero verifier exit status")
        result.reward = reward
        result.evaluation_completed = True
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result
