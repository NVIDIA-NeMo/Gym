# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grade a candidate in a sandbox: the one lent to ``/verify``, or a fresh one.

This module runs in the Gym server. Each verification uploads ``sandbox/sandbox_runner.py``, which
runs only in the sandbox, and the unchanged grading harness from ``lcb_integration/``, the same
grading code used for every candidate; nothing else from this server is uploaded. The upload and run go through
``run_verification_in_sandbox`` for a lent sandbox and ``run_verification_sandbox`` for a fresh one; both raise
``SandboxVerificationError`` on a sandbox or runner failure so the caller can mask the sample.
Candidate failures (wrong answer, timeout, crash) come back as normal harness results.
"""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, StrictBool, StrictInt

from nemo_gym.sandbox import AsyncSandbox, SandboxProvider, cpu_cap_env
from nemo_gym.verification import (
    RUNNER_FILE,
    SandboxVerificationConfig,
    run_verification_in_sandbox,
    run_verification_sandbox,
)


_CODE_GEN_DIR = Path(__file__).parent
_RUNNER = _CODE_GEN_DIR / "sandbox" / RUNNER_FILE
# The grading harnesses, uploaded under the same relative paths so the runner can import them.
_HARNESS_FILES = (
    "lcb_integration/__init__.py",
    "lcb_integration/correctness.py",
    "lcb_integration/testing_util.py",
    "assertion_harness/__init__.py",
    "assertion_harness/check.py",
    "sandbox_limits.py",
)
DEFAULT_WORK_DIR = "/tmp/code_gen_verify"
# Interpreter start-up, Manager start/stop and result serialization on top of the
# harness's own global deadline.
_EXEC_OVERHEAD_S = 60
# Sandbox memory on top of a task's own limit, for the runner, the harness and the tests.
_SANDBOX_MEMORY_HEADROOM_MIB = 2048


class _RunnerResult(BaseModel):
    result: list[StrictBool | StrictInt]
    metadata: dict[str, Any] | None = None


def exec_timeout_s(timeout: float, num_tests: int, *, deadline_s: float | None = None) -> float:
    """The deadline ``check_correctness`` enforces on its child (its own, or *deadline_s*), plus runner overhead."""
    grading_deadline_s = deadline_s if deadline_s is not None else (timeout + 1) * num_tests + 5
    return grading_deadline_s + _EXEC_OVERHEAD_S


def task_deadline_s(time_limit_s: float, num_tests: int) -> float:
    """The grading deadline for a task's own time limit: compilation, then every test case."""
    return time_limit_s + (time_limit_s + 1) * num_tests + 5


def _with_memory(config: SandboxVerificationConfig, memory_limit_mb: int) -> SandboxVerificationConfig:
    needed = memory_limit_mb + _SANDBOX_MEMORY_HEADROOM_MIB
    if (config.resources.memory_mib or 0) >= needed:
        return config
    return config.model_copy(update={"resources": dataclasses.replace(config.resources, memory_mib=needed)})


def _runner_files() -> dict[str, str]:
    return {
        RUNNER_FILE: _RUNNER.read_text(encoding="utf-8"),
        **{name: (_CODE_GEN_DIR / name).read_text(encoding="utf-8") for name in _HARNESS_FILES},
    }


async def check_correctness_in_sandbox(
    sample: dict[str, str],
    generation: str,
    timeout: float,
    debug: bool,
    *,
    num_tests: int,
    config: SandboxVerificationConfig,
    sandbox: AsyncSandbox | None = None,
    provider: Mapping[str, Any] | SandboxProvider | None = None,
    metadata: Mapping[str, str] | None = None,
    work_dir: str = DEFAULT_WORK_DIR,
    memory_limit_mb: int | None = None,
    deadline_s: float | None = None,
) -> tuple[list[bool | int], dict[str, Any] | None]:
    """Sandboxed equivalent of ``check_correctness``; raises ``SandboxVerificationError`` on infra failure.

    Exactly one of ``sandbox`` and ``provider`` is given. ``sandbox`` is one the caller holds, such as the attempt's
    sandbox lent to ``/verify``: the tests run in it with ``config.python``, and it is left running for its owner.
    ``provider`` starts a fresh sandbox from ``config`` for this one grade, with ``metadata``, and stops it after.
    *memory_limit_mb* caps the task's memory; *deadline_s* replaces ``check_correctness``'s own deadline.
    """
    request: dict[str, Any] = {"sample": sample, "generation": generation, "timeout": timeout, "debug": debug}
    if memory_limit_mb is not None:
        request["memory_limit_mb"] = memory_limit_mb
        config = _with_memory(config, memory_limit_mb)
    if deadline_s is not None:
        request["deadline_s"] = deadline_s
    return await _grade(
        request,
        exec_timeout=exec_timeout_s(timeout, num_tests, deadline_s=deadline_s),
        config=config,
        sandbox=sandbox,
        provider=provider,
        metadata=metadata,
        work_dir=work_dir,
    )


async def check_assertions_in_sandbox(
    generation: str,
    tests: list[str],
    *,
    time_limit_s: float,
    memory_limit_mb: int,
    config: SandboxVerificationConfig,
    sandbox: AsyncSandbox | None = None,
    provider: Mapping[str, Any] | SandboxProvider | None = None,
    metadata: Mapping[str, str] | None = None,
    work_dir: str = DEFAULT_WORK_DIR,
) -> tuple[list[bool | int], dict[str, Any] | None]:
    """Sandboxed equivalent of ``check_assertions``; raises ``SandboxVerificationError`` on infra failure.

    ``sandbox``, ``provider`` and ``metadata`` are as for ``check_correctness_in_sandbox``.
    """
    # One time limit covers the candidate and all its tests; the 5 s slack matches check_correctness's.
    deadline_s = time_limit_s + 5
    return await _grade(
        {
            "format": "assertions",
            "generation": generation,
            "tests": tests,
            "time_limit_s": time_limit_s,
            "memory_limit_mb": memory_limit_mb,
            "deadline_s": deadline_s,
        },
        exec_timeout=exec_timeout_s(time_limit_s, len(tests), deadline_s=deadline_s),
        config=_with_memory(config, memory_limit_mb),
        sandbox=sandbox,
        provider=provider,
        metadata=metadata,
        work_dir=work_dir,
    )


async def _grade(
    request: dict[str, Any],
    *,
    exec_timeout: float,
    config: SandboxVerificationConfig,
    sandbox: AsyncSandbox | None,
    provider: Mapping[str, Any] | SandboxProvider | None,
    metadata: Mapping[str, str] | None,
    work_dir: str,
) -> tuple[list[bool | int], dict[str, Any] | None]:
    """Run the grading harness on ``request`` in the lent ``sandbox``, or in a fresh one from ``provider``."""
    if (sandbox is None) == (provider is None):
        raise ValueError("pass exactly one of sandbox and provider")
    if sandbox is not None:
        # A lent sandbox has the resources its owner started it with; config.resources only caps the test threads.
        parsed = await run_verification_in_sandbox(
            sandbox,
            files=_runner_files(),
            request=request,
            exec_timeout_s=exec_timeout,
            parse_result=_RunnerResult.model_validate_json,
            work_dir=work_dir,
            python=config.python,
            # The same thread caps a fresh sandbox of config.resources gets.
            env=cpu_cap_env(config.resources.cpu) | config.env,
        )
    else:
        parsed = await run_verification_sandbox(
            provider=provider,
            config=config,
            files=_runner_files(),
            request=request,
            exec_timeout_s=exec_timeout,
            parse_result=_RunnerResult.model_validate_json,
            metadata=metadata or {},
            work_dir=work_dir,
        )
    return parsed.result, parsed.metadata
