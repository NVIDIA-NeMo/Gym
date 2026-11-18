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

"""Grade a candidate in a fresh provider sandbox.

This module runs in the Gym server. Each verification uploads ``sandbox/sandbox_runner.py``, which
runs only in the sandbox, and the unchanged grading harness from ``lcb_integration/``, the same
grading code used for every candidate; nothing else from this server is uploaded. The upload and run go through
``nemo_gym.verification.sandbox_verification.run_verification_sandbox``, which raises
``SandboxVerificationError`` on a sandbox or runner failure so the caller can mask the sample.
Candidate failures (wrong answer, timeout, crash) come back as normal harness results.
"""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, StrictBool, StrictInt

from nemo_gym.sandbox import SandboxProvider
from nemo_gym.verification import (
    RUNNER_FILE,
    SandboxVerificationConfig,
    run_verification_sandbox,
)


_CODE_GEN_DIR = Path(__file__).parent
_RUNNER = _CODE_GEN_DIR / "sandbox" / RUNNER_FILE
# The grading harness, uploaded under the same relative paths so the runner can import it.
_HARNESS_FILES = (
    "lcb_integration/__init__.py",
    "lcb_integration/correctness.py",
    "lcb_integration/testing_util.py",
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


async def check_correctness_in_sandbox(
    sample: dict[str, str],
    generation: str,
    timeout: float,
    debug: bool,
    *,
    num_tests: int,
    provider: Mapping[str, Any] | SandboxProvider,
    config: SandboxVerificationConfig,
    metadata: Mapping[str, str],
    work_dir: str = DEFAULT_WORK_DIR,
    memory_limit_mb: int | None = None,
    deadline_s: float | None = None,
) -> tuple[list[bool | int], dict[str, Any] | None]:
    """Sandboxed equivalent of ``check_correctness``; raises ``SandboxVerificationError`` on infra failure.

    *memory_limit_mb* caps the task's memory; *deadline_s* replaces ``check_correctness``'s own deadline.
    """
    request: dict[str, Any] = {"sample": sample, "generation": generation, "timeout": timeout, "debug": debug}
    if memory_limit_mb is not None:
        request["memory_limit_mb"] = memory_limit_mb
        config = _with_memory(config, memory_limit_mb)
    if deadline_s is not None:
        request["deadline_s"] = deadline_s
    parsed = await run_verification_sandbox(
        provider=provider,
        config=config,
        files={
            RUNNER_FILE: _RUNNER.read_text(encoding="utf-8"),
            **{name: (_CODE_GEN_DIR / name).read_text(encoding="utf-8") for name in _HARNESS_FILES},
        },
        request=request,
        exec_timeout_s=exec_timeout_s(timeout, num_tests, deadline_s=deadline_s),
        parse_result=_RunnerResult.model_validate_json,
        metadata=metadata,
        work_dir=work_dir,
    )
    return parsed.result, parsed.metadata
