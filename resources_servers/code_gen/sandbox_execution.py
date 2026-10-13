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

"""Grade a candidate in a fresh provider sandbox instead of on a Ray worker.

Each verification gets its own sandbox: the runner and the unchanged grading harness are
uploaded with the request, ``sandbox_runner.py`` writes a result file, and the sandbox is
stopped. Only the provider-neutral sandbox API is used, so any provider works.

Candidate failures (wrong answer, timeout, crash) come back as normal harness results. A
failure of the sandbox or the runner raises ``SandboxVerificationError`` so the caller can
mask the sample instead of scoring it zero.
"""

import json
import logging
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, StrictBool, StrictInt, field_validator

from nemo_gym.failure_kinds import PROVIDER_UNAVAILABLE, VERIFIER_ERROR
from nemo_gym.sandbox import AsyncSandbox, SandboxProvider, SandboxResources, SandboxSpec
from nemo_gym.sandbox.utils import cpu_cap_env, read_text


LOG = logging.getLogger(__name__)

_CODE_GEN_DIR = Path(__file__).parent
# The runner plus the exact grading code the Ray path runs.
_HARNESS_FILES = (
    "sandbox_runner.py",
    "lcb_integration/__init__.py",
    "lcb_integration/correctness.py",
    "lcb_integration/testing_util.py",
)
DEFAULT_WORK_DIR = "/tmp/code_gen_verify"
# Interpreter start-up, Manager start/stop and result serialization on top of the
# harness's own global deadline.
_EXEC_OVERHEAD_S = 60
# Server-side backstop for a sandbox whose stop() never reaches the provider.
_TTL_MARGIN_S = 300
_LOG_TAIL_BYTES = 2000


class CodeGenSandboxConfig(BaseModel):
    """Sandbox settings for running unit tests; used only when ``sandbox_provider`` is set."""

    image: str | None = None
    python: str = "python3"
    resources: SandboxResources = Field(default_factory=SandboxResources)
    ready_timeout_s: float = 600
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("resources", mode="before")
    @classmethod
    def _parse_resources(cls, value: object) -> object:
        # from_mapping rejects unknown keys, so a typo fails at startup instead of being ignored.
        return SandboxResources.from_mapping(value) if isinstance(value, Mapping) else value


class SandboxVerificationError(Exception):
    """The sandbox or runner failed, so the sample has no valid reward."""

    def __init__(self, failure_kind: str, message: str) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


class _RunnerResult(BaseModel):
    result: list[StrictBool | StrictInt]
    metadata: dict[str, Any] | None = None


def exec_timeout_s(timeout: int, num_tests: int) -> float:
    """The deadline ``check_correctness`` enforces on its child, plus runner overhead."""
    return (timeout + 1) * num_tests + 5 + _EXEC_OVERHEAD_S


async def check_correctness_in_sandbox(
    sample: dict[str, str],
    generation: str,
    timeout: int,
    debug: bool,
    *,
    num_tests: int,
    provider: Mapping[str, Any] | SandboxProvider,
    config: CodeGenSandboxConfig,
    metadata: Mapping[str, str],
    work_dir: str = DEFAULT_WORK_DIR,
) -> tuple[list[bool | int], dict[str, Any] | None]:
    """Sandboxed equivalent of ``check_correctness``; raises ``SandboxVerificationError`` on infra failure."""
    deadline_s = exec_timeout_s(timeout, num_tests)
    files = {f"{work_dir}/{name}": (_CODE_GEN_DIR / name).read_text(encoding="utf-8") for name in _HARNESS_FILES}
    files[f"{work_dir}/request.json"] = json.dumps(
        {"sample": sample, "generation": generation, "timeout": timeout, "debug": debug}
    )
    spec = SandboxSpec(
        image=config.image,
        ttl_s=config.ready_timeout_s + deadline_s + _TTL_MARGIN_S,
        ready_timeout_s=config.ready_timeout_s,
        env=cpu_cap_env(config.resources.cpu) | config.env,
        files=files,
        metadata={**metadata, **config.metadata},
        resources=config.resources,
        provider_options=config.provider_options,
    )

    sandbox = AsyncSandbox(provider)
    try:
        try:
            # Covers allocation, readiness and the file upload.
            await sandbox.start(spec)
        except Exception as e:
            raise SandboxVerificationError(PROVIDER_UNAVAILABLE, f"sandbox start failed: {e!r}") from e

        # The candidate shares the runner's stdout, so keep all output in the sandbox.
        command = (
            f"cd {shlex.quote(work_dir)} && "
            f"{shlex.quote(config.python)} sandbox_runner.py request.json result.json > runner.log 2>&1"
        )
        execution = await sandbox.exec(command, timeout_s=deadline_s)
        # Providers report a timeout either as error_type or as exit code 124; both are nonzero here.
        if execution.error_type or execution.return_code != 0:
            raise SandboxVerificationError(
                VERIFIER_ERROR,
                f"runner failed (exit={execution.return_code}, error_type={execution.error_type}): "
                f"{(execution.stderr or '')[-_LOG_TAIL_BYTES:]}{await _runner_log_tail(sandbox, work_dir)}",
            )

        parsed = _RunnerResult.model_validate_json(await read_text(sandbox, path=f"{work_dir}/result.json"))
        return parsed.result, parsed.metadata
    except SandboxVerificationError:
        raise
    except Exception as e:
        # Request boundary: any other sandbox or result failure invalidates only this sample.
        raise SandboxVerificationError(VERIFIER_ERROR, f"sandbox verification failed: {e!r}") from e
    finally:
        try:
            await sandbox.stop()
        except Exception:
            LOG.warning("Failed to stop code_gen verification sandbox", exc_info=True)


async def _runner_log_tail(sandbox: AsyncSandbox, work_dir: str) -> str:
    try:
        tail = await sandbox.exec(f"tail -c {_LOG_TAIL_BYTES} {shlex.quote(work_dir)}/runner.log", timeout_s=30)
    except Exception:
        return ""
    return f"\nrunner.log tail:\n{tail.stdout}" if tail.stdout else ""
