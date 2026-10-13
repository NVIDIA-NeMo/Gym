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

"""Grade a submission in a fresh provider sandbox instead of the NeMo Skills sandbox.

Each verification gets its own sandbox. ``ccc_eval.py`` and ``sandbox_runner.py`` are
uploaded with the request; the problem's grader files and tests are read from
``problem_data_dir``, which must already exist in the sandbox (baked into the image or
mounted), because full-problem test sets reach hundreds of megabytes. Write it with
``split_problem_metadata.py``.

Submission failures (compile error, wrong answer, time limit) come back as normal
results. A failure of the sandbox or the runner raises ``SandboxVerificationError`` so the
caller can mask the sample instead of scoring it zero.
"""

import json
import logging
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from nemo_gym.failure_kinds import PROVIDER_UNAVAILABLE, VERIFIER_ERROR
from nemo_gym.sandbox import AsyncSandbox, SandboxProvider, SandboxResources, SandboxSpec
from nemo_gym.sandbox.utils import cpu_cap_env, read_text


LOG = logging.getLogger(__name__)

_CCC_DIR = Path(__file__).parent
_HARNESS_FILES = ("ccc_eval.py", "sandbox_runner.py")
DEFAULT_WORK_DIR = "/tmp/ccc_verify"
# Server-side backstop for a sandbox whose stop() never reaches the provider.
_TTL_MARGIN_S = 300
_LOG_TAIL_BYTES = 2000


class CCCSandboxConfig(BaseModel):
    """Sandbox settings for compiling and running submissions; used only when ``sandbox_provider`` is set."""

    image: str | None = None
    python: str = "python3"
    resources: SandboxResources = Field(default_factory=SandboxResources)
    # Read-only per-problem metadata inside the sandbox, laid out by split_problem_metadata.py.
    problem_data_dir: str = "/data/ccc"
    ready_timeout_s: float = 600
    # Whole verification: grader and solution compiles plus every test that runs.
    exec_timeout_s: float = 3600
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


class _SubtaskResult(BaseModel):
    score: float
    outputs: list[dict[str, Any]]


class _RunnerResult(BaseModel, extra="allow"):
    test_case_results: dict[str, _SubtaskResult]


def problem_file(problem_data_dir: str, competition_id: str | None, problem_id: str) -> str:
    """Where ``split_problem_metadata.py`` writes, and the runner reads, one problem's metadata."""

    def safe(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)

    return f"{problem_data_dir}/{safe(competition_id or '_')}/{safe(problem_id)}.jsonl"


async def evaluate_in_sandbox(
    entry: dict[str, Any],
    *,
    evaluator_config: Mapping[str, Any],
    provider: Mapping[str, Any] | SandboxProvider,
    config: CCCSandboxConfig,
    metadata: Mapping[str, str],
    work_dir: str = DEFAULT_WORK_DIR,
) -> dict[str, Any]:
    """Sandboxed equivalent of ``CCCEvaluator.eval_single``; raises ``SandboxVerificationError`` on infra failure."""
    request = {
        "entry": entry,
        "problem_file": problem_file(config.problem_data_dir, entry.get("competition_id"), entry["problem_id"]),
        "config": dict(evaluator_config),
    }
    files = {f"{work_dir}/{name}": (_CCC_DIR / name).read_text(encoding="utf-8") for name in _HARNESS_FILES}
    files[f"{work_dir}/request.json"] = json.dumps(request)
    spec = SandboxSpec(
        image=config.image,
        ttl_s=config.ready_timeout_s + config.exec_timeout_s + _TTL_MARGIN_S,
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

        command = (
            f"cd {shlex.quote(work_dir)} && "
            f"{shlex.quote(config.python)} sandbox_runner.py request.json result.json > runner.log 2>&1"
        )
        execution = await sandbox.exec(command, timeout_s=config.exec_timeout_s)
        # Providers report a timeout either as error_type or as exit code 124; both are nonzero here.
        if execution.error_type or execution.return_code != 0:
            raise SandboxVerificationError(
                VERIFIER_ERROR,
                f"runner failed (exit={execution.return_code}, error_type={execution.error_type}): "
                f"{(execution.stderr or '')[-_LOG_TAIL_BYTES:]}{await _runner_log_tail(sandbox, work_dir)}",
            )

        raw = await read_text(sandbox, path=f"{work_dir}/result.json")
        _RunnerResult.model_validate_json(raw)
        return json.loads(raw)
    except SandboxVerificationError:
        raise
    except Exception as e:
        # Request boundary: any other sandbox or result failure invalidates only this sample.
        raise SandboxVerificationError(VERIFIER_ERROR, f"sandbox verification failed: {e!r}") from e
    finally:
        try:
            await sandbox.stop()
        except Exception:
            LOG.warning("Failed to stop competitive-coding verification sandbox", exc_info=True)


async def _runner_log_tail(sandbox: AsyncSandbox, work_dir: str) -> str:
    try:
        tail = await sandbox.exec(f"tail -c {_LOG_TAIL_BYTES} {shlex.quote(work_dir)}/runner.log", timeout_s=30)
    except Exception:
        return ""
    return f"\nrunner.log tail:\n{tail.stdout}" if tail.stdout else ""
