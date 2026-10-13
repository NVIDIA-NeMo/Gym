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

The runner and the unchanged grading harness are uploaded with the request and run by
``nemo_gym.sandbox.verification.run_verification_sandbox``, which raises
``SandboxVerificationError`` on a sandbox or runner failure so the caller can mask the sample.
Candidate failures (wrong answer, timeout, crash) come back as normal harness results.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, StrictBool, StrictInt

from nemo_gym.sandbox import SandboxProvider
from nemo_gym.sandbox.verification import SandboxVerificationConfig, run_verification_sandbox


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
    config: SandboxVerificationConfig,
    metadata: Mapping[str, str],
    work_dir: str = DEFAULT_WORK_DIR,
) -> tuple[list[bool | int], dict[str, Any] | None]:
    """Sandboxed equivalent of ``check_correctness``; raises ``SandboxVerificationError`` on infra failure."""
    parsed = await run_verification_sandbox(
        provider=provider,
        config=config,
        files={name: (_CODE_GEN_DIR / name).read_text(encoding="utf-8") for name in _HARNESS_FILES},
        request={"sample": sample, "generation": generation, "timeout": timeout, "debug": debug},
        exec_timeout_s=exec_timeout_s(timeout, num_tests),
        parse_result=_RunnerResult.model_validate_json,
        metadata=metadata,
        work_dir=work_dir,
    )
    return parsed.result, parsed.metadata
