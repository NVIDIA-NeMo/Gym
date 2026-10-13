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

``ccc_eval.py`` and ``sandbox_runner.py`` are uploaded with the request and run by
``nemo_gym.sandbox.verification.run_verification_sandbox``, which raises
``SandboxVerificationError`` on a sandbox or runner failure so the caller can mask the sample.
The problem's grader files and tests are read from ``problem_data_dir``, which must already
exist in the sandbox (baked into the image or mounted), because full-problem test sets reach
hundreds of megabytes. Write it with ``split_problem_metadata.py``.

Submission failures (compile error, wrong answer, time limit) come back as normal results.
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nemo_gym.sandbox import SandboxProvider
from nemo_gym.sandbox.verification import SandboxVerificationConfig, run_verification_sandbox


_CCC_DIR = Path(__file__).parent
_HARNESS_FILES = ("ccc_eval.py", "sandbox_runner.py")
DEFAULT_WORK_DIR = "/tmp/ccc_verify"


class CCCSandboxConfig(SandboxVerificationConfig):
    """Sandbox settings for compiling and running submissions; used only when ``sandbox_provider`` is set."""

    # Read-only per-problem metadata inside the sandbox, laid out by split_problem_metadata.py.
    problem_data_dir: str = "/data/ccc"
    # Whole verification: grader and solution compiles plus every test that runs.
    exec_timeout_s: float = 3600


class _SubtaskResult(BaseModel):
    score: float
    outputs: list[dict[str, Any]]


class _RunnerResult(BaseModel, extra="allow"):
    test_case_results: dict[str, _SubtaskResult]


def _parse_result(raw: str) -> dict[str, Any]:
    # Validate the fields the server scores from, but return the runner's full output.
    _RunnerResult.model_validate_json(raw)
    return json.loads(raw)


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
    return await run_verification_sandbox(
        provider=provider,
        config=config,
        files={name: (_CCC_DIR / name).read_text(encoding="utf-8") for name in _HARNESS_FILES},
        request={
            "entry": entry,
            "problem_file": problem_file(config.problem_data_dir, entry.get("competition_id"), entry["problem_id"]),
            "config": dict(evaluator_config),
        },
        exec_timeout_s=config.exec_timeout_s,
        parse_result=_parse_result,
        metadata=metadata,
        work_dir=work_dir,
    )
