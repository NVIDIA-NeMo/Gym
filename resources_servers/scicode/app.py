# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""SciCode resources server.

Runs the agent's accumulated per-sub-step Python solutions against each sub-step's test cases
(targets loaded from test_data.h5) and returns a binary reward: 1.0 iff every sub-step passes.
Per-sub-step counts are also returned so sub-step accuracy can be computed downstream.

Each sub-step is executed in a subprocess in this server's own process (instead of a
Docker sandbox), so the subprocess inherits this server's interpreter and dependencies.
"""

import asyncio
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ConfigDict
from scicode_integration.runner import build_test_program, run_substep, sanitize_test

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


# Agent sentinel for a sub-step it could not generate (ran out of context); always fails.
_OUT_OF_CONTEXT = "_ran_out_of_context_"


class ScicodeResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int = 20
    # Per-sub-step execution timeout
    timeout_secs: float = 30.0
    # Local path to SciCode's test_data.h5 (staged manually)
    test_data_fpath: Optional[str] = None
    # Opt-in: keep each scored sub-step's execution record (exit code, timeout, duration, exception class
    # and bounded stdout/stderr tails) in the verify response as `step_outputs`, aligned with `step_results`.
    # Off by default: the record carries model-generated output text and grows every rollout row.
    record_step_outputs: bool = False
    # Last N characters of stdout and of stderr kept per sub-step when record_step_outputs is on.
    output_tail_chars: int = 2000
    # Opt-in on top of record_step_outputs: each record also carries `code`, the accumulated solution the
    # sub-step's tests ran against (the agent's solutions[step_id]); None for a sentinel step that never ran.
    # Off by default: the code is model-generated and grows a row by the size of every scored program.
    record_step_code: bool = False


class ScicodeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    problem_id: str
    sub_steps: List[dict]
    # {"<problem_id>.<step>": accumulated_code} produced by the agent.
    solutions: Optional[Dict[str, str]] = None


class ScicodeVerifyRequest(ScicodeRunRequest, BaseVerifyRequest):
    pass


class ScicodeVerifyResponse(BaseVerifyResponse):
    # Retain the agent's compact accounting records through verification. None
    # distinguishes historical final-step-only rollouts from whole-problem usage.
    token_usage_version: Optional[int] = None
    step_usage: Optional[List[dict]] = None
    # Declared so it survives into the rollout output (identifies the problem); the request's
    # sub_steps/solutions are intentionally not carried through to keep rollout rows small.
    problem_id: str = ""
    step_results: List[bool] = []
    # Per scored sub-step, aligned with step_results, only when record_step_outputs is on (None otherwise):
    # {step_id, step_number, status (executed | out_of_context | empty), passed, returncode, timed_out,
    #  duration_s, exception_class, stdout_tail, stderr_tail} plus `code` when record_step_code is on too.
    step_outputs: Optional[List[dict]] = None
    num_steps_passed: int = 0
    num_steps_total: int = 0
    problem_accuracy: bool = False
    # Per-rollout sub-step pass fraction. As a numeric verify-response field it
    # gets the full generic statistics (mean/std/min/max/median) from the
    # RewardProfiler, which the aggregate-only pooled scalar cannot provide.
    subtask_accuracy: float = 0.0


class ScicodeResourcesServer(SimpleResourcesServer):
    config: ScicodeResourcesServerConfig

    def model_post_init(self, context):
        self._semaphore = asyncio.Semaphore(value=self.config.num_processes)

    def _resolve_test_data(self) -> str:
        if not self.config.test_data_fpath:
            raise RuntimeError(
                "test_data_fpath is not configured. Stage SciCode's test_data.h5 and set "
                "test_data_fpath (see benchmarks/scicode/README.md)."
            )
        path = Path(self.config.test_data_fpath).expanduser()
        # Resolve relative paths against the Gym root, since the server's cwd is its own dir.
        if not path.is_absolute():
            path = PARENT_DIR / path
        if not path.is_file():
            raise RuntimeError(
                f"SciCode test_data.h5 not found at {path}. Download and stage it "
                "(see benchmarks/scicode/README.md) before running."
            )
        return str(path)

    async def verify(self, body: ScicodeVerifyRequest) -> ScicodeVerifyResponse:
        solutions = body.solutions or {}
        record = self.config.record_step_outputs
        record_code = record and self.config.record_step_code
        # Score only sub-steps the agent produced a solution for. Sub-steps absent from solutions
        # (prefilled steps) are excluded from the denominator entirely; out-of-context sentinels
        # are present and counted as failures.
        scored = [i for i in range(len(body.sub_steps)) if f"{body.problem_id}.{i + 1}" in solutions]
        if not scored:
            return ScicodeVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                step_results=[],
                step_outputs=[] if record else None,
                num_steps_passed=0,
                num_steps_total=0,
                problem_accuracy=False,
            )

        h5_path = self._resolve_test_data()
        loop = asyncio.get_running_loop()

        def _step_output(
            i: int, status: str, passed: bool, result: Optional[dict] = None, code: Optional[str] = None
        ) -> dict:
            result = result or {}
            output = {
                "step_id": f"{body.problem_id}.{i + 1}",
                "step_number": body.sub_steps[i]["step_number"],
                "status": status,
                "passed": passed,
                "returncode": result.get("returncode"),
                "timed_out": bool(result.get("timed_out", False)),
                "duration_s": result.get("duration_s"),
                "exception_class": result.get("exception_class"),
                "stdout_tail": result.get("stdout_tail", ""),
                "stderr_tail": result.get("stderr_tail", ""),
            }
            if record_code:
                output["code"] = code
            return output

        async def _run_substep(i: int) -> tuple:
            sub_step = body.sub_steps[i]
            code = solutions[f"{body.problem_id}.{i + 1}"]
            if not code or code == _OUT_OF_CONTEXT:
                status = "out_of_context" if code == _OUT_OF_CONTEXT else "empty"
                return False, _step_output(i, status, False)
            sanitized = [sanitize_test(tc) for tc in sub_step["test_cases"]]
            program = build_test_program(code, h5_path, sub_step["step_number"], sanitized)
            if record:
                call = partial(
                    run_substep, program, self.config.timeout_secs, output_tail_chars=self.config.output_tail_chars
                )
            else:
                call = partial(run_substep, program, self.config.timeout_secs)
            async with self._semaphore:
                result = await loop.run_in_executor(None, call)
            passed = bool(result["passed"])
            return passed, _step_output(i, "executed", passed, result, code=code)

        outcomes = await asyncio.gather(*[_run_substep(i) for i in scored])
        step_results = [passed for passed, _ in outcomes]
        num_passed = sum(step_results)
        all_passed = num_passed == len(scored)

        return ScicodeVerifyResponse(
            **body.model_dump(),
            reward=1.0 if all_passed else 0.0,
            step_results=step_results,
            step_outputs=[output for _, output in outcomes] if record else None,
            num_steps_passed=num_passed,
            num_steps_total=len(scored),
            problem_accuracy=all_passed,
            subtask_accuracy=num_passed / len(scored),
        )


if __name__ == "__main__":
    ScicodeResourcesServer.run_webserver()
