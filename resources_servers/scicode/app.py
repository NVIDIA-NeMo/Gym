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


class ScicodeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    problem_id: str
    sub_steps: List[dict]
    # {"<problem_id>.<step>": accumulated_code} produced by the agent.
    solutions: Optional[Dict[str, str]] = None


class ScicodeVerifyRequest(ScicodeRunRequest, BaseVerifyRequest):
    pass


class ScicodeVerifyResponse(BaseVerifyResponse):
    # Declared so it survives into the rollout output (identifies the problem); the request's
    # sub_steps/solutions are intentionally not carried through to keep rollout rows small.
    problem_id: str = ""
    step_results: List[bool] = []
    num_steps_passed: int = 0
    num_steps_total: int = 0
    problem_accuracy: bool = False


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
        # Score only sub-steps the agent produced a solution for. Sub-steps absent from solutions
        # (prefilled steps) are excluded from the denominator entirely; out-of-context sentinels
        # are present and counted as failures.
        scored = [i for i in range(len(body.sub_steps)) if f"{body.problem_id}.{i + 1}" in solutions]
        if not scored:
            return ScicodeVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                step_results=[],
                num_steps_passed=0,
                num_steps_total=0,
                problem_accuracy=False,
            )

        h5_path = self._resolve_test_data()
        loop = asyncio.get_running_loop()

        async def _run_substep(i: int) -> bool:
            sub_step = body.sub_steps[i]
            code = solutions[f"{body.problem_id}.{i + 1}"]
            if not code or code == _OUT_OF_CONTEXT:
                return False
            sanitized = [sanitize_test(tc) for tc in sub_step["test_cases"]]
            program = build_test_program(code, h5_path, sub_step["step_number"], sanitized)
            async with self._semaphore:
                result = await loop.run_in_executor(None, run_substep, program, self.config.timeout_secs)
            return bool(result["passed"])

        step_results = list(await asyncio.gather(*[_run_substep(i) for i in scored]))
        num_passed = sum(step_results)
        all_passed = num_passed == len(scored)

        return ScicodeVerifyResponse(
            **body.model_dump(),
            reward=1.0 if all_passed else 0.0,
            step_results=step_results,
            num_steps_passed=num_passed,
            num_steps_total=len(scored),
            problem_accuracy=all_passed,
        )


if __name__ == "__main__":
    ScicodeResourcesServer.run_webserver()
