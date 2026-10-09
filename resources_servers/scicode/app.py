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
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
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


class ScicodeGradingInterpreter(BaseModel):
    """One Python interpreter used to grade generated SciCode programs."""

    name: str
    python_executable: str


class ScicodeResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int = 20
    # Per-sub-step execution timeout
    timeout_secs: float = 30.0
    # Local path to SciCode's test_data.h5 (staged manually)
    test_data_fpath: Optional[str] = None
    # Optional checksum for binding problem/test releases. Existing SciCode configs leave this unset.
    test_data_md5: Optional[str] = None
    # Empty preserves the existing behavior of grading with this server's interpreter.
    grading_interpreters: List[ScicodeGradingInterpreter] = Field(default_factory=list)
    # SciCode-Verified sets this to two so a single-interpreter run cannot be mistaken for canonical.
    required_grading_interpreters: int = Field(default=1, ge=1)
    # The canonical OR rule short-circuits after the first pass. Enable this only for diagnostics.
    run_all_grading_interpreters: bool = False


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
    num_steps_passed: int = 0
    num_steps_total: int = 0
    problem_accuracy: bool = False
    # Per-rollout sub-step pass fraction. As a numeric verify-response field it
    # gets the full generic statistics (mean/std/min/max/median) from the
    # RewardProfiler, which the aggregate-only pooled scalar cannot provide.
    subtask_accuracy: float = 0.0
    # Audit fields are aligned with step_results and contain only scored (non-prefilled) steps.
    scored_step_ids: List[str] = Field(default_factory=list)
    step_environment_results: List[Dict[str, bool]] = Field(default_factory=list)
    step_environment_errors: List[Dict[str, str]] = Field(default_factory=list)


class ScicodeResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: ScicodeResourcesServerConfig
    _resolved_test_data_path: Optional[str] = PrivateAttr(default=None)
    _resolved_grading_interpreters: Optional[List[tuple[str, str]]] = PrivateAttr(default=None)

    def model_post_init(self, context):
        self._semaphore = asyncio.Semaphore(value=self.config.num_processes)
        # Fail before accepting rollouts when a benchmark pins its data or requires an explicit
        # multi-environment grading protocol. Legacy SciCode keeps both checks lazy.
        if self.config.test_data_md5:
            self._resolve_test_data()
        if self.config.grading_interpreters or self.config.required_grading_interpreters > 1:
            self._resolve_grading_interpreters()

    def _resolve_test_data(self) -> str:
        if self._resolved_test_data_path is not None:
            return self._resolved_test_data_path
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
        if self.config.test_data_md5:
            digest = hashlib.md5()  # noqa: S324 - release integrity, not cryptographic security
            with path.open("rb") as test_data:
                for chunk in iter(lambda: test_data.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_md5 = digest.hexdigest()
            if actual_md5 != self.config.test_data_md5.lower():
                raise RuntimeError(
                    f"SciCode test data checksum mismatch at {path}: expected "
                    f"{self.config.test_data_md5.lower()}, got {actual_md5}."
                )
        self._resolved_test_data_path = str(path)
        return self._resolved_test_data_path

    def _resolve_grading_interpreters(self) -> List[tuple[str, str]]:
        if self._resolved_grading_interpreters is not None:
            return self._resolved_grading_interpreters

        configured = self.config.grading_interpreters
        if configured:
            resolved = []
            seen_names = set()
            for interpreter in configured:
                if interpreter.name in seen_names:
                    raise RuntimeError(f"Duplicate SciCode grading interpreter name: {interpreter.name!r}.")
                seen_names.add(interpreter.name)

                configured_path = Path(interpreter.python_executable).expanduser()
                if configured_path.is_absolute() or configured_path.parent != Path("."):
                    if not configured_path.is_absolute():
                        configured_path = PARENT_DIR / configured_path
                    # Keep the configured path rather than resolving symlinks. A virtual
                    # environment's ``bin/python`` is normally a symlink; dereferencing it
                    # changes argv[0] to the base interpreter and prevents Python from finding
                    # the venv's pyvenv.cfg and site-packages.
                    executable = os.path.abspath(configured_path)
                else:
                    executable = shutil.which(interpreter.python_executable) or ""
                if not executable or not Path(executable).is_file() or not os.access(executable, os.X_OK):
                    raise RuntimeError(
                        f"SciCode grading interpreter {interpreter.name!r} is not executable: "
                        f"{interpreter.python_executable!r}."
                    )
                resolved.append((interpreter.name, executable))
        else:
            resolved = [("current", sys.executable)]

        if len(resolved) < self.config.required_grading_interpreters:
            raise RuntimeError(
                "SciCode requires at least "
                f"{self.config.required_grading_interpreters} grading interpreters, but "
                f"{len(resolved)} resolved. Configure grading_interpreters explicitly."
            )
        self._resolved_grading_interpreters = resolved
        return self._resolved_grading_interpreters

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
        grading_interpreters = self._resolve_grading_interpreters()
        loop = asyncio.get_running_loop()

        async def _run_substep(i: int) -> tuple[bool, Dict[str, bool], Dict[str, str]]:
            sub_step = body.sub_steps[i]
            code = solutions[f"{body.problem_id}.{i + 1}"]
            if not code or code == _OUT_OF_CONTEXT:
                return False, {}, {}
            sanitized = [sanitize_test(tc) for tc in sub_step["test_cases"]]
            program = build_test_program(code, h5_path, sub_step["step_number"], sanitized)
            per_environment = {}
            per_environment_errors = {}
            for name, executable in grading_interpreters:
                async with self._semaphore:
                    result = await loop.run_in_executor(
                        None,
                        run_substep,
                        program,
                        self.config.timeout_secs,
                        executable,
                    )
                per_environment[name] = bool(result["passed"])
                if result["error"]:
                    per_environment_errors[name] = result["error"]
                if result.get("infrastructure_error"):
                    raise RuntimeError(
                        "SciCode grading infrastructure failure for "
                        f"problem {body.problem_id}, step {sub_step['step_number']}, "
                        f"environment {name}:\n{result['error']}"
                    )
                if per_environment[name] and not self.config.run_all_grading_interpreters:
                    break
            return any(per_environment.values()), per_environment, per_environment_errors

        evaluated_steps = list(await asyncio.gather(*[_run_substep(i) for i in scored]))
        step_results = [passed for passed, _, _ in evaluated_steps]
        step_environment_results = [per_environment for _, per_environment, _ in evaluated_steps]
        step_environment_errors = [per_environment_errors for _, _, per_environment_errors in evaluated_steps]
        scored_step_ids = [body.sub_steps[i]["step_number"] for i in scored]
        num_passed = sum(step_results)
        all_passed = num_passed == len(scored)

        return ScicodeVerifyResponse(
            **body.model_dump(),
            reward=1.0 if all_passed else 0.0,
            step_results=step_results,
            num_steps_passed=num_passed,
            num_steps_total=len(scored),
            problem_accuracy=all_passed,
            subtask_accuracy=num_passed / len(scored),
            scored_step_ids=scored_step_ids,
            step_environment_results=step_environment_results,
            step_environment_errors=step_environment_errors,
        )


if __name__ == "__main__":
    ScicodeResourcesServer.run_webserver()
