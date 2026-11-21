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
"""Verified code generation in Lean 4, on Verina tasks.

The model receives a task description and its formal specification (a precondition and a postcondition written in
Lean). It writes the function and a proof that the function meets the specification. The verifier compiles the
solution with Lean, runs the task's hidden test cases, and checks which axioms the proof depends on.
"""

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.verina.lean_file import (
    SOLUTION_REGIONS,
    build_check_file,
    extract_solution,
    forbidden_constructs,
)
from resources_servers.verina.lean_runner import LeanRunner
from resources_servers.verina.setup_lean import ensure_lean_workspace, lean_env, lean_search_path
from resources_servers.verina.task_data import VerinaTest


_SERVER_DIR = Path(__file__).parent


class VerinaResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    lean_workspace: str = Field(
        default=str(_SERVER_DIR / ".lean_workspace"),
        description="Directory holding elan, the Lean toolchain and the Mathlib project; installed on first start.",
    )
    max_concurrency: int = Field(
        default=max(1, (os.cpu_count() or 2) // 2),
        description="Lean processes run at once. Each maps Mathlib (about 5 GB, shared through the page cache).",
    )
    memory_limit_gb: float | None = Field(
        default=8.0,
        description="Heap limit for one Lean process; a check that exceeds it aborts and earns no reward. None: no limit.",
    )
    timeout_s: float = Field(default=120.0, description="Wall-clock limit for one Lean check, including import.")
    tests_only_reward: float = Field(
        default=0.5,
        description="Reward when every test passes but the proof is missing or does not check.",
    )
    import_mathlib: bool = Field(
        default=True, description="Import Mathlib in every check so proofs can use its tactics and lemmas."
    )


class VerinaVerifyRequest(BaseVerifyRequest):
    task_id: str
    lean_template: str
    function_name: str
    theorem_name: str
    tests: list[VerinaTest]


class VerinaVerifyResponse(BaseVerifyResponse):
    task_id: str
    found_sections: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    compiled: bool = False
    timed_out: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    proof_verified: bool = False
    axioms: list[str] | None = None
    lean_errors: list[str] = Field(default_factory=list)


class VerinaVerifier:
    config: VerinaResourcesServerConfig
    _runner: LeanRunner

    async def verify(self, body: VerinaVerifyRequest) -> VerinaVerifyResponse:
        base = body.model_dump(exclude={"lean_template", "tests", "function_name", "theorem_name"})
        solution = extract_solution(body.response.output_text or "")
        found = [name for name in SOLUTION_REGIONS if name in solution]
        tests_total = len(body.tests)
        if "code" not in solution:
            return VerinaVerifyResponse(**base, reward=0.0, found_sections=found, tests_total=tests_total)
        forbidden = forbidden_constructs(solution)
        if forbidden:
            return VerinaVerifyResponse(
                **base, reward=0.0, found_sections=found, forbidden=forbidden, tests_total=tests_total
            )

        check = build_check_file(
            body.lean_template,
            solution,
            function_name=body.function_name,
            theorem_name=body.theorem_name,
            tests=[test.model_dump() for test in body.tests],
            import_mathlib=self.config.import_mathlib,
        )
        outcome = await self._runner.check(check, timeout_s=self.config.timeout_s)
        # Full reward needs both: a proof alone could rest on a specification that admits a wrong answer.
        if outcome.proof_verified and outcome.all_tests_passed:
            reward = 1.0
        elif outcome.all_tests_passed:
            reward = self.config.tests_only_reward
        else:
            reward = 0.0
        return VerinaVerifyResponse(
            **base,
            reward=reward,
            found_sections=found,
            compiled=outcome.compiled,
            timed_out=outcome.timed_out,
            tests_passed=outcome.tests_passed,
            tests_total=outcome.tests_total,
            proof_verified=outcome.proof_verified,
            axioms=outcome.axioms,
            lean_errors=[f"line {error.line}: {error.text}" for error in outcome.errors[:5] + outcome.test_errors[:2]],
        )


def build_runner(config: VerinaResourcesServerConfig) -> LeanRunner:
    """Install the Lean workspace if needed and return a runner for it."""
    workspace = Path(config.lean_workspace)
    if not workspace.is_absolute():
        workspace = _SERVER_DIR.parents[1] / workspace  # relative to the repository root, like the dataset paths
    project = ensure_lean_workspace(workspace)
    return LeanRunner(
        project_dir=project,
        lean_path=lean_search_path(workspace),
        env=lean_env(workspace),
        max_concurrency=config.max_concurrency,
        memory_limit_bytes=None if config.memory_limit_gb is None else int(config.memory_limit_gb * 2**30),
    )


class VerinaResourcesServer(VerinaVerifier, SimpleResourcesServer):
    config: VerinaResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._runner = build_runner(self.config)


def _fixture_verifier() -> VerinaVerifier:
    verifier = VerinaVerifier()
    verifier.config = VerinaResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="app.py", name="verina")
    verifier._runner = build_runner(verifier.config)
    return verifier


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_verifier,
    request_model=VerinaVerifyRequest,
    cases_path=_SERVER_DIR / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    VerinaResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = VerinaResourcesServer.run_webserver()  # noqa: F401
