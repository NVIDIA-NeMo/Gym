# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic native-result fixtures; these attest classification, not GPU execution."""

from pydantic import BaseModel, JsonValue

from resources_servers.sol_execbench.problem_store import Problem, problem_digest
from resources_servers.sol_execbench.scoring import WorkloadAnchor


class NativeFixtureRequest(BaseModel):
    traces: list[dict[str, JsonValue]]
    return_code: int


class FixtureReward(BaseModel):
    reward: float


def synthetic_problem() -> Problem:
    """Return original tiny metadata with no redistributed corpus content."""
    definition = {"name": "synthetic_identity"}
    workloads = [{"uuid": "synthetic-workload", "axes": {}, "inputs": {}}]
    return Problem(
        task_id="synthetic/identity",
        definition=definition,
        workloads=workloads,
        assets=[],
        problem_digest=problem_digest("synthetic/identity", definition, workloads, []),
    )


class NativeVerifierFixture:
    async def verify(self, body: NativeFixtureRequest) -> FixtureReward:
        from resources_servers.sol_execbench.app import classify_native_result, score_native_result

        result = classify_native_result(
            problem=synthetic_problem(),
            solution_name="synthetic_solution",
            return_code=body.return_code,
            traces=body.traces,
            benchmark_reference=False,
        )
        result = score_native_result(result, {"synthetic-workload": WorkloadAnchor(baseline_ms=0.018, sol_ms=0.01)})
        return FixtureReward(reward=result.sol_score if result.sol_score is not None else 0.0)
