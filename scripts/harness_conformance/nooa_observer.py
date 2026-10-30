# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent native execution witness installed only in the conformance runtime."""

import json
from pathlib import Path
from typing import Any

from nooa import Agent
from nooa.runtime.hooks import InstrumentationHooks, hooks_scope

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.bench_agent_adapter import invoke_bench_agent


class ExecutionWitness(InstrumentationHooks):
    """Observe native code boundaries without reading Gym's canonical projection."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def before_code_execution(self, *, code: str, tool_call_id: str, **_: Any) -> dict:
        return {"id": tool_call_id, "arguments": {"code": code}}

    def after_code_execution(self, *, context: dict, exception: BaseException | None, **_: Any) -> None:
        # This is a native Python execution outcome, not a subprocess exit status.
        # The common witness schema uses zero/nonzero to distinguish success/failure.
        record = {**context, "exit_code": int(exception is not None), "executed": True}
        with self.path.open("a") as output:
            output.write(json.dumps(record) + "\n")


async def invoke_with_witness(agent: Agent, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    """Call the production adapter once with an additional independent observer."""
    # The production adapter resets shell.cwd to the worker task directory.
    # Before that reset BenchAgent may have selected an image default such as /app.
    path = Path.cwd().parent / "native_executions.jsonl"
    with hooks_scope(ExecutionWitness(path)):
        return await invoke_bench_agent(agent, request)
