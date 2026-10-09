# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Harbor agent for tests, loaded through AgentConfig.import_path like any custom Harbor agent."""

import asyncio
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Trajectory


class FakeHarborAgent(BaseAgent):
    SUPPORTS_ATIF = True
    instances: list["FakeHarborAgent"] = []

    def __init__(self, logs_dir: Path, model_name: str | None = None, logger=None, **kwargs) -> None:
        self.behavior = kwargs.pop("behavior", "finish")
        self.api_base = kwargs.pop("api_base", None)
        super().__init__(logs_dir, model_name, logger, **kwargs)
        self.setup_calls = 0
        self.runs: list[tuple[str, str | int | None]] = []
        FakeHarborAgent.instances.append(self)

    @staticmethod
    def name() -> str:
        return "fake-harbor-agent"

    def version(self) -> str | None:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        self.setup_calls += 1
        await environment.exec("echo setup")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self.runs.append((instruction, environment.default_user))
        await environment.exec("echo run")
        if self.behavior == "hang":
            await asyncio.sleep(3600)
        trajectory = Trajectory.model_validate(
            {
                "agent": {"name": self.name(), "version": "1"},
                "steps": [
                    {"step_id": 1, "source": "user", "message": instruction},
                    {"step_id": 2, "source": "agent", "message": "done"},
                ],
            }
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "trajectory.json").write_text(trajectory.model_dump_json())
        context.n_input_tokens = 10
        context.n_output_tokens = 5
