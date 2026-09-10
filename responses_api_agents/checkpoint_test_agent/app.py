# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

from nemo_gym._checkpoint.agent import (
    AgentBoundaryRecord,
    AgentCheckpointParticipant,
    AgentExecution,
    AgentExecutionState,
)
from responses_api_agents.simple_agent.app import SimpleAgent


class CheckpointTestParticipant(AgentCheckpointParticipant):
    """Hold one execution at its first real turn boundary."""

    def __init__(self, instance_name: str | None = None) -> None:
        super().__init__(instance_name)
        self._hold_first_boundary = (
            os.environ.get("NEMO_GYM_TEST_HOLD_FIRST_BOUNDARY") == "1"
        )

    async def commit_boundary(
        self,
        execution: AgentExecution,
        record: AgentBoundaryRecord,
    ) -> None:
        hold = self._hold_first_boundary
        if hold:
            self._hold_first_boundary = False

            # Wait until the real checkpoint prepare operation requests parking.
            async with self._changed:
                while (
                    self._owns(execution)
                    and execution.state == AgentExecutionState.RUNNING
                ):
                    await self._changed.wait()

            self._require_owner(execution)

        # Exercise the normal production boundary and parking implementation.
        await super().commit_boundary(execution, record)

        if hold:
            # Keep phase one alive until the test kills it after selecting
            # the published snapshot.
            await asyncio.Event().wait()


class CheckpointTestAgent(SimpleAgent):
    def checkpoint_participant(self) -> AgentCheckpointParticipant:
        if self._checkpoint_participant is None:
            self._checkpoint_participant = CheckpointTestParticipant(
                self.config.name
            )
        return self._checkpoint_participant


if __name__ == "__main__":
    CheckpointTestAgent.run_webserver()
