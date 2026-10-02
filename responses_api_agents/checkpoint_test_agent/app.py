# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Agent that holds one session at a deterministic turn boundary for checkpoint tests.

With ``NEMO_GYM_TEST_HOLD_FIRST_MUTATED_BOUNDARY=1``, the first session that reaches a boundary after its
first tool call waits there until a real checkpoint prepare asks it to park, parks through the production
path, and then stays parked until the process is killed. A crash/restore test therefore always finds a
published checkpoint that holds a turn whose resources mutation has already been applied, independent of
how fast the policy model is.
"""

import asyncio
import os
from typing import Any, Callable

from fastapi import FastAPI

from nemo_gym._checkpoint.agent import AgentSessionHooks, AgentSessionParticipant, _Session
from nemo_gym._checkpoint.control import install_participant
from nemo_gym._checkpoint.settings import checkpoint_settings
from responses_api_agents.simple_agent.app import SimpleAgent


HOLD_FIRST_MUTATED_BOUNDARY_ENV = "NEMO_GYM_TEST_HOLD_FIRST_MUTATED_BOUNDARY"


def _after_first_tool_call(snapshot: Callable[[], dict[str, Any]]) -> bool:
    """Whether a Simple Agent loop boundary follows at least one executed tool call."""
    state = snapshot()
    return state["step"] >= 1 and not state["pending_tools"]


class CheckpointTestParticipant(AgentSessionParticipant):
    """Hold the first session that reaches a boundary after its first tool call."""

    def __init__(self, hooks: AgentSessionHooks) -> None:
        super().__init__(hooks)
        self._hold_first_mutated_boundary = os.environ.get(HOLD_FIRST_MUTATED_BOUNDARY_ENV) == "1"

    async def at_boundary(self, session: _Session, snapshot: Callable[[], dict[str, Any]]) -> None:
        if not self._hold_first_mutated_boundary or not _after_first_tool_call(snapshot):
            await super().at_boundary(session, snapshot)
            return
        self._hold_first_mutated_boundary = False

        # Wait until a real checkpoint prepare asks this session to park.
        while not session.park_requested:
            self._require_live(session)
            await self.wait_changed(1.0)

        # Record the boundary and park through the production implementation.
        await super().at_boundary(session, snapshot)

        # Stay parked at this boundary until the test kills the process, so every later checkpoint still
        # exports it and the training step that needs this rollout never completes.
        session.state = "at_boundary"
        await self.notify()
        await asyncio.Event().wait()


class CheckpointTestAgent(SimpleAgent):
    def setup_agent_checkpoint(self, app: FastAPI) -> None:
        settings = checkpoint_settings(getattr(self.server_client, "global_config_dict", None))
        if settings is None:
            return
        if (self.config.num_workers or 1) != 1:
            raise ValueError("agent checkpointing requires num_workers=1: sessions live in one process")
        self._checkpoint_participant = CheckpointTestParticipant(self)
        install_participant(
            app,
            self._checkpoint_participant,
            auth_token=settings.control_auth_token,
            lease_grace_seconds=settings.lease_grace_seconds,
            instance_name=self.config.name,
        )

    ray_enabled = False


if __name__ == "__main__":
    CheckpointTestAgent.run_webserver()
