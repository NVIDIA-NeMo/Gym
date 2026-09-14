# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor 0.23 lifecycle bridge; evaluated harnesses live in the Gym agent worker.

The adapter uses Harbor's public custom-agent extension, not private phase calls.
Its setup/run methods only rendezvous with the resources server. Harbor retains
its official artifact collection, separate verifier, timeout and cleanup logic.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.trial.errors import AgentTimeoutError

from nemo_gym.sandbox.handoff import AgentTermination


@dataclass
class ExternalEpisode:
    prepared: asyncio.Event = field(default_factory=asyncio.Event)
    setup_complete: asyncio.Event = field(default_factory=asyncio.Event)
    running: asyncio.Event = field(default_factory=asyncio.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    environment: object = None
    termination: AgentTermination | None = None
    started_at: float | None = None
    phase: str = "preparing"
    on_phase_change: Callable[[], None] | None = None

    def set_phase(self, phase):
        self.phase = phase
        if self.on_phase_change:
            self.on_phase_change()


class ExternalAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "gym-external"

    def version(self) -> str:
        return "1"

    async def setup(self, environment):
        self.episode.environment = environment
        self.episode.set_phase("ready")
        self.episode.prepared.set()
        await self.episode.setup_complete.wait()

    async def run(self, instruction, environment, context):
        self.episode.set_phase("agent_running")
        self.episode.started_at = monotonic()
        self.episode.running.set()
        try:
            await self.episode.finished.wait()
        except asyncio.CancelledError:
            if self.episode.termination is None:
                self.episode.termination = AgentTermination(
                    reason="timeout", detail="Resources agent deadline reached"
                )
            raise
        finally:
            await environment.quiesce_agent(self.session_id)
            self.episode.set_phase("verifying")
        termination = self.episode.termination
        if termination.reason == "timeout":
            raise AgentTimeoutError(termination.detail or "Gym agent timed out")
        if termination.reason in {"nonzero_exit", "cancelled", "infrastructure_error"}:
            # This Harbor exception preserves official grading after agent failure.
            # The public response separately masks infrastructure failures.
            raise NonZeroAgentExitCodeError(termination.detail or termination.reason)
