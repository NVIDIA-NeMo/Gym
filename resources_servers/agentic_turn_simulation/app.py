# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-turn simulation of agent-harness conversations: the policy plays one agent turn against simulated tools.

A row is a recorded conversation up to a user message. `/seed_session` stores its history and replays it into a model
of the agent's files (`workspace`), `/harness` tells the agent the row's stop tools and tool-call protocol,
`/simulate_tool_call` answers each call in code where the result is knowable and otherwise with a simulator LLM, and
`/verify` scores the turn from the request alone: reward 1 when it passes the code gates and the judges' combined
`correct_ladder` is "yes", else 0. A turn cut off by the wall-clock cap, a row too long for the policy's context, a
turn that got a failed simulation, and one no judge could score are masked instead. The row's harness (`harnesses`)
supplies everything that depends on the agent harness it was recorded in. Optionally, `obstacles` fails one web call
of the turn on purpose, recoverably.

The turn itself is set up, simulated and scored in `turns`, which the ``agent_arena`` environment server runs too.
This server serves it to the legacy ``/run`` relay and re-scores saved rows (``gym eval reverify``).
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from fastapi import FastAPI, HTTPException, Request
from pydantic import PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.agentic_turn_simulation.schemas import (
    AgenticTurnSimulationRunRequest,
    AgenticTurnSimulationSettings,
    AgenticTurnSimulationVerifyRequest,
    AgenticTurnSimulationVerifyResponse,
    HarnessResponse,
    SimulateToolCallRequest,
    SimulateToolCallResponse,
)
from resources_servers.agentic_turn_simulation.turns import TurnSession, TurnSimulation


class AgenticTurnSimulationResourcesServerConfig(BaseResourcesServerConfig, AgenticTurnSimulationSettings):
    # /verify recomputes the reward from the request alone (only `simulated_calls` reads the session).
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # Sessions that never reach /verify (e.g. a crashed rollout) are dropped after this long.
    session_ttl_seconds: float = 6 * 3600


class AgenticTurnSimulationResourcesServer(SimpleResourcesServer):
    config: AgenticTurnSimulationResourcesServerConfig

    _sessions: dict[str, TurnSession] = PrivateAttr(default_factory=dict)
    _turns: TurnSimulation = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._turns = TurnSimulation(self.config, self.server_client)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/harness")(self.describe_harness)
        app.post("/simulate_tool_call")(self.simulate_tool_call)
        return app

    # --- session --------------------------------------------------------------------------------------------

    def _evict_stale_sessions(self) -> None:
        cutoff = time.monotonic() - self.config.session_ttl_seconds
        for sid in [s for s, state in self._sessions.items() if state.created < cutoff]:
            del self._sessions[sid]

    async def seed_session(self, request: Request, body: AgenticTurnSimulationRunRequest) -> BaseSeedSessionResponse:
        self._evict_stale_sessions()
        self._sessions[request.session[SESSION_ID_KEY]] = self._turns.new_session(body)
        return BaseSeedSessionResponse()

    async def describe_harness(self, request: Request) -> HarnessResponse:
        state = self._sessions.get(request.session.get(SESSION_ID_KEY))
        if state is None:
            raise HTTPException(
                status_code=404, detail="No seeded session for this request: call /seed_session first."
            )
        return self._turns.describe_harness(state)

    # --- tool simulation ------------------------------------------------------------------------------------

    async def simulate_tool_call(self, request: Request, body: SimulateToolCallRequest) -> SimulateToolCallResponse:
        return await self._turns.simulate_tool_call(self._sessions.get(request.session.get(SESSION_ID_KEY)), body)

    # --- verify ---------------------------------------------------------------------------------------------

    async def verify(
        self, request: Request, body: AgenticTurnSimulationVerifyRequest
    ) -> AgenticTurnSimulationVerifyResponse:
        return await self._turns.verify(body, self._sessions.pop(request.session.get(SESSION_ID_KEY), None))

    # --- aggregate metrics ----------------------------------------------------------------------------------

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self._turns.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return self._turns.get_key_metrics(agent_metrics)


if __name__ == "__main__":
    AgenticTurnSimulationResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = AgenticTurnSimulationResourcesServer.run_webserver()  # noqa: F401
