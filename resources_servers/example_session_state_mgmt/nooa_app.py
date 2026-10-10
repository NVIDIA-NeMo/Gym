# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native resource sessions for the NOOA stateful-counter example."""

from fastapi import HTTPException, Request
from pydantic import Field

from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.example_session_state_mgmt.app import (
    GetCounterValueResponse,
    IncrementCounterRequest,
    IncrementCounterResponse,
    StatefulCounterResourcesServer,
    StatefulCounterSeedSessionRequest,
)


class NOOAStatefulCounterResourcesServer(StatefulCounterResourcesServer):
    """Keep NOOA session retries idempotent and prevent reuse after close."""

    native_sessions: dict[str, ResourcesSeedSessionRequest] = Field(default_factory=dict)
    closed_sessions: dict[str, ResourcesCloseSessionRequest] = Field(default_factory=dict)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        session_id = body.resources_session_id
        if session_id in self.closed_sessions:
            raise HTTPException(409, "Resources session is closed")
        previous = self.native_sessions.get(session_id)
        if previous is not None and previous != body:
            raise HTTPException(409, "Resources session has different task data or identity")
        initial = StatefulCounterSeedSessionRequest.model_validate(body.task_data)
        self.native_sessions[session_id] = body.model_copy(deep=True)
        request.session[SESSION_ID_KEY] = session_id
        await super().seed_session(request, initial)
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        previous = self.native_sessions.get(body.resources_session_id)
        closed = self.closed_sessions.get(body.resources_session_id)
        if (previous is not None and previous.episode_id != body.episode_id) or (
            closed is not None and closed != body
        ):
            raise HTTPException(409, "Resources session belongs to another episode")
        self.session_id_to_counter.pop(body.resources_session_id, None)
        self.native_sessions.pop(body.resources_session_id, None)
        self.closed_sessions[body.resources_session_id] = body.model_copy(deep=True)
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def increment_counter(self, request: Request, body: IncrementCounterRequest) -> IncrementCounterResponse:
        if request.session[SESSION_ID_KEY] in self.closed_sessions:
            raise HTTPException(409, "Resources session is closed")
        return await super().increment_counter(request, body)

    async def get_counter_value(self, request: Request) -> GetCounterValueResponse:
        if request.session[SESSION_ID_KEY] in self.closed_sessions:
            raise HTTPException(409, "Resources session is closed")
        return await super().get_counter_value(request)


if __name__ == "__main__":
    NOOAStatefulCounterResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NOOAStatefulCounterResourcesServer.run_webserver()
