# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource sessions for the NOOA stateful counter example."""

from typing import Any

from fastapi import HTTPException, Request
from pydantic import Field, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.server_runtime import (
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.example_session_state_mgmt.app import (
    GetCounterValueResponse,
    IncrementCounterRequest,
    IncrementCounterResponse,
    StatefulCounterResourcesServer,
    StatefulCounterSeedSessionRequest,
)


class NOOAStatefulCounterResourcesServer(StatefulCounterResourcesServer):
    """Keep NOOA session retries idempotent and refuse reuse for session_close_retry_window_seconds after close."""

    ray_enabled = False

    resource_session_seeds: dict[str, ResourcesSeedSessionRequest] = Field(default_factory=dict)
    _closed_sessions: ClosedResourcesSessionRecords[ResourcesCloseSessionRequest] = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._closed_sessions = ClosedResourcesSessionRecords(self.config.session_close_retry_window_seconds)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        session_id = body.resources_session_id
        if session_id in self._closed_sessions:
            raise HTTPException(409, "Resources session is closed")
        previous = self.resource_session_seeds.get(session_id)
        if previous is not None and previous != body:
            raise HTTPException(409, "Resources session has different task data or identity")
        initial = StatefulCounterSeedSessionRequest.model_validate(body.task_data)
        self.resource_session_seeds[session_id] = body.model_copy(deep=True)
        request.session[SESSION_ID_KEY] = session_id
        await super().seed_session(request, initial)
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        previous = self.resource_session_seeds.get(body.resources_session_id)
        closed = self._closed_sessions.get(body.resources_session_id)
        if (previous is not None and previous.episode_id != body.episode_id) or (
            closed is not None and closed != body
        ):
            raise HTTPException(409, "Resources session belongs to another episode")
        self.session_id_to_counter.pop(body.resources_session_id, None)
        self.resource_session_seeds.pop(body.resources_session_id, None)
        self._closed_sessions.remember(body.resources_session_id, body.model_copy(deep=True))
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def increment_counter(self, request: Request, body: IncrementCounterRequest) -> IncrementCounterResponse:
        self._require_open_session(request)
        return await super().increment_counter(request, body)

    async def get_counter_value(self, request: Request) -> GetCounterValueResponse:
        self._require_open_session(request)
        return await super().get_counter_value(request)

    def _require_open_session(self, request: Request) -> None:
        # Judged by the counter, not the closed records: those expire,
        # and the parent's tool routes would then recreate a counter for a stale cookie that nothing closes.
        if request.session.get(SESSION_ID_KEY) not in self.session_id_to_counter:
            raise HTTPException(409, "Resources session is not open")


if __name__ == "__main__":
    NOOAStatefulCounterResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NOOAStatefulCounterResourcesServer.run_webserver()
