# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource sessions for the NOOA stateful counter example."""

from fastapi import Request
from pydantic import PrivateAttr

from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.example_session_state_mgmt.app import StatefulCounterResourcesServer


class NOOAStatefulCounterResourcesServer(StatefulCounterResourcesServer):
    """The counter, whose seed retry must repeat the first seed exactly.

    The base keeps a seed retry idempotent, refuses a seed or close from another episode, refuses reuse for
    session_close_retry_window_seconds after close, and answers a tool call for a session it does not hold with 404.
    This server also refuses a retry whose task data changed.
    """

    ray_enabled = False

    _seeds: dict[str, ResourcesSeedSessionRequest] = PrivateAttr(default_factory=dict)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        previous = self._seeds.get(body.resources_session_id)
        if previous is not None and previous != body:
            raise ResourcesSessionConflictError("Resources session has different task data or identity")
        reply = await super().seed_session(body, request=request)
        self._seeds[body.resources_session_id] = body.model_copy(deep=True)
        return reply

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        reply = await super().close_resources_session(body)
        self._seeds.pop(body.resources_session_id, None)
        return reply


if __name__ == "__main__":
    NOOAStatefulCounterResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NOOAStatefulCounterResourcesServer.run_webserver()
