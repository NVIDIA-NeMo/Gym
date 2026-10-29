# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The NOOA counter resources server, with a seed failure to inject and a report of what each worker holds.

It also takes untyped seeds, as an agent's own ``/run`` sends them, which never read the session,
and keeps a second counter per cookie session for them, created by the first call that reads the session.

Only the session routing stress test runs it.
Its directory is on that test's PYTHONPATH, under the ``resources_servers`` namespace,
so uvicorn can import it as ``resources_servers.session_routing_stress.app`` in each worker.
"""

import os
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.example_session_state_mgmt.nooa_app import NOOAStatefulCounterResourcesServer


class PlainAddRequest(BaseModel):
    count: int


class SessionRoutingStressServer(NOOAStatefulCounterResourcesServer):
    # Every session this worker ever created a counter for, to find a session created on two workers.
    created: set[str] = Field(default_factory=set)
    failed_once: set[str] = Field(default_factory=set)
    # The counters of untyped seeds' sessions, and every such session this worker ever created one for.
    plain: dict[str, int] = Field(default_factory=dict)
    plain_created: set[str] = Field(default_factory=set)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/stress_stats")(self.stress_stats)
        app.post("/plain_add")(self.plain_add)
        app.post("/plain_end")(self.plain_end)
        return app

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | BaseSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        if not isinstance(body, ResourcesSeedSessionRequest):
            # As SimpleResourcesServer's default seed does: nothing is kept, and the session is never read.
            return BaseSeedSessionResponse()
        session_id = body.resources_session_id
        if body.task_data.get("fail_seed_once") and session_id not in self.failed_once:
            # Setup creates the counter and then fails, so only the caller's cleanup close removes it.
            self.failed_once.add(session_id)
            self.session_id_to_counter[session_id] = -1
            self.created.add(session_id)
            raise RuntimeError("injected seed failure")
        response = await super().seed_session(request, body)
        self.created.add(session_id)
        return response

    async def plain_add(self, request: Request, body: PlainAddRequest) -> dict[str, int]:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.plain:
            self.plain[session_id] = 0
            self.plain_created.add(session_id)
        self.plain[session_id] += body.count
        return {"count": self.plain[session_id]}

    async def plain_end(self, request: Request) -> dict[str, int]:
        session_id = request.session[SESSION_ID_KEY]
        count = self.plain.pop(session_id, 0)
        self.checkpoint_session_ended(request)
        return {"count": count}

    async def stress_stats(self, request: Request) -> dict[str, Any]:
        """What this worker holds; a GET without a session is always answered by the worker that receives it."""
        return {
            "pid": os.getpid(),
            "routing_id": getattr(request.app.state, "nemo_gym_routing_id", None),
            "live": sorted(self.session_id_to_counter),
            "created": sorted(self.created),
            "plain_live": sorted(self.plain),
            "plain_created": sorted(self.plain_created),
            "seed_records": len(self.resource_session_seeds),
            "closed_records": len(self._closed_sessions),
        }


if __name__ == "__main__":
    SessionRoutingStressServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SessionRoutingStressServer.run_webserver()
