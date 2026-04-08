# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Resources server for the customer service user model's tools.

Provides lookup_order, check_account, get_policy endpoints.
The user model (simulated customer) calls these via simple_agent.
Tool state is initialized per session from the JSONL data passed through seed_session.
"""

import json
from typing import Any, Dict

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


class UserToolsSeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class CustomerServiceUserToolsServer(SimpleResourcesServer):
    config: BaseResourcesServerConfig
    session_state: Dict[str, Any] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/lookup_order")(self.lookup_order)
        app.post("/check_account")(self.check_account)
        app.post("/get_policy")(self.get_policy)
        return app

    async def seed_session(self, request: Request, body: UserToolsSeedRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.session_state[session_id] = body.model_extra or {}
        return BaseSeedSessionResponse()

    async def lookup_order(self, body: ToolRequest, request: Request) -> ToolResponse:
        state = self.session_state.get(request.session[SESSION_ID_KEY], {})
        order = state.get("order", {})
        order_id = body.model_dump().get("order_id", "")
        if order_id == order.get("order_id"):
            return ToolResponse(**{"output": json.dumps(order)})
        return ToolResponse(**{"output": json.dumps({"error": "Order not found"})})

    async def check_account(self, body: ToolRequest, request: Request) -> ToolResponse:
        state = self.session_state.get(request.session[SESSION_ID_KEY], {})
        return ToolResponse(**{"output": json.dumps(state.get("customer", {}))})

    async def get_policy(self, body: ToolRequest, request: Request) -> ToolResponse:
        state = self.session_state.get(request.session[SESSION_ID_KEY], {})
        policies = state.get("policies", {})
        policy_type = body.model_dump().get("policy_type", "")
        return ToolResponse(**{"output": policies.get(policy_type, "Policy not found.")})

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


if __name__ == "__main__":
    CustomerServiceUserToolsServer.run_webserver()
