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

"""blackbox_echo_tool — a tiny MCP tool provider (no task, no grading).

Exists to demonstrate the multi-resources-server pattern: an external harness
can be given tools from SEVERAL resources servers at once. This server only
lends a tool (``echo_upper``) over MCP; it never seeds a task or verifies. When
listed under an agent's ``tool_providers``, it is seeded with ``tool_only=true``
and its tool is namespaced ``mcp__blackbox_echo_tool__echo_upper`` so it cannot
clash with the verifier's tools.
"""

from __future__ import annotations

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    MCPResourcesServer,
    gym_tool,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint


class BlackboxEchoToolConfig(BaseResourcesServerConfig):
    pass


class EchoSeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")


class EchoSeedResponse(BaseSeedSessionResponse):
    model_config = ConfigDict(extra="allow")
    mcp: dict = {}


class EchoVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: dict = {}


class EchoVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: dict = {}


class BlackboxEchoToolResourcesServer(MCPResourcesServer):
    config: BlackboxEchoToolConfig

    async def seed_session(self, body: EchoSeedRequest, request: Request) -> EchoSeedResponse:
        # Tool-only: hand back the per-rollout MCP metadata + signed token; no
        # task is materialized (tool_only=true rides on the request, ignored here).
        return EchoSeedResponse(mcp=self.build_mcp_session_metadata(request).model_dump())

    @gym_tool
    def echo_upper(self, text: str) -> str:
        """Return the given text uppercased."""
        return (text or "").upper()

    async def verify(self, body: EchoVerifyRequest) -> EchoVerifyResponse:
        # A tool-only provider is never the verifier; return a no-op if called.
        return EchoVerifyResponse(**body.model_dump(), reward=0.0, is_resolved=False)


if __name__ == "__main__":
    BlackboxEchoToolResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = BlackboxEchoToolResourcesServer.run_webserver()  # noqa: F401
