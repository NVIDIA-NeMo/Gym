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
"""Shared MCP metadata, session-token helpers and resource seed replies."""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

from nemo_gym.sandbox.access import SandboxAccess


if TYPE_CHECKING:
    from itsdangerous import URLSafeSerializer


NEMO_GYM_MCP_SESSION_TOKEN_HEADER = "X-NeMo-Gym-Session-Token"


NEMO_GYM_MCP_METADATA_KEY = "mcp"


# The MCP session token claim that names the attempt whose seed minted the token, by its capture key.
# Only a server with checkpointing on adds it.
MCP_TOKEN_EPISODE_CLAIM = "episode"


# Salt namespacing the signed MCP session token, so it can't be confused with another signer
# that happens to share the same session-middleware secret.
_MCP_TOKEN_SALT = "nemo-gym-mcp-session-token"


def mcp_session_token_serializer(secret: str) -> "URLSafeSerializer":
    """The signer of Gym MCP session tokens; build it once per server."""
    from itsdangerous import URLSafeSerializer

    return URLSafeSerializer(secret, salt=_MCP_TOKEN_SALT)


def mcp_session_token_payload(serializer: "URLSafeSerializer", token: str) -> Any:
    """The payload of a Gym MCP session token whose signature verifies, or ``None`` if it does not."""
    from itsdangerous import BadSignature

    try:
        return serializer.loads(token)
    except BadSignature:
        return None


# Tool names that would collide with the resources server's own endpoints if advertised over MCP.
RESERVED_MCP_TOOL_NAMES = frozenset({"verify", "seed_session", "close_session", "aggregate_metrics", "mcp"})


class MCPServerMetadata(BaseModel):
    """Metadata returned from /seed_session for per-rollout Gym MCP access."""

    server_name: str
    url_path: str = "/mcp"
    transport: str = "http"
    headers: dict[str, str]


class ResourcesSeedSessionResponse(BaseModel):
    """Confirm resources state and return optional agent access and an initial observation."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    resources_tools: MCPServerMetadata | None = None
    sandbox_access: SandboxAccess | None = None
    # The single_agent_turn Environment Server appends this to the agent's input as one user message.
    observation: str | None = None

    # A return annotation would replace the model's serialization schema with that container schema.
    @model_serializer(mode="wrap")
    def _serialize_observation(self, handler: SerializerFunctionWrapHandler):
        """Preserve the seed wire shape when the server supplies no observation field."""
        result = handler(self)
        if "observation" not in self.model_fields_set:
            result.pop("observation", None)
        return result
