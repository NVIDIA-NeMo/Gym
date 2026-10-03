# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entrypoint for existing sandboxed OpenCode configurations."""

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_agent.legacy import (
    LegacyOpenCodeAgent as OpenCodeSandboxedAgent,
)
from responses_api_agents.opencode_agent.legacy import (
    LegacyOpenCodeAgentConfig as OpenCodeSandboxedAgentConfig,
)
from responses_api_agents.opencode_agent.legacy import (
    LegacyOpenCodeAgentRunRequest as OpenCodeSandboxedAgentRunRequest,
)
from responses_api_agents.opencode_agent.legacy import (
    LegacyOpenCodeAgentVerifyRequest as OpenCodeSandboxedAgentVerifyRequest,
)
from responses_api_agents.opencode_agent.legacy import (
    LegacyOpenCodeAgentVerifyResponse as OpenCodeSandboxedAgentVerifyResponse,
)


__all__ = [
    "OpenCodeSandboxedAgent",
    "OpenCodeSandboxedAgentConfig",
    "OpenCodeSandboxedAgentRunRequest",
    "OpenCodeSandboxedAgentVerifyRequest",
    "OpenCodeSandboxedAgentVerifyResponse",
]


if is_nemo_gym_fastapi_entrypoint(__name__):
    OpenCodeSandboxedAgent.run_webserver()
