# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entrypoint for existing sandboxed OpenCode configurations."""

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_agent.legacy import (
    LegacyOpenCodeAgent as OpenCodeSandboxedAgent,
    LegacyOpenCodeAgentConfig as OpenCodeSandboxedAgentConfig,
    LegacyOpenCodeAgentRunRequest as OpenCodeSandboxedAgentRunRequest,
    LegacyOpenCodeAgentVerifyRequest as OpenCodeSandboxedAgentVerifyRequest,
    LegacyOpenCodeAgentVerifyResponse as OpenCodeSandboxedAgentVerifyResponse,
)


if is_nemo_gym_fastapi_entrypoint(__name__):
    OpenCodeSandboxedAgent.run_webserver()
