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
"""What runs inside every Gym server: the resources and environment server bases, request handling, sessions, and partial-rollout checkpoints."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.server_runtime._checkpoint.agent import (
        Activation,
        ActivationOutOfOrderError,
        LegacyRun,
        RestoredAgentSession,
        require_rollout,
    )
    from nemo_gym.server_runtime._checkpoint.control import CheckpointMode
    from nemo_gym.server_runtime._checkpoint.errors import RolloutIdRequiredError
    from nemo_gym.server_runtime._checkpoint.steps import (
        CHECKPOINT_CLOSE_FOLLOWS_HEADER,
        StepType,
        seed_restarts,
        seed_verify_step_type,
    )
    from nemo_gym.server_runtime.base_environment_server import (
        BaseEnvironmentServer,
        BaseEnvironmentServerConfig,
        CleanupContext,
        CleanupHandle,
        HandledEpisodeError,
        dependency_failure,
        is_retryable_dependency_error,
        is_retryable_dependency_status,
    )
    from nemo_gym.server_runtime.base_resources_server import (
        NEMO_GYM_MCP_METADATA_KEY,
        NEMO_GYM_MCP_SESSION_TOKEN_HEADER,
        BaseMultiRewardVerifyResponse,
        BaseResourcesServerConfig,
        BaseRunRequest,
        BaseSeedSessionRequest,
        BaseSeedSessionResponse,
        BaseVerifyRequest,
        BaseVerifyResponse,
        ClosedResourcesSessionRecords,
        ResourcesCloseSessionRequest,
        ResourcesCloseSessionResponse,
        ResourcesSeedRejectedError,
        ResourcesSeedSessionRequest,
        ResourcesSeedSessionResponse,
        ResourcesSessionConflictError,
        ResourcesSessionFence,
        ResourcesSessionLostError,
        ResourcesVerifyRequest,
        ReverifyMode,
        SimpleResourcesServer,
        lost_session_verification,
    )
    from nemo_gym.server_runtime.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
    from nemo_gym.server_runtime.mcp_auto_exposure import MCPTool
    from nemo_gym.server_runtime.request_handling.rollout_context import RolloutContextMiddleware
    from nemo_gym.server_runtime.request_handling.session_routing import SESSION_OWNER_HEADER
    from nemo_gym.server_runtime.single_agent_turn_types import (
        SingleAgentTurnFailure,
        SingleAgentTurnRequest,
        SingleAgentTurnResponse,
        SingleAgentTurnResult,
    )
    from nemo_gym.server_runtime.tool_access import (
        DirectHTTPToolAccess,
        MCPStreamableHTTPConnection,
        MCPToolAccess,
        ToolAccess,
    )
    from nemo_gym.server_runtime.verifier_observability import observe_verifier


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "Activation": "_checkpoint.agent",
    "ActivationOutOfOrderError": "_checkpoint.agent",
    "BaseEnvironmentServer": "base_environment_server",
    "BaseEnvironmentServerConfig": "base_environment_server",
    "BaseEpisodeRequest": "episode_types",
    "BaseEpisodeResponse": "episode_types",
    "BaseMultiRewardVerifyResponse": "base_resources_server",
    "BaseResourcesServerConfig": "base_resources_server",
    "BaseRunRequest": "base_resources_server",
    "BaseSeedSessionRequest": "base_resources_server",
    "BaseSeedSessionResponse": "base_resources_server",
    "BaseVerifyRequest": "base_resources_server",
    "BaseVerifyResponse": "base_resources_server",
    "CHECKPOINT_CLOSE_FOLLOWS_HEADER": "_checkpoint.steps",
    "CheckpointMode": "_checkpoint.control",
    "CleanupContext": "base_environment_server",
    "CleanupHandle": "base_environment_server",
    "ClosedResourcesSessionRecords": "base_resources_server",
    "DirectHTTPToolAccess": "tool_access",
    "EpisodeFailure": "episode_types",
    "HandledEpisodeError": "base_environment_server",
    "LegacyRun": "_checkpoint.agent",
    "MCPStreamableHTTPConnection": "tool_access",
    "MCPTool": "mcp_auto_exposure",
    "MCPToolAccess": "tool_access",
    "NEMO_GYM_MCP_METADATA_KEY": "base_resources_server",
    "NEMO_GYM_MCP_SESSION_TOKEN_HEADER": "base_resources_server",
    "ResourcesCloseSessionRequest": "base_resources_server",
    "ResourcesCloseSessionResponse": "base_resources_server",
    "ResourcesSeedRejectedError": "base_resources_server",
    "ResourcesSeedSessionRequest": "base_resources_server",
    "ResourcesSeedSessionResponse": "base_resources_server",
    "ResourcesSessionConflictError": "base_resources_server",
    "ResourcesSessionFence": "base_resources_server",
    "ResourcesSessionLostError": "base_resources_server",
    "ResourcesVerifyRequest": "base_resources_server",
    "RestoredAgentSession": "_checkpoint.agent",
    "ReverifyMode": "base_resources_server",
    "RolloutContextMiddleware": "request_handling.rollout_context",
    "RolloutIdRequiredError": "_checkpoint.errors",
    "SESSION_OWNER_HEADER": "request_handling.session_routing",
    "SimpleResourcesServer": "base_resources_server",
    "SingleAgentTurnFailure": "single_agent_turn_types",
    "SingleAgentTurnRequest": "single_agent_turn_types",
    "SingleAgentTurnResponse": "single_agent_turn_types",
    "SingleAgentTurnResult": "single_agent_turn_types",
    "StepType": "_checkpoint.steps",
    "ToolAccess": "tool_access",
    "dependency_failure": "base_environment_server",
    "is_retryable_dependency_error": "base_environment_server",
    "is_retryable_dependency_status": "base_environment_server",
    "lost_session_verification": "base_resources_server",
    "observe_verifier": "verifier_observability",
    "require_rollout": "_checkpoint.agent",
    "seed_restarts": "_checkpoint.steps",
    "seed_verify_step_type": "_checkpoint.steps",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
