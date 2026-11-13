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
        AgentSessionParticipant,
        AgentSessionRecord,
        LegacyRun,
        RestoredAgentSession,
        require_rollout,
    )
    from nemo_gym.server_runtime._checkpoint.control import (
        CheckpointMode,
        CheckpointRequest,
        CommitRequest,
        ParticipantControlPlane,
        RestoreRequest,
        next_attempt,
    )
    from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant, EpisodeRecord, task_digest
    from nemo_gym.server_runtime._checkpoint.errors import RolloutIdRequiredError, StaleAttemptError
    from nemo_gym.server_runtime._checkpoint.generation_cut import (
        GenerationCutInventory,
        GenerationCutPrefixAck,
        GenerationCutReceipt,
    )
    from nemo_gym.server_runtime._checkpoint.resources import ResourcesParticipant
    from nemo_gym.server_runtime._checkpoint.steps import (
        CHECKPOINT_CLOSE_FOLLOWS_HEADER,
        StepType,
        seed_rebinds_tool_accesses,
        seed_restarts,
        seed_verify_step_type,
    )
    from nemo_gym.server_runtime._checkpoint.store import write_participant_state
    from nemo_gym.server_runtime.base_environment_server import (
        BaseEnvironmentServer,
        BaseEnvironmentServerConfig,
        CleanupContext,
        CleanupHandle,
        dependency_failure,
        handled_episode_failure,
        is_handled_episode_failure,
        is_retryable_dependency_error,
        is_retryable_dependency_status,
    )
    from nemo_gym.server_runtime.base_resources_server import (
        EPISODE_STATUS_PATH,
        NEMO_GYM_MCP_METADATA_KEY,
        NEMO_GYM_MCP_SESSION_TOKEN_HEADER,
        BaseMultiRewardVerifyResponse,
        BaseResourcesServerConfig,
        BaseRunRequest,
        BaseSeedSessionRequest,
        BaseSeedSessionResponse,
        BaseVerifyRequest,
        BaseVerifyResponse,
        ResourcesCloseSessionRequest,
        ResourcesCloseSessionResponse,
        ResourcesEpisodeStatusRequest,
        ResourcesEpisodeStatusResponse,
        ResourcesSeedRejectedError,
        ResourcesSeedSessionRequest,
        ResourcesSeedSessionResponse,
        ResourcesSessionConflictError,
        ResourcesSessionFences,
        ResourcesSessionLocks,
        ResourcesSessionLostError,
        ResourcesVerification,
        ResourcesVerifyRequest,
        ReverifyMode,
        SimpleResourcesServer,
        lost_session_verification,
        reject_partial_typed_seed,
    )
    from nemo_gym.server_runtime.base_server import (
        SimpleServer,
        async_rollout_prefix_hook,
        rollout_prefix_hook,
        with_rollout_prefix,
    )
    from nemo_gym.server_runtime.mcp_auto_exposure import (
        TOKEN_HEADER,
        MCPTool,
        harvest_tools,
        install_auto_exposure,
        maybe_auto_expose,
    )
    from nemo_gym.server_runtime.ray_support import entrypoint_ray_enabled
    from nemo_gym.server_runtime.request_handling.model_sampling import ModelSamplingMiddleware
    from nemo_gym.server_runtime.request_handling.rollout_context import RolloutContextMiddleware
    from nemo_gym.server_runtime.request_handling.session_routing import SESSION_OWNER_HEADER, install_session_routing
    from nemo_gym.server_runtime.sandbox_lending import LentSandboxes
    from nemo_gym.server_runtime.single_agent_turn_types import (
        SingleAgentTurnFailure,
        SingleAgentTurnRequest,
        SingleAgentTurnResponse,
        SingleAgentTurnResult,
        SingleAgentTurnTaskInput,
    )
    from nemo_gym.server_runtime.testing.session_conformance import check_resources_session_contract
    from nemo_gym.server_runtime.tool_access import (
        DirectHTTPToolAccess,
        MCPStreamableHTTPConnection,
        MCPToolAccess,
        ToolAccess,
    )
    from nemo_gym.server_runtime.verifier_observability import (
        observe_llm_judge_grading,
        observe_programmatic_grading,
        observe_verifier,
        observe_verifier_sandbox,
    )


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "async_rollout_prefix_hook": "base_server",
    "rollout_prefix_hook": "base_server",
    "with_rollout_prefix": "base_server",
    "Activation": "_checkpoint.agent",
    "ActivationOutOfOrderError": "_checkpoint.agent",
    "AgentSessionParticipant": "_checkpoint.agent",
    "AgentSessionRecord": "_checkpoint.agent",
    "BaseEnvironmentServer": "base_environment_server",
    "BaseEnvironmentServerConfig": "base_environment_server",
    "BaseMultiRewardVerifyResponse": "base_resources_server",
    "BaseResourcesServerConfig": "base_resources_server",
    "BaseRunRequest": "base_resources_server",
    "BaseSeedSessionRequest": "base_resources_server",
    "BaseSeedSessionResponse": "base_resources_server",
    "BaseVerifyRequest": "base_resources_server",
    "BaseVerifyResponse": "base_resources_server",
    "CHECKPOINT_CLOSE_FOLLOWS_HEADER": "_checkpoint.steps",
    "CheckpointMode": "_checkpoint.control",
    "CheckpointRequest": "_checkpoint.control",
    "CleanupContext": "base_environment_server",
    "CleanupHandle": "base_environment_server",
    "CommitRequest": "_checkpoint.control",
    "DirectHTTPToolAccess": "tool_access",
    "EPISODE_STATUS_PATH": "base_resources_server",
    "EnvironmentParticipant": "_checkpoint.environment",
    "EpisodeRecord": "_checkpoint.environment",
    "GenerationCutInventory": "_checkpoint.generation_cut",
    "GenerationCutPrefixAck": "_checkpoint.generation_cut",
    "GenerationCutReceipt": "_checkpoint.generation_cut",
    "LegacyRun": "_checkpoint.agent",
    "MCPStreamableHTTPConnection": "tool_access",
    "MCPTool": "mcp_auto_exposure",
    "LentSandboxes": "sandbox_lending",
    "MCPToolAccess": "tool_access",
    "ModelSamplingMiddleware": "request_handling.model_sampling",
    "NEMO_GYM_MCP_METADATA_KEY": "base_resources_server",
    "NEMO_GYM_MCP_SESSION_TOKEN_HEADER": "base_resources_server",
    "ParticipantControlPlane": "_checkpoint.control",
    "ResourcesCloseSessionRequest": "base_resources_server",
    "ResourcesCloseSessionResponse": "base_resources_server",
    "ResourcesEpisodeStatusRequest": "base_resources_server",
    "ResourcesEpisodeStatusResponse": "base_resources_server",
    "ResourcesParticipant": "_checkpoint.resources",
    "ResourcesSeedRejectedError": "base_resources_server",
    "ResourcesSeedSessionRequest": "base_resources_server",
    "ResourcesSeedSessionResponse": "base_resources_server",
    "ResourcesSessionConflictError": "base_resources_server",
    "ResourcesSessionFences": "base_resources_server",
    "ResourcesSessionLocks": "base_resources_server",
    "ResourcesSessionLostError": "base_resources_server",
    "ResourcesVerification": "base_resources_server",
    "ResourcesVerifyRequest": "base_resources_server",
    "RestoreRequest": "_checkpoint.control",
    "RestoredAgentSession": "_checkpoint.agent",
    "ReverifyMode": "base_resources_server",
    "RolloutContextMiddleware": "request_handling.rollout_context",
    "RolloutIdRequiredError": "_checkpoint.errors",
    "SESSION_OWNER_HEADER": "request_handling.session_routing",
    "SimpleResourcesServer": "base_resources_server",
    "SimpleServer": "base_server",
    "SingleAgentTurnFailure": "single_agent_turn_types",
    "SingleAgentTurnRequest": "single_agent_turn_types",
    "SingleAgentTurnResponse": "single_agent_turn_types",
    "SingleAgentTurnResult": "single_agent_turn_types",
    "SingleAgentTurnTaskInput": "single_agent_turn_types",
    "StaleAttemptError": "_checkpoint.errors",
    "StepType": "_checkpoint.steps",
    "TOKEN_HEADER": "mcp_auto_exposure",
    "ToolAccess": "tool_access",
    "check_resources_session_contract": "testing.session_conformance",
    "dependency_failure": "base_environment_server",
    "entrypoint_ray_enabled": "ray_support",
    "handled_episode_failure": "base_environment_server",
    "harvest_tools": "mcp_auto_exposure",
    "install_auto_exposure": "mcp_auto_exposure",
    "install_session_routing": "request_handling.session_routing",
    "is_handled_episode_failure": "base_environment_server",
    "is_retryable_dependency_error": "base_environment_server",
    "is_retryable_dependency_status": "base_environment_server",
    "lost_session_verification": "base_resources_server",
    "maybe_auto_expose": "mcp_auto_exposure",
    "next_attempt": "_checkpoint.control",
    "reject_partial_typed_seed": "base_resources_server",
    "observe_llm_judge_grading": "verifier_observability",
    "observe_programmatic_grading": "verifier_observability",
    "observe_verifier": "verifier_observability",
    "observe_verifier_sandbox": "verifier_observability",
    "require_rollout": "_checkpoint.agent",
    "seed_rebinds_tool_accesses": "_checkpoint.steps",
    "seed_restarts": "_checkpoint.steps",
    "seed_verify_step_type": "_checkpoint.steps",
    "task_digest": "_checkpoint.environment",
    "write_participant_state": "_checkpoint.store",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
