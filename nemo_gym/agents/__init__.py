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
"""What agent servers share: the agent server base, agent outcomes and activations, sandbox sessions and tools, and skills."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.agents.agent_activation_types import AgentContinuationCapabilities, AgentContinuationRequirements
    from nemo_gym.agents.agent_outcome import (
        AGENT_OUTCOME_METADATA_KEY,
        HARNESS_STEP_LIMIT_METADATA_KEY,
        AgentOutcome,
        AgentOutcomePolicyConfig,
        OutcomeAction,
        OutcomePolicyConfig,
        agent_outcome_policy,
        confirmed_policy_response_timeout,
        failure_outcome_policy,
    )
    from nemo_gym.agents.agent_tools import (
        restricted_network_policy,
        sandbox_server_url,
        seed_mcp_servers,
        verification_failure_fields,
        verify_agent_response,
    )
    from nemo_gym.agents.base_responses_api_agent import (
        AGENT_ACTIVATION_HEADER,
        AGENT_EPISODE_STEP_LIMIT_HEADER,
        AGENT_SESSION_COOKIE_KEY,
        ROLLOUT_STOP_METADATA_KEY,
        AgentCloseSessionRequest,
        AgentCloseSessionResponse,
        AgentSeedRejectedError,
        AgentSeedSessionRequest,
        AgentSeedSessionResponse,
        AgentSessionCapabilitiesResponse,
        AgentSessionConflictError,
        AgentSessionSetupError,
        AgentSessionState,
        BaseResponsesAPIAgentConfig,
        Body,
        RolloutStopRequest,
        SimpleResponsesAPIAgent,
        assert_model_url_reachable_from_sandbox,
    )
    from nemo_gym.agents.skills import stage_skills
    from nemo_gym.agents.testing.agent_session_conformance import check_agent_seed_rejected
    from nemo_gym.agents.testing.failure_assertions import raises_agent_failure
    from nemo_gym.agents.utils.agent_context import resolve_agent_context_artifact
    from nemo_gym.agents.utils.ordered_operations import ActivationConflictError, OrderedOperationLedger
    from nemo_gym.agents.utils.sandbox_network import SandboxNetworkAccess, restrict_sandbox_network
    from nemo_gym.agents.utils.sandbox_observation import lifecycle_observation
    from nemo_gym.agents.utils.sandbox_python import SandboxPythonRuntime, probe_sandbox_platform
    from nemo_gym.agents.utils.sandbox_session import (
        HARNESS_COMMAND_NOT_STARTED,
        HARNESS_RUNNER_FAILED,
        HARNESS_RUNNER_FILE,
        HARNESS_RUNNER_SCRIPT,
        SandboxCommand,
        SandboxSession,
        shell_assignments,
    )
    from nemo_gym.agents.utils.supervisor_client import (
        CLEANUP_PHASE_COUNT,
        DEFAULT_CLEANUP_TIMEOUT,
        OUTPUT_LOG_FILE,
        STOP_REQUEST_FILE,
        SUPERVISOR_FILE,
        SUPERVISOR_SCRIPT,
        CleanupReceipt,
        exec_timeout,
        harness_termination_failure,
        is_harness_termination_failure,
        parse_cleanup_receipt,
    )
    from nemo_gym.agents.utils.task_timeout import AGENT_TIMEOUT_METADATA_KEY, resolve_agent_timeout


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "ActivationConflictError": "utils.ordered_operations",
    "AGENT_ACTIVATION_HEADER": "base_responses_api_agent",
    "AGENT_EPISODE_STEP_LIMIT_HEADER": "base_responses_api_agent",
    "AGENT_OUTCOME_METADATA_KEY": "agent_outcome",
    "AGENT_SESSION_COOKIE_KEY": "base_responses_api_agent",
    "AGENT_TIMEOUT_METADATA_KEY": "utils.task_timeout",
    "AgentCloseSessionRequest": "base_responses_api_agent",
    "AgentCloseSessionResponse": "base_responses_api_agent",
    "AgentContinuationCapabilities": "agent_activation_types",
    "AgentContinuationRequirements": "agent_activation_types",
    "AgentOutcome": "agent_outcome",
    "AgentOutcomePolicyConfig": "agent_outcome",
    "AgentSeedRejectedError": "base_responses_api_agent",
    "AgentSeedSessionRequest": "base_responses_api_agent",
    "AgentSeedSessionResponse": "base_responses_api_agent",
    "AgentSessionCapabilitiesResponse": "base_responses_api_agent",
    "AgentSessionConflictError": "base_responses_api_agent",
    "AgentSessionSetupError": "base_responses_api_agent",
    "AgentSessionState": "base_responses_api_agent",
    "BaseResponsesAPIAgentConfig": "base_responses_api_agent",
    "Body": "base_responses_api_agent",
    "CLEANUP_PHASE_COUNT": "utils.supervisor_client",
    "CleanupReceipt": "utils.supervisor_client",
    "DEFAULT_CLEANUP_TIMEOUT": "utils.supervisor_client",
    "HARNESS_COMMAND_NOT_STARTED": "utils.sandbox_session",
    "HARNESS_RUNNER_FAILED": "utils.sandbox_session",
    "HARNESS_RUNNER_FILE": "utils.sandbox_session",
    "HARNESS_RUNNER_SCRIPT": "utils.sandbox_session",
    "HARNESS_STEP_LIMIT_METADATA_KEY": "agent_outcome",
    "OUTPUT_LOG_FILE": "utils.supervisor_client",
    "OrderedOperationLedger": "utils.ordered_operations",
    "OutcomeAction": "agent_outcome",
    "OutcomePolicyConfig": "agent_outcome",
    "ROLLOUT_STOP_METADATA_KEY": "base_responses_api_agent",
    "RolloutStopRequest": "base_responses_api_agent",
    "STOP_REQUEST_FILE": "utils.supervisor_client",
    "SUPERVISOR_FILE": "utils.supervisor_client",
    "SUPERVISOR_SCRIPT": "utils.supervisor_client",
    "SandboxCommand": "utils.sandbox_session",
    "SandboxNetworkAccess": "utils.sandbox_network",
    "SandboxPythonRuntime": "utils.sandbox_python",
    "SandboxSession": "utils.sandbox_session",
    "SimpleResponsesAPIAgent": "base_responses_api_agent",
    "agent_outcome_policy": "agent_outcome",
    "assert_model_url_reachable_from_sandbox": "base_responses_api_agent",
    "check_agent_seed_rejected": "testing.agent_session_conformance",
    "confirmed_policy_response_timeout": "agent_outcome",
    "exec_timeout": "utils.supervisor_client",
    "failure_outcome_policy": "agent_outcome",
    "harness_termination_failure": "utils.supervisor_client",
    "is_harness_termination_failure": "utils.supervisor_client",
    "lifecycle_observation": "utils.sandbox_observation",
    "parse_cleanup_receipt": "utils.supervisor_client",
    "probe_sandbox_platform": "utils.sandbox_python",
    "raises_agent_failure": "testing.failure_assertions",
    "resolve_agent_context_artifact": "utils.agent_context",
    "resolve_agent_timeout": "utils.task_timeout",
    "restrict_sandbox_network": "utils.sandbox_network",
    "restricted_network_policy": "agent_tools",
    "sandbox_server_url": "agent_tools",
    "seed_mcp_servers": "agent_tools",
    "shell_assignments": "utils.sandbox_session",
    "stage_skills": "skills",
    "verification_failure_fields": "agent_tools",
    "verify_agent_response": "agent_tools",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
