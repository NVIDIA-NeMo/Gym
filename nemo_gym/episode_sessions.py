# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared state and opt-in resources-session routes."""

from dataclasses import dataclass

from nemo_gym.episode import AgentSeedSessionRequest
from nemo_gym.rollout_observability import AgentObservationBundle


AGENT_SESSION_ACTIVE_KEY = "nemo_gym_agent_session"


@dataclass
class AgentSession:
    """Store state owned by one agent-server worker."""

    request: AgentSeedSessionRequest
    state: object
    activation_started: bool = False


@dataclass
class AgentCloseSessionResult:
    """Return data harvested while closing agent-owned session state."""

    agent_observations: AgentObservationBundle | None = None
    resources_cookies: dict[str, str] | None = None
