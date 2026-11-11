# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from responses_api_agents.gymv_agent.app import GymVAgent, GymVAgentConfig, GymVAgentRunRequest
from tests.helpers.visual_context_contract import VisualContextContract


class TestCompaction(VisualContextContract):
    agent_type = GymVAgent
    config_type = GymVAgentConfig
    request_type = GymVAgentRunRequest
    component = "gymv_agent"
    overlay_path = Path(__file__).parents[1] / "configs/first_last_two.yaml"
