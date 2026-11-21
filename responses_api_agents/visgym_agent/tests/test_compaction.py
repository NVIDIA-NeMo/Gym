# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from responses_api_agents.visgym_agent.app import TextActionAgent, TextActionAgentConfig, TextActionAgentRunRequest
from tests.helpers.visual_context_contract import VisualContextContract


class TestCompaction(VisualContextContract):
    agent_type = TextActionAgent
    config_type = TextActionAgentConfig
    request_type = TextActionAgentRunRequest
    component = "visgym_agent"
    overlay_path = Path(__file__).parents[1] / "configs/first_last_two.yaml"
