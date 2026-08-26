# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from responses_api_agents.hermes_agent import runner
from responses_api_agents.hermes_agent.mcp_names import hermes_mcp_wire_name


def test_wire_name_matches_hermes_provider_safe_alias():
    assert hermes_mcp_wire_name("work-place", "reply-email") == "mcp__work_place__reply_email"


def test_wire_name_preserves_existing_underscores():
    assert hermes_mcp_wire_name("work_place", "reply_email") == "mcp__work_place__reply_email"


def test_runner_directory_does_not_shadow_upstream_mcp_sdk():
    assert not Path(runner.__file__).with_name("mcp.py").exists()
