# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from responses_api_agents.hermes_agent.mcp import hermes_mcp_wire_name


def test_wire_name_matches_hermes_provider_safe_alias():
    assert hermes_mcp_wire_name("work-place", "reply-email") == "mcp__work_place__reply_email"


def test_wire_name_preserves_existing_underscores():
    assert hermes_mcp_wire_name("work_place", "reply_email") == "mcp__work_place__reply_email"
