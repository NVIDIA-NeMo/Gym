# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.synthetic_tool_use_policy_tool_generation.app import PolicyToolGenerationResourcesServer
from resources_servers.synthetic_tool_use_policy_tool_generation.profiles import load_profile


@pytest.mark.parametrize("profile_name", ["general", "proactive"])
def test_policy_tool_generation_server_contract(profile_name: str) -> None:
    server = PolicyToolGenerationResourcesServer.model_construct(config=None, server_client=None)
    routes = {route.path for route in server.setup_webserver().routes}
    profile = load_profile(profile_name)

    assert routes >= {"/generate"}
    assert "/seed_session" not in routes
    assert "/verify" not in routes
    assert profile.policy_prompt
    assert profile.tools_prompt
