# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from resources_servers.synthetic_tool_use_scenario_generation.app import ScenarioGenerationResourcesServer
from resources_servers.synthetic_tool_use_scenario_generation.assets import load_scenario_prompts
from resources_servers.synthetic_tool_use_scenario_generation.schema import scenario_schema_json


def test_scenario_generation_server_contract() -> None:
    server = ScenarioGenerationResourcesServer.model_construct(config=None, server_client=None)
    routes = {route.path for route in server.setup_webserver().routes}
    prompts = load_scenario_prompts()

    assert routes >= {"/generate"}
    assert "/seed_session" not in routes
    assert "/verify" not in routes
    assert prompts.system
    assert prompts.user
    assert scenario_schema_json()
