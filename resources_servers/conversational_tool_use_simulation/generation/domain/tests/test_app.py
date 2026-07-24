# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from resources_servers.conversational_tool_use_simulation.generation.domain.app import DomainGenerationResourcesServer
from resources_servers.conversational_tool_use_simulation.generation.domain.assets import load_domain_prompt


def test_domain_generation_server_contract() -> None:
    server = DomainGenerationResourcesServer.model_construct(config=None, server_client=None)
    routes = {route.path for route in server.setup_webserver().routes}

    assert routes >= {"/generate"}
    assert "/seed_session" not in routes
    assert "/verify" not in routes
    assert load_domain_prompt()
