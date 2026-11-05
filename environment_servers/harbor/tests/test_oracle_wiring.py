# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server names the wiring, not the task, when it cannot tell which harness it serves."""

from omegaconf import OmegaConf

from environment_servers.harbor.tests.conftest import make_server, run_episode


async def test_an_unresolvable_implementation_warns_and_places_no_solution(tmp_path, monkeypatch, caplog):
    """The wiring, not the task, is named when the server cannot tell which harness it serves."""
    server, task, sandbox, _ = make_server(
        tmp_path, monkeypatch, agent_implementation="harbor_oracle_agent", solution=True
    )
    # Two implementations under the instance: the server cannot tell which harness it serves.
    block = OmegaConf.to_container(server.server_client.global_config_dict)
    block["agent"]["responses_api_agents"]["other_agent"] = dict(
        block["agent"]["responses_api_agents"]["harbor_oracle_agent"]
    )
    server.server_client.global_config_dict = OmegaConf.create(block)
    with caplog.at_level("WARNING"):
        response = await run_episode(server, task)
    assert response.failure is None, response.failure
    assert [c for c in sandbox.execs if "tar -xzf" in c["command"] and "-C /solution" in c["command"]] == []
    assert "no single responses_api_agents implementation" in caplog.text and "no solution is placed" in caplog.text
