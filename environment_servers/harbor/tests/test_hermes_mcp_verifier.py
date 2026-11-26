# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from environment_servers.harbor.tests.conftest import agent_response, make_server, run_episode
from nemo_gym.utils import hermes_mcp_metadata


@pytest.mark.parametrize("authored_name", ["mcp_claw_media_v1_read", "mcp__claw_media_v1__read"])
async def test_hermes_verification_copy_preserves_returned_model_tool_names(tmp_path, monkeypatch, authored_name):
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    response = agent_response().model_dump(mode="json")
    response["metadata"] = hermes_mcp_metadata(["claw", "claw-media.v1"])
    response["output"].insert(
        0,
        {"type": "function_call", "id": "tool1", "call_id": "tool1", "name": authored_name, "arguments": "{}"},
    )
    server.server_client.turn_response = response
    seen = {}
    export = server._export_trajectory
    verify = server._run_verifier

    def export_copy(request, verifier_response):
        seen["export"] = verifier_response.output[0].name
        return export(request, verifier_response)

    async def verify_copy(session, trajectory, *, response):
        seen["verify"] = response.output[0].name
        return await verify(session, trajectory, response=response)

    monkeypatch.setattr(server, "_export_trajectory", export_copy)
    monkeypatch.setattr(server, "_run_verifier", verify_copy)
    result = await run_episode(server, task)

    assert result.failure is None
    assert result.result.reward == 1.0
    assert result.result.mask_sample is False
    assert seen == {"export": "mcp__claw-media.v1__read", "verify": "mcp__claw-media.v1__read"}
    assert result.result.response.output[0].name == authored_name
    assert response["output"][0]["name"] == authored_name


async def test_invalid_hermes_metadata_refuses_before_export_or_verification(tmp_path, monkeypatch):
    server, task, _, _ = make_server(tmp_path, monkeypatch)
    response = agent_response().model_dump(mode="json")
    response["metadata"] = {"hermes_mcp_servers": '["claw","claw"]'}
    response["output"].insert(
        0,
        {"type": "function_call", "id": "tool1", "call_id": "tool1", "name": "mcp_claw_read", "arguments": "{}"},
    )
    server.server_client.turn_response = response

    def forbidden_export(*args):
        pytest.fail("Malformed metadata must not reach the verifier")

    monkeypatch.setattr(server, "_export_trajectory", forbidden_export)
    result = await run_episode(server, task)
    assert result.failure is not None
    assert result.result is None
