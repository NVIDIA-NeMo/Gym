# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP grant staging in a continuation activation spends the activation's own deadline."""

import asyncio
import json
from time import monotonic

import pytest
from fastapi.testclient import TestClient

from nemo_gym.agent_activation_types import AgentContinuationRequirements
from responses_api_agents.opencode_agent.tests.test_continuation import (
    activate,
    close,
    install_artifact_runner,
    roots,
)
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import seed, setup  # noqa: F401


GRANT = {
    "kind": "mcp",
    "name": "task_tools",
    "required": True,
    "connection": {"url": "http://resources.example:8000/task_tools/mcp", "headers": {"x-session": "token"}},
}
# A transfer this slow would hold the request well past its 0.01 s budget if staging ignored the deadline.
STALL_SECONDS = 2.0


def start_with_grant(client):
    request = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    body = request.model_dump(mode="json")
    body["tool_accesses"] = [GRANT]
    created = client.post("/v1/agent_sessions", json=body)
    assert created.status_code == 200, created.text
    return request


@pytest.mark.parametrize("stalled", ["required_mcp.mjs", "mcp-grants.json", "mcp-status.json"])
def test_mcp_staging_expiry_is_incomplete_and_next_activation_recovers(setup, tmp_path, stalled):  # noqa: F811
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    upload, launch = sandbox.upload, sandbox.launch.side_effect
    stall = {"on": True}

    async def slow_upload(source, destination):
        if stall["on"] and destination.endswith("/" + stalled):
            await asyncio.sleep(STALL_SECONDS)
        await upload(source, destination)

    async def connected_launch(**kwargs):
        # Stand-in for required_mcp.mjs, which records the grant status before the first model call.
        path = [path for path in sandbox.files if path.endswith("/params.sh")][-1]
        directory = path.rsplit("/", 1)[0]
        sandbox.files[f"{directory}/mcp-status.json"] = json.dumps({"task_tools": {"status": "connected"}})
        return await launch(**kwargs)

    sandbox.launch.side_effect = connected_launch
    with TestClient(agent.setup_webserver()) as client:
        request = start_with_grant(client)
        sandbox.upload = slow_upload
        started = monotonic()
        first = activate(client, 1, timeout=0.01)
        elapsed = monotonic() - started
        assert first.status_code == 200, first.text
        result = first.json()
        assert result["status"] == "incomplete" and result["output"] == []
        assert roots(result["_ng_agent_observations"])[0]["error_type"] == "timeout"
        assert elapsed < STALL_SECONDS / 2, f"staging ignored the activation deadline: {elapsed:.2f} s"
        sandbox.launch.assert_not_awaited()
        stall["on"] = False
        second = activate(client, 2, "Start now")
        assert second.status_code == 200 and second.json()["status"] == "completed", second.text
        sandbox.launch.assert_awaited_once()
        receipt = close(client, request)
        assert [r["status"] for r in roots(receipt["agent_observations"])] == ["incomplete", "completed"]
        assert receipt["cleanup_confirmed"]
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
