# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pinned Pi CLI forwards prompt and tool controls to a loopback model."""

from unittest.mock import patch

import pytest

from nemo_gym.config import ModelServerRef
from responses_api_agents.pi_agent.tests.test_app import _make_agent
from responses_api_agents.pi_agent.tests.test_sandbox import (
    _finish_tool_server,
    _finishing_model,
    _http_server,
    _pinned_pi_package,
)


@pytest.mark.parametrize("mode", ["append", "replace"])
@pytest.mark.parametrize("enable_tools", [True, False])
async def test_pinned_pi_receives_prompt_and_builtin_tool_controls(tmp_path, monkeypatch, mode, enable_tools):
    node, package = _pinned_pi_package()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(key, raising=False)
    requests = []
    with _http_server(_finishing_model(requests, [])) as model_url:
        agent = _make_agent(
            command=f"{node} {package / 'dist/cli.js'}",
            workspace_root=str(tmp_path),
            model="synthetic",
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            system_prompt_mode=mode,
            enable_tools=enable_tools,
        )
        with (
            patch("responses_api_agents.pi_agent.app.ensure_pi"),
            patch.object(agent, "_resolve_model_base_url", return_value=model_url),
        ):
            output, _, _, _ = await agent._run_pi(
                "Synthetic user task", "G10 replacement marker", collect_observations=False, timeout=30
            )
    assert output[0].content[0].text == "Done."
    assert len(requests) == 1
    system = "\n".join(message["content"] for message in requests[0]["messages"] if message["role"] == "system")
    assert "G10 replacement marker" in system
    assert ("You are an expert coding assistant operating inside pi" in system) is (mode == "append")
    assert requests[0]["messages"][-1]["content"] == [{"type": "text", "text": "Synthetic user task"}]
    assert bool(requests[0].get("tools")) is enable_tools


async def test_pinned_pi_keeps_explicit_mcp_tools_when_builtins_are_disabled(tmp_path, monkeypatch):
    node, package = _pinned_pi_package()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(key, raising=False)
    requests = []
    with _http_server(_finish_tool_server([])) as mcp_url, _http_server(_finishing_model(requests, [])) as model_url:
        agent = _make_agent(
            command=f"{node} {package / 'dist/cli.js'}",
            workspace_root=str(tmp_path),
            model="synthetic",
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            enable_tools=False,
            mcp_servers={"resources": {"url": mcp_url}},
        )
        with (
            patch("responses_api_agents.pi_agent.app.ensure_pi"),
            patch.object(agent, "_resolve_model_base_url", return_value=model_url),
        ):
            output, _, _, _ = await agent._run_pi("Synthetic task", None, collect_observations=False, timeout=30)
    assert output[0].content[0].text == "Done."
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["resources_finish"]
