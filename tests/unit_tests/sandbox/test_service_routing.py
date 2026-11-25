# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from nemo_gym.agents import agent_tools
from nemo_gym.agents.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.client.server_client import get_server_url


def config():
    return OmegaConf.create(
        {
            "model": {
                "responses_api_models": {
                    "test": {"host": "127.0.0.1", "port": 8000, "sandbox_url": "http://127.0.0.1:18000"}
                }
            },
        }
    )


def model_agent(cfg, *, sandbox=True):
    return SimpleNamespace(
        # An unused provider reference must not interfere with URL selection.
        config=SimpleNamespace(sandbox_provider="unused"),
        server_client=SimpleNamespace(global_config_dict=cfg, _build_server_base_url=lambda _: "http://host:8000"),
        calls_model_from_sandbox=sandbox,
        _token_id_capture_enabled=lambda: False,
    )


def test_only_sandbox_model_calls_use_the_service_override(monkeypatch):
    cfg = config()
    cfg.model.responses_api_models.test.sandbox_url = "https://relay.example/run/model"
    monkeypatch.setattr("nemo_gym.client.server_client.get_global_config_dict", lambda: cfg)
    assert get_server_url("model") == "http://127.0.0.1:8000"
    agent = model_agent(cfg, sandbox=False)
    resolve = SimpleResponsesAPIAgent.resolve_model_base_url
    assert resolve(agent, "model", trace_context=False) == "http://host:8000/v1"
    agent.calls_model_from_sandbox = True
    assert resolve(agent, "model", trace_context=False) == "https://relay.example/run/model/v1"
    del cfg.model.responses_api_models.test["sandbox_url"]
    assert resolve(agent, "model", trace_context=False) == "http://host:8000/v1"


async def test_seeded_mcp_uses_static_override_and_preserves_session_authentication():
    from nemo_gym.config.config_types import ResourcesServerRef
    from nemo_gym.server_runtime.base_resources_server import BaseRunRequest

    cfg = OmegaConf.create(
        {
            "tools": {
                "resources_servers": {
                    "test": {"host": "127.0.0.1", "port": 8001, "sandbox_url": "https://relay.example/run/tools"}
                }
            }
        }
    )
    headers = {"Authorization": "Bearer test-session"}
    response = SimpleNamespace(
        ok=True,
        read=AsyncMock(
            return_value=json.dumps(
                {agent_tools.NEMO_GYM_MCP_METADATA_KEY: {"headers": headers, "url_path": "/mcp"}}
            ).encode()
        ),
        raise_for_status=lambda: None,
    )
    client = SimpleNamespace(global_config_dict=cfg, post=AsyncMock(return_value=response))
    result = await agent_tools.seed_mcp_servers(
        client,
        [ResourcesServerRef(type="resources_servers", name="tools")],
        BaseRunRequest(responses_create_params={"input": "test"}),
        {},
        timeout_s=30,
        require_reachable=True,
    )
    assert result["tools"] == {
        "url": "https://relay.example/run/tools/mcp",
        "headers": headers,
        "enabled": True,
        "timeout": 30000,
    }
    assert client.post.call_args.kwargs["server_name"] == "tools"
    assert client.post.call_args.kwargs["url_path"] == "/seed_session"


def test_loopback_override_requires_reachability():
    with pytest.raises(ValueError, match="sandbox-reachable"):
        agent_tools.sandbox_server_url("model", require_reachable=True, global_config=config())
