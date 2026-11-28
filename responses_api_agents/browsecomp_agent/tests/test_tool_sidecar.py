# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from the Apache-2.0 BrowseComp tool-sidecar tests in NeMo Gym.
# https://github.com/NVIDIA-NeMo/Gym/blob/73b48526c6d91be5faa07c68f433b27241509f2a/responses_api_agents/browsecomp_agent/tests/test_tool_sidecar.py

import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.agents import AgentSeedRejectedError, SandboxSession
from nemo_gym.sandbox import SandboxSpec
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent, BrowsecompAgentConfig
from responses_api_agents.browsecomp_agent.tests.test_agent_sessions import GRANT, _agent, _seed
from responses_api_agents.browsecomp_agent.tool_sidecar import (
    BrowsecompToolSidecar,
    BrowsecompToolSidecarConfig,
    ToolSidecarImageError,
    sidecar_spec,
)


SANDBOX_RESPONSES = BrowsecompAgent._sandbox_responses


CONFIG = {
    "resources_server": {"type": "resources_servers", "name": "configured_harness"},
    "image": "example/tools:pinned",
}


@pytest.mark.parametrize(
    "model,targets",
    [
        ("http://model:8000/v1", ["api.tavily.com", "model"]),
        ("http://10.20.30.40:8000/v1", ["10.20.30.40", "api.tavily.com"]),
    ],
)
def test_native_policy_allows_only_search_and_inference(model, targets):
    spec = sidecar_spec(
        SandboxSpec(image="python:3.13-slim", ttl_s=600, metadata={"episode": "one"}),
        config=BrowsecompToolSidecarConfig(**CONFIG),
        model_url=model,
        outside_urls=["http://resources:8000", "http://agent:8001", "http://environment:8002"],
        settings={"search_provider": "tavily"},
        tool_token="test-grant",
    )
    assert spec.metadata == {"episode": "one"}
    assert len(spec.sidecars) == 1 and spec.sidecars[0].name == "tools"
    assert spec.sidecars[0].env["BROWSECOMP_TOOL_TOKEN"] == "test-grant"
    assert spec.provider_options == {
        "network_policy": {
            "defaultAction": "deny",
            "egress": [{"action": "allow", "target": target} for target in targets],
        }
    }
    assert not spec.ports, "Tools need no externally exposed pod port"


@pytest.mark.parametrize(
    "model,options,ttl,reason",
    [
        ("http://agent:9000/v1", {}, 600, "dedicated"),
        ("http://127.0.0.1:9000/v1", {}, 600, "reachable"),
        ("http://model/v1", {"network_policy": {"defaultAction": "allow"}}, 600, "network rules"),
        ("http://model/v1", {}, None, "ttl_s"),
        ("http://model/v1", {}, float("nan"), "ttl_s"),
    ],
)
def test_invalid_egress_or_lifetime_fails_closed(model, options, ttl, reason):
    with pytest.raises(ValueError, match=reason):
        sidecar_spec(
            SandboxSpec(image="main", ttl_s=ttl, provider_options=options),
            config=BrowsecompToolSidecarConfig(**CONFIG),
            model_url=model,
            outside_urls=["http://agent:8000"],
            settings={"search_provider": "tavily"},
            tool_token="test-grant",
        )


def test_only_selected_tool_settings_cross_the_container_boundary():
    config = BrowsecompToolSidecarConfig(**CONFIG)
    provider_limits = {"provider_rate_limit_wait_seconds": 3, "provider_request_timeout_seconds": 15}
    selected = config.tool_settings(
        {
            "tavily_api_key": "search-key",
            "exa_api_key": "unused-key",
            "max_results": 7,
            **provider_limits,
            "ground_truth": "private-answer",
            "judge_model_server": {"host": "private-judge"},
            "sandbox_provider": "another-pod",
            "workspace_root": "/private",
            "use_judge": True,
        }
    )
    assert selected["tavily_api_key"] == "search-key"
    assert selected["max_results"] == 7
    assert {name: selected[name] for name in provider_limits} == provider_limits
    assert not {"exa_api_key", "ground_truth", "judge_model_server", "sandbox_provider", "use_judge"}.intersection(
        selected
    )


def test_token_count_callback_cannot_be_enabled_with_tool_sidecar():
    with pytest.raises(ValidationError, match="token-count callback"):
        BrowsecompAgentConfig(
            name="agent",
            host="agent",
            port=1,
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy"},
            tool_sidecar=CONFIG,
            save_model_call_using_vllm_tokenize_endpoint=True,
        )


@pytest.mark.parametrize(
    "url",
    ["http://localhost/v1", "http://user@model/v1", "http://model/v1?key=secret", "http://model/v1#fragment"],
)
def test_direct_model_url_rejects_credentials_and_ambiguous_origins(url):
    with pytest.raises(ValidationError):
        BrowsecompToolSidecarConfig(**CONFIG, model_base_url=url)


async def test_direct_model_url_still_rejects_control_hosts(seeded_agent):
    agent, app, _, _ = seeded_agent
    agent.config.tool_sidecar.model_base_url = "http://head/v0/models/deployment/v1"
    with pytest.raises(AgentSeedRejectedError, match="dedicated model-serving host"):
        await agent._seed_agent_session_state(_seed())
    app.AsyncSandbox.return_value.start.assert_not_awaited()


async def test_direct_model_url_cannot_bypass_required_token_capture(seeded_agent, monkeypatch):
    agent, app, _, _ = seeded_agent
    agent.config.tool_sidecar.model_base_url = "http://model/v0/models/deployment/v1"
    monkeypatch.setattr(type(agent), "_token_id_capture_enabled", lambda self: True)
    with pytest.raises(AgentSeedRejectedError, match="token capture"):
        await agent._seed_agent_session_state(_seed())
    app.AsyncSandbox.return_value.start.assert_not_awaited()


@pytest.fixture
def seeded_agent(monkeypatch):
    from responses_api_agents.browsecomp_agent import app

    client, server_client, _ = _agent([], monkeypatch, tool_sidecar=CONFIG)
    agent = next(route.endpoint.__self__ for route in client.app.routes if route.path == "/v1/browsecomp/token_count")
    server_client.global_config_dict = OmegaConf.create(
        {
            "configured_harness": {
                "resources_servers": {
                    "browsecomp_advanced_harness": {
                        "host": "harness",
                        "port": 9000,
                        "workspace": "none",
                        "use_tool_sidecar_metrics": True,
                        "tavily_api_key": "private-search-key",
                        "ground_truth": "private-answer",
                    }
                }
            },
            "browsecomp_agent": {"responses_api_agents": {"browsecomp_agent": {"host": "agent", "port": 8000}}},
        }
    )
    server_client._resolve_base_url.side_effect = lambda name: (
        "http://harness:9000" if name == "configured_harness" else "http://agent:8000"
    )
    server_client.head_server_config = SimpleNamespace(host="head", port=11000)
    server_client._build_server_base_url.return_value = "http://head:11000"
    monkeypatch.setattr(
        app, "resolve_provider_config", lambda *args: {"opensandbox": {"networking": {"enabled": True}}}
    )
    sidecar = MagicMock(
        start=AsyncMock(), close=AsyncMock(), metrics=AsyncMock(return_value={"async_tavily_calls": []})
    )
    factory = MagicMock(return_value=sidecar)
    monkeypatch.setattr(app, "BrowsecompToolSidecar", factory)
    return agent, app, sidecar, factory


async def test_seed_creates_tools_sidecar_and_replaces_host_tool_access(seeded_agent):
    agent, app, sidecar, factory = seeded_agent
    state = await agent._seed_agent_session_state(_seed())
    spec = app.AsyncSandbox.return_value.start.await_args.args[0]
    assert len(spec.sidecars) == 1 and spec.sidecars[0].name == "tools"
    app.AsyncSandbox.return_value.start.assert_awaited_once()
    assert str(state.tool_access.base_url) == "http://tools:8080/"
    assert state.tool_access.cookies == {} and state.tool_access.headers == {
        "Authorization": "Bearer " + spec.sidecars[0].env["BROWSECOMP_TOOL_TOKEN"]
    }
    assert state.resources_cookies == dict(GRANT.cookies)
    assert "private-answer" not in str(factory.call_args)
    assert "private-search-key" not in str(spec.env)
    assert json.loads(spec.sidecars[0].env["BROWSECOMP_TOOL_SETTINGS"])["tavily_api_key"] == "private-search-key"
    sidecar.start.assert_awaited_once()
    state.session.artifacts = {"resources_cookies": {"tool-cookie": "pod-only"}}
    closed = await agent._close_agent_session_state(state)
    assert closed.resources_cookies == dict(GRANT.cookies), "Outside verifier must retain its original seed cookie"
    sidecar.close.assert_awaited_once()


async def test_disabled_peer_networking_rejects_seed_before_allocation(seeded_agent, monkeypatch):
    agent, app, _, _ = seeded_agent
    monkeypatch.setattr(app, "resolve_provider_config", lambda *args: {"opensandbox": {}})
    with pytest.raises(AgentSeedRejectedError, match="networking.enabled"):
        await agent._seed_agent_session_state(_seed())
    app.AsyncSandbox.assert_not_called()


async def test_sidecar_startup_failure_closes_owned_sandbox(seeded_agent):
    agent, app, sidecar, _ = seeded_agent
    sidecar.start.side_effect = RuntimeError("tools not ready")
    with pytest.raises(RuntimeError, match="tools not ready"):
        await agent._seed_agent_session_state(_seed())
    SandboxSession.close.assert_awaited_once(), "The session must stop the pod it owns"
    sidecar.close.assert_awaited_once()


async def test_unrunnable_tools_image_rejects_the_seed(seeded_agent):
    agent, app, sidecar, _ = seeded_agent
    sidecar.start.side_effect = ToolSidecarImageError("exit code 127")
    with pytest.raises(AgentSeedRejectedError, match="exit code 127"):
        await agent._seed_agent_session_state(_seed())
    SandboxSession.close.assert_awaited_once(), "The session must stop the pod it owns"


async def test_model_on_the_head_server_host_is_rejected(seeded_agent, monkeypatch):
    agent, app, _, _ = seeded_agent
    agent.server_client._build_server_base_url.return_value = "http://policy-host:11000"
    monkeypatch.setattr(type(agent), "resolve_model_base_url", lambda self, *args: "http://policy-host:8000/v1")
    with pytest.raises(AgentSeedRejectedError, match="dedicated model-serving host"):
        await agent._seed_agent_session_state(_seed())
    app.AsyncSandbox.return_value.start.assert_not_awaited()


def _exec_result(return_code=0, stderr=None, stdout=None):
    return SimpleNamespace(return_code=return_code, error_type=None, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize(
    "service,error,match",
    [
        (_exec_result(127, "python: not found"), ToolSidecarImageError, "(exit code 127): python: not found"),
        (_exec_result(1, "RuntimeError: startup failed"), RuntimeError, "startup failed"),
    ],
)
async def test_tools_startup_failures_name_their_cause(service, error, match):
    sandbox = MagicMock()

    async def execute(*argv, **kwargs):
        if "-m" in argv:
            return service
        if "/health" in argv[-1]:
            await asyncio.sleep(0.01)
            return _exec_result(7)
        return _exec_result()

    sandbox.sidecar.return_value.exec = AsyncMock(side_effect=execute)
    sidecar = BrowsecompToolSidecar(sandbox, BrowsecompToolSidecarConfig(**CONFIG))
    with pytest.raises(error, match=re.escape(match)) as raised:
        await sidecar.start(lifetime_seconds=600)
    assert isinstance(raised.value, ToolSidecarImageError) == (error is ToolSidecarImageError)
    await sidecar.close()


async def test_tools_readiness_timeout_names_the_probe():
    sandbox = MagicMock()

    async def execute(*argv, **kwargs):
        if "-m" in argv:
            await asyncio.Event().wait()
        return _exec_result(0 if "-m" not in argv and "/health" not in argv[-1] else 7)

    sandbox.sidecar.return_value.exec = AsyncMock(side_effect=execute)
    config = BrowsecompToolSidecarConfig(**CONFIG, startup_timeout_seconds=0.05)
    sidecar = BrowsecompToolSidecar(sandbox, config)
    with pytest.raises(TimeoutError, match="not ready on http://127.0.0.1:8080/health"):
        await sidecar.start(lifetime_seconds=600)
    await sidecar.close()
    assert sidecar.service.done()


async def test_borrowed_sandbox_rejects_tool_sidecar(seeded_agent):
    from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess

    agent, app, _, _ = seeded_agent
    with pytest.raises(AgentSeedRejectedError, match="agent-owned OpenSandbox"):
        await agent._seed_agent_session_state(
            _seed(
                sandbox_access=SandboxAccess(
                    connection=DirectSandboxConnection(
                        provider_config_ref="sandbox", descriptor={"sandbox_id": "existing"}
                    ),
                    workdir="/workspace",
                )
            )
        )
    app.AsyncSandbox.assert_not_called()
    app.AsyncSandbox.connect.assert_not_awaited()


async def test_management_requests_use_loopback_inside_tools():
    sandbox = MagicMock()
    commands = []

    async def execute(*argv, **kwargs):
        commands.append(argv)
        if "-m" in argv:
            await asyncio.Event().wait()
        stdout = '{"async_tavily_calls": []}' if "/_metrics" in argv[-1] else ""
        return SimpleNamespace(return_code=0, error_type=None, stdout=stdout)

    sandbox.sidecar.return_value.exec = AsyncMock(side_effect=execute)
    sidecar = BrowsecompToolSidecar(sandbox, BrowsecompToolSidecarConfig(**CONFIG))
    try:
        await sidecar.start(lifetime_seconds=600)
        await asyncio.sleep(0)
        assert await sidecar.metrics() == {"async_tavily_calls": []}
        assert any("resources_servers.browsecomp_advanced_harness.tools_sidecar" in cmd for cmd in commands)
        assert all("127.0.0.1:8080" in cmd[-1] for cmd in commands if "urllib.request" in " ".join(cmd))
        sandbox.start.assert_not_called()
        sandbox.stop.assert_not_called()
    finally:
        await sidecar.close()
    assert sidecar.service.done()


@pytest.mark.parametrize("model_base_url", [None, "http://10.20.30.40/v0/models/deployment/v1"])
async def test_runner_receives_no_host_cookies_or_callbacks_and_returns_metrics(seeded_agent, model_base_url):
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
    from responses_api_agents.browsecomp_agent.tests.test_agent_sessions import _model_response

    agent, app, sidecar, _ = seeded_agent
    agent.config.tool_sidecar.model_base_url = model_base_url
    state = await agent._seed_agent_session_state(_seed())
    captured = []

    async def stage(payload):
        captured.append(payload)

    async def execute(**kwargs):
        await kwargs["stage_activation"]()
        return {"response": _model_response(), "resources_cookies": {"pod": "cookie"}}

    state.stage_activation = stage
    state.session.execute = AsyncMock(side_effect=execute)
    result = await SANDBOX_RESPONSES(
        agent,
        MagicMock(cookies={"agent": "private-cookie"}),
        NeMoGymResponseCreateParamsNonStreaming(input=[]),
        state,
    )
    assert result.model_dump()["browsecomp_tool_metrics"] == {"async_tavily_calls": []}
    assert state.resources_cookies == dict(GRANT.cookies)
    payload = captured[0]
    assert payload["token_count_url"] is None and payload["agent_cookies"] == {}
    assert payload["seed"]["tool_accesses"] == []
    assert "tool_sidecar" not in payload["config"]
    assert "private" not in json.dumps(payload)
    assert payload["tool_access"]["base_url"] == "http://tools:8080/"
    assert payload["model_url"] == (model_base_url or "http://model:9001/v1")
    spec = app.AsyncSandbox.return_value.start.await_args.args[0]
    assert {rule["target"] for rule in spec.provider_options["network_policy"]["egress"]} == {
        "api.tavily.com",
        "10.20.30.40" if model_base_url else "model",
    }
    sidecar.metrics.assert_awaited_once()


def test_example_config_launches_policy_and_existing_environment(monkeypatch):
    from pathlib import Path

    from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
    from nemo_gym.server_runtime.supervisor.server_supervisor import _configured_servers

    monkeypatch.setenv("OPENSANDBOX_API_KEY", "unused-in-offline-test")
    monkeypatch.setenv("GYM_BROWSECOMP_TOOLS_IMAGE", "example/tools:pinned")
    monkeypatch.setenv("BROWSECOMP_POLICY_MODEL_HOST", "dedicated-model")
    path = (
        Path(__file__).resolve().parents[3]
        / "resources_servers/browsecomp_advanced_harness/configs/browsecomp_opensandbox_tool_sidecar.yaml"
    )
    config = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        OmegaConf.load(path),
        {
            "tavily_api_key": "test-key",
            "policy_base_url": "https://inference.example/v1",
            "policy_api_key": "test-policy-key",
            "policy_model_name": "test-policy",
            "exclude_domains_file_path": "/tmp/domains.json",
            "search_judge_model_base_url": "http://judge:8000/v1",
            "search_judge_model_api_key": "test-key",
            "search_judge_model_name": "fixture",
        },
    )
    parsed = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=config,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    policy = parsed.policy_model.responses_api_models.vllm_model
    assert policy.host == "dedicated-model" and policy.port == 8000
    assert policy.entrypoint == "app.py"
    launched = {server.top_level_path for server in _configured_servers(parsed)}
    assert "policy_model" in launched
    assert "browsecomp_advanced_harness_agent" in launched
    assert "judge_model" in launched
    assert parsed.sandbox.opensandbox.networking.enabled
    environment = parsed.browsecomp_advanced_harness_environment_server.environment_servers.single_agent_turn_legacy
    assert environment.resources_server.name == "browsecomp_advanced_harness_resources_server"
    agent = parsed.browsecomp_advanced_harness_agent.responses_api_agents.browsecomp_agent
    assert agent.tool_sidecar.image == "example/tools:pinned"
    assert not agent.save_model_call_using_vllm_tokenize_endpoint
