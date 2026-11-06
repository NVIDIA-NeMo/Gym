# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import seed
from responses_api_agents.opencode_multimodal_agent.app import OpenCodeMultimodalAgent
from responses_api_agents.opencode_multimodal_agent.tests.test_app import agent, body, config, database
from responses_api_agents.opencode_multimodal_agent.tests.test_eval import evaluation
from responses_api_agents.opencode_multimodal_agent.tests.test_sessions import (
    opencode_setup,  # noqa: F401
    visual_setup,  # noqa: F401
)


def test_network_is_restricted_by_default():
    from responses_api_agents.opencode_multimodal_agent.eval import OpenCodeMultimodalEvalConfig

    default = config()
    assert default.network_access == "model_only"
    assert OpenCodeMultimodalEvalConfig.model_validate(default.model_dump()).network_access == "model_only"


@pytest.mark.parametrize("mode", ["model_only", "model_and_websearch", "inherit"])
def test_network_policy_preserves_all_and_restricts_web_tools(monkeypatch, tmp_path, mode):
    agent = evaluation(monkeypatch, tmp_path)
    agent.config.network_access = mode
    agent.config.opencode_config["tools"] = {"webfetch": True, "websearch": True}
    agent._write_opencode_config(tmp_path)
    config = json.loads((tmp_path / "opencode.json").read_text())
    assert config["tools"]["webfetch"] is (mode == "inherit")
    assert config["tools"]["websearch"] is (mode != "model_only")
    assert config["permission"]["bash"]["*pkill*"] == "deny"
    assert config["agent"]["build"]["prompt"] == "Exact combined prompt\n"
    if mode != "inherit":
        for agent_config in config["agent"].values():
            assert agent_config["permission"]["webfetch"] == "deny"
            assert agent_config["permission"]["websearch"] == ("allow" if mode == "model_and_websearch" else "deny")


@pytest.mark.parametrize("mode", ["model_only", "model_and_websearch"])
def test_restricted_borrowed_session_fails_before_connect(visual_setup, mode):
    agent, sandbox = visual_setup
    agent.config.network_access = mode
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert response.status_code == 422
        assert "owned sandbox" in response.text
    sandbox.launch.assert_not_awaited()


def test_namespace_failure_never_falls_back_to_open_network(tmp_path):
    launcher = tmp_path / "unshare"
    launcher.write_text("#!/bin/sh\nexit 37\n")
    launcher.chmod(0o700)
    marker = tmp_path / "ran"
    runner = Path(__file__).parents[1] / "network_runner.py"
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--mode",
            "model_only",
            "--model-url",
            "http://127.0.0.1:9",
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ],
        env=os.environ | {"PATH": str(tmp_path)},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 37, result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("ceiling", ["model_only", "model_and_websearch", "inherit"])
@pytest.mark.parametrize("selected", [None, "model_only", "model_and_websearch", "inherit", "invalid"])
def test_sample_policy_and_run_ceiling(monkeypatch, ceiling, selected):
    obj = agent(monkeypatch, network_access=ceiling)
    request = body(metadata={"network_access": selected} if selected is not None else None)
    levels = {"model_only": 0, "model_and_websearch": 1, "inherit": 2}
    if selected == "invalid" or (selected is not None and levels[selected] > levels[ceiling]):
        with pytest.raises(HTTPException) as error:
            obj._network_access_for_request(request)
        assert error.value.status_code == 422
    else:
        assert obj._network_access_for_request(request) == (selected or ceiling)
    assert obj.config.network_access == ceiling


@pytest.mark.asyncio
async def test_mixed_sample_policies_do_not_leak_between_training_episodes(monkeypatch, tmp_path):
    saved_database = database(tmp_path)
    seen = {}

    async def run(self, *args, artifact_path, rollout_id, **kwargs):
        directory = artifact_path.parent
        self._write_opencode_config(directory, rollout_id)
        await asyncio.sleep(0)
        seen[rollout_id] = (self.config.network_access, json.loads((directory / "opencode.json").read_text()))
        shutil.copyfile(saved_database, artifact_path)

    monkeypatch.setattr(OpenCodeMultimodalAgent, "_run_opencode", run)
    obj = agent(monkeypatch, network_access="inherit")
    await asyncio.gather(
        *[
            obj._create_episode(body(metadata={"network_access": mode}), rollout_id=mode)
            for mode in ("model_only", "model_and_websearch", "inherit")
        ]
    )
    assert obj.config.network_access == "inherit"
    for mode, (effective, config) in seen.items():
        assert effective == mode
        if mode != "inherit":
            assert config["tools"]["websearch"] is (mode == "model_and_websearch")
            assert config["tools"]["webfetch"] is False


def test_session_consumes_only_its_network_metadata(monkeypatch, tmp_path):
    obj = evaluation(monkeypatch, tmp_path)
    request = body(metadata={"network_access": "model_and_websearch"})
    assert obj._session_input(request) == obj._session_input(body())
    assert request.metadata == {"network_access": "model_and_websearch"}
    with pytest.raises(HTTPException):
        obj._session_input(body(metadata={"network_access": "model_and_websearch", "unknown_control": "bad"}))


DEFAULT_TOOLS = ["bash", "edit", "glob", "grep", "read", "task", "todowrite", "write"]


@pytest.mark.parametrize(
    "mode,extra", [("model_only", []), ("model_and_websearch", ["websearch"]), ("inherit", ["websearch", "webfetch"])]
)
def test_sample_tool_config(monkeypatch, tmp_path, mode, extra):
    obj = evaluation(monkeypatch, tmp_path)
    obj.config.allowed_tools = DEFAULT_TOOLS + ["websearch", "webfetch"]
    request = body(metadata={"network_access": mode, "allowed_tools": json.dumps(DEFAULT_TOOLS + extra)})
    selected = obj._tools_for_request(request)
    assert selected == DEFAULT_TOOLS + extra
    assert obj._session_input(request) == obj._session_input(
        body(metadata={"allowed_tools": json.dumps(DEFAULT_TOOLS)})
    )
    runner = obj.model_copy(
        update={"config": obj.config.model_copy(update={"network_access": mode, "allowed_tools": selected})}
    )
    runner._write_opencode_config(tmp_path)
    config = json.loads((tmp_path / "opencode.json").read_text())
    assert config["tools"]["websearch"] is ("websearch" in extra)
    assert config["tools"]["webfetch"] is ("webfetch" in extra)
    assert config["permission"]["bash"]["*pkill*"] == "deny"
    assert config["agent"]["build"]["permission"]["bash"]["*pkill*"] == "deny"
    assert config["agent"]["build"]["permission"]["external_directory"] == "allow"
    assert config["agent"]["build"]["permission"]["doom_loop"] == "allow"
    assert config["tools"]["skill"] is False
    assert len(obj.config.allowed_tools) == 10


def test_tool_allowlist_preserves_agent_specific_directory_restriction(monkeypatch, tmp_path):
    obj = evaluation(monkeypatch, tmp_path)
    obj.config.allowed_tools = DEFAULT_TOOLS
    config = {
        "permission": {"external_directory": "allow", "doom_loop": "allow"},
        "agent": {"build": {"permission": {"external_directory": {"/private/*": "deny"}}}},
    }
    obj._apply_tool_allowlist(config)
    assert config["agent"]["build"]["permission"]["external_directory"] == {"/private/*": "deny"}
    assert config["agent"]["build"]["permission"]["doom_loop"] == "allow"


@pytest.mark.parametrize("raw", ["null", "{}", '"bash"', "[1]", '["unknown"]', '["bash","bash"]', '["edit"]', "bad"])
def test_invalid_sample_tools(monkeypatch, raw):
    with pytest.raises(HTTPException):
        agent(monkeypatch)._tools_for_request(body(metadata={"allowed_tools": raw}))


@pytest.mark.parametrize("mode,tool", [("model_only", "websearch"), ("model_and_websearch", "webfetch")])
def test_sample_tools_cannot_exceed_network(monkeypatch, mode, tool):
    with pytest.raises(HTTPException):
        agent(monkeypatch)._tools_for_request(
            body(metadata={"network_access": mode, "allowed_tools": json.dumps([tool])})
        )


def test_sample_tools_respect_run_denies(monkeypatch):
    request = body(metadata={"allowed_tools": '["webfetch"]'})
    with pytest.raises(HTTPException):
        agent(monkeypatch, allowed_tools=DEFAULT_TOOLS)._tools_for_request(request)
    with pytest.raises(HTTPException):
        agent(monkeypatch, opencode_config={"tools": {"webfetch": False}})._tools_for_request(request)
    assert agent(monkeypatch)._tools_for_request(body(metadata={"allowed_tools": "[]"})) == []


@pytest.mark.asyncio
async def test_sample_tools_are_isolated_between_concurrent_training_episodes(monkeypatch, tmp_path):
    saved_database = database(tmp_path)
    seen = {}

    async def run(self, *args, artifact_path, rollout_id, **kwargs):
        self._write_opencode_config(artifact_path.parent, rollout_id)
        await asyncio.sleep(0)
        seen[rollout_id] = json.loads((artifact_path.parent / "opencode.json").read_text())
        shutil.copyfile(saved_database, artifact_path)

    monkeypatch.setattr(OpenCodeMultimodalAgent, "_run_opencode", run)
    obj = agent(monkeypatch, allowed_tools=DEFAULT_TOOLS + ["websearch", "webfetch"])
    profiles = [("model_only", []), ("model_and_websearch", ["websearch"]), ("inherit", ["websearch", "webfetch"])]
    await asyncio.gather(
        *[
            obj._create_episode(
                body(metadata={"network_access": mode, "allowed_tools": json.dumps(DEFAULT_TOOLS + extra)}),
                rollout_id=mode,
            )
            for mode, extra in profiles
        ]
    )
    for mode, extra in profiles:
        assert seen[mode]["tools"]["webfetch"] is ("webfetch" in extra)
        assert seen[mode]["tools"]["websearch"] is ("websearch" in extra)
    assert obj.config.allowed_tools == DEFAULT_TOOLS + ["websearch", "webfetch"]


def test_sample_tools_preserve_explicit_permission_denies(monkeypatch):
    request = body(metadata={"allowed_tools": '["bash"]'})
    for overrides in [{"permission": {"bash": "deny"}}, {"agent": {"build": {"permission": {"bash": "deny"}}}}]:
        with pytest.raises(HTTPException):
            agent(monkeypatch, opencode_config=overrides)._tools_for_request(request)
    obj = agent(monkeypatch, allowed_tools=["bash"])
    config = {"agent": {"plan": {"permission": {"bash": "deny"}}}}
    obj._apply_tool_allowlist(config)
    assert config["agent"]["plan"]["permission"]["bash"] == "deny"


@pytest.mark.parametrize("provider_name", ["opensandbox", "enroot"])
@pytest.mark.parametrize("mode", ["model_only", "model_and_websearch", "inherit"])
def test_owned_network_uses_upstream_policy_for_opensandbox(monkeypatch, provider_name, mode):
    from types import SimpleNamespace

    from nemo_gym.sandbox import SandboxSpec

    obj = agent(monkeypatch, network_access=mode)
    monkeypatch.setattr(type(obj), "resolve_model_base_url", lambda *args, **kwargs: "http://model.example:8000/v1")
    request = seed()
    request.sandbox_access = None
    request.tool_accesses = []
    spec = SandboxSpec(image="test-image", workdir="/workspace")
    result = obj._restrict_owned_sandbox_network(spec, SimpleNamespace(name=provider_name), request)
    assert obj.config.network_access == mode
    assert (
        obj._sandbox_network_access()
        == {"model_only": "model_only", "model_and_websearch": "model_and_tools", "inherit": "inherit"}[mode]
    )
    if provider_name == "enroot" or mode == "inherit":
        assert result is spec
    else:
        targets = ["model.example"] if mode == "model_only" else ["mcp.exa.ai", "model.example"]
        assert result.provider_options["network_policy"] == {
            "defaultAction": "deny",
            "egress": [{"action": "allow", "target": target} for target in targets],
        }
        assert "network_policy" not in (spec.provider_options or {})


def test_owned_opensandbox_conflicting_policy_is_terminal(visual_setup):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    obj, sandbox = visual_setup
    obj.config.network_access = "model_only"
    obj.config.sandbox_provider = "agent-provider"
    obj.config.sandbox_config = {
        "image": "test-image",
        "workdir": "/workspace",
        "provider_options": {"network_policy": {}},
    }
    request = seed()
    request.sandbox_access = None
    request.tool_accesses = []
    provider = SimpleNamespace(name="opensandbox", aclose=AsyncMock())
    with (
        patch("responses_api_agents.opencode_agent.app.create_provider", return_value=provider),
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox") as factory,
        TestClient(obj.setup_webserver()) as client,
    ):
        response = client.post("/v1/agent_sessions", json=request.model_dump(mode="json"))
    assert response.status_code == 422
    assert "not both" in response.text
    factory.assert_not_called()
    provider.aclose.assert_awaited_once()


@pytest.mark.parametrize("mode", ["model_only", "model_and_websearch"])
def test_restricted_network_rejects_required_external_mcp(monkeypatch, mode):
    from types import SimpleNamespace

    from nemo_gym.agents import AgentSeedRejectedError, AgentSeedSessionRequest
    from nemo_gym.sandbox import SandboxSpec
    from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import owned_seed_with_several_grants

    obj = agent(monkeypatch, network_access=mode)
    request = AgentSeedSessionRequest.model_validate(owned_seed_with_several_grants(required=True))
    with pytest.raises(AgentSeedRejectedError, match="required MCP"):
        obj._restrict_owned_sandbox_network(SandboxSpec(image="test-image"), SimpleNamespace(name="enroot"), request)


@pytest.mark.parametrize(
    "legacy,canonical", [("none", "model_only"), ("websearch", "model_and_websearch"), ("all", "inherit")]
)
def test_legacy_network_config_and_metadata_normalize(monkeypatch, legacy, canonical):
    obj = agent(monkeypatch, network_access=legacy)
    assert obj.config.network_access == canonical
    assert obj.config.model_dump()["network_access"] == canonical
    assert obj._network_access_for_request(body(metadata={"network_access": legacy})) == canonical
    assert obj._network_access_for_request(body(metadata={"network_access": canonical})) == canonical


def test_legacy_metadata_cannot_expand_canonical_ceiling(monkeypatch):
    obj = agent(monkeypatch, network_access="model_only")
    for mode in ("all", "websearch", "model_and_websearch", "inherit"):
        with pytest.raises(HTTPException) as error:
            obj._network_access_for_request(body(metadata={"network_access": mode}))
        assert error.value.status_code == 422


def test_arbitrary_tool_networking_is_not_silently_mapped(monkeypatch):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        config(network_access="model_and_tools")
    with pytest.raises(HTTPException) as error:
        agent(monkeypatch)._network_access_for_request(body(metadata={"network_access": "model_and_tools"}))
    assert error.value.status_code == 422


@pytest.mark.parametrize("provider_name", ["opensandbox", "enroot", "another-provider"])
@pytest.mark.parametrize(
    "policy",
    [
        {"defaultAction": "deny", "egress": []},
        {"defaultAction": "deny", "egress": [{"action": "allow", "target": "restricted.example"}]},
    ],
)
def test_inherit_preserves_existing_provider_restrictions(monkeypatch, provider_name, policy):
    from types import SimpleNamespace

    from nemo_gym.sandbox import SandboxSpec

    obj = agent(monkeypatch, network_access="inherit")
    monkeypatch.setattr(type(obj), "resolve_model_base_url", lambda *args, **kwargs: "http://model.example:8000/v1")
    request = seed()
    request.sandbox_access = None
    request.tool_accesses = []
    spec = SandboxSpec(image="test-image", provider_options={"network_policy": policy, "custom": "preserved"})
    result = obj._restrict_owned_sandbox_network(spec, SimpleNamespace(name=provider_name), request)
    assert result is spec
    assert result.provider_options == {"network_policy": policy, "custom": "preserved"}


@pytest.mark.parametrize("mode", ["inherit", "all"])
def test_inherit_runner_preserves_outer_network_namespace_and_proxy(tmp_path, mode):
    runner = Path(__file__).parents[1] / "network_runner.py"
    expected_ns = os.readlink("/proc/self/ns/net")
    probe = "import os,json; print(json.dumps([os.readlink('/proc/self/ns/net'),os.environ['HTTPS_PROXY']]))"
    result = subprocess.run(
        [sys.executable, str(runner), "--mode", mode, "--", sys.executable, "-c", probe],
        env=os.environ | {"HTTPS_PROXY": "http://existing-proxy.example:8080"},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert json.loads(result.stdout) == [expected_ns, "http://existing-proxy.example:8080"]


@pytest.mark.parametrize("overrides", [{"tools": {"websearch": False}}, {"permission": {"websearch": "deny"}}])
def test_search_network_mode_respects_explicit_tool_disable(monkeypatch, tmp_path, overrides):
    obj = evaluation(monkeypatch, tmp_path)
    obj.config.network_access = "model_and_websearch"
    obj.config.opencode_config = overrides
    obj._write_opencode_config(tmp_path)
    written = json.loads((tmp_path / "opencode.json").read_text())
    assert written["tools"]["websearch"] is False
    assert written["permission"]["websearch"] == "deny"
    assert obj._env(str(tmp_path))["OPENCODE_ENABLE_EXA"] == "0"
    assert obj.config.network_access == "model_and_websearch"


def test_search_network_mode_preserves_agent_specific_tool_disable(monkeypatch, tmp_path):
    obj = evaluation(monkeypatch, tmp_path)
    obj.config.network_access = "model_and_websearch"
    obj.config.opencode_config = {"agent": {"plan": {"permission": {"websearch": "deny"}}}}
    obj._write_opencode_config(tmp_path)
    written = json.loads((tmp_path / "opencode.json").read_text())
    assert written["tools"]["websearch"] is True
    assert written["agent"]["build"]["permission"]["websearch"] == "allow"
    assert written["agent"]["plan"]["permission"]["websearch"] == "deny"
