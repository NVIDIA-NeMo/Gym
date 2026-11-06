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
    assert default.network_access == "none"
    assert OpenCodeMultimodalEvalConfig.model_validate(default.model_dump()).network_access == "none"


@pytest.mark.parametrize("mode", ["none", "websearch", "all"])
def test_network_policy_preserves_all_and_restricts_web_tools(monkeypatch, tmp_path, mode):
    agent = evaluation(monkeypatch, tmp_path)
    agent.config.network_access = mode
    agent.config.opencode_config["tools"] = {"webfetch": True, "websearch": True}
    agent._write_opencode_config(tmp_path)
    config = json.loads((tmp_path / "opencode.json").read_text())
    assert config["tools"]["webfetch"] is (mode == "all")
    assert config["tools"]["websearch"] is (mode != "none")
    assert config["permission"]["bash"]["*pkill*"] == "deny"
    assert config["agent"]["build"]["prompt"] == "Exact combined prompt\n"
    if mode != "all":
        for agent_config in config["agent"].values():
            assert agent_config["permission"]["webfetch"] == "deny"
            assert agent_config["permission"]["websearch"] == ("allow" if mode == "websearch" else "deny")


@pytest.mark.parametrize("mode", ["none", "websearch"])
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
            "none",
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


@pytest.mark.parametrize("ceiling", ["none", "websearch", "all"])
@pytest.mark.parametrize("selected", [None, "none", "websearch", "all", "invalid"])
def test_sample_policy_and_run_ceiling(monkeypatch, ceiling, selected):
    obj = agent(monkeypatch, network_access=ceiling)
    request = body(metadata={"network_access": selected} if selected is not None else None)
    levels = {"none": 0, "websearch": 1, "all": 2}
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
    obj = agent(monkeypatch, network_access="all")
    await asyncio.gather(
        *[
            obj._create_episode(body(metadata={"network_access": mode}), rollout_id=mode)
            for mode in ("none", "websearch", "all")
        ]
    )
    assert obj.config.network_access == "all"
    for mode, (effective, config) in seen.items():
        assert effective == mode
        if mode != "all":
            assert config["tools"]["websearch"] is (mode == "websearch")
            assert config["tools"]["webfetch"] is False


def test_session_consumes_only_its_network_metadata(monkeypatch, tmp_path):
    obj = evaluation(monkeypatch, tmp_path)
    request = body(metadata={"network_access": "websearch"})
    assert obj._session_input(request) == obj._session_input(body())
    assert request.metadata == {"network_access": "websearch"}
    with pytest.raises(HTTPException):
        obj._session_input(body(metadata={"network_access": "websearch", "unknown_control": "bad"}))


DEFAULT_TOOLS = ["bash", "edit", "glob", "grep", "read", "task", "todowrite", "write"]


@pytest.mark.parametrize(
    "mode,extra", [("none", []), ("websearch", ["websearch"]), ("all", ["websearch", "webfetch"])]
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


@pytest.mark.parametrize("mode,tool", [("none", "websearch"), ("websearch", "webfetch")])
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
    profiles = [("none", []), ("websearch", ["websearch"]), ("all", ["websearch", "webfetch"])]
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
