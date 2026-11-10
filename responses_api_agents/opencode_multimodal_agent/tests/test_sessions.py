# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import (
    capture_observations,
    close_body,
    seed,
)
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import setup as opencode_setup  # noqa: F401
from responses_api_agents.opencode_multimodal_agent.app import OpenCodeMultimodalAgent, OpenCodeMultimodalAgentConfig
from responses_api_agents.opencode_multimodal_agent.eval import (
    OpenCodeMultimodalEvalAgent,
    OpenCodeMultimodalEvalConfig,
)


def image_url():
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def request_body():
    return {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Count red objects"},
                    {"type": "input_image", "image_url": image_url(), "detail": "auto"},
                ],
            }
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_output_tokens": 2048,
    }


@pytest.fixture
def visual_setup(opencode_setup, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from responses_api_agents.opencode_agent import app as base_app
    from responses_api_agents.opencode_multimodal_agent.sessions import _NETWORK_RUNTIME

    monkeypatch.setattr(type(_NETWORK_RUNTIME), "is_installed", AsyncMock(return_value=True))
    # Owned cases use local-provider setup semantics; OpenSandbox's offline
    # contract is exercised explicitly below.
    base_app.create_provider.return_value.name = "local"
    original, sandbox = opencode_setup
    sandbox.provider_name = "local"
    config = OpenCodeMultimodalAgentConfig(
        **(
            original.config.model_dump()
            | {
                "model": "vision-model",
                # This fixture supplies a borrowed fake sandbox, with no namespace support.
                "network_access": "inherit",
                "system_prompt": "Inspect carefully.",
                "tool_rewards": None,
            }
        )
    )
    agent = OpenCodeMultimodalAgent(config=config, server_client=original.server_client)
    original_upload = sandbox.upload

    async def upload(source, destination):
        if str(source).endswith(".png"):
            sandbox.files[destination] = Path(source).read_bytes()
        else:
            await original_upload(source, destination)

    sandbox.upload = upload
    export = json.loads(sandbox.events)
    export["session_id"] = "root"
    export["messages"][0]["parts"].append({"type": "file", "mime": "image/png", "url": image_url()})
    tool = export["messages"][1]["parts"][1]
    tool["state"].update(status="completed", attachments=[{"mime": "image/png", "url": image_url()}])
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    return agent, sandbox


def test_restricted_opensandbox_missing_bundle_refuses_before_allocation(visual_setup):
    from unittest.mock import AsyncMock, patch

    from responses_api_agents.opencode_agent import app as base_app

    agent, sandbox = visual_setup
    base_app.create_provider.return_value.name = "opensandbox"
    base_app.create_provider.return_value.aclose = AsyncMock()
    agent.config.network_access = "model_only"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/app"}
    sandbox.start = AsyncMock()
    body = seed()
    body.sandbox_access = None
    body.tool_accesses = []
    with patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox):
        with TestClient(agent.setup_webserver()) as client:
            result = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
    assert result.status_code == 422, result.text
    assert "pinned offline runtime bundle" in result.text
    sandbox.start.assert_not_awaited()
    sandbox.exec.assert_not_awaited()
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("asset", ["executable", "plugin_dependencies", "ripgrep"])
@pytest.mark.parametrize("field", ["path", "sha256"])
def test_restricted_opensandbox_requires_each_offline_pin(visual_setup, asset, field):
    from types import SimpleNamespace

    from nemo_gym.agents import AgentSeedRejectedError
    from nemo_gym.sandbox import SandboxSpec

    agent, _ = visual_setup
    agent.config.network_access = "model_only"
    for item in ("executable", "plugin_dependencies", "ripgrep"):
        setattr(agent.config, f"sandbox_{item}_path", f"/opt/runtime/{item}")
        setattr(agent.config, f"sandbox_{item}_sha256", "a" * 64)
    missing = f"sandbox_{asset}_{field}"
    setattr(agent.config, missing, None)
    with pytest.raises(AgentSeedRejectedError, match=missing):
        agent._restrict_owned_sandbox_network(SandboxSpec(image="test"), SimpleNamespace(name="opensandbox"), seed())


def test_offline_bundle_preserves_opensandbox_model_only_policy(visual_setup):
    from types import SimpleNamespace

    from nemo_gym.sandbox import SandboxSpec

    agent, _ = visual_setup
    agent.config.network_access = "model_only"
    for item in ("executable", "plugin_dependencies", "ripgrep"):
        setattr(agent.config, f"sandbox_{item}_path", f"/opt/runtime/{item}")
        setattr(agent.config, f"sandbox_{item}_sha256", "a" * 64)
    body = seed()
    body.tool_accesses = []
    spec = agent._restrict_owned_sandbox_network(SandboxSpec(image="test"), SimpleNamespace(name="opensandbox"), body)
    assert spec.provider_options["network_policy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "model.example"}],
    }


@pytest.mark.parametrize("mode", ["model_only", "model_and_tools", "inherit"])
@pytest.mark.parametrize("narrow_request", [False, True])
def test_owned_opensandbox_uses_creation_policy_without_nested_wrapper(
    visual_setup, monkeypatch, mode, narrow_request
):
    from unittest.mock import AsyncMock, patch

    from responses_api_agents.opencode_agent import app as base_app
    from responses_api_agents.opencode_multimodal_agent.sessions import _NETWORK_RUNTIME

    if narrow_request and mode == "model_only":
        pytest.skip("No narrower network mode")
    agent, sandbox = visual_setup
    base_app.create_provider.return_value.name = "opensandbox"
    sandbox.provider_name = "opensandbox"
    sandbox.start = AsyncMock()
    agent.config.network_access = mode
    agent.config.sandbox_config = {"image": "offline-image", "workdir": "/app"}
    for asset in ("executable", "plugin_dependencies", "ripgrep"):
        setattr(agent.config, f"sandbox_{asset}_path", f"/opt/runtime/{asset}")
        setattr(agent.config, f"sandbox_{asset}_sha256", "a" * 64)
    install = AsyncMock(side_effect=AssertionError("OpenSandbox needs no nested namespace runtime"))
    monkeypatch.setattr(type(_NETWORK_RUNTIME), "install", install)
    monkeypatch.setattr(type(_NETWORK_RUNTIME), "is_installed", AsyncMock(return_value=False))
    body = seed()
    body.sandbox_access = None
    body.tool_accesses = []
    with (
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox),
        patch.object(type(agent), "_stage_archive", AsyncMock()),
        TestClient(agent.setup_webserver()) as client,
    ):
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        request = request_body()
        if narrow_request:
            request["metadata"] = {"network_access": "model_only"}
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=request)
        if narrow_request:
            assert response.status_code == 422, response.text
            assert "fixed at sandbox creation" in response.text
            sandbox.launch.assert_not_awaited()
        else:
            assert response.status_code == 200, response.text
            assert not any("network_runner.py" in arg for arg in sandbox.params()["command"])
        assert not any(path.endswith("network_runner.py") for path in sandbox.files)
        install.assert_not_awaited()
        assert not any("--model-url" in call.args[0] for call in sandbox.exec.call_args_list)
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
        sandbox.stop.assert_awaited()


@pytest.mark.parametrize("error_type", [None, "timeout"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_namespace_preflight_failure_closes_before_activation(visual_setup, error_type, cleanup_fails):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    agent, sandbox = visual_setup
    agent.config.network_access = "model_only"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/app"}
    sandbox.start = AsyncMock()
    original_exec = sandbox.run_exec

    async def execute(command, **kwargs):
        if "network_runner.py" in command and "--model-url" in command:
            return SimpleNamespace(
                return_code=1 if error_type is None else 0,
                error_type=error_type,
                stdout="",
                stderr="unshare unavailable or forbidden",
            )
        return await original_exec(command, **kwargs)

    sandbox.exec.side_effect = execute
    if cleanup_fails:
        sandbox.stop = AsyncMock(side_effect=[RuntimeError("cleanup failed"), None])
    body = seed()
    body.sandbox_access = None
    body.tool_accesses = []
    with patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox):
        with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
            result = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
            assert result.status_code >= 400
            sandbox.launch.assert_not_awaited()
            sandbox.stop.assert_awaited()
    if cleanup_fails:
        assert sandbox.stop.await_count >= 2


def test_session_stages_images_preserves_outputs_and_replays_once(visual_setup):
    agent, sandbox = visual_setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        path = "/ng-rollout/opencode-smoke-a2/v1/responses"
        first = client.post(path, json=request_body())
        assert first.status_code == 200, first.text
        retry = client.post(path, json=request_body())
        assert retry.status_code == 200 and retry.json() == first.json()
        sandbox.launch.assert_awaited_once()
        response = first.json()
        assert response["status"] == "completed"
        assert response["output"][0]["type"] == "reasoning"
        output = next(item for item in response["output"] if item["type"] == "function_call_output")
        assert output["output"][-1]["image_url"] == image_url()
        assert response["opencode_evidence"]["turns_used"] == 1
        payload = sandbox.params()
        assert payload["env"]["OPENCODE_ENABLE_EXA"] == "0"
        assert payload["prompt"] == "Count red objects"
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        assert config["agent"]["build"]["temperature"] == 0.6
        assert config["agent"]["build"]["top_p"] == 0.95
        provider = config["provider"]["nemo"]
        assert "/ng-rollout/opencode-smoke-a2/" in provider["options"]["baseURL"]
        assert provider["models"]["vision-model"]["limit"]["output"] == 2048
        assert any(path.endswith(".visual-final-answer.mjs") for path in sandbox.files)
        from nemo_gym.core import MediaFile
        from nemo_gym.sandbox import media_path

        image_path = media_path(MediaFile.from_data_uri(image_url()))
        assert sandbox.files[image_path] == MediaFile.from_data_uri(image_url()).data
        assert image_path in payload["command"]
        assert not any(path.startswith("/app/") for path in sandbox.files)
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200 and closed.json()["agent_observations"] is not None
        assert client.post(path, json=request_body()).status_code == 409
    sandbox.stop.assert_not_awaited()
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("webfetch", [False, True])
@pytest.mark.parametrize("websearch", [False, True])
def test_session_web_tool_policy_reaches_runtime(visual_setup, webfetch, websearch):
    original, sandbox = visual_setup
    tools = {"webfetch": webfetch, "websearch": websearch}
    config = OpenCodeMultimodalEvalConfig.model_validate(
        original.config.model_dump()
        | {
            "entrypoint": "eval.py",
            "final_answer_reserve": None,
            "opencode_config": original.config.opencode_config | {"tools": tools},
        }
    )
    agent = OpenCodeMultimodalEvalAgent(config=config, server_client=original.server_client)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=request_body())
        assert response.status_code == 200, response.text
        payload = sandbox.params()
        assert payload["env"]["OPENCODE_ENABLE_EXA"] == ("1" if websearch else "0")
        policy = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        for tool, enabled in tools.items():
            assert policy["tools"][tool] is enabled
            assert policy["permission"][tool] == ("allow" if enabled else "deny")
        assert policy["permission"]["bash"]["*pkill*"] == "deny"
        assert "*" not in policy["permission"]
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", None),
        ("max_output_tokens", 999999),
        ("tools", [{"type": "function", "name": "extra", "parameters": {}, "strict": False}]),
    ],
)
def test_invalid_request_does_not_launch(visual_setup, field, value):
    agent, sandbox = visual_setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=request_body() | {field: value})
        assert response.status_code == 422, response.text
        sandbox.launch.assert_not_awaited()
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )


@pytest.mark.parametrize("field,value", [("tool_rewards", {}), ("token_id_capture", True)])
def test_session_rejects_training_controls_before_allocation(visual_setup, field, value):
    agent, sandbox = visual_setup
    setattr(agent.config, field, value)
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert response.status_code == 422, response.text
        assert "evaluation-only" in response.text
        sandbox.launch.assert_not_awaited()
        sandbox.exec.assert_not_awaited()


def test_upload_failure_is_not_relaunched_and_session_can_close(visual_setup):
    from unittest.mock import AsyncMock

    agent, sandbox = visual_setup
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200
        original_upload = sandbox.upload
        sandbox.upload = AsyncMock(side_effect=OSError("sandbox transfer failed"))
        path = "/ng-rollout/opencode-smoke-a2/v1/responses"
        assert client.post(path, json=request_body()).status_code == 500
        assert client.post(path, json=request_body()).status_code == 500
        sandbox.upload.assert_awaited_once()
        sandbox.launch.assert_not_awaited()
        sandbox.upload = original_upload
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )
    sandbox.disconnect.assert_awaited_once()


def test_missing_initial_image_cannot_produce_success(visual_setup):
    agent, sandbox = visual_setup
    export = json.loads(sandbox.events)
    export["messages"][0]["parts"] = [part for part in export["messages"][0]["parts"] if part["type"] != "file"]
    sandbox.events = json.dumps(export)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=request_body())
        assert response.status_code == 502, response.text
        assert "dropped initial images" in response.text
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )


@pytest.mark.parametrize("owned", [False, True])
def test_visual_session_releases_only_its_owned_sandbox(visual_setup, owned):
    from unittest.mock import AsyncMock, patch

    agent, sandbox = visual_setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/agent-workspace"}
    sandbox.expected_workdir = "/agent-workspace" if owned else "/app"
    body = seed()
    if owned:
        body.sandbox_access = None
    sandbox.start = AsyncMock()
    with (
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox) as factory,
        patch("responses_api_agents.opencode_agent.app.resolve_provider_config"),
        TestClient(agent.setup_webserver()) as client,
    ):
        factory.connect = AsyncMock(return_value=sandbox)
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=request_body())
        assert response.status_code == 200, response.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
        again = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert again.json() == closed.json()
    (sandbox.stop if owned else sandbox.disconnect).assert_awaited_once()
    (sandbox.disconnect if owned else sandbox.stop).assert_not_awaited()


@pytest.mark.parametrize("mode", ["model_only", "model_and_tools", "inherit"])
@pytest.mark.parametrize("websearch", [False, True])
def test_owned_session_network_and_websearch_policy(visual_setup, monkeypatch, mode, websearch):
    from unittest.mock import AsyncMock, patch

    from responses_api_agents.opencode_multimodal_agent.sessions import _NETWORK_RUNTIME

    installed = AsyncMock(return_value=False)
    install = AsyncMock()
    monkeypatch.setattr(type(_NETWORK_RUNTIME), "is_installed", installed)
    monkeypatch.setattr(type(_NETWORK_RUNTIME), "install", install)
    original, sandbox = visual_setup
    agent = OpenCodeMultimodalEvalAgent(
        config=OpenCodeMultimodalEvalConfig.model_validate(
            original.config.model_dump() | {"entrypoint": "eval.py", "final_answer_reserve": None}
        ),
        server_client=original.server_client,
    )
    agent.config.network_access = mode
    agent.config.opencode_config["tools"] = {"websearch": websearch, "webfetch": False}
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/agent-workspace"}
    sandbox.expected_workdir = "/agent-workspace"
    sandbox.start = AsyncMock()
    request = seed()
    request.sandbox_access = None
    request.tool_accesses = []
    with (
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox),
        patch("responses_api_agents.opencode_agent.app.resolve_provider_config"),
        TestClient(agent.setup_webserver()) as client,
    ):
        created = client.post("/v1/agent_sessions", json=request.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=request_body())
        assert response.status_code == 200, response.text
        payload = sandbox.params()
        enabled = websearch and mode == "inherit"
        assert payload["env"]["OPENCODE_ENABLE_EXA"] == ("1" if enabled else "0")
        policy = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        assert policy["tools"]["websearch"] is enabled
        assert policy["permission"]["websearch"] == ("allow" if enabled else "deny")
        install.assert_awaited_once()
        assert payload["command"][:2] == [_NETWORK_RUNTIME.python, "-I"]
        assert payload["command"][3:5] == ["--mode", mode]
        assert "--protect-url" in payload["command"]
        probes = [c for c in sandbox.exec.call_args_list if "--model-url" in c.args[0]]
        assert len(probes) == 1
        assert probes[0].kwargs["timeout_s"] == 30
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("mode", ["model_only", "inherit"])
def test_private_runtime_setup_failure_never_activates(visual_setup, monkeypatch, cleanup_fails, mode):
    from unittest.mock import AsyncMock, patch

    from responses_api_agents.opencode_multimodal_agent.sessions import _NETWORK_RUNTIME

    agent, sandbox = visual_setup
    agent.config.network_access = mode
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/agent-workspace"}
    sandbox.expected_workdir = "/agent-workspace"
    sandbox.start = AsyncMock()
    monkeypatch.setattr(type(_NETWORK_RUNTIME), "is_installed", AsyncMock(return_value=False))
    install = AsyncMock(side_effect=RuntimeError("scripted runtime install failure"))
    monkeypatch.setattr(type(_NETWORK_RUNTIME), "install", install)
    if cleanup_fails:
        sandbox.stop = AsyncMock(side_effect=[RuntimeError("scripted cleanup failure"), None])
    body = seed()
    body.sandbox_access = None
    body.tool_accesses = []
    with (
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox),
        patch("responses_api_agents.opencode_agent.app.resolve_provider_config"),
        TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client,
    ):
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code >= 400
        install.assert_awaited_once()
        sandbox.launch.assert_not_awaited()
        sandbox.stop.assert_awaited()
    if cleanup_fails:
        assert sandbox.stop.await_count >= 2


@pytest.mark.parametrize("mode", ["model_only", "model_and_tools"])
def test_session_wrapper_uses_only_effective_mcp_grants(visual_setup, monkeypatch, mode):
    from unittest.mock import AsyncMock, patch

    from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import owned_seed_with_several_grants
    from responses_api_agents.opencode_multimodal_agent.sessions import _NETWORK_RUNTIME

    monkeypatch.setattr(type(_NETWORK_RUNTIME), "is_installed", AsyncMock(return_value=False))
    install = AsyncMock()
    monkeypatch.setattr(type(_NETWORK_RUNTIME), "install", install)
    agent, sandbox = visual_setup
    agent.config.sandbox_provider = "agent-provider"
    agent.config.sandbox_config = {"image": "test-image", "workdir": "/agent-workspace"}
    sandbox.expected_workdir = "/agent-workspace"
    sandbox.start = AsyncMock()
    request = owned_seed_with_several_grants(required=False)
    with (
        patch("responses_api_agents.opencode_agent.app.AsyncSandbox", return_value=sandbox),
        patch("responses_api_agents.opencode_agent.app.resolve_provider_config"),
        TestClient(agent.setup_webserver()) as client,
    ):
        created = client.post("/v1/agent_sessions", json=request)
        assert created.status_code == 200, created.text
        install.assert_awaited_once()
        body = request_body() | {"metadata": {"network_access": mode}}
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=body)
        assert response.status_code == 200, response.text
        install.assert_awaited_once()
        command = sandbox.params()["command"]
        assert sum("--model-url" in c.args[0] for c in sandbox.exec.call_args_list) == 1
        urls = [command[index + 1] for index, arg in enumerate(command) if arg == "--tool-url"]
        assert urls == (
            ["http://resources.example:8000/search/mcp", "http://docs.example:9000/mcp"]
            if mode == "model_and_tools"
            else []
        )
        client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))


@pytest.mark.parametrize("qualified", [False, True])
def test_tool_images_follow_call_identity_without_crossing_invocations(visual_setup, qualified):
    from nemo_gym.core import NeMoGymFunctionCallOutput
    from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ToolCallObservation

    agent, sandbox = visual_setup
    parent_id = "normalized-parent" if qualified else "tool-1"
    child_id = "normalized-child"
    parent = AgentInvocation(
        invocation_id="root",
        conversation=[NeMoGymFunctionCallOutput(call_id=parent_id, output="parent result")],
    )
    child = AgentInvocation(
        invocation_id="child",
        conversation=[NeMoGymFunctionCallOutput(call_id=child_id, output="child result")],
    )
    observations = AgentObservationBundle(
        source="opencode",
        schema_version="2.0" if qualified else "1.0",
        records=[
            parent,
            child,
            ToolCallObservation(
                invocation_id="root",
                tool_call_id=parent_id,
                native_tool_call_id="tool-1" if qualified else None,
            ),
            # Deliberately reuse the native ID after the parent record.
            ToolCallObservation(invocation_id="child", tool_call_id=child_id, native_tool_call_id="tool-1"),
        ],
    )
    output = agent._session_output(json.loads(sandbox.events), observations)
    assert len(output) == 1
    assert output[0].call_id == parent_id
    assert output[0].output == [
        {"type": "input_text", "text": "parent result"},
        {"type": "input_image", "image_url": image_url(), "detail": "auto"},
    ]
    assert parent.conversation[0].output == "parent result"
    assert child.conversation[0].output == "child result"


def test_session_decodes_once_and_preserves_image_bytes(visual_setup):
    from unittest.mock import patch

    from nemo_gym.core import MediaFile, NeMoGymResponseCreateParamsNonStreaming

    agent, _ = visual_setup
    agent = agent.model_copy(update={"config": agent.config.model_copy(update={"preserve_image_format": True})})
    body = request_body()
    # The shared decoder trusts the bytes rather than a mislabeled MIME header.
    body["input"][0]["content"][1]["image_url"] = image_url().replace("image/png", "image/jpeg")
    with patch.object(MediaFile, "from_data_uri", wraps=MediaFile.from_data_uri) as decode:
        prompt, _, media = agent._session_input(NeMoGymResponseCreateParamsNonStreaming.model_validate(body))
    assert decode.call_count == 1
    assert prompt == "Count red objects"
    assert media == [MediaFile.from_data_uri(image_url())]


@pytest.mark.parametrize("source", ["https://example.com/image.png", "file:///tmp/image.png"])
def test_session_rejects_non_inline_images(visual_setup, source):
    from fastapi import HTTPException

    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

    agent, sandbox = visual_setup
    body = request_body()
    body["input"][0]["content"][1]["image_url"] = source
    with pytest.raises(HTTPException) as error:
        agent._session_input(NeMoGymResponseCreateParamsNonStreaming.model_validate(body))
    assert error.value.status_code == 422
    sandbox.launch.assert_not_awaited()
