# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from nemo_gym.core import MediaFile, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import media_path
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import capture_observations, close_body
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import seed as base_seed
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import setup as opencode_setup  # noqa: F401
from responses_api_agents.opencode_multimodal_agent.app import OpenCodeMultimodalAgent, OpenCodeMultimodalAgentConfig
from responses_api_agents.opencode_multimodal_agent.rewards import OpenCodeToolRewardConfig
from responses_api_agents.opencode_multimodal_agent.tests.test_app import body, image_url


def seed():
    return base_seed().model_copy(update={"sandbox_access": None})


@pytest.fixture
def visual_setup(opencode_setup, tmp_path, monkeypatch):
    from responses_api_agents.opencode_agent import app as base

    original, sandbox = opencode_setup
    sandbox.provider_name = "opensandbox"
    sandbox.start = AsyncMock()
    base.create_provider.return_value.name = "opensandbox"
    base.create_provider.return_value.aclose = AsyncMock()
    monkeypatch.setattr(base, "AsyncSandbox", lambda provider: sandbox)
    cfg = OpenCodeMultimodalAgentConfig.model_validate(
        original.config.model_dump()
        | {
            "model": "vision",
            "tool_rewards": None,
            "system_prompt": "Inspect carefully.",
            "network_access": "inherit",
            "sandbox_config": {"image": "test-image", "workdir": "/app"},
        }
    )
    agent = OpenCodeMultimodalAgent(config=cfg, server_client=original.server_client)
    export = json.loads(sandbox.events)
    export["session_id"] = "root"
    export["messages"][0]["parts"].append({"type": "file", "mime": "image/png", "url": image_url()})
    export["messages"][1]["parts"][1]["state"].update(status="completed", metadata={"exit": 0})
    sandbox.events = json.dumps(export)
    capture_observations(sandbox, tmp_path)
    original_launch = sandbox.launch.side_effect

    async def launch_with_receipt(*args, **kwargs):
        result = await original_launch(*args, **kwargs)
        params = sandbox.params()
        config = json.loads(params["env"]["OPENCODE_CONFIG_CONTENT"])
        if any(path.endswith("/visual-final-answer.mjs") for path in config.get("plugin", [])):
            sandbox.files[sandbox.directory + "/visual-final-answer.ready"] = "ready\n"
        return result

    sandbox.launch.side_effect = launch_with_receipt

    return agent, sandbox


def test_session_stages_original_image_replays_and_closes(visual_setup):
    agent, sandbox = visual_setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        url = "/ng-rollout/opencode-smoke-a2/v1/responses"
        first = client.post(url, json=body().model_dump(mode="json"))
        assert first.status_code == 200, first.text
        retry = client.post(url, json=body().model_dump(mode="json"))
        assert retry.json() == first.json()
        sandbox.launch.assert_awaited_once()
        params = sandbox.params()
        policy = json.loads(params["env"]["OPENCODE_CONFIG_CONTENT"])
        assert policy["enabled_providers"] == ["nemo_gym"]
        assert set(policy["provider"]) == {"nemo_gym"}
        image = MediaFile.from_data_uri(image_url())
        assert sandbox.files[media_path(image)] == image.data
        assert media_path(image) in params["command"]
        assert policy["agent"]["build"]["temperature"] == 0.6
        assert policy["attachment"]["image"]["max_width"] == 2**31 - 1
        assert not any("network_runner.py" in path for path in sandbox.files)
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text
    sandbox.stop.assert_awaited()


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("exit_code", [0, 1, None])
def test_capture_routing_and_native_reward_evidence(visual_setup, capture, exit_code):
    agent, sandbox = visual_setup
    agent.config.token_id_capture = capture
    agent.server_client.global_config_dict.token_id_capture = {"enabled": capture}
    agent.config.tool_rewards = OpenCodeToolRewardConfig()
    export = json.loads(sandbox.events)
    export["messages"][1]["parts"][1]["state"]["metadata"] = {"exit": exit_code}
    sandbox.events = json.dumps(export)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=body().model_dump(mode="json"))
        assert result.status_code == 200, result.text
        assert "opencode_evidence" in result.json(), result.text
        evidence = result.json()["opencode_evidence"]["tool_reward"]
        assert evidence["bash_success_count"] == int(exit_code == 0)
        assert evidence["tool_error_count"] == int(exit_code != 0)
        config = json.loads(sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"])
        assert ("training-token-capture" in config["provider"]["nemo_gym"]["options"]["baseURL"]) is capture
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )


@pytest.mark.parametrize("provider", ["local", "enroot"])
def test_non_opensandbox_provider_is_rejected_before_allocation(visual_setup, provider):
    from responses_api_agents.opencode_agent import app as base

    agent, sandbox = visual_setup
    base.create_provider.return_value.name = provider
    with TestClient(agent.setup_webserver()) as client:
        result = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert result.status_code == 422
    sandbox.start.assert_not_awaited()


def test_borrowed_session_is_rejected(visual_setup):
    agent, sandbox = visual_setup
    with TestClient(agent.setup_webserver()) as client:
        result = client.post("/v1/agent_sessions", json=base_seed().model_dump(mode="json"))
    assert result.status_code == 422
    sandbox.start.assert_not_awaited()


@pytest.mark.parametrize(
    "url", ["https://example.com/a.png", "file:///tmp/a.png", "data:image/png;base64,aW52YWxpZA=="]
)
def test_unsupported_images_are_rejected(visual_setup, url):
    agent, _ = visual_setup
    params = body().model_dump()
    params["input"][0]["content"][1]["image_url"] = url
    with pytest.raises(HTTPException) as error:
        agent._session_input(NeMoGymResponseCreateParamsNonStreaming.model_validate(params))
    assert error.value.status_code == 422


def test_large_image_is_decoded_once_and_preserved(visual_setup, monkeypatch):
    agent, _ = visual_setup
    stream = io.BytesIO()
    Image.new("RGB", (9001, 2), "red").save(stream, format="PNG")
    params = body().model_dump()
    params["input"][0]["content"][1]["image_url"] = (
        "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()
    )
    from unittest.mock import patch

    with patch.object(MediaFile, "from_data_uri", wraps=MediaFile.from_data_uri) as decode:
        _, _, media = agent._session_input(NeMoGymResponseCreateParamsNonStreaming.model_validate(params))
    assert decode.call_count == 1
    assert media[0].data == stream.getvalue()


def test_truncated_image_returns_422(visual_setup):
    agent, _ = visual_setup
    params = body().model_dump()
    params["input"][0]["content"][1]["image_url"] = (
        "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").decode()
    )
    with pytest.raises(HTTPException) as error:
        agent._session_input(NeMoGymResponseCreateParamsNonStreaming.model_validate(params))
    assert error.value.status_code == 422


@pytest.mark.parametrize("key", ["network_access", "allowed_tools"])
def test_request_cannot_change_creation_time_profiles(visual_setup, key):
    agent, _ = visual_setup
    with pytest.raises(HTTPException):
        agent._session_input(body(metadata={key: "inherit"}))


def test_staged_final_answer_plugin_loads_and_blocks_tools(visual_setup, tmp_path):
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the staged plugin")
    agent, sandbox = visual_setup
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        reply = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=body().model_dump(mode="json"))
        assert reply.status_code == 200, reply.text
        plugin_source = next(data for path, data in sandbox.files.items() if path.endswith("/visual-final-answer.mjs"))
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )
    plugin = tmp_path / "plugin.mjs"
    plugin.write_text(plugin_source)
    script = """
const module = await import(process.argv[1]);
const hooks = await module.default();
await hooks["experimental.chat.messages.transform"]({}, {messages: [
 {info: {role: "user", sessionID: "s"}},
 ...Array.from({length: 8}, () => ({info: {role: "assistant", sessionID: "s", time: {completed: 1}, tokens: {total: 100}}}))
]});
try { await hooks["tool.execute.before"]({sessionID: "s"}); process.exit(3); }
catch (error) { if (!error.message.includes("No tools")) throw error; }
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", script, plugin.as_uri()], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_missing_final_answer_policy_receipt_cannot_return_a_success(visual_setup):
    agent, sandbox = visual_setup
    original_launch = sandbox.launch.side_effect

    async def launch_without_receipt(*args, **kwargs):
        result = await original_launch(*args, **kwargs)
        sandbox.files.pop(sandbox.directory + "/visual-final-answer.ready", None)
        return result

    sandbox.launch.side_effect = launch_without_receipt
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        reply = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=body().model_dump(mode="json"))
        assert reply.status_code >= 400
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )


@pytest.mark.parametrize(
    "network,permission,enabled", [("inherit", "allow", "1"), ("inherit", "deny", "0"), ("model_only", "allow", "0")]
)
async def test_native_search_requires_explicit_permission_and_network(visual_setup, network, permission, enabled):
    from types import SimpleNamespace

    agent, sandbox = visual_setup
    agent.config.network_access = network
    agent.config.final_answer_reserve = None
    state = SimpleNamespace(session=SimpleNamespace(session_dir="/session", sandbox=sandbox))
    payload = {
        "env": {
            "OPENCODE_CONFIG_CONTENT": json.dumps(
                {
                    "model": "nemo_gym/dummy",
                    "provider": {"nemo_gym": {"models": {agent.config.sandbox_model_id: {"limit": {}}}}},
                    "permission": {"websearch": permission},
                }
            )
        }
    }
    await agent._prepare_session_launch(state, body(), payload)
    assert payload["env"]["OPENCODE_ENABLE_EXA"] == enabled


@pytest.mark.parametrize("image_input", [False, True])
@pytest.mark.parametrize("string_input", [False, True])
def test_text_prompt_uses_shared_conditional_image_declaration(visual_setup, image_input, string_input):
    agent, sandbox = visual_setup
    agent.config.image_input = image_input
    export = json.loads(sandbox.events)
    # No initial image; a tool can attach one later without being mistaken for a dropped prompt image.
    export["messages"][0]["parts"] = [p for p in export["messages"][0]["parts"] if p.get("type") != "file"]
    if image_input:
        export["messages"][1]["parts"][1]["state"]["attachments"] = [
            {"type": "file", "mime": "image/png", "url": image_url()}
        ]
    sandbox.events = json.dumps(export)
    request = body(input="Inspect the task" if string_input else [{"role": "user", "content": "Inspect the task"}])
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=request.model_dump(mode="json"))
        assert result.status_code == 200, result.text
        launch = sandbox.params()
        model_config = json.loads(launch["env"]["OPENCODE_CONFIG_CONTENT"])
        model = model_config["provider"]["nemo_gym"]["models"][agent.config.sandbox_model_id]
        assert ("modalities" in model) is image_input
        if image_input:
            assert "image" in model["modalities"]["input"]
            assert model_config["attachment"]["image"]["max_width"] == 2**31 - 1
        else:
            assert "attachment" not in model_config
        assert "--file" not in launch["command"]
        assert not any(path.startswith("/tmp/nemo-gym-media/") for path in sandbox.files)
        retry = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=request.model_dump(mode="json"))
        assert retry.json() == result.json()
        sandbox.launch.assert_awaited_once()
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )
