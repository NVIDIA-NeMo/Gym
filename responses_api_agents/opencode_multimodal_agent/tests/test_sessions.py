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
def visual_setup(opencode_setup, tmp_path):
    original, sandbox = opencode_setup
    config = OpenCodeMultimodalAgentConfig(
        **(
            original.config.model_dump()
            | {
                "model": "vision-model",
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
        assert "Inspect carefully.\n\nCount red objects" in payload["prompt"]
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        assert config["agent"]["build"]["temperature"] == 0.6
        assert config["agent"]["build"]["top_p"] == 0.95
        provider = config["provider"]["nemo"]
        assert "/ng-rollout/opencode-smoke-a2/" in provider["options"]["baseURL"]
        assert provider["models"]["vision-model"]["limit"]["output"] == 2048
        assert any(path.endswith(".visual-final-answer.mjs") for path in sandbox.files)
        image_path = next(path for path in sandbox.files if path.endswith("input-0.png"))
        assert image_path.startswith(sandbox.directory + "/")
        assert not any(path.startswith("/app/") for path in sandbox.files)
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200 and closed.json()["agent_observations"] is not None
        assert client.post(path, json=request_body()).status_code == 409
    sandbox.stop.assert_not_awaited()
    sandbox.disconnect.assert_awaited_once()


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
