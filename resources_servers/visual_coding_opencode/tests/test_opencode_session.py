# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The example rows in an opencode_agent sandbox session, with the OpenCode settings of this environment's config."""

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import Sandbox, close_body, seed


SERVER = Path(__file__).resolve().parents[1]
SESSION_CONFIG = SERVER / "configs" / "visual_coding_opencode.yaml"
EXAMPLE_ROWS = SERVER / "data" / "example.jsonl"
# The settings OpenCode runs with inside the sandbox. The provider, image and model name come from the run.
SESSION_KEYS = (
    "opencode_version",
    "thinking",
    "timeout",
    "context_window",
    "max_output_tokens",
    "image_input",
    "opencode_config",
)


def session_agent_config() -> dict:
    config = yaml.safe_load(SESSION_CONFIG.read_text())
    return config["visual_coding_opencode_agent"]["responses_api_agents"]["opencode_agent"]


def example_rows() -> list[dict]:
    return [json.loads(line) for line in EXAMPLE_ROWS.read_text().splitlines() if line.strip()]


@pytest.fixture
def setup():
    sandbox = Sandbox()
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {"policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}}
    )
    client._build_server_base_url.return_value = "http://model.example:9000"
    settings = session_agent_config()
    config = OpenCodeAgentConfig(
        name="visual_coding_opencode_agent",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        model_server={"type": "responses_api_models", "name": "policy"},
        resources_server={"type": "resources_servers", "name": "visual_coding_opencode"},
        sandbox_provider="unused",
        sandbox_config={},
        session_close_timeout_seconds=1,
        **{key: settings[key] for key in SESSION_KEYS},
    )
    module = "responses_api_agents.opencode_agent.app"
    with (
        patch(
            f"{module}.ensure_opencode", side_effect=AssertionError("Sandbox sessions must not install on the host")
        ),
        patch(f"{module}.resolve_provider_config"),
        patch(f"{module}.get_global_config_dict", return_value={}),
        patch(f"{module}.create_provider") as provider_factory,
        patch(f"{module}.AsyncSandbox.connect", AsyncMock(return_value=sandbox)),
        patch("responses_api_agents.opencode_agent.sandbox.snapshot", sandbox.snapshot),
    ):
        provider_factory.return_value.name = "opensandbox"
        agent = OpenCodeAgent(config=config, server_client=client)
        yield agent, sandbox


@pytest.mark.parametrize("row", example_rows(), ids=lambda row: row["source_id"])
def test_a_visual_coding_row_runs_in_a_session_with_its_image_attached(setup, row):
    agent, sandbox = setup
    [message] = row["responses_create_params"]["input"]
    [image] = [part for part in message["content"] if part["type"] == "input_image"]
    [text] = [part["text"] for part in message["content"] if part["type"] == "input_text"]
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=row["responses_create_params"])
        assert result.status_code == 200, result.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text

    payload = sandbox.params()
    assert payload["prompt"] == text
    command = payload["command"]
    assert "--thinking" in command
    [image_path] = [command[index + 1] for index, arg in enumerate(command) if arg == "--file"]
    assert sandbox.files[image_path] == base64.b64decode(image["image_url"].partition(",")[2])

    opencode = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    model = opencode["provider"]["nemo_gym"]["models"]["dummy_model"]
    assert model["modalities"]["input"] == ["text", "image"]
    assert opencode["compaction"]["auto"] is False
    assert opencode["permission"]["external_directory"] == "allow"
    assert opencode["permission"]["doom_loop"] == "allow"
    assert opencode["tools"]["task"] is False


def test_image_input_declares_images_for_a_row_without_a_prompt_image(setup):
    # The example rows all carry a prompt image, which declares image input on its own. A text-only row shows that
    # image_input: true declares it too, so read can return the model's crops and plots.
    agent, sandbox = setup
    params = example_rows()[0]["responses_create_params"]
    [message] = params["input"]
    message["content"] = [part for part in message["content"] if part["type"] != "input_image"]
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=params)
        assert result.status_code == 200, result.text
        closed = client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"]))
        assert closed.status_code == 200, closed.text

    payload = sandbox.params()
    assert "--file" not in payload["command"]
    opencode = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    assert opencode["provider"]["nemo_gym"]["models"]["dummy_model"]["modalities"]["input"] == ["text", "image"]
