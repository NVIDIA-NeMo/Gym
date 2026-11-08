# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual-coding rows (BabyVision's format) under the OpenCode settings of the visual_coding_opencode session config."""

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


ROOT = Path(__file__).resolve().parents[3]
SESSION_CONFIG = ROOT / "resources_servers" / "visual_coding_opencode" / "configs" / "sandbox_sessions.yaml"
EXAMPLE_ROWS = ROOT / "resources_servers" / "visual_coding_opencode" / "data" / "example.jsonl"
# The settings OpenCode runs with inside the sandbox. The provider, image and model name come from the run.
SESSION_KEYS = ("opencode_version", "thinking", "timeout", "context_window", "max_output_tokens", "opencode_config")


def session_agent_config() -> dict:
    config = yaml.safe_load(SESSION_CONFIG.read_text())
    return config["visual_coding_opencode_session_agent"]["responses_api_agents"]["opencode_agent"]


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
        name="visual_coding_opencode_session_agent",
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


def test_the_session_config_runs_without_a_step_limit_or_compaction():
    """Token capture refuses agent.build.steps, and compaction's summary calls cannot join the scored call."""
    opencode_config = session_agent_config()["opencode_config"]
    assert "agent" not in opencode_config
    assert opencode_config["compaction"] == {"auto": False}


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
    assert "--thinking" not in command
    [image_path] = [command[index + 1] for index, arg in enumerate(command) if arg == "--file"]
    assert sandbox.files[image_path] == base64.b64decode(image["image_url"].partition(",")[2])

    opencode = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    model = opencode["provider"]["nemo_gym"]["models"]["dummy_model"]
    assert model["modalities"]["input"] == ["text", "image"]
    assert opencode["compaction"]["auto"] is False
    assert opencode["permission"]["external_directory"] == "allow"
    assert opencode["permission"]["doom_loop"] == "allow"
    assert opencode["tools"]["task"] is False
