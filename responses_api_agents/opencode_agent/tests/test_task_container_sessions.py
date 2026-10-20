# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in sandbox-session settings for running OpenCode as a harness started in the task container."""

import json

import pytest
from fastapi.testclient import TestClient

from responses_api_agents.opencode_agent.runtime import OBSERVABILITY_PATCH
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import close_body, seed, setup  # noqa: F401


ROLLOUT = "/ng-rollout/opencode-smoke-a2/v1/responses"


def configure_direct(agent, **overrides):
    """A direct OpenAI-compatible endpoint instead of a Gym model server."""
    settings = {
        "model_server": None,
        "model_base_url": "http://gateway.example:16100/v1/",
        "model_api_key": "gateway-key",  # pragma: allowlist secret
        "model": "deepseek-flash",
    } | overrides
    for name, value in settings.items():
        setattr(agent.config, name, value)


def run_session(agent, sandbox, body=None, request=None):
    """Seed, activate and close one session; return the response and the runner input."""
    body = body or seed()
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=body.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        response = client.post(ROLLOUT, json=request or {"input": "Fix the code"})
        payload = json.loads(sandbox.files[f"{sandbox.directory}/input.json"])
        closed = client.post("/v1/agent_sessions/close", json=close_body(body.agent_session_id))
        assert closed.status_code == 200, closed.text
    return response, payload, closed.json()


def test_direct_endpoint_session_calls_the_configured_endpoint(setup):
    agent, sandbox = setup
    configure_direct(agent)
    # Gym capture is on, but calls to a direct endpoint never pass through Gym: no correlation plugin.
    agent.server_client.global_config_dict["observability_enabled"] = True
    response, payload, _ = run_session(agent, sandbox)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "deepseek-flash"
    config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    assert (config["model"], config["small_model"]) == ("nemo_gym/deepseek-flash", "nemo_gym/deepseek-flash")
    provider = config["provider"]["nemo_gym"]
    assert provider["options"] == {
        "baseURL": "http://gateway.example:16100/v1",
        "apiKey": "gateway-key",  # pragma: allowlist secret
        "timeout": False,
    }
    assert list(provider["models"]) == ["deepseek-flash"]
    assert not any(OBSERVABILITY_PATCH.name in plugin for plugin in config["plugin"])
    assert not any(path.endswith(OBSERVABILITY_PATCH.name) for path in sandbox.files)
    agent.server_client.post.assert_not_called()


def test_direct_endpoint_key_defaults_to_a_placeholder(setup):
    agent, sandbox = setup
    configure_direct(agent, model_api_key=None)
    response, payload, _ = run_session(agent, sandbox)
    assert response.status_code == 200, response.text
    config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"]["nemo_gym"]["options"]["apiKey"] == "EMPTY"  # pragma: allowlist secret


@pytest.mark.parametrize("routes", ["neither", "both"])
def test_seed_requires_exactly_one_model_route(setup, routes):
    agent, sandbox = setup
    if routes == "neither":
        agent.config.model_server = None
    else:
        agent.config.model_base_url = "http://gateway.example/v1"
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert response.status_code == 422
    assert "model_server or model_base_url" in response.text
    sandbox.exec.assert_not_awaited()


def test_direct_endpoint_accepts_its_own_model_name_in_requests(setup):
    agent, sandbox = setup
    configure_direct(agent)
    response, _, _ = run_session(agent, sandbox, request={"input": "Fix the code", "model": "deepseek-flash"})
    assert response.status_code == 200, response.text
