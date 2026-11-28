# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for RoboLabAgent session lifecycle and sandbox refusal."""

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from responses_api_agents.robolab_agent.app import RoboLabAgent, RoboLabAgentConfig


def _build_agent() -> RoboLabAgent:
    config = RoboLabAgentConfig(
        name="robolab",
        host="127.0.0.1",
        port=8014,
        entrypoint="app.py",
        model_server={"type": "responses_api_models", "name": "policy_model"},
    )
    client = ServerClient(
        head_server_config={"host": "127.0.0.1", "port": 8000},
        global_config_dict=OmegaConf.create(
            {"policy_model": {"responses_api_models": {"vllm_model": {"host": "127.0.0.1", "port": 8011}}}}
        ),
    )
    return RoboLabAgent(config=config, server_client=client)


def test_robolab_agent_seeds_closes_and_refuses_lent_sandbox() -> None:
    agent = _build_agent()
    client = TestClient(agent.setup_webserver())

    seed_payload = {
        "agent_session_id": "rl-sess-1",
        "episode_id": {"rollout_id": "robolab-0", "attempt": 0},
        "task_id": {"taskset": "robolab", "task_id": "case-0"},
        "agent_context": {
            "values": {
                "case_id": "GrabABagelTask@run-010",
                "seed": 0,
                "manifest_sha256": "a" * 64,
            }
        },
    }
    seeded = client.post("/v1/agent_sessions", json=seed_payload)
    assert seeded.status_code == 200

    closed = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "rl-sess-1", "episode_id": seed_payload["episode_id"]},
    )
    assert closed.status_code == 200
    assert closed.json()["cleanup_confirmed"] is True

    fresh_client = TestClient(agent.setup_webserver())
    lent = fresh_client.post(
        "/v1/agent_sessions",
        json={
            **seed_payload,
            "agent_session_id": "rl-sess-2",
            "episode_id": {"rollout_id": "robolab-2", "attempt": 0},
            "task_id": {"taskset": "robolab", "task_id": "case-2"},
            "sandbox_access": {
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "docker",
                    "descriptor": {"container_id": "abc"},
                },
                "workdir": "/workspace",
            },
        },
    )
    assert lent.status_code == 422
    assert "does not support sandbox access" in lent.text.lower()


@pytest.mark.parametrize(
    "bad_context",
    [{"case_id": "x"}, {"case_id": "GrabABagelTask@run-010", "seed": -1, "manifest_sha256": "a" * 64}],
)
def test_robolab_agent_rejects_invalid_context(bad_context: dict[str, object]) -> None:
    agent = _build_agent()
    client = TestClient(agent.setup_webserver())
    resp = client.post(
        "/v1/agent_sessions",
        json={
            "agent_session_id": "rl-sess-bad",
            "episode_id": {"rollout_id": "robolab-1", "attempt": 0},
            "task_id": {"taskset": "robolab", "task_id": "case-1"},
            "agent_context": {"values": bad_context},
        },
    )
    assert resp.status_code == 422
