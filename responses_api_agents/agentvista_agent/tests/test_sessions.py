# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the public session API, including retries, refused grants, and closed activation."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig, ModelServerRef
from responses_api_agents.agentvista_agent.app import AgentVistaAgent, AgentVistaConfig
from responses_api_agents.agentvista_agent.runtime import HARNESS_VERSION


@pytest.mark.parametrize("harness", ["default", "opencode"])
def test_session_activation_is_ungraded_replayed_and_closed(tmp_path, monkeypatch, harness):
    import nemo_gym.config.global_config as global_config

    cfg = OmegaConf.create({"observability_enabled": False})
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", cfg)
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=18181), global_config_dict=cfg)
    agent = AgentVistaAgent(
        config=AgentVistaConfig(
            host="127.0.0.1",
            port=18182,
            entrypoint="app.py",
            name="agentvista_agent",
            source_root=str(tmp_path),
            dataset=str(tmp_path / "tasks.json"),
            image_folder=str(tmp_path),
            output_root=str(tmp_path / "outputs"),
            model_name="fixture",
            harness=harness,
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        ),
        server_client=client,
    )
    raw = {
        "artifacts": str(tmp_path),
        "verification_pending": True,
        "mask_sample": True,
        "verify": {"status": "ERROR"},
        "judge_responses": ["<answer>42</answer>"],
        "agent_output": "42",
    }
    execute = AsyncMock(return_value=raw)
    monkeypatch.setattr(AgentVistaAgent, "_execute", execute)
    metadata = {
        "task_id": "task",
        "category_id": "fixture",
        "source_sha256": "hash",
        "harness_version": HARNESS_VERSION,
    }
    seed = {
        "agent_session_id": "session-1",
        "episode_id": {"rollout_id": "rollout-1"},
        "task_id": {"taskset": "fixture", "task_id": "task"},
        "agent_context": {"values": {"verifier_metadata": metadata}},
    }
    with TestClient(agent.setup_webserver()) as api:
        assert api.post("/v1/responses", json={"input": "Act"}).status_code == 409
        for _ in range(2):
            response = api.post("/v1/agent_sessions", json=seed)
            assert response.status_code == 200, response.text
        response = api.post("/v1/responses", json={"input": "Act"})
        assert response.status_code == 200, response.text
        assert response.json()["agentvista_result"]["verification_pending"] is True
        assert response.json()["agentvista_result"]["mask_sample"] is True
        assert execute.await_args.args[0].capture_rollout_id == "rollout-1"
        close = {"agent_session_id": "session-1", "episode_id": seed["episode_id"]}
        first = api.post("/v1/agent_sessions/close", json=close)
        second = api.post("/v1/agent_sessions/close", json=close)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert first.json()["agent_observations"]["gaps"]
        assert first.json()["cleanup_confirmed"] is True
        assert api.post("/v1/responses", json={"input": "Act"}).status_code == 409
        assert api.post("/v1/agent_sessions", json=seed).status_code == 409
        assert execute.await_count == 1
