# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from fastapi.testclient import TestClient

from responses_api_agents.opencode_multimodal_agent.eval import (
    OpenCodeMultimodalEvalAgent,
    OpenCodeMultimodalEvalConfig,
)
from responses_api_agents.opencode_multimodal_agent.tests.test_sessions import (  # noqa: F401
    body,
    close_body,
    opencode_setup,
    seed,
    visual_setup,
)


@pytest.mark.parametrize("headroom,steps", [(0, 8), (2, 6)])
def test_eval_uses_shared_session_with_advisory_steps(visual_setup, headroom, steps):
    native, sandbox = visual_setup
    cfg = OpenCodeMultimodalEvalConfig.model_validate(
        native.config.model_dump()
        | {"final_answer_reserve": None, "final_answer_turn_headroom": headroom, "tool_rewards": None}
    )
    agent = OpenCodeMultimodalEvalAgent(config=cfg, server_client=native.server_client)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=body().model_dump(mode="json"))
        assert result.status_code == 200, result.text
        config = json.loads(sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"])
        assert config["agent"]["build"]["steps"] == steps
        assert config["enabled_providers"] == ["nemo_gym"]
        assert not any("visual-final-answer" in p for p in config["plugin"])
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )


async def test_enforced_budget_stages_session_header_and_removes_stock_steps(visual_setup, monkeypatch):
    from types import SimpleNamespace

    native, sandbox = visual_setup
    cfg = OpenCodeMultimodalEvalConfig.model_validate(
        native.config.model_dump()
        | {
            "final_answer_reserve": None,
            "max_turns": 8,
            "turn_budget": {"regular_turns": 6, "grace_turns": 2, "grace_tool_turns": 1},
            "tool_rewards": None,
        }
    )
    # The constructor's matching-gateway checks are covered by the gateway suite.
    monkeypatch.setattr(OpenCodeMultimodalEvalAgent, "_validate_turn_budget_gateway", lambda self: None)
    agent = OpenCodeMultimodalEvalAgent(config=cfg, server_client=native.server_client)
    state = SimpleNamespace(session=SimpleNamespace(session_dir="/session", sandbox=sandbox))
    payload = {
        "env": {
            "OPENCODE_CONFIG_CONTENT": json.dumps(
                {
                    "model": "nemo_gym/dummy",
                    "provider": {"nemo_gym": {"models": {cfg.sandbox_model_id: {"limit": {}}}}},
                }
            )
        }
    }
    await agent._prepare_session_launch(state, body(), payload)
    config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    assert "steps" not in config["agent"]["build"]
    plugin = sandbox.files["/session/eval-budget-headers.mjs"]
    assert 'output.headers["x-opencode-session-id"] = input.sessionID' in plugin
    assert "file:///session/eval-budget-headers.mjs" in config["plugin"]
