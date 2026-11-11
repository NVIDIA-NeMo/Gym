# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
from fastapi.testclient import TestClient

from responses_api_agents.opencode_multimodal_agent.combined import (
    OpenCodeCombinedPromptAgent,
    OpenCodeCombinedPromptConfig,
)
from responses_api_agents.opencode_multimodal_agent.tests.test_sessions import (  # noqa: F401
    body,
    close_body,
    opencode_setup,
    seed,
    visual_setup,
)


def test_combined_prompt_uses_shared_replacement_and_keeps_literal_braces(visual_setup):
    native, sandbox = visual_setup
    prompt = "Use {literal} and {env:DO_NOT_SUBSTITUTE} without changing them."
    cfg = OpenCodeCombinedPromptConfig.model_validate(
        native.config.model_dump() | {"system_prompt_mode": "replace", "system_prompt": prompt}
    )
    agent = OpenCodeCombinedPromptAgent(config=cfg, server_client=native.server_client)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json=body().model_dump(mode="json"))
        assert result.status_code == 200, result.text
        raw = sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"]
        assert "{env:DO_NOT_SUBSTITUTE}" not in raw
        assert prompt in json.loads(raw)["agent"]["build"]["prompt"]
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("instructions", [None, "Use {env:DO_NOT_SUBSTITUTE} literally."])
def test_replace_without_configured_prompt_uses_optional_request_instructions(visual_setup, evaluation, instructions):
    from responses_api_agents.opencode_multimodal_agent.app import (
        OpenCodeMultimodalAgent,
        OpenCodeMultimodalAgentConfig,
    )
    from responses_api_agents.opencode_multimodal_agent.eval import (
        OpenCodeMultimodalEvalAgent,
        OpenCodeMultimodalEvalConfig,
    )

    native, sandbox = visual_setup
    config_cls = OpenCodeMultimodalEvalConfig if evaluation else OpenCodeMultimodalAgentConfig
    agent_cls = OpenCodeMultimodalEvalAgent if evaluation else OpenCodeMultimodalAgent
    cfg = config_cls.model_validate(
        native.config.model_dump() | {"system_prompt": None, "system_prompt_mode": "replace"}
    )
    agent = agent_cls(config=cfg, server_client=native.server_client)
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        result = client.post(
            "/ng-rollout/opencode-smoke-a2/v1/responses", json=body(instructions=instructions).model_dump(mode="json")
        )
        assert result.status_code == 200, result.text
        raw = sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"]
        build = json.loads(raw)["agent"]["build"]
        if instructions is None:
            assert "prompt" not in build
        else:
            assert instructions in build["prompt"]
            assert "{env:DO_NOT_SUBSTITUTE}" not in raw
        assert (
            client.post("/v1/agent_sessions/close", json=close_body(created.json()["agent_session_id"])).status_code
            == 200
        )
