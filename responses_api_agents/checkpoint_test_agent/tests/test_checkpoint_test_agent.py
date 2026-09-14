# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the checkpoint-specific Simple Agent fixture."""

import json
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.checkpoint_test_agent.app import CheckpointTestAgent
from responses_api_agents.simple_agent.app import SimpleAgentConfig


def _agent() -> CheckpointTestAgent:
    config = SimpleAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="checkpoint_test_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        resources_server=ResourcesServerRef(
            type="resources_servers",
            name="resources",
        ),
        max_steps=2,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    return CheckpointTestAgent(config=config, server_client=client)


def test_workplace_prefix_mode_rewrites_only_later_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_GYM_TEST_WORKPLACE_PREFIX_AFTER_MUTATION", "1")
    monkeypatch.setenv("NEMO_GYM_TEST_PREFIX_MIN_TOKENS", "512")
    agent = _agent()
    request = NeMoGymResponseCreateParamsNonStreaming(
        input="create the event",
        tool_choice="auto",
        parallel_tool_calls=True,
        max_output_tokens=64,
        metadata={"existing": "value"},
    )

    first = agent._prepare_model_request_for_turn(request, turn_index=1)
    assert first == request

    second = agent._prepare_model_request_for_turn(request, turn_index=2)
    assert second.tools == []
    assert second.tool_choice == "none"
    assert second.parallel_tool_calls is False
    assert second.max_output_tokens == 512
    assert second.metadata["existing"] == "value"
    assert json.loads(second.metadata["extra_body"]) == {"min_tokens": 512}


def test_workplace_prefix_mode_rejects_invalid_min_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_GYM_TEST_WORKPLACE_PREFIX_AFTER_MUTATION", "1")
    monkeypatch.setenv("NEMO_GYM_TEST_PREFIX_MIN_TOKENS", "zero")

    with pytest.raises(ValueError, match="must be an integer"):
        _agent()._prepare_model_request_for_turn(
            NeMoGymResponseCreateParamsNonStreaming(input="test"),
            turn_index=2,
        )
