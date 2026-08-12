# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.claude_code_sandboxed_agent.app import (
    ClaudeCodeSandboxedAgent,
    ClaudeCodeSandboxedAgentConfig,
    ClaudeCodeSandboxedAgentRunRequest,
    _settings_json,
)


def _config(**overrides) -> ClaudeCodeSandboxedAgentConfig:
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "name": "claude_code_sandboxed_agent",
        "resources_server": ResourcesServerRef(type="resources_servers", name="benchmark"),
        "model_server": ModelServerRef(type="responses_api_models", name="policy_model"),
        "sandbox_provider": "sandbox",
        "sandbox_timeout": 300,
    }
    return ClaudeCodeSandboxedAgentConfig(**(values | overrides))


def _agent(**overrides) -> ClaudeCodeSandboxedAgent:
    return ClaudeCodeSandboxedAgent(
        config=_config(**overrides),
        server_client=MagicMock(spec=ServerClient),
    )


def test_settings_preserve_telemetry_defaults() -> None:
    settings = json.loads(_settings_json({"env": {"CUSTOM": "1"}, "permissions": {}}))
    assert settings["env"]["CLAUDE_CODE_ENABLE_TELEMETRY"] == "0"
    assert settings["env"]["CUSTOM"] == "1"
    assert settings["permissions"] == {}


def test_command_matches_claude_code_harness_controls() -> None:
    agent = _agent(max_turns=17, allowed_tools="Bash,Read", system_prompt="be precise")
    command = agent._command("test-model", "solve it", "be precise")
    assert "--output-format stream-json" in command
    assert "--bare" in command
    assert "--max-turns 17" in command
    assert "--allowedTools Bash,Read" in command
    assert "--append-system-prompt 'be precise'" in command
    assert command.rstrip().endswith("-- 'solve it'")


async def test_responses_runs_claude_code_in_seeded_sandbox() -> None:
    agent = _agent(model="test-model")
    sandbox = AsyncMock()
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "done"}],
                        "usage": {"input_tokens": 4, "output_tokens": 2},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "num_turns": 1,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            ),
        ]
    )
    sandbox.exec.return_value = MagicMock(stdout=stdout, stderr="", return_code=0, error_type=None)
    agent._sandboxes["session"] = sandbox
    request = MagicMock(cookies={"sandbox_id": "session"}, path_params={"rollout_id": "rollout"})
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "solve it"}])

    with (
        patch(
            "responses_api_agents.claude_code_sandboxed_agent.app.get_server_url",
            return_value="http://model",
        ),
        patch(
            "responses_api_agents.claude_code_sandboxed_agent.app.apply_rollout_prefix",
            return_value="http://model/rollout",
        ),
    ):
        response = await agent.responses(request=request, body=body)

    assert response.model == "test-model"
    assert response.output[-1].content[0].text == "done"
    assert agent._run_results["session"]["claude_code_finished"] is True
    assert agent._run_results["session"]["turns_used"] == 1
    assert sandbox.exec.call_args.kwargs["env"]["ANTHROPIC_BASE_URL"] == "http://model/rollout"


async def test_responses_surfaces_sandbox_timeout() -> None:
    agent = _agent(model="test-model")
    sandbox = AsyncMock()
    sandbox.exec.return_value = MagicMock(stdout="", stderr="", return_code=None, error_type="timeout")
    agent._sandboxes["session"] = sandbox
    request = MagicMock(cookies={"sandbox_id": "session"}, path_params={})
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "solve it"}])

    with (
        patch(
            "responses_api_agents.claude_code_sandboxed_agent.app.get_server_url",
            return_value="http://model",
        ),
        pytest.raises(TimeoutError, match="timed out"),
    ):
        await agent.responses(request=request, body=body)


async def test_run_uses_resources_server_for_session_and_verification() -> None:
    agent = _agent()
    agent.server_client.post = AsyncMock()
    sandbox = AsyncMock()
    seed = MagicMock(cookies={"seed": "cookie"})
    seed.json = AsyncMock(return_value={"sandbox_handle": {"sandbox_id": "box"}})
    model_response = MagicMock(cookies={"model": "cookie"})
    verify_response = MagicMock(cookies={})
    agent.server_client.post.side_effect = [seed, model_response, verify_response]
    response_json = {
        "id": "response",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    verify_json = {
        "responses_create_params": {"input": [{"role": "user", "content": "solve"}]},
        "response": response_json,
        "reward": 1.0,
    }
    body = ClaudeCodeSandboxedAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "solve"}])
    )
    request = MagicMock(session={SESSION_ID_KEY: "session"}, cookies={})

    with (
        patch.object(agent, "_connect_sandbox", new=AsyncMock(return_value=sandbox)),
        patch(
            "responses_api_agents.claude_code_sandboxed_agent.app.raise_for_status",
            new=AsyncMock(),
        ),
        patch(
            "responses_api_agents.claude_code_sandboxed_agent.app.get_response_json",
            new=AsyncMock(side_effect=[response_json, verify_json]),
        ),
    ):
        result = await agent.run(request, body)

    assert result.reward == 1.0
    assert [call.kwargs["url_path"] for call in agent.server_client.post.await_args_list] == [
        "/seed_session",
        "/v1/responses",
        "/verify",
    ]
    sandbox.stop.assert_awaited_once()
