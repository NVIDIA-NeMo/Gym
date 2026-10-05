# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from run_agent import AIAgent

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from responses_api_agents.hermes_agent.model_kwargs import install_summary_compat
from responses_api_agents.hermes_agent.observability import HermesAgentObserver
from responses_api_agents.hermes_agent.sandbox_observer import SandboxHermesObserver


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("retry", [False, True])
def test_pinned_hermes_summary_passes_gym_schema_and_retains_call_ownership(
    sandbox, retry, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("run_agent.get_tool_definitions", return_value=[]), patch("run_agent.OpenAI"):
        agent = AIAgent(
            base_url="http://gym:8000/v1",
            api_key="test-key",  # pragma: allowlist secret
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "Summarize the work."
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "call_id": "call_1",
                    "response_item_id": "fc_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "patch applied"},
    ]
    requests = []

    def complete(**kwargs):
        requests.append(NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(kwargs))
        content = "" if retry and len(requests) == 1 else "Applied a partial patch."
        return SimpleNamespace(
            id=f"summary-{len(requests)}", choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = complete
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *args, **kwargs: None)
    install_summary_compat(agent, preserve_reasoning_history=False)
    observer = (
        SandboxHermesObserver()
        if sandbox
        else HermesAgentObserver(model_ref=ModelServerRef(type="responses_api_models", name="policy_model"))
    ).instrument(agent)

    summary = agent._handle_max_iterations(messages, 1)

    assert summary == "Applied a partial patch."
    assert agent._gym_iteration_limit_reached is True
    assert len(requests) == 1 + retry
    assert messages[0]["tool_calls"][0]["response_item_id"] == "fc_1"
    result = {"messages": messages, "completed": False}
    if sandbox:
        observations = observer.finish(result, None)
        ids = observations["invocations"][0]["model_response_ids"]
    else:
        observations = observer.finish(result)
        ids = [call.response_id for call in observations.records[0].model_calls]
    assert ids == [f"summary-{i + 1}" for i in range(len(requests))]
