# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from run_agent import AIAgent
from tools.delegate_tool import _build_child_agent

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
        assert observations["invocations"][0]["stop_reason"] == "max_iterations"
        assert observations["invocations"][0]["status"] == "incomplete"
    else:
        observations = observer.finish(result)
        ids = [call.response_id for call in observations.records[0].model_calls]
        assert observations.records[0].stop_reason == "max_iterations"
        assert observations.records[0].status == "incomplete"
    assert ids == [f"summary-{i + 1}" for i in range(len(requests))]


@pytest.mark.parametrize("observer_kind", ["host", "sandbox", "none"])
@pytest.mark.parametrize("install_before_observer", [False, True])
def test_delegated_child_limit_summary_uses_gym_and_belongs_to_child(
    observer_kind, install_before_observer, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("run_agent.get_tool_definitions", return_value=[]), patch("run_agent.OpenAI"):
        parent = AIAgent(
            base_url="http://gym:8000/v1",
            api_key="test-key",  # pragma: allowlist secret
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        if install_before_observer:
            install_summary_compat(parent, preserve_reasoning_history=False)
        observer = None
        if observer_kind == "host":
            observer = HermesAgentObserver(
                model_ref=ModelServerRef(type="responses_api_models", name="policy_model")
            ).instrument(parent)
        elif observer_kind == "sandbox":
            observer = SandboxHermesObserver().instrument(parent)
        if not install_before_observer:
            install_summary_compat(parent, preserve_reasoning_history=False)
        child = _build_child_agent(
            task_index=0,
            goal="Inspect a file",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=1,
            parent_agent=parent,
        )

    child._cached_system_prompt = "Inspect the file."
    requests = []

    def complete(**kwargs):
        requests.append(NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(kwargs))
        return SimpleNamespace(
            id="child-summary", choices=[SimpleNamespace(message=SimpleNamespace(content="Partial findings."))]
        )

    client = MagicMock()
    client.chat.completions.create.side_effect = complete
    monkeypatch.setattr(child, "_create_request_openai_client", lambda **kwargs: client)
    monkeypatch.setattr(child, "_close_request_openai_client", lambda *args, **kwargs: None)
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
        {"role": "tool", "tool_call_id": "call_1", "content": "file inspected"},
    ]

    assert child._handle_max_iterations(messages, 1) == "Partial findings."
    assert child._gym_iteration_limit_reached is True
    assert not getattr(parent, "_gym_iteration_limit_reached", False)
    assert len(requests) == 1
    assert messages[0]["tool_calls"][0]["response_item_id"] == "fc_1"

    if observer_kind == "sandbox":
        invocations = observer.finish({"messages": [], "completed": True}, None)["invocations"]
        assert invocations[0]["model_response_ids"] == []
        assert invocations[1]["model_response_ids"] == ["child-summary"]
        assert invocations[0].get("stop_reason") is None
        assert invocations[0]["status"] == "completed"
        assert invocations[1]["stop_reason"] == "max_iterations"
        assert invocations[1]["status"] == "incomplete"
    elif observer_kind == "host":
        invocations = [
            r for r in observer.finish({"messages": [], "completed": True}).records if hasattr(r, "model_calls")
        ]
        assert invocations[0].model_calls == []
        assert [call.response_id for call in invocations[1].model_calls] == ["child-summary"]
        assert invocations[0].stop_reason is None
        assert invocations[0].status == "completed"
        assert invocations[1].stop_reason == "max_iterations"
        assert invocations[1].status == "incomplete"


def test_iteration_limit_hook_fires_once_before_summary_even_when_summary_fails() -> None:
    events = []

    def fail_summary(_messages, _api_call_count):
        events.append("summary")
        raise RuntimeError("summary unavailable")

    agent = SimpleNamespace(
        _ensure_primary_openai_client=lambda *, reason: None,
        _handle_max_iterations=fail_summary,
        _gym_invocation_id="root.child-1",
        _gym_on_iteration_limit_reached=lambda *, invocation_id: events.append(invocation_id),
    )
    install_summary_compat(agent, preserve_reasoning_history=False)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="summary unavailable"):
            agent._handle_max_iterations([], 1)

    assert events == ["root.child-1", "summary", "summary"]
