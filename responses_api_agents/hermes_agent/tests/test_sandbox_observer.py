# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from responses_api_agents.hermes_agent.sandbox_observer import SandboxHermesObserver


class _Agent:
    tool_start_callback = None
    tool_complete_callback = None

    def __init__(self) -> None:
        self._active_children = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=4)

    def _interruptible_api_call(self, kwargs):
        return {"id": "response-1"}

    def _compress_context(self, *args, **kwargs):
        return [], "summary"


def test_sandbox_observer_records_model_tool_and_compaction_events() -> None:
    agent = _Agent()
    observer = SandboxHermesObserver().instrument(agent)

    agent.tool_start_callback("call-1", "terminal", {"command": "false"})
    agent._interruptible_api_call({})
    agent._compress_context([], "", approx_tokens=10)
    agent.tool_complete_callback(
        "call-1",
        "terminal",
        {"command": "false"},
        '{"exit_code": 1}',
    )
    observations = observer.finish(
        {"messages": [{"role": "assistant", "content": "done"}], "completed": True},
        None,
    )

    assert observations["invocations"][0]["model_response_ids"] == ["response-1"]
    assert observations["invocations"][0]["status"] == "completed"
    assert observations["tools"][0]["status"] == "failed"
    assert observations["tools"][0]["duration_ms"] >= 0
    assert observations["compactions"][0]["tokens_before"] == 10
    assert observations["compactions"][0]["tokens_after"] == 4


@pytest.mark.parametrize("child_call", [False, True])
@pytest.mark.parametrize("headers", [None, {"custom": "preserved", "x-session-id": "stale"}])
def test_failed_model_call_carries_its_invocation_without_mutating_headers(
    child_call: bool, headers: dict | None
) -> None:
    class FailingAgent(_Agent):
        def _interruptible_api_call(self, api_kwargs):
            self.request_headers = api_kwargs["extra_headers"]
            raise RuntimeError("HTTP 400: rejected")

    agent = FailingAgent()
    observer = SandboxHermesObserver().instrument(agent)
    caller = agent
    if child_call:
        caller = FailingAgent()
        agent._active_children.append(caller)
    api_kwargs = {"extra_headers": headers}

    with pytest.raises(RuntimeError, match="HTTP 400"):
        caller._interruptible_api_call(api_kwargs=api_kwargs)

    invocation_id = "root.child-1" if child_call else "root"
    assert caller.request_headers == {**(headers or {}), "x-session-id": invocation_id}
    assert api_kwargs == {"extra_headers": headers}
    invocation = next(
        item for item in observer.finish(None, None)["invocations"] if item["invocation_id"] == invocation_id
    )
    assert invocation["model_response_ids"] == []


@pytest.mark.parametrize("kwargs", [{"changed_argument": {}}, {"api_kwargs": {"extra_headers": 42}}])
def test_header_observation_failure_preserves_original_call(kwargs: dict) -> None:
    calls = []
    response = {"id": "response-1"}

    class ChangedAgent(_Agent):
        def _interruptible_api_call(self, **received):
            calls.append(received)
            return response

    agent = ChangedAgent()
    observer = SandboxHermesObserver().instrument(agent)
    assert agent._interruptible_api_call(**kwargs) is response
    assert calls == [kwargs]
    evidence = observer.finish(None, None)
    assert evidence["invocations"][0]["model_response_ids"] == ["response-1"]
    assert any(gap["code"] == "hermes_observer_error" for gap in evidence["gaps"])
