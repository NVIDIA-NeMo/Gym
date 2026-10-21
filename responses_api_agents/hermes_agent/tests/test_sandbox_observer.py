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
def test_failed_model_call_carries_its_invocation_without_mutating_headers(child_call: bool) -> None:
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
    api_kwargs = {"extra_headers": {"custom": "preserved", "x-session-id": "stale"}}

    with pytest.raises(RuntimeError, match="HTTP 400"):
        caller._interruptible_api_call(api_kwargs=api_kwargs)

    invocation_id = "root.child-1" if child_call else "root"
    assert caller.request_headers == {"custom": "preserved", "x-session-id": invocation_id}
    assert api_kwargs == {"extra_headers": {"custom": "preserved", "x-session-id": "stale"}}
    invocation = next(
        item for item in observer.finish(None, None)["invocations"] if item["invocation_id"] == invocation_id
    )
    assert invocation["model_response_ids"] == []
