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
        self.context_compressor = SimpleNamespace(last_prompt_tokens=4, compression_count=0)

    def _interruptible_api_call(self, kwargs):
        return {"id": "response-1"}

    def _compress_context(self, *args, **kwargs):
        self.context_compressor.compression_count += 1
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


def test_protected_history_noop_does_not_emit_completed_compaction():
    class NoOpAgent(_Agent):
        def _compress_context(self, messages, *args, **kwargs):
            return messages, "system"

    agent = NoOpAgent()
    observer = SandboxHermesObserver().instrument(agent)
    messages = [{"role": "user", "content": "hello"}]
    agent._compress_context(messages, "system", approx_tokens=27000)
    assert observer.finish({"messages": messages, "completed": True}, None)["compactions"] == []


def test_export_keeps_history_and_output_across_multiple_compactions():
    class CompactingAgent(_Agent):
        def _compress_context(self, messages, *args, **kwargs):
            self.context_compressor.compression_count += 1
            return [{"role": "user", "content": "summary"}], "system"

    agent = CompactingAgent()
    observer = SandboxHermesObserver().instrument(agent)
    original = [{"role": "user", "content": "request"}, {"role": "assistant", "content": "first"}]
    compressed, _ = agent._compress_context(original, "system")
    compressed.append({"role": "assistant", "content": "second"})
    compacted_again, _ = agent._compress_context(compressed, "system")
    compacted_again.append({"role": "assistant", "content": "final"})
    result = {"messages": compacted_again, "completed": True}
    expected = original + [{"role": "assistant", "content": "second"}, {"role": "assistant", "content": "final"}]
    assert observer.restore_messages(result)["messages"] == expected
    assert observer.finish(result, None)["invocations"][0]["messages"] == expected
    assert result["messages"] == compacted_again


@pytest.mark.parametrize("usage", [True, False])
def test_summary_calls_keep_invocation_identity_and_usage(monkeypatch, usage):
    import agent.context_compressor as compressor
    from openai.types.chat import ChatCompletion

    response = ChatCompletion.model_validate(
        {
            "id": "summary-response",
            "object": "chat.completion",
            "created": 0,
            "model": "policy",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Preserve progress."},
                }
            ],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 11,
                "total_tokens": 511,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 0},
            }
            if usage
            else None,
        }
    )
    monkeypatch.setattr(compressor, "call_llm", lambda **kwargs: response)

    class SummarizingAgent(_Agent):
        session_prompt_tokens = 100
        session_completion_tokens = 8
        session_cache_read_tokens = 0
        session_reasoning_tokens = 0

        def _compress_context(self, *args, **kwargs):
            compressor.call_llm(task="compression", messages=[])
            return super()._compress_context(*args, **kwargs)

    agent = SummarizingAgent()
    observer = SandboxHermesObserver().instrument(agent)
    agent._compress_context([], "system")
    result = {
        "messages": [],
        "completed": True,
        "prompt_tokens": agent.session_prompt_tokens,
        "completion_tokens": agent.session_completion_tokens,
    }
    exported = observer.restore_messages(result)
    assert exported["prompt_tokens"] == (600 if usage else None)
    assert exported["completion_tokens"] == (19 if usage else None)
    observations = observer.finish(result, None)
    assert observations["invocations"][0]["model_response_ids"] == ["summary-response"]
    assert observations["compactions"][0]["model_calls"] == [{"response_id": "summary-response"}]
    assert observations["compactions"][0]["summary"] == "Preserve progress."
