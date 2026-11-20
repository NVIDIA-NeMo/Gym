# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import pytest

from nemo_gym.trajectory import read_tool_events
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


def test_sandbox_observer_records_model_tool_and_compaction_events(tool_recorder, tool_events_path) -> None:
    agent = _Agent()
    observer = SandboxHermesObserver(tool_recorder=tool_recorder).instrument(agent)

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
    (tool,), gaps = read_tool_events(tool_events_path)
    assert gaps == [] and "tools" not in observations
    assert (tool.invocation_id, tool.tool_call_id, tool.tool_name, tool.status) == (
        "root",
        "call-1",
        "terminal",
        "failed",
    )
    (timing,) = tool.timings
    assert (timing.boundary, timing.timing_source) == ("lifecycle", "hermes.tool_callback")
    assert timing.duration_ms >= 0
    assert observations["compactions"][0]["tokens_before"] == 10
    assert observations["compactions"][0]["tokens_after"] == 4


@pytest.mark.parametrize("child_call", [False, True])
@pytest.mark.parametrize("headers", [None, {"custom": "preserved", "x-session-id": "stale"}])
def test_failed_model_call_carries_its_invocation_without_mutating_headers(
    child_call: bool, headers: dict | None, tool_recorder
) -> None:
    class FailingAgent(_Agent):
        def _interruptible_api_call(self, api_kwargs):
            self.request_headers = api_kwargs["extra_headers"]
            raise RuntimeError("HTTP 400: rejected")

    agent = FailingAgent()
    observer = SandboxHermesObserver(tool_recorder=tool_recorder).instrument(agent)
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
def test_header_observation_failure_preserves_original_call(kwargs: dict, tool_recorder) -> None:
    calls = []
    response = {"id": "response-1"}

    class ChangedAgent(_Agent):
        def _interruptible_api_call(self, **received):
            calls.append(received)
            return response

    agent = ChangedAgent()
    observer = SandboxHermesObserver(tool_recorder=tool_recorder).instrument(agent)
    assert agent._interruptible_api_call(**kwargs) is response
    assert calls == [kwargs]
    evidence = observer.finish(None, None)
    assert evidence["invocations"][0]["model_response_ids"] == ["response-1"]
    assert any(gap["code"] == "hermes_observer_error" for gap in evidence["gaps"])


def test_protected_history_noop_does_not_emit_completed_compaction(tool_recorder):
    class NoOpAgent(_Agent):
        def _compress_context(self, messages, *args, **kwargs):
            return messages, "system"

    agent = NoOpAgent()
    observer = SandboxHermesObserver(tool_recorder=tool_recorder).instrument(agent)
    messages = [{"role": "user", "content": "hello"}]
    agent._compress_context(messages, "system", approx_tokens=27000)
    assert observer.finish({"messages": messages, "completed": True}, None)["compactions"] == []


def test_export_keeps_history_and_output_across_multiple_compactions(tool_recorder):
    class CompactingAgent(_Agent):
        def _compress_context(self, messages, *args, **kwargs):
            self.context_compressor.compression_count += 1
            return [{"role": "user", "content": "summary"}], "system"

    agent = CompactingAgent()
    observer = SandboxHermesObserver(tool_recorder=tool_recorder).instrument(agent)
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


def test_export_keeps_turns_after_hermes_merges_the_summary_into_the_next_message(tool_recorder):
    summary = {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"}
    tail_call = {"role": "assistant", "content": "", "tool_calls": [{"id": "call-2"}]}
    tail = [tail_call, {"role": "tool", "tool_call_id": "call-2", "content": "two"}]
    restated = {"role": "user", "content": "[STILL IN PROGRESS] request"}

    class MergingAgent(_Agent):
        def _compress_context(self, messages, *args, **kwargs):
            self.context_compressor.compression_count += 1
            return [messages[0], summary, *tail, restated], "system"

    agent = MergingAgent()
    observer = SandboxHermesObserver(tool_recorder=tool_recorder).instrument(agent)
    original = [
        {"role": "user", "content": "request"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "one"},
        *tail,
    ]
    agent._compress_context(original, "system")
    added = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-3"}]},
        {"role": "tool", "tool_call_id": "call-3", "content": "three"},
        {"role": "assistant", "content": "done"},
    ]
    # Hermes folds the summary into the assistant message that follows it, one message fewer than it returned.
    merged = [original[0], {**tail_call, "content": summary["content"]}, tail[1], restated, *added]
    result = {"messages": merged, "completed": True}
    assert observer.restore_messages(result)["messages"] == original + added


@pytest.mark.parametrize("usage", [True, False])
def test_summary_calls_keep_invocation_identity_and_usage(monkeypatch, usage, tool_recorder):
    from types import SimpleNamespace

    from openai.types.chat import ChatCompletion

    from responses_api_agents.hermes_agent.sandbox_observer import _install_summary_capture

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
    compressor = SimpleNamespace(call_llm=lambda **kwargs: response)
    _install_summary_capture(compressor)

    class SummarizingAgent(_Agent):
        session_prompt_tokens = 100
        session_completion_tokens = 8
        session_cache_read_tokens = 0
        session_reasoning_tokens = 0

        def _compress_context(self, *args, **kwargs):
            compressor.call_llm(task="compression", messages=[])
            return super()._compress_context(*args, **kwargs)

    agent = SummarizingAgent()
    observer = SandboxHermesObserver(tool_recorder=tool_recorder).instrument(agent)
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


def _on_worker_thread(function, *args):
    """Run a Hermes callback or tool on another thread, as Hermes' concurrent path does."""
    errors = []

    def run():
        try:
            function(*args)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]


def test_worker_attributes_child_and_executor_interval_to_hermes_tool_call_id(tool_recorder, tool_events_path):
    agent = _Agent()
    agent._invoke_tool = lambda *_: agent._active_children.append(_Agent()) or "done"
    observer = SandboxHermesObserver(tool_recorder=tool_recorder).instrument(agent)

    def concurrent_call():
        agent.tool_start_callback("d1", "delegate_task", {"goal": "inspect"})
        agent._invoke_tool("delegate_task", {"goal": "inspect"}, "task-id", "d1")

    _on_worker_thread(concurrent_call)
    (executed,), _ = read_tool_events(tool_events_path)
    agent.tool_complete_callback("d1", "delegate_task", {}, "done")
    observation = observer.finish({"completed": True}, None)
    assert observation["invocations"][1]["spawned_by_tool_call_id"] == "d1"
    (tool,), gaps = read_tool_events(tool_events_path)
    # The executor interval ends when `_invoke_tool` returns; the later completion callback adds nothing.
    assert gaps == []
    assert [(t.boundary, t.timing_source) for t in tool.timings] == [("executor", "hermes.invoke_tool")]
    assert tool.timings[0].completed_at == executed.timings[0].completed_at
    assert tool.status == "completed"


def test_each_call_has_one_timing_at_the_boundary_its_path_reaches(tool_recorder, tool_events_path):
    """Sequential calls are timed at the callbacks, concurrent calls in `_invoke_tool`, and a concurrent call that
    never reaches `_invoke_tool` at its completion."""
    agent = _Agent()
    agent._invoke_tool = lambda *_: "done"
    observer = SandboxHermesObserver(tool_recorder=tool_recorder).instrument(agent)

    agent.tool_start_callback("sequential", "terminal", {})
    agent.tool_complete_callback("sequential", "terminal", {}, "done")

    def concurrent_call():
        agent.tool_start_callback("concurrent", "read_file", {})
        agent._invoke_tool("read_file", {}, "task-id", "concurrent")

    _on_worker_thread(concurrent_call)
    agent.tool_complete_callback("concurrent", "read_file", {}, "done")
    _on_worker_thread(agent.tool_start_callback, "unreached", "read_file", {})
    # Left to `_invoke_tool`, the unreached call has no record until its completion.
    assert "unreached" not in {t.tool_call_id for t in read_tool_events(tool_events_path)[0]}
    agent.tool_complete_callback("unreached", "read_file", {}, "Error: refused")
    observer.finish({"completed": True}, None)

    tools, gaps = read_tool_events(tool_events_path)
    assert gaps == []
    assert {t.tool_call_id: ([timing.boundary for timing in t.timings], t.status) for t in tools} == {
        "sequential": (["lifecycle"], "completed"),
        "concurrent": (["executor"], "completed"),
        "unreached": (["lifecycle"], "failed"),
    }


def test_a_concurrent_call_that_raises_is_a_failed_executor_attempt(tool_recorder, tool_events_path):
    agent = _Agent()

    def refuse(*_):
        raise RuntimeError("tool crashed")

    agent._invoke_tool = refuse
    SandboxHermesObserver(tool_recorder=tool_recorder).instrument(agent)

    def concurrent_call():
        agent.tool_start_callback("c1", "terminal", {})
        with pytest.raises(RuntimeError, match="tool crashed"):
            agent._invoke_tool("terminal", {}, "task-id", "c1")

    _on_worker_thread(concurrent_call)
    agent.tool_complete_callback("c1", "terminal", {}, "Error executing tool 'terminal': tool crashed")

    (tool,), _ = read_tool_events(tool_events_path)
    assert ([t.boundary for t in tool.timings], tool.status) == (["executor"], "failed")
