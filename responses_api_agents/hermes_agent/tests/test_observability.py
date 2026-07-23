# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ContextCompactionObservation,
    ToolCallObservation,
)
from responses_api_agents.hermes_agent.observability import (
    HermesAgentObserver,
    normalize_hermes_messages,
)


class _FakeAgent:
    def __init__(self, conversation=None):
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self._active_children = []
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self._conversation = conversation or {"completed": True, "messages": []}

    def _invoke_tool(self, function_name, function_args, effective_task_id):
        return function_args.get("result", function_name)

    def _interruptible_api_call(self, api_kwargs):
        return SimpleNamespace(id=api_kwargs.get("response_id", "resp-1"))

    def _compress_context(self, messages, system_message, **kwargs):
        self.context_compressor.last_prompt_tokens = 7
        return messages[-1:], system_message

    def run_conversation(self, *args, **kwargs):
        return self._conversation


def _invocation(bundle, invocation_id):
    return next(
        item for item in bundle.records if isinstance(item, AgentInvocation) and item.invocation_id == invocation_id
    )


def _tool(bundle, tool_call_id):
    return next(
        item for item in bundle.records if isinstance(item, ToolCallObservation) and item.tool_call_id == tool_call_id
    )


def _compactions(bundle):
    return [item for item in bundle.records if isinstance(item, ContextCompactionObservation)]


def test_normalize_hermes_messages_preserves_conversation_order():
    items = normalize_hermes_messages(
        [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "terminal", "arguments": {"command": "pwd"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "/workspace"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        ],
        id_prefix="root",
    )

    assert [item.type for item in items] == [
        "message",
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert items[2].arguments == '{"command": "pwd"}'
    assert items[3].call_id == "call-1"
    assert items[4].content[0].text == "done"


def test_matching_invoke_tool_records_executor_interval():
    calls = []
    agent = _FakeAgent()
    agent.tool_start_callback = lambda *args: calls.append(("start", *args))
    agent.tool_complete_callback = lambda *args: calls.append(("complete", *args))
    observer = HermesAgentObserver().instrument(agent)
    args = {"command": "pwd"}

    agent.tool_start_callback("call-1", "terminal", args)
    agent._invoke_tool("terminal", args, "task")
    agent.tool_complete_callback("call-1", "terminal", args, "/workspace")
    bundle = observer.finish({"completed": True, "messages": []})

    tool = _tool(bundle, "call-1")
    assert tool.status == "completed"
    assert tool.started_at is not None
    assert tool.completed_at >= tool.started_at
    assert tool.duration_ms >= 0
    assert tool.timing_source == "executor"
    assert [call[0] for call in calls] == ["start", "complete"]


@pytest.mark.parametrize(
    "result",
    [
        "Error executing tool: boom",
        '{"error":"boom"}',
        '{"status":"error","message":"boom"}',
    ],
)
def test_tool_error_payloads_are_reported_as_failed(result):
    agent = _FakeAgent()
    observer = HermesAgentObserver().instrument(agent)
    args = {"command": "pwd"}

    agent.tool_start_callback("call-1", "terminal", args)
    agent.tool_complete_callback("call-1", "terminal", args, result)

    assert _tool(observer.finish({"completed": True, "messages": []}), "call-1").status == "failed"


def test_callback_only_tool_records_harness_interval():
    agent = _FakeAgent()
    observer = HermesAgentObserver().instrument(agent)
    callback_args = {"command": "pwd"}

    agent.tool_start_callback("call-1", "terminal", callback_args)
    agent._invoke_tool("terminal", {"command": "pwd"}, "task")
    agent.tool_complete_callback("call-1", "terminal", callback_args, "/workspace")
    bundle = observer.finish({"completed": True, "messages": []})

    tool = _tool(bundle, "call-1")
    assert tool.status == "completed"
    assert tool.started_at is not None
    assert tool.completed_at >= tool.started_at
    assert tool.duration_ms >= 0
    assert tool.timing_source == "harness"


def test_model_response_id_is_joined_to_the_owning_invocation():
    model_ref = ModelServerRef(type="responses_api_models", name="policy")
    agent = _FakeAgent()
    observer = HermesAgentObserver(model_ref=model_ref).instrument(agent)

    response = agent._interruptible_api_call({"response_id": "resp-7"})
    bundle = observer.finish({"completed": True, "messages": []})

    assert response.id == "resp-7"
    [reference] = _invocation(bundle, "root").model_calls
    assert reference.model_ref == model_ref
    assert reference.response_id == "resp-7"
    assert "model_call_ownership_unavailable" not in {gap.code for gap in bundle.gaps}


def test_concurrent_tool_intervals_end_at_each_worker_not_completion_callback():
    from run_agent import AIAgent

    agent = _FakeAgent()
    release = {"fast": threading.Event(), "slow": threading.Event()}
    entered = {"fast": threading.Event(), "slow": threading.Event()}

    def invoke(function_name, function_args, effective_task_id):
        entered[function_name].set()
        assert release[function_name].wait(timeout=2)
        return function_name

    agent._invoke_tool = invoke
    agent._interrupt_requested = False
    agent.quiet_mode = False
    agent.verbose_logging = False
    agent.log_prefix_chars = 100
    agent.tool_progress_callback = None
    agent._checkpoint_mgr = SimpleNamespace(enabled=False)
    agent._get_budget_warning = lambda _count: None
    observer = HermesAgentObserver().instrument(agent)
    calls = [
        SimpleNamespace(id=f"{name}-call", function=SimpleNamespace(name=name, arguments="{}"))
        for name in ("fast", "slow")
    ]
    execute = AIAgent._execute_tool_calls_concurrent.__get__(agent, _FakeAgent)
    execution = threading.Thread(target=execute, args=(SimpleNamespace(tool_calls=calls), [], "task"))
    execution.start()
    assert entered["fast"].wait(timeout=2)
    assert entered["slow"].wait(timeout=2)
    release["fast"].set()
    assert not threading.Event().wait(timeout=0.02)
    release["slow"].set()
    execution.join(timeout=2)
    assert not execution.is_alive()
    bundle = observer.finish({"completed": True, "messages": []})

    fast = _tool(bundle, "fast-call")
    slow = _tool(bundle, "slow-call")
    assert fast.completed_at < slow.completed_at
    assert fast.duration_ms < slow.duration_ms


def test_delegate_children_retain_exact_tree_and_full_conversations():
    root = _FakeAgent()
    observer = HermesAgentObserver(root_invocation_id="root").instrument(root)
    delegate_args = {"tasks": [{"goal": "one"}]}
    root.tool_start_callback("delegate-1", "delegate_task", delegate_args)

    child = _FakeAgent(
        {
            "completed": True,
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "child answer"},
            ],
        }
    )
    root._active_children.append(child)
    child.run_conversation("one")
    child_delegate_args = {"goal": "nested"}
    child.tool_start_callback("delegate-2", "delegate_task", child_delegate_args)
    grandchild = _FakeAgent(
        {
            "completed": True,
            "messages": [{"role": "assistant", "content": "nested answer"}],
        }
    )
    child._active_children.append(grandchild)
    grandchild.run_conversation("nested")
    child.tool_complete_callback("delegate-2", "delegate_task", child_delegate_args, "ok")
    root.tool_complete_callback("delegate-1", "delegate_task", delegate_args, "ok")

    bundle = observer.finish(
        {
            "completed": True,
            "messages": [
                {"role": "user", "content": "root task"},
                {"role": "assistant", "content": "root answer"},
            ],
        }
    )

    root_invocation = _invocation(bundle, "root")
    child_invocation = _invocation(bundle, "root.child-1")
    grandchild_invocation = _invocation(bundle, "root.child-1.child-2")
    assert root_invocation.status == "completed"
    assert child_invocation.parent_invocation_id == "root"
    assert child_invocation.spawned_by_tool_call_id == "delegate-1"
    assert [item.type for item in child_invocation.conversation] == ["message", "message"]
    assert grandchild_invocation.parent_invocation_id == "root.child-1"
    assert grandchild_invocation.spawned_by_tool_call_id == "delegate-2"
    assert all(not invocation.model_calls for invocation in bundle.records if isinstance(invocation, AgentInvocation))
    ownership_gaps = [gap for gap in bundle.gaps if gap.code == "model_call_ownership_unavailable"]
    assert {gap.invocation_id for gap in ownership_gaps} == {
        "root",
        "root.child-1",
        "root.child-1.child-2",
    }


def test_compaction_is_explicit_and_hook_failures_do_not_break_execution():
    agent = _FakeAgent()
    agent.tool_start_callback = MagicMock(side_effect=RuntimeError("callback"))
    observer = HermesAgentObserver().instrument(agent)

    assert agent._invoke_tool("terminal", {"result": "ok"}, "task") == "ok"
    compressed, prompt = agent._compress_context([{"role": "user", "content": "x"}], "system", approx_tokens=42)
    assert compressed == [{"role": "user", "content": "x"}]
    assert prompt == "system"
    bundle = observer.finish({"interrupted": True, "messages": [{"role": "user", "content": "x"}]})

    assert _invocation(bundle, "root").status == "incomplete"
    [compaction] = _compactions(bundle)
    assert compaction.tokens_before == 42
    assert compaction.tokens_after == 7
    assert compaction.outcome == "completed"
    assert {
        "compaction_model_call_boundary_unavailable",
        "compaction_summary_unavailable",
    } <= {gap.code for gap in bundle.gaps}


def test_failed_compaction_is_recorded_without_swallowing_the_error():
    agent = _FakeAgent()
    agent._compress_context = MagicMock(side_effect=RuntimeError("compression failed"))
    observer = HermesAgentObserver().instrument(agent)

    with pytest.raises(RuntimeError, match="compression failed"):
        agent._compress_context([], "system", approx_tokens=42)

    [compaction] = _compactions(observer.finish({"completed": True, "messages": []}))
    assert compaction.outcome == "failed"


def test_missing_private_hooks_are_reported_without_raising():
    class MinimalAgent:
        pass

    observer = HermesAgentObserver().instrument(MinimalAgent())
    bundle = observer.finish({"completed": True, "messages": []})

    unavailable = {gap.detail for gap in bundle.gaps if gap.code == "hermes_hook_unavailable"}
    assert unavailable == {
        "tool_start_callback",
        "tool_complete_callback",
        "_interruptible_api_call",
        "_invoke_tool",
        "_compress_context",
        "_active_children",
    }


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"completed": True, "messages": []}, "completed"),
        ({"final_response": "ok", "messages": []}, "completed"),
        ({"interrupted": True, "messages": []}, "incomplete"),
        ({"error": "boom", "messages": []}, "failed"),
    ],
)
def test_root_status(result, expected):
    observer = HermesAgentObserver().instrument(_FakeAgent())
    assert _invocation(observer.finish(result), "root").status == expected
