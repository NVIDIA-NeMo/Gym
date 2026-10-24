# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason

from nemo_gym.rollout_collection import _build_trajectory_record
from responses_api_agents.tau2.observability import build_tool_observations


def request(owner: str = "assistant", call_id: str = "call-1") -> AssistantMessage | UserMessage:
    cls = AssistantMessage if owner == "assistant" else UserMessage
    return cls(role=owner, tool_calls=[ToolCall(id=call_id, name="lookup", arguments={"query": "x"}, requestor=owner)])


def output(owner: str = "assistant", call_id: str = "call-1", **kwargs) -> ToolMessage:
    return ToolMessage(role="tool", id=call_id, requestor=owner, **({"content": "result"} | kwargs))


def trajectory(messages: list[Message]):
    result = SimulationRun(
        id="simulation",
        task_id="task",
        start_time="2026-10-08T10:00:00",
        end_time="2026-10-08T10:00:01",
        duration=1,
        num_steps=len(messages),
        termination_reason=TerminationReason.USER_STOP,
        messages=messages,
    )
    original = result.model_dump()
    bundle = build_tool_observations(result)
    assert result.model_dump() == original
    return _build_trajectory_record(
        {"_ng_task_index": 0, "_ng_rollout_index": 0}, {"ng_agent_observations": bundle.model_dump(mode="json")}
    )


def test_both_participants_can_use_the_same_tool_id() -> None:
    record = trajectory([request(), output(content="policy"), request("user"), output("user", content="simulator")])
    tools = {tool.invocation_id: tool for tool in record.tool_calls}
    assert tools["simulation:agent"].output == "policy"
    assert tools["simulation:user_simulator"].output == "simulator"
    assert len(record.invocations) == 2
    for invocation in record.invocations:
        assert invocation.parent_invocation_id is None
        assert invocation.status == "unknown"
        assert invocation.model_calls == []
        call = invocation.conversation[0]
        assert call.call_id == "call-1"
        assert call.name == "lookup"
        assert json.loads(call.arguments) == {"query": "x"}


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("content", ["result", "", None])
def test_native_outcome_and_output_without_fabricated_timing(failed: bool, content: str | None) -> None:
    record = trajectory([request(), output(error=failed, content=content)])
    (tool,) = record.tool_calls
    assert tool.output == content
    assert tool.status == ("failed" if failed else "completed")
    assert tool.started_at is tool.completed_at is tool.duration_ms is tool.timing_source is None
    assert tool.tool_name == "lookup"


@pytest.mark.parametrize(
    "messages",
    [
        [request()],
        [output()],
        [request(), request(), output()],
        [request(), output(), output()],
        [output(), request()],
        [request(call_id=""), output(call_id="")],
        [request(), output("user")],
    ],
)
def test_missing_or_ambiguous_joins_leave_explicit_gaps(messages: list[Message]) -> None:
    record = trajectory(messages)
    assert record.tool_calls == []
    assert any(gap.code in {"tau_tool_join_unavailable", "tau_tool_result_missing"} for gap in record.gaps)


def test_parallel_tool_outputs_keep_native_ids() -> None:
    first = request()
    first.tool_calls.append(ToolCall(id="call-2", name="other", arguments={}))
    record = trajectory(
        [
            first,
            MultiToolMessage(role="tool", tool_messages=[output(call_id="call-2", error=True), output()]),
        ]
    )
    assert {tool.tool_call_id: tool.status for tool in record.tool_calls} == {
        "call-1": "completed",
        "call-2": "failed",
    }


def test_conflicting_requestor_does_not_attribute_execution_to_policy() -> None:
    message = request()
    message.tool_calls[0].requestor = "user"
    record = trajectory([message, output("user")])
    assert not record.tool_calls
    assert any(gap.code == "tau_tool_requestor_mismatch" for gap in record.gaps)


@pytest.mark.parametrize("missing_capture", [False, True])
def test_native_model_turns_supply_perf_and_keep_tool_observations(missing_capture):
    from nemo_gym.config_types import ModelServerRef
    from nemo_gym.rollout_collection import _attach_ng_perf, _attach_trajectory_record
    from responses_api_agents.tau2.observability import build_trajectory

    policy = ModelServerRef(type="responses_api_models", name="policy")
    user = ModelServerRef(type="responses_api_models", name="user")
    tool_request = request()
    tool_request.raw_data = {"id": "policy-1"}
    result = SimulationRun(
        id="simulation",
        task_id="task",
        start_time="2026-10-08T10:00:00",
        end_time="2026-10-08T10:00:01",
        duration=1,
        num_steps=5,
        termination_reason=TerminationReason.USER_STOP,
        messages=[
            AssistantMessage(role="assistant", content="Scripted greeting"),
            UserMessage(role="user", content="Please look it up", raw_data={"id": "user-1"}),
            tool_request,
            output(),
            AssistantMessage(role="assistant", content="Done", raw_data={"id": "policy-2"}),
        ],
    )
    observations = build_tool_observations(result)
    trajectory = build_trajectory(
        result, observations=observations, task_id="task", rollout_id="0-0", policy_model=policy, user_model=user
    )
    # Model-only participants must also reach the observation join, not just the trajectory.
    simulator = next(
        r for r in observations.records if getattr(r, "invocation_id", None) == "simulation:user_simulator"
    )
    assert simulator.model_calls[0].response_id == "user-1"
    calls = [
        {"model_call_id": "c0", "response_id": "user-1", "model_ref": user.model_dump(), "tokens_out": 10},
        {"model_call_id": "c1", "response_id": "policy-1", "model_ref": policy.model_dump(), "tokens_out": 20},
        {"model_call_id": "c2", "response_id": "policy-2", "model_ref": policy.model_dump(), "tokens_out": 30},
    ]
    record = {
        "ng_trajectory": trajectory.model_dump(mode="json"),
        "ng_agent_observations": observations.model_dump(mode="json"),
        "ng_model_call_capture": {"calls": calls[:-1] if missing_capture else calls},
    }
    _attach_trajectory_record({"task_id": "task", "_ng_task_index": 0, "_ng_rollout_index": 0}, record)
    _attach_ng_perf(record, observability_enabled=True, rollout_latency_ms=1000)
    assert record["ng_perf"] == {
        "num_turns": 3,
        "num_tool_calls": 1,
        "completion_tokens": 30 if missing_capture else 60,
        "token_observability_coverage": 2 / 3 if missing_capture else 1.0,
        "total_latency_ms": 1000,
    }
    turns = record["ng_trajectory"]["turns"]
    assert [(t["invocation_id"], t["turn_no"]) for t in turns] == [
        ("simulation:agent", 1),
        ("simulation:user_simulator", 1),
        ("simulation:agent", 2),
    ]  # Sorted by native timestamps: tool_request was constructed first.
    assert record["ng_trajectory"]["tool_calls"][0]["output"] == "result"
    assert turns[0]["answer"]["tool_calls"][0]["id"] == "call-1"
