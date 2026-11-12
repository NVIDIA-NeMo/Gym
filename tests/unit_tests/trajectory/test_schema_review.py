# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.trajectory.tool_timing import normalize_tool_calls
from nemo_gym.trajectory.trajectory_record import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ToolCallObservation,
    TrajectoryModelCall,
    TrajectoryRecord,
)


@pytest.mark.parametrize("version", ["1.0", "2.0"])
def test_sparse_envelope_serialization(version):
    record = TrajectoryRecord(schema_version=version, task_id="task", rollout_id="capture")
    assert record.model_dump(include={"task_id"}) == {"task_id": "task"}
    sparse = record.model_dump(exclude_unset=True)
    assert "tool_calls" not in sparse
    assert sparse["capture_key" if version == "2.0" else "rollout_id"] == "capture"
    assert "tool_calls" in record.model_dump()
    bundle = AgentObservationBundle(schema_version=version, source="test")
    assert bundle.model_dump(include={"source"}) == {"source": "test"}
    assert "records" not in bundle.model_dump(exclude_unset=True)
    assert bundle.model_dump()["records"] == []


@pytest.mark.parametrize("second_producer", ["one", "two"])
def test_output_attribution_counts_producer_qualified_attempts(second_producer):
    invocation = AgentInvocation(
        invocation_id="root", conversation=[{"type": "function_call_output", "call_id": "call", "output": "result"}]
    )
    observed = [
        ToolCallObservation(
            invocation_id="root", tool_call_id="call", span_id="local", producer_id=producer, status="completed"
        )
        for producer in ["one", second_producer]
    ]
    retained, gaps = normalize_tool_calls(
        observed, capture_key="capture", producer_id="default", model_calls=[], invocations=[invocation]
    )
    assert len(retained) == (1 if second_producer == "one" else 2)
    assert [tool.output for tool in retained] == (["result"] if second_producer == "one" else [None, None])
    ambiguous = [gap for gap in gaps if gap.code == "tool_output_ambiguous"]
    assert len(ambiguous) == (0 if second_producer == "one" else 2)
    assert {gap.span_id for gap in ambiguous} <= {tool.span_id for tool in retained}


@pytest.mark.parametrize("reference_kind", ["pair", "id", "contradictory", "unmatched", "ambiguous"])
@pytest.mark.parametrize("instance", ["owner", "other", None])
def test_invocation_model_reference_ownership(reference_kind, instance):
    model_ref = {"name": "policy", "type": "responses_api_models"}
    metadata = {"model_ref": model_ref, "response_id": "response"}
    calls = [TrajectoryModelCall(model_call_id="model", agent_instance=instance, response_metadata=metadata)]
    ref = {"model_ref": model_ref, "response_id": "response"}
    if reference_kind == "id":
        ref = {"model_call_id": "model"}
    elif reference_kind == "contradictory":
        ref.update(model_call_id="model", response_id="different")
    elif reference_kind == "unmatched":
        ref["response_id"] = "missing"
    elif reference_kind == "ambiguous":
        calls.append(TrajectoryModelCall(model_call_id="second", agent_instance="other", response_metadata=metadata))
    kwargs = dict(
        schema_version="2.0",
        task_id="task",
        capture_key="capture",
        model_calls=calls,
        invocations=[AgentInvocation(invocation_id="root", agent_instance="owner", model_calls=[ModelCallRef(**ref)])],
    )
    if reference_kind in {"pair", "id"} and instance == "other":
        with pytest.raises(ValidationError, match="owning invocation"):
            TrajectoryRecord(**kwargs)
    else:
        TrajectoryRecord(**kwargs)


@pytest.mark.parametrize("explicit_version", [False, True])
def test_v1_sandbox_failure_reference_round_trips(explicit_version):
    payload = {
        "source": "opencode",
        "records": [{"kind": "sandbox", "role": "agent", "outcome": "oom", "failure_id": "oom-event"}],
    }
    if explicit_version:
        payload["schema_version"] = "1.0"
    bundle = AgentObservationBundle.model_validate(payload)
    assert bundle.records[0].failure_id == "oom-event"
    assert AgentObservationBundle.model_validate_json(bundle.model_dump_json()) == bundle


@pytest.mark.parametrize("same_attempt", [False, True])
@pytest.mark.parametrize("canonical_first", [False, True])
def test_output_attribution_counts_retained_and_raw_attempts(same_attempt, canonical_first):
    invocation = AgentInvocation(
        invocation_id="root", conversation=[{"type": "function_call_output", "call_id": "request", "output": "result"}]
    )
    kwargs = dict(capture_key="capture", producer_id="producer", model_calls=[], invocations=[invocation])
    raw = ToolCallObservation(invocation_id="root", tool_call_id="request", span_id="a", status="completed")
    [canonical], _ = normalize_tool_calls([raw], **kwargs)
    incoming = raw.model_copy(update={"span_id": "a" if same_attempt else "b"})
    observations = [canonical, incoming] if canonical_first else [incoming, canonical]
    tools, gaps = normalize_tool_calls(observations, **kwargs)
    assert next(tool for tool in tools if tool.span_id == canonical.span_id).output == "result"
    ambiguous = [gap for gap in gaps if gap.code == "tool_output_ambiguous"]
    if same_attempt:
        assert len(tools) == 1
        assert not ambiguous
    else:
        [new] = [tool for tool in tools if tool.span_id != canonical.span_id]
        assert new.output is None
        assert [gap.span_id for gap in ambiguous] == [new.span_id]
