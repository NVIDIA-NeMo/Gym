# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.base_responses_api_model import ModelCallRecord
from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
    join_model_call_observations,
)


@pytest.mark.parametrize(
    "value",
    ({}, {"response_id": "resp-1"}, {"model_ref": {"name": "policy", "type": "responses_api_models"}}),
)
def test_model_call_ref_rejects_incomplete_join_keys(value: dict) -> None:
    with pytest.raises(ValidationError, match="model_call_id or both model_ref and response_id"):
        ModelCallRef.model_validate(value)


def test_observation_bundle_rejects_duplicate_invocation_ids() -> None:
    with pytest.raises(ValidationError, match="invocation_id must be unique"):
        AgentObservationBundle(
            source="test",
            records=[AgentInvocation(invocation_id="root"), AgentInvocation(invocation_id="root")],
        )


@pytest.mark.parametrize(
    "records",
    (
        [AgentInvocation(invocation_id="root", parent_invocation_id="root")],
        [
            AgentInvocation(invocation_id="a", parent_invocation_id="b"),
            AgentInvocation(invocation_id="b", parent_invocation_id="a"),
        ],
    ),
)
def test_observation_bundle_rejects_parent_cycles(records: list[AgentInvocation]) -> None:
    with pytest.raises(ValidationError, match="parent_invocation_id must not form a cycle"):
        AgentObservationBundle(source="test", records=records)


def test_observation_bundle_allows_missing_parent() -> None:
    bundle = AgentObservationBundle(
        source="test",
        records=[AgentInvocation(invocation_id="child", parent_invocation_id="unobserved")],
    )
    assert bundle.records[0].parent_invocation_id == "unobserved"


def test_observation_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="producer_extension"):
        ModelCallRef.model_validate({"model_call_id": "call-1", "producer_extension": "unexpected"})


@pytest.mark.parametrize(
    "timing",
    (
        {"duration_ms": -1},
        {"started_at": 2.0, "completed_at": 1.0},
    ),
)
def test_tool_call_observation_rejects_invalid_timing(timing: dict) -> None:
    with pytest.raises(ValidationError):
        ToolCallObservation(invocation_id="root", tool_call_id="call-1", **timing)


def test_agent_invocation_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        AgentInvocation(invocation_id="root", duration_ms=-1)


@pytest.mark.parametrize(
    "values",
    (
        {"tokens_before": -1},
        {"tokens_after": -1},
    ),
)
def test_context_compaction_rejects_invalid_token_counts(values: dict) -> None:
    with pytest.raises(ValidationError):
        ContextCompactionObservation(invocation_id="root", **values)


def test_context_compaction_preserves_producer_reported_token_counts() -> None:
    observation = ContextCompactionObservation(
        invocation_id="root",
        tokens_before=10,
        tokens_after=11,
        outcome="completed",
    )
    assert observation.tokens_after == 11


def test_join_model_calls_resolves_exact_references_and_reports_unowned_calls() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(
                invocation_id="root",
                model_calls=[ModelCallRef(model_ref=model_ref, response_id="resp-1")],
            )
        ],
        gaps=[
            ObservationGap(code="model_call_ownership_unavailable", detail="direct_provider"),
            ObservationGap(code="model_call_ownership_unavailable", detail="capture:stale"),
        ],
    )
    calls = [
        ModelCallRecord(
            model_call_id="call-1",
            response_id="resp-1",
            model_ref=model_ref,
            call_index=0,
        ),
        ModelCallRecord(model_call_id="call-2", model_ref=model_ref, call_index=1),
    ]

    joined = join_model_call_observations(bundle, calls)

    [invocation] = [record for record in joined.records if isinstance(record, AgentInvocation)]
    [joined_call] = invocation.model_calls
    assert joined_call.model_call_id == "call-1"
    assert joined_call.model_ref == model_ref
    assert joined_call.response_id == "resp-1"
    ownership_gaps = [gap for gap in joined.gaps if gap.code == "model_call_ownership_unavailable"]
    assert [gap.detail for gap in ownership_gaps] == ["direct_provider", "capture:call-2:call_index=1"]


def test_join_model_calls_does_not_guess_ambiguous_response_ids() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(
                invocation_id="root",
                model_calls=[ModelCallRef(model_ref=model_ref, response_id="resp-1")],
            )
        ],
    )
    calls = [
        ModelCallRecord(model_call_id="call-1", response_id="resp-1", model_ref=model_ref, call_index=0),
        ModelCallRecord(model_call_id="call-2", response_id="resp-1", model_ref=model_ref, call_index=1),
    ]

    joined = join_model_call_observations(bundle, calls)

    [invocation] = [record for record in joined.records if isinstance(record, AgentInvocation)]
    assert invocation.model_calls[0].model_call_id is None
    assert "model_call_reference_ambiguous" in {gap.code for gap in joined.gaps}
    assert [gap.detail for gap in joined.gaps if gap.code == "model_call_ownership_unavailable"] == [
        "capture:call-1:call_index=0",
        "capture:call-2:call_index=1",
    ]


def test_join_model_calls_reports_conflicting_and_unmatched_references() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(
                invocation_id="root",
                model_calls=[
                    ModelCallRef(model_call_id="call-1"),
                    ModelCallRef(model_ref=model_ref, response_id="resp-1"),
                    ModelCallRef(model_call_id="missing"),
                ],
            )
        ],
        gaps=[
            ObservationGap(
                code="model_call_ownership_unavailable",
                invocation_id="root",
                detail="producer gap",
            )
        ],
    )
    calls = [
        ModelCallRecord(
            model_call_id="call-1",
            response_id="resp-1",
            model_ref=model_ref,
            call_index=0,
        )
    ]

    joined = join_model_call_observations(bundle, calls)
    joined_again = join_model_call_observations(joined, calls)

    assert joined_again.model_dump() == joined.model_dump()
    assert {gap.code for gap in joined.gaps} == {
        "model_call_ownership_unavailable",
        "model_call_reference_conflict",
        "model_call_reference_unmatched",
    }
    assert any(gap.detail == "producer gap" for gap in joined.gaps)


def test_join_model_calls_resolves_compaction_boundaries() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    references = {
        response_id: ModelCallRef(model_ref=model_ref, response_id=response_id)
        for response_id in ("before", "compaction", "after")
    }
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(
                invocation_id="root",
                model_calls=list(references.values()),
            ),
            ContextCompactionObservation(
                invocation_id="root",
                before_model_call=references["before"],
                model_calls=[references["compaction"]],
                after_model_call=references["after"],
            ),
        ],
    )
    calls = [
        ModelCallRecord(
            model_call_id="call-before",
            response_id="before",
            model_ref=model_ref,
            call_index=0,
        ),
        ModelCallRecord(
            model_call_id="call-compaction",
            response_id="compaction",
            model_ref=model_ref,
            call_index=1,
        ),
        ModelCallRecord(
            model_call_id="call-after",
            response_id="after",
            model_ref=model_ref,
            call_index=2,
        ),
    ]

    joined = join_model_call_observations(bundle, calls)
    [compaction] = [record for record in joined.records if isinstance(record, ContextCompactionObservation)]

    assert compaction.before_model_call.model_call_id == "call-before"
    assert compaction.model_calls[0].model_call_id == "call-compaction"
    assert compaction.after_model_call.model_call_id == "call-after"
    assert joined.gaps == []


def test_join_model_calls_rejects_cross_invocation_compaction_ownership() -> None:
    model_ref = ModelServerRef(name="policy", type="responses_api_models")
    owned = ModelCallRef(model_ref=model_ref, response_id="owned")
    unowned = ModelCallRef(model_ref=model_ref, response_id="unowned")
    bundle = AgentObservationBundle(
        source="test",
        records=[
            AgentInvocation(invocation_id="root"),
            AgentInvocation(invocation_id="other", model_calls=[owned]),
            ContextCompactionObservation(
                invocation_id="root",
                before_model_call=owned,
                model_calls=[owned, unowned],
            ),
        ],
    )
    calls = [
        ModelCallRecord(model_call_id="call-owned", response_id="owned", model_ref=model_ref, call_index=0),
        ModelCallRecord(model_call_id="call-unowned", response_id="unowned", model_ref=model_ref, call_index=1),
    ]

    joined = join_model_call_observations(bundle, calls)
    compaction = next(record for record in joined.records if isinstance(record, ContextCompactionObservation))

    assert compaction.before_model_call.model_call_id is None
    assert all(reference.model_call_id is None for reference in compaction.model_calls)
    assert {(gap.code, gap.detail) for gap in joined.gaps if gap.invocation_id == "root"} == {
        ("model_call_reference_conflict", "before_model_call:call-owned"),
        ("model_call_reference_conflict", "model_calls:call-owned"),
        ("model_call_ownership_unavailable", "model_calls:call-unowned"),
    }


def test_sandbox_observation_rejects_negative_usage() -> None:
    with pytest.raises(ValidationError):
        SandboxObservation(role="agent", cpu_time_s=-1)
