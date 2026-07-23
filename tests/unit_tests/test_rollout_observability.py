# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_gym.base_responses_api_model import ModelCallRecord
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
    join_model_call_observations,
    link_tool_calls_to_sandbox,
    pop_response_observations,
    response_with_observations,
)


@pytest.mark.parametrize(
    "value",
    ({}, {"response_id": "resp-1"}, {"model_ref": {"name": "policy", "type": "responses_api_models"}}),
)
def test_model_call_ref_rejects_incomplete_join_keys(value: dict) -> None:
    with pytest.raises(ValidationError, match="model_call_id or both model_ref and response_id"):
        ModelCallRef.model_validate(value)


def test_observations_round_trip_through_internal_response() -> None:
    response = NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 0,
            "model": "model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )
    carried = response_with_observations(
        AgentEpisode(response=response, observations=AgentObservationBundle(source="test"))
    ).model_dump(mode="json")

    observations = pop_response_observations(carried)

    assert observations == AgentObservationBundle(source="test")
    assert "ng_agent_observations" not in carried


def test_parallel_tool_calls_share_enclosing_sandbox_without_overwriting_existing_link() -> None:
    bundle = AgentObservationBundle(
        source="test",
        records=[
            ToolCallObservation(
                invocation_id="root",
                tool_call_id="call-1",
                started_at=1.0,
                completed_at=3.0,
            ),
            ToolCallObservation(
                invocation_id="root",
                tool_call_id="call-2",
                started_at=2.0,
                completed_at=4.0,
            ),
            ToolCallObservation(
                invocation_id="child",
                tool_call_id="call-3",
                sandbox_id="other-sandbox",
            ),
        ],
    )

    link_tool_calls_to_sandbox(bundle, "shared-sandbox")

    tools = [record for record in bundle.records if isinstance(record, ToolCallObservation)]
    assert [tool.sandbox_id for tool in tools] == [
        "shared-sandbox",
        "shared-sandbox",
        "other-sandbox",
    ]


def test_observation_bundle_rejects_duplicate_invocation_ids() -> None:
    with pytest.raises(ValidationError, match="invocation_id must be unique"):
        AgentObservationBundle(
            source="test",
            records=[AgentInvocation(invocation_id="root"), AgentInvocation(invocation_id="root")],
        )


def test_observation_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="producer_extension"):
        ModelCallRef.model_validate({"model_call_id": "call-1", "producer_extension": "unexpected"})


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
        gaps=[ObservationGap(code="model_call_ownership_unavailable")],
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
    assert [gap.detail for gap in ownership_gaps] == ["capture:call-2:call_index=1"]


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
    bundle = AgentObservationBundle(
        source="test",
        records=[
            ContextCompactionObservation(
                invocation_id="root",
                before_model_call=ModelCallRef(model_ref=model_ref, response_id="before"),
                after_model_call=ModelCallRef(model_ref=model_ref, response_id="after"),
            )
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
            model_call_id="call-after",
            response_id="after",
            model_ref=model_ref,
            call_index=1,
        ),
    ]

    joined = join_model_call_observations(bundle, calls)
    [compaction] = [record for record in joined.records if isinstance(record, ContextCompactionObservation)]

    assert compaction.before_model_call.model_call_id == "call-before"
    assert compaction.after_model_call.model_call_id == "call-after"


def test_sandbox_observation_rejects_negative_usage() -> None:
    with pytest.raises(ValidationError):
        SandboxObservation(role="agent", cpu_time_s=-1)
