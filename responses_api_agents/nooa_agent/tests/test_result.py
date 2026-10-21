# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from responses_api_agents.nooa_agent.result import finalize_run_result
from responses_api_agents.nooa_agent.runner import NOOARunResult
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_gym_llm import model_response


@pytest.mark.parametrize(
    ("reason", "wire_reason", "expected_status"),
    [
        (None, None, "completed"),
        (NOOATerminationReason.POLICY_BUDGET_EXCEEDED, "policy_budget_exceeded", "incomplete"),
        (NOOATerminationReason.MAX_ITERATIONS_EXCEEDED, "max_iterations_exceeded", "incomplete"),
        (NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED, "max_output_tokens_exceeded", "incomplete"),
        (NOOATerminationReason.LOOP_DETECTED, "loop_detected", "incomplete"),
        (NOOATerminationReason.INVALID_POLICY_OUTPUT, "invalid_policy_output", "incomplete"),
        (NOOATerminationReason.TIMEOUT_EXCEEDED, "timeout_exceeded", "incomplete"),
        (NOOATerminationReason.CANCELLED, "cancelled", "cancelled"),
        (NOOATerminationReason.INFRASTRUCTURE_ERROR, "infrastructure_error", "failed"),
    ],
)
def test_termination_projection_preserves_status_and_wire_values(
    reason: NOOATerminationReason | None, wire_reason: str | None, expected_status: str
) -> None:
    episode = AgentEpisode(
        response=NeMoGymResponse.model_validate(model_response()),
        observations=AgentObservationBundle(source="nooa", records=[], gaps=[]),
    )
    result = NOOARunResult(
        episode=episode,
        return_value=None,
        model_cookies={},
        resource_cookies={},
        termination_reason=reason,
    )

    response, observations = finalize_run_result(result)

    serialized_response = json.loads(response.model_dump_json())
    assert serialized_response["status"] == expected_status
    assert response.output == episode.response.output
    assert response.incomplete_details is None
    serialized_observations = json.loads(observations.model_dump_json())
    if wire_reason is None:
        assert serialized_response["error"] is None
        assert serialized_observations["gaps"] == []
    else:
        if wire_reason in {
            "policy_budget_exceeded",
            "invalid_policy_output",
            "max_iterations_exceeded",
            "max_output_tokens_exceeded",
            "loop_detected",
            "cancelled",
        }:
            assert serialized_response["error"] is None
        else:
            assert serialized_response["error"] == {
                "code": "server_error",
                "message": f"NOOA execution terminated with {wire_reason}.",
            }
        assert serialized_observations["gaps"] == [
            {"code": wire_reason, "invocation_id": None, "detail": f"NOOA execution terminated with {wire_reason}."}
        ]
        assert json.loads(json.dumps({"nooa_termination_reason": reason})) == {"nooa_termination_reason": wire_reason}


def test_cancellation_clears_response_error_and_preserves_partial_output_and_diagnostic() -> None:
    episode = AgentEpisode(
        response=NeMoGymResponse.model_validate(
            {**model_response(), "error": {"code": "server_error", "message": "stale model error"}}
        ),
        observations=AgentObservationBundle(source="nooa"),
    )
    diagnostic = "Caller cancelled after partial output."
    result = NOOARunResult(
        episode=episode,
        return_value=None,
        model_cookies={},
        resource_cookies={},
        termination_reason=NOOATerminationReason.CANCELLED,
        termination_error=diagnostic,
    )

    response, observations = finalize_run_result(result)

    assert response.status == "cancelled"
    assert response.error is None
    assert response.output == episode.response.output
    assert [gap.model_dump(mode="json") for gap in observations.gaps] == [
        {"code": "cancelled", "invocation_id": None, "detail": diagnostic}
    ]


@pytest.mark.parametrize("reason", ["timeout", "agent_run_timeout", "transient", "legitimate", "unknown"])
def test_termination_reason_rejects_aliases_and_collector_routing_classes(reason: str) -> None:
    with pytest.raises(ValueError):
        NOOATerminationReason(reason)


@pytest.mark.parametrize("primary_transient", [False, True])
@pytest.mark.parametrize("cleanup_transient", [False, True])
def test_retry_classification_checks_primary_and_cleanup_with_cycles(
    primary_transient: bool, cleanup_transient: bool
) -> None:
    from responses_api_agents.nooa_agent.result import is_transient_infrastructure_error

    primary = ConnectionError("primary transport") if primary_transient else ValueError("primary failure")
    cleanup = TimeoutError("cleanup timeout") if cleanup_transient else ValueError("cleanup failure")
    primary.nooa_cleanup_error = cleanup
    cleanup.__context__ = primary
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = primary
    assert is_transient_infrastructure_error(wrapper) is (primary_transient or cleanup_transient)


def test_close_only_nontransient_cleanup_marker_does_not_loop() -> None:
    from responses_api_agents.nooa_agent.result import is_transient_infrastructure_error

    error = ValueError("close-only failure")
    error.nooa_cleanup_error = error
    assert not is_transient_infrastructure_error(error)


@pytest.mark.parametrize("first_transient", [False, True])
@pytest.mark.parametrize("second_transient", [False, True])
def test_retry_classification_checks_all_cleanup_errors_with_cycles(
    first_transient: bool, second_transient: bool
) -> None:
    from responses_api_agents.nooa_agent.result import is_transient_infrastructure_error

    primary = ValueError("task failure")
    first = ConnectionError("agent close") if first_transient else ValueError("agent close")
    second = TimeoutError("scope close") if second_transient else ValueError("scope close")
    primary.nooa_cleanup_errors = (first, second, first)
    primary.nooa_cleanup_error = second
    first.__context__ = primary
    second.__cause__ = primary
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = primary
    assert is_transient_infrastructure_error(wrapper) is (first_transient or second_transient)
