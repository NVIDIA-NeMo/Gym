# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.agents.agent_outcome import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentOutcome,
    AgentOutcomePolicyConfig,
    agent_outcome_policy,
)
from nemo_gym.agents.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.failure_kinds import FAILURE_KINDS, MODEL_REQUEST_ERROR, MODEL_REQUEST_REJECTED
from nemo_gym.core.responses_api_types import NeMoGymResponse


def test_inherited_outcome_reporting_preserves_response_and_marks_unknown():
    class Agent(SimpleResponsesAPIAgent):
        async def responses(self, body):
            pass

        async def run(self, body):
            pass

    agent = Agent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )
    response = NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
        status="failed",
        error={"code": "server_error", "message": "Unclassified native failure"},
        metadata={"existing": "retained"},
    )
    reported = agent.report_outcome(response, {"error": "unrecognized"})
    assert reported.metadata == {"existing": "retained", AGENT_OUTCOME_METADATA_KEY: '{"failure_kind":"unknown"}'}
    assert reported.model_dump(exclude={"metadata"}) == response.model_dump(exclude={"metadata"})
    for status in ("completed", "incomplete"):
        response = response.model_copy(update={"status": status})
        assert agent.report_outcome(response, {}) is response


@pytest.mark.parametrize(
    "fields",
    [
        {"failure_kind": "model_request_error"},
        {"failure_kind": "unknown", "status_code": 503},
        {"failure_kind": "model_request_error", "status_code": True},
        {"failure_kind": "model_request_error", "status_code": 200},
    ],
)
def test_rejects_inconsistent_model_request_evidence(fields):
    with pytest.raises(ValidationError):
        AgentOutcome.model_validate(fields)


@pytest.mark.parametrize("kind", sorted(FAILURE_KINDS | {"unknown"}))
def test_accepts_shared_failure_vocabulary(kind):
    fields = {"failure_kind": kind}
    if kind == MODEL_REQUEST_ERROR:
        fields["status_code"] = 503
    outcome = AgentOutcome(**fields)
    assert AgentOutcome.model_validate_json(outcome.model_dump_json()) == outcome


@pytest.mark.parametrize(
    "kind", ["", "typo", "model_timeout_typo", "agent:custom", "unknown ", "limit", "invalid_output"]
)
def test_rejects_unregistered_failure_kinds(kind):
    import json

    fields = {"failure_kind": kind}
    with pytest.raises(ValidationError, match="Unregistered failure kind"):
        AgentOutcome(**fields)
    with pytest.raises(ValidationError, match="Unregistered failure kind"):
        AgentOutcome.model_validate_json(json.dumps(fields))


def test_class_action_cannot_bypass_environment_supported_actions():
    from pydantic import ValidationError

    from nemo_gym.agents.agent_outcome import AgentOutcomePolicyConfig

    with pytest.raises(ValidationError, match="does not support count_as_zero"):
        AgentOutcomePolicyConfig(by_failure_class={"model": "count_as_zero"})


def test_agent_class_action_uses_its_configured_shared_mapping():
    from nemo_gym.agents.agent_outcome import (
        AGENT_OUTCOME_METADATA_KEY,
        AgentOutcomePolicyConfig,
        agent_outcome_policy,
    )
    from nemo_gym.core.responses_api_types import NeMoGymResponse

    response = NeMoGymResponse(
        id="test",
        object="response",
        created_at=0,
        model="test",
        output=[],
        status="failed",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        metadata={AGENT_OUTCOME_METADATA_KEY: '{"failure_kind":"model_policy_response_timeout"}'},
    )
    policy = AgentOutcomePolicyConfig(by_failure_class={"evaluation": "retry"})
    assert agent_outcome_policy(response, config=policy) == "fail"
    policy.classification.by_failure_kind["model_policy_response_timeout"] = "evaluation"
    assert agent_outcome_policy(response, config=policy) == "retry"


# Recorded from Pi against openai_model and gpt-4.1 with Pi's default output budget.
OUTPUT_CAP_REJECTION = (
    "400 max_tokens is too large: 131072. This model supports at most 32768 completion tokens, whereas you provided "
    "131072."
)


class _StatusAgent(SimpleResponsesAPIAgent):
    """A harness whose native result names the HTTP status of the model request that ended the run."""

    async def responses(self, body):
        pass

    async def run(self, body):
        pass

    def classify_outcome(self, native_result):
        return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=native_result["status"])


def _report(status: int, message: str, *, model_answered: bool | None, error_as_dict: bool = False):
    agent = _StatusAgent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )
    error = {"code": "server_error", "message": message}
    response = NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
        status="failed",
        error=None if error_as_dict else error,
    )
    if error_as_dict:
        # model_copy(update=...) keeps the error a dict, unvalidated, as Simple Strands sets it.
        response = response.model_copy(update={"error": error})
    return agent.report_outcome(response, {"status": status}, model_answered=model_answered)


REQUEST_ERROR_400 = {"failure_kind": "model_request_error", "status_code": 400}


@pytest.mark.parametrize("error_as_dict", [False, True])
@pytest.mark.parametrize(
    ("status", "model_answered", "message", "expected", "action"),
    [
        # Before any answer the endpoint refused the run's own request: not graded, not retried.
        (400, False, OUTPUT_CAP_REJECTION, {"failure_kind": "model_request_rejected"}, "fail"),
        # After an answer, or without evidence either way, the 400 stays graded.
        (400, True, OUTPUT_CAP_REJECTION, REQUEST_ERROR_400, "verify"),
        (400, None, OUTPUT_CAP_REJECTION, REQUEST_ERROR_400, "verify"),
        # A prompt over the context window is the model's own limit, as vLLM and OpenAI word it.
        (
            400,
            False,
            "This model's maximum context length is 32768 tokens. However, you requested 32818 tokens.",
            REQUEST_ERROR_400,
            "verify",
        ),
        (
            400,
            False,
            '{"error":{"message":"Your input exceeds the limit.","code":"context_length_exceeded"}}',
            REQUEST_ERROR_400,
            "verify",
        ),
        (413, False, "payload too large", {"failure_kind": "model_request_error", "status_code": 413}, "verify"),
        (422, False, "invalid request", {"failure_kind": "model_request_error", "status_code": 422}, "verify"),
        (429, False, "slow down", {"failure_kind": "model_request_error", "status_code": 429}, "retry"),
    ],
)
def test_a_400_before_any_model_answer_is_reported_as_a_rejected_request(
    status, model_answered, message, expected, action, error_as_dict
):
    reported = _report(status, message, model_answered=model_answered, error_as_dict=error_as_dict)
    assert json.loads(reported.metadata[AGENT_OUTCOME_METADATA_KEY]) == expected
    assert agent_outcome_policy(reported) == action


def test_a_rejected_request_follows_its_kind_rule_not_the_status_rules():
    """A rejected request carries no status, so a deployment grades it again by its kind, not by a 400 rule."""
    reported = _report(400, OUTPUT_CAP_REJECTION, model_answered=False)
    assert agent_outcome_policy(reported, config=AgentOutcomePolicyConfig(by_status_code={400: "retry"})) == "fail"
    policy = AgentOutcomePolicyConfig()
    policy.by_failure_kind[MODEL_REQUEST_REJECTED] = "verify"
    assert agent_outcome_policy(reported, config=policy) == "verify"
