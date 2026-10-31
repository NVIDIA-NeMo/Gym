# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy validation at the real environment boundary; its server client is a labelled protocol stub."""

import pytest

from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.core.failure_kinds import MODEL_POLICY_RESPONSE_TIMEOUT
from nemo_gym.openai_utils import MODEL_RESPONSE_TIMEOUT_REASON
from tests.single_agent_turn_helpers import _agent_response, _environment_server, _request, _Response


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "omitted",
        "wrong_id",
        "wrong_kind",
        "wrong_operation",
        "missing_upload",
        "connect",
        "mixed",
        "duplicate_attempt",
        "foreign_operation",
    ],
)
async def test_unproven_timeout_never_counts_as_zero_or_verifies(damage):
    environment, client = _environment_server()
    client.global_config_dict.agent.responses_api_agents.test.final_on_model_timeout = True
    model_event = {
        "event_id": "model-attempt",
        "boundary": "model",
        "failure_kind": "model_request_error",
        "failure_reason": MODEL_RESPONSE_TIMEOUT_REASON,
        "operation": "model_request",
        "operation_id": "model-operation",
        "attempt_id": "model-operation-1",
        "request_upload_complete": True,
    }
    agent_event = {
        "event_id": "agent-timeout",
        "boundary": "agent",
        "failure_kind": MODEL_POLICY_RESPONSE_TIMEOUT,
        "operation": "agent.policy_request",
        "operation_id": "model-operation",
        "cause_id": "model-attempt",
    }
    report = {"failure_id": "agent-timeout", "events": {"model-attempt": model_event, "agent-timeout": agent_event}}
    if damage == "omitted":
        report["omitted"] = True
    elif damage == "wrong_id":
        report["failure_id"] = "not-present"
    elif damage == "wrong_kind":
        agent_event["failure_kind"] = "model_request_error"
    elif damage == "wrong_operation":
        agent_event["operation"] = "unrelated"
    elif damage == "missing_upload":
        model_event.pop("request_upload_complete")
    elif damage == "connect":
        model_event["request_upload_complete"] = False
    elif damage == "mixed":
        report["events"]["previous"] = dict(
            model_event,
            event_id="previous",
            attempt_id="model-operation-0",
            failure_reason="connection failed",
            request_upload_complete=False,
        )
    elif damage == "duplicate_attempt":
        report["events"]["previous"] = dict(model_event, event_id="previous")
    elif damage == "foreign_operation":
        model_event["operation_id"] = "other-operation"
    response = _agent_response().model_dump(mode="json")
    response.update(
        status="failed",
        output=[],
        error={"code": "server_error", "message": "scripted timeout"},
        metadata={
            AGENT_OUTCOME_METADATA_KEY: AgentOutcome(failure_kind=MODEL_POLICY_RESPONSE_TIMEOUT).model_dump_json()
        },
    )
    if damage != "missing":
        response["failure_report"] = report
    client.responses[2] = _Response(response)
    del client.responses[4]
    result = await environment.run_request(_request())
    assert result.result is None
    assert result.failure.terminal is True
    assert not any(path == "/verify" for _, path, _ in client.calls)
    assert result.failure.partial_response.model_dump(mode="json") == response
    assert result.failure.failure_kind is None
