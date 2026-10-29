# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentOutcome,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import MODEL_REQUEST_ERROR, NeMoGymResponse
from nemo_gym.core.failure_kinds import FAILURE_KINDS


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
