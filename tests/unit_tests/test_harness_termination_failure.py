# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.agent_utils.supervisor_client import harness_termination_failure
from nemo_gym.base_responses_api_agent import AgentSessionSetupError, AgentSessionState
from nemo_gym.failure_transport import FailureHeader
from nemo_gym.typed_exceptions import NgFailureEvent
from tests.unit_tests.test_agent_seed_failure_boundary import _served


def _receipt() -> dict:
    return {"cleanup_confirmed": False, "error": "descendants remain: pid 41"}


def test_an_unconfirmed_termination_is_still_the_same_runtime_error() -> None:
    error = harness_termination_failure(harness="codex", receipt=_receipt())

    assert type(error) is NgFailureEvent
    assert str(error) == "codex harness termination was not confirmed: descendants remain: pid 41"


def test_a_seed_that_cannot_confirm_harness_termination_reports_agent_then_harness_failure() -> None:
    worker = _served()
    termination = harness_termination_failure(harness="codex", receipt=_receipt())

    async def initialize(body):
        try:
            raise termination
        except NgFailureEvent as error:
            raise AgentSessionSetupError(AgentSessionState(request=body), error=error) from error

    worker.agent._seed_agent_session_state = initialize
    with worker.client:
        failed = worker.seed()

    assert failed.status_code == 500
    failure = FailureHeader.decode(failed.headers)
    events = failure.events
    agent = events[failure.failure_id]
    harness = events[agent.cause_id]
    assert (agent.boundary, harness.boundary, harness.event_id) == ("agent", "harness", termination.event.event_id)
    assert (harness.operation, harness.failure_kind) == ("harness.terminate", "harness_termination_unconfirmed")
    # The receipt's text stays on the exception; the event names only the harness.
    assert harness.failure_reason == "codex harness termination was not confirmed"
