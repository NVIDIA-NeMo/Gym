# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pickle

from nemo_gym.agent_utils.supervisor_client import HarnessTerminationError
from nemo_gym.base_responses_api_agent import AgentSessionSetupError, AgentSessionState
from nemo_gym.typed_exceptions import HarnessFailure, decode_header
from tests.unit_tests.test_agent_seed_failure_boundary import _served


def _receipt() -> dict:
    return {"cleanup_confirmed": False, "error": "descendants remain: pid 41"}


def test_an_unconfirmed_termination_is_still_the_same_runtime_error() -> None:
    error = HarnessTerminationError(harness="codex", receipt=_receipt())

    assert isinstance(error, RuntimeError) and isinstance(error, HarnessFailure)
    assert str(error) == "codex harness termination was not confirmed: descendants remain: pid 41"
    restored = pickle.loads(pickle.dumps(error))
    assert str(restored) == str(error) and restored.event_id == error.event_id


def test_a_seed_that_cannot_confirm_harness_termination_reports_agent_then_harness_failure() -> None:
    worker = _served()
    termination = HarnessTerminationError(harness="codex", receipt=_receipt())

    async def initialize(body):
        try:
            raise termination
        except HarnessTerminationError as error:
            raise AgentSessionSetupError(AgentSessionState(request=body), error=error) from error

    worker.agent._seed_agent_session_state = initialize
    with worker.client:
        failed = worker.seed()

    assert failed.status_code == 500
    failure = decode_header(failed.headers)
    events = failure.failure_events
    agent = events[failure.failure_id]
    harness = events[agent.cause_id]
    assert (agent.type, harness.type, harness.event_id) == ("AgentFailure", "HarnessFailure", termination.event_id)
    assert (harness.operation, harness.failure_kind) == ("harness.terminate", None)
    # The receipt's text stays on the exception; the event names only the harness.
    assert harness.failure_reason == "codex harness termination was not confirmed"
