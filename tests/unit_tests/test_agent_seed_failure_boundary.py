# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_gym.base_responses_api_agent import AgentSessionSetupError, AgentSessionState
from nemo_gym.episode_types import FailureBoundary, FailureEvent
from nemo_gym.failure_transport import FailureHeader
from nemo_gym.typed_exceptions import BoundaryFailure
from tests.unit_tests.test_agent_session_retries import _seed
from tests.unit_tests.test_session_routing import AgentWorker


def _served(label: str = "a") -> AgentWorker:
    worker = AgentWorker(label, None)
    # run_webserver installs the exception middleware after setup_webserver; mirror it.
    worker.agent.setup_exception_middleware(worker.app)
    return worker


def _failure(response):
    failure = FailureHeader.decode(response.headers)
    assert failure is not None and failure.failure_id is not None
    return failure, failure.events


def test_a_failed_seed_leaves_the_agent_server_as_an_agent_failure_caused_by_the_setup_error() -> None:
    worker = _served()
    with worker.client:
        worker.agent.fail_seed = True
        failed = worker.seed()

    assert failed.status_code == 500
    failure, events = _failure(failed)
    own = events[failure.failure_id]
    assert (own.boundary, own.operation, own.stage, own.failure_kind) == (
        "agent",
        "agent.seed_session",
        "seed",
        None,
    )
    cause = events[own.cause_id]
    # The foreign setup error is recorded by type only; its text stays out of the failure events.
    assert (cause.boundary, cause.exception_type, cause.failure_reason) == (None, "RuntimeError", "")


def test_a_setup_errors_own_cause_reaches_the_caller_with_its_kind() -> None:
    worker = _served()
    model = BoundaryFailure(
        FailureEvent(
            boundary=FailureBoundary.MODEL,
            failure_reason="model unavailable",
            operation="model.responses",
            http_status=429,
        )
    )
    setup = RuntimeError("setup failed")
    setup.__cause__ = model

    async def initialize(body):
        raise AgentSessionSetupError(AgentSessionState(request=body), error=setup)

    worker.agent._seed_agent_session_state = initialize
    with worker.client:
        failed = worker.seed()

    failure, events = _failure(failed)
    own = events[failure.failure_id]
    inner = events[events[own.cause_id].cause_id]
    assert (inner.boundary, inner.event_id, inner.http_status) == ("model", model.event.event_id, 429)
    assert own.failure_kind is None  # The agent boundary does not take its cause's kind.


async def test_a_direct_seed_call_still_raises_the_setup_error_itself_with_its_cause() -> None:
    worker = _served()
    root = ValueError("root")
    setup = RuntimeError("setup failed")
    setup.__cause__ = root

    async def initialize(body):
        raise AgentSessionSetupError(AgentSessionState(request=body), error=setup)

    worker.agent._seed_agent_session_state = initialize

    class _Request:
        session: dict = {}

    with pytest.raises(RuntimeError, match="setup failed") as raised:
        await worker.agent.seed_agent_session(_Request(), _seed())
    assert raised.value is setup and raised.value.__cause__ is root


def test_successful_and_refused_seeds_are_unchanged() -> None:
    worker = _served()
    with worker.client:
        assert worker.seed().status_code == 200
        assert worker.close().status_code == 200
        refused = worker.seed()

    assert refused.status_code == 409 and "already closed" in refused.json()["detail"]
    assert FailureHeader.decode(refused.headers) is None


def test_over_http_a_rejected_sandbox_credential_keeps_the_selected_sdk_message_and_status() -> None:
    from nemo_gym.sandbox.providers.base import SandboxCredentialsRejected

    class SdkError(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.status_code = 401

        def __str__(self) -> str:
            return f"{self.args[0]} request=POST /sandboxes api_key=secret"

    worker = _served()
    sdk = SdkError("Authentication credentials are invalid.")
    rejected = SandboxCredentialsRejected.from_provider_error(
        sdk, boundary=FailureBoundary.ENVIRONMENT, operation="sandbox.setup", sdk_message=sdk.args[0]
    )
    setup = RuntimeError("setup failed")
    setup.__cause__ = rejected

    async def initialize(body):
        raise AgentSessionSetupError(AgentSessionState(request=body), error=setup)

    worker.agent._seed_agent_session_state = initialize
    with worker.client:
        failed = worker.seed()

    assert failed.status_code == 500
    assert "secret" not in failed.headers[FailureHeader.NAME]
    failure, events = _failure(failed)
    credentials = events[rejected.event.event_id]
    assert (credentials.failure_kind, credentials.http_status) == ("provider_credentials_rejected", 401)
    assert credentials.foreign_error.message == "Authentication credentials are invalid."
    assert events[credentials.cause_id].http_status == 401 and events[credentials.cause_id].failure_reason == ""
