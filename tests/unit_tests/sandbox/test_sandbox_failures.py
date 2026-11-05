# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox boundary contracts, retry safety and recovered failure reporting."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from nemo_gym.core.failure_kinds import (
    AGENT_ROUTE_FAILED,
    PROVIDER_COMMAND_OUTCOME_UNKNOWN,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_OOM_KILLED,
    PROVIDER_REQUEST_FAILED,
    PROVIDER_STATUS_UNAVAILABLE,
    PROVIDER_UNAVAILABLE,
)
from nemo_gym.core.failure_transport import FailureHeader, RequestFailureLog
from nemo_gym.core.outcome_policy import OutcomeAction, failure_outcome_policy
from nemo_gym.core.typed_exceptions import NgFailureEvent, rebuild_failure
from nemo_gym.sandbox.failures import sandbox_failure
from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.sandbox.providers.opensandbox import provider as module
from nemo_gym.sandbox.providers.opensandbox.failures import opensandbox_failure


pytestmark = pytest.mark.sandbox


class HttpError(Exception):
    def __init__(self, status):
        self.status_code = status
        self.message = "request refused"
        super().__init__(self.message)


def test_causal_chain_roundtrips_sandbox_evidence_and_origin_ids():
    sdk = HttpError(502)
    inner = opensandbox_failure(operation="command status", sandbox_id="sb", execution_id="exec-1", cause=sdk)
    try:
        raise inner from sdk
    except NgFailureEvent as cause:
        try:
            raise NgFailureEvent(
                boundary="agent", failure_kind=AGENT_ROUTE_FAILED, failure_reason="Agent execution failed"
            ) from cause
        except NgFailureEvent as outer:
            encoded = json.loads(json.dumps(outer.to_record()))
    restored = NgFailureEvent.from_record(encoded)
    assert restored.__cause__.event_id == inner.event_id
    assert restored.__cause__.details.execution_id == "exec-1"
    assert restored.__cause__.details.provider == "opensandbox"
    assert "sandbox" not in encoded["failure_events"][inner.event_id]
    assert restored.__cause__.http_status == 502
    assert restored.__cause__.foreign_error.message == "request refused"
    assert restored.to_record() == encoded
    assert len(encoded["failure_events"]) == 2


def test_policy_overrides_merge_and_do_not_change_failure_facts():
    policy = module.OpenSandboxOutcomePolicy.model_validate(
        {
            "by_status_code": {"503": "fail"},
            "by_failure_kind": {PROVIDER_UNAVAILABLE: "fail"},
        }
    )
    assert policy.by_status_code[429] == OutcomeAction.RETRY
    error = opensandbox_failure(operation="status", sandbox_id="sb", cause=HttpError(502))
    before = error.to_record()
    assert failure_outcome_policy(error, config=policy) == OutcomeAction.RETRY
    error503 = opensandbox_failure(operation="status", sandbox_id="sb", cause=HttpError(503))
    assert failure_outcome_policy(error503, config=policy) == OutcomeAction.FAIL
    assert error.to_record() == before
    with pytest.raises(ValidationError):
        module.OpenSandboxOutcomePolicy(default_action="count_as_zero")
    with pytest.raises(ValidationError):
        module.OpenSandboxOutcomePolicy(by_failure_kind={"misspelled_kind": "retry"})


@pytest.mark.parametrize("message", ["HTTP 401 OOMKilled", "exit 137", "command timeout"])
def test_text_cannot_establish_cause_or_status(message):
    failure = opensandbox_failure(operation="run", sandbox_id="sb", cause=RuntimeError(message))
    assert failure.failure_kind == PROVIDER_REQUEST_FAILED
    assert failure.http_status is None
    assert failure.details.provider_status_reason is None


def provider_and_handle(monkeypatch, commands, **operations):
    monkeypatch.setattr(module, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object))
    provider = module.OpenSandboxProvider(
        operations={
            "background_exec": True,
            "background_poll_initial_s": 0.001,
            "background_poll_interval_s": 0.001,
            **operations,
        },
        diagnostics={"timeout_s": 0.01, "request_timeout_s": 0.005, "poll_interval_s": 0.001},
    )
    return provider, SandboxHandle("sb", "opensandbox", SimpleNamespace(commands=commands))


@pytest.mark.parametrize("error", [HttpError(502), TimeoutError("waiting for submission")])
async def test_policy_cannot_replay_uncertain_command_submission(monkeypatch, error):
    commands = SimpleNamespace(run=AsyncMock(side_effect=error))
    provider, handle = provider_and_handle(
        monkeypatch, commands, retries=10, command_retries=10, outcome_policy={"default_action": "retry"}
    )
    with pytest.raises(NgFailureEvent) as caught:
        await provider.exec(handle, "increment-counter")
    commands.run.assert_awaited_once()
    assert caught.value.__cause__ is error
    assert caught.value.failure_kind == (
        PROVIDER_IO_TIMEOUT if isinstance(error, TimeoutError) else PROVIDER_UNAVAILABLE
    )


async def test_recovered_status_failure_reaches_success_report(monkeypatch):
    commands = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(id="exec-1")),
        get_command_status=AsyncMock(
            side_effect=[TimeoutError("read"), SimpleNamespace(running=False, exit_code=0, error=None)]
        ),
        get_background_command_logs=AsyncMock(return_value=SimpleNamespace(content="ok")),
    )
    provider, handle = provider_and_handle(monkeypatch, commands)
    with RequestFailureLog.capture() as log:
        result = await provider.exec(handle, "echo ok")
        report = FailureHeader.decode(FailureHeader.encode(log.reply_report(None)))
    assert result.return_code == 0
    assert report.failure_id is None
    assert len(report.events) == 1
    event = next(iter(report.events.values()))
    assert event.failure_kind == PROVIDER_IO_TIMEOUT
    assert event.details.execution_id == "exec-1"
    assert event.foreign_error.exception_type == "TimeoutError"
    commands.run.assert_awaited_once()
    assert [call.args for call in commands.get_command_status.await_args_list] == [("exec-1",), ("exec-1",)]


async def test_recovery_and_diagnostics_share_remaining_deadline(monkeypatch):
    commands = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(id="exec-1")),
        get_command_status=AsyncMock(side_effect=TimeoutError("read")),
    )
    provider, handle = provider_and_handle(monkeypatch, commands, status_recovery_timeout_s=0.025)

    async def unavailable():
        await asyncio.sleep(60)

    handle.raw.get_info = unavailable
    start = asyncio.get_running_loop().time()
    with RequestFailureLog.capture() as log:
        with pytest.raises(NgFailureEvent) as caught:
            await provider.exec(handle, "long run", timeout_s=3600)
        final = log.reply_report(caught.value)
    assert asyncio.get_running_loop().time() - start < 0.2
    assert caught.value.failure_kind == PROVIDER_STATUS_UNAVAILABLE
    assert caught.value.__cause__.failure_kind == PROVIDER_IO_TIMEOUT
    restored = rebuild_failure(final)
    assert restored.event_id == caught.value.event_id
    assert restored.__cause__.event_id == caught.value.__cause__.event_id
    assert len(final.events) >= 2
    assert all(event.boundary.value == "sandbox" for event in final.events.values())
    commands.run.assert_awaited_once()


async def test_expired_evidence_deadline_does_not_start_another_request():
    provider = module.OpenSandboxProvider(diagnostics={"inspect_oom": True})
    raw = SimpleNamespace(get_info=AsyncMock())
    handle = SandboxHandle("sb", "opensandbox", raw)
    provider._control_plane_inspect_oom_evidence = AsyncMock()
    assert await provider._sandbox_death_evidence(handle, deadline=0) is None
    raw.get_info.assert_not_awaited()
    provider._control_plane_inspect_oom_evidence.assert_not_awaited()


@pytest.mark.parametrize("operation", ["status", "endpoint", "network_address"])
@pytest.mark.parametrize("action", ["retry", "fail"])
async def test_idempotent_lifecycle_reads_recover_wrapped_timeouts(operation, action):
    import httpx
    from opensandbox.exceptions import SandboxInternalException

    cause = httpx.ReadTimeout("read timed out")
    wrapped = SandboxInternalException("SDK read failed", cause=cause)
    response = (
        SimpleNamespace(status=SimpleNamespace(state="Running"))
        if operation == "status"
        else SimpleNamespace(endpoint="http://10.0.0.5:44772", headers={})
    )
    read = AsyncMock(side_effect=[wrapped, response])
    provider = module.OpenSandboxProvider(
        networking={"enabled": True},
        operations={
            "retries": 1,
            "retry_delay_s": 0,
            "retry_max_delay_s": 0,
            "outcome_policy": {"by_failure_kind": {PROVIDER_IO_TIMEOUT: action}},
        },
    )
    handle = SandboxHandle(
        "sb",
        "opensandbox",
        SimpleNamespace(
            get_info=read,
            get_endpoint=read,
            _sandbox_service=SimpleNamespace(get_sandbox_endpoint=read),
        ),
    )
    with RequestFailureLog.capture() as log:
        call = (
            getattr(provider, operation)(handle, 44772)
            if operation == "endpoint"
            else getattr(provider, operation)(handle)
        )
        if action == "fail":
            with pytest.raises(NgFailureEvent) as caught:
                await call
            assert caught.value.__cause__ is wrapped
        else:
            result = await call
            if operation == "status":
                assert result == module.SandboxStatus.RUNNING
            elif operation == "endpoint":
                assert result.endpoint == "http://10.0.0.5:44772"
            else:
                assert result == "10.0.0.5"
        assert read.await_count == (2 if action == "retry" else 1)
        event = next(iter(log.report.events.values()))
        assert event.failure_kind == PROVIDER_IO_TIMEOUT
        assert event.details.sandbox_id == "sb"
        assert event.foreign_error.exception_type.endswith("SandboxInternalException")


@pytest.mark.parametrize(
    "expire_outer_cap", [True, False], ids=["execution-budget-expired", "diagnostic-budget-available"]
)
async def test_foreground_diagnostics_cannot_extend_the_execution_deadline(monkeypatch, expire_outer_cap):
    monkeypatch.setattr(module, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object))
    loop = asyncio.get_running_loop()
    real_time = loop.time
    elapsed = 0.0
    request_cancelled = asyncio.Event()

    async def run(*args, **kwargs):
        nonlocal elapsed
        if expire_outer_cap:
            elapsed += 61
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # A wedged SDK ignores its request deadline; the outer cap still owns diagnosis.
                request_cancelled.set()
                elapsed += 60
                await asyncio.Event().wait()
        raise TimeoutError("provider read timed out")

    async def get_info():
        nonlocal elapsed
        # One successful read uses the diagnostic budget, without real polling sleeps.
        elapsed += 5
        return SimpleNamespace(status=SimpleNamespace(state="Running", reason=None))

    commands = SimpleNamespace(run=AsyncMock(side_effect=run))
    lifecycle_read = AsyncMock(side_effect=get_info)
    provider = module.OpenSandboxProvider(
        operations={"background_exec": False, "command_timeout_headroom_s": 0, "exec_hard_cap_slack_s": 0},
        diagnostics={"timeout_s": 5, "request_timeout_s": 1, "poll_interval_s": 0.001},
    )
    handle = SandboxHandle("sb", "opensandbox", SimpleNamespace(commands=commands, get_info=lifecycle_read))
    # Cross the SDK's 60s request deadline, then the 120s hard cap in separate
    # event-loop turns. Scheduling load cannot reverse the cancellation order.
    with monkeypatch.context() as clock, pytest.raises(NgFailureEvent) as caught:
        clock.setattr(loop, "time", lambda: real_time() + elapsed)
        await asyncio.wait_for(provider.exec(handle, "one command", timeout_s=60), 125)
    assert caught.value.failure_kind == PROVIDER_IO_TIMEOUT
    assert isinstance(caught.value.__cause__, TimeoutError)
    commands.run.assert_awaited_once()
    if expire_outer_cap:
        assert request_cancelled.is_set()
        lifecycle_read.assert_not_awaited()
    else:
        # With execution budget remaining, diagnosis runs within its own shorter budget.
        lifecycle_read.assert_awaited_once()


@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize("state,reason", [("Failed", "Evicted"), ("Stopped", "Expired")])
async def test_confirmed_non_oom_terminal_state_preserves_the_original_failure(monkeypatch, background, state, reason):
    request_error = HttpError(502)
    commands = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(id="exec-1"))
        if background
        else AsyncMock(side_effect=request_error),
        get_command_status=AsyncMock(side_effect=request_error),
    )
    provider, handle = provider_and_handle(
        monkeypatch, commands, background_exec=background, status_recovery_timeout_s=0.025
    )
    handle.raw.get_info = AsyncMock(return_value=SimpleNamespace(status=SimpleNamespace(state=state, reason=reason)))
    with RequestFailureLog.capture() as log:
        with pytest.raises(NgFailureEvent) as caught:
            await provider.exec(handle, "one command", timeout_s=1)
        report = log.reply_report(caught.value)
    failure = caught.value
    assert failure.failure_kind == PROVIDER_UNAVAILABLE
    assert failure.details.provider_status_reason == reason
    assert isinstance(failure.__cause__, NgFailureEvent)
    assert failure.__cause__.__cause__ is request_error
    assert failure.__cause__.http_status == 502
    assert report.events[failure.event_id].cause_id == failure.__cause__.event_id
    commands.run.assert_awaited_once()
    if background:
        commands.get_command_status.assert_awaited_once_with("exec-1")


@pytest.mark.parametrize(
    "state,reason,message,kind",
    [
        ("Failed", "ContainerFailed", "sandbox OOMKilled", PROVIDER_UNAVAILABLE),
        ("Running", "OOMKilled", "", PROVIDER_COMMAND_OUTCOME_UNKNOWN),
        ("Failed", "OOMKilled", "", PROVIDER_UNAVAILABLE),
    ],
)
async def test_pty_uses_the_same_explicit_evidence(state, reason, message, kind):
    provider = module.OpenSandboxProvider(diagnostics={"timeout_s": 0.01, "poll_interval_s": 0.001})
    handle = SandboxHandle(
        "sb",
        "opensandbox",
        SimpleNamespace(
            get_info=AsyncMock(
                return_value=SimpleNamespace(status=SimpleNamespace(state=state, reason=reason, message=message))
            )
        ),
    )
    cause = ConnectionError("socket closed")
    failure = await provider._pty_failure(handle, cause)
    assert failure.failure_kind == kind
    assert failure.__cause__ is cause
    if state == "Failed":
        assert failure.details.provider_status_reason == reason
        assert "lifecycle status" in failure.failure_reason


@pytest.mark.parametrize("command", [True, False])
async def test_renewal_failure_prevents_new_provider_operation(command):
    provider = module.OpenSandboxProvider()

    async def fail():
        raise ConnectionError("renewal unavailable")

    renewal = asyncio.create_task(fail())
    with pytest.raises(ConnectionError):
        await renewal
    provider._renewals["sb"] = renewal
    submit = AsyncMock()
    with pytest.raises(NgFailureEvent) as caught:
        if command:
            await provider._submit_command(submit, operation="command run", sandbox_id="sb", timeout_s=1, retries=9)
        else:
            await provider._await_sdk_operation(submit, operation="write_file", sandbox_id="sb", timeout_s=1)
    assert caught.value.failure_kind == PROVIDER_UNAVAILABLE
    assert caught.value.operation == "lifetime renewal"
    assert caught.value.__cause__ is renewal.exception()
    submit.assert_not_awaited()


async def test_owned_session_deadline_is_observed_without_claiming_a_provider_request_timeout():
    from nemo_gym.core.failure_kinds import PROVIDER_COMMAND_TIMEOUT

    stopped = asyncio.Event()

    async def run(*args, **kwargs):
        await stopped.wait()
        return SimpleNamespace(exit_code=-1, error=None, logs=SimpleNamespace(stdout=[], stderr=[]))

    async def stop(*args):
        stopped.set()

    commands = SimpleNamespace(
        create_session=AsyncMock(return_value="session"), run_in_session=run, delete_session=stop
    )
    provider = module.OpenSandboxProvider()
    handle = SandboxHandle("sb", "opensandbox", SimpleNamespace(commands=commands))
    with RequestFailureLog.capture() as log:
        result = await provider.exec_with_background_services(handle, "sleep 10", timeout_s=0.01)
        report = log.reply_report(None)
    assert result.error_type == "timeout"
    assert report.failure_id is None
    assert [event.failure_kind for event in report.events.values()] == [PROVIDER_COMMAND_TIMEOUT]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_name, expected_kind", [("ReadTimeout", "provider_io_timeout"), ("ConnectError", "provider_unavailable")]
)
async def test_real_sdk_wrapped_transport_failure_recovers_and_retains_observation(transport_name, expected_kind):
    import httpx
    from opensandbox.adapters.command_adapter import CommandsAdapter
    from opensandbox.config import ConnectionConfig
    from opensandbox.models.sandboxes import SandboxEndpoint

    calls = []

    async def respond(request):
        calls.append(request)
        if len(calls) == 1:
            raise getattr(httpx, transport_name)("injected network failure", request=request)
        return httpx.Response(200, json={"id": "exec-1", "running": False, "exit_code": 0})

    adapter = CommandsAdapter(
        ConnectionConfig(transport=httpx.MockTransport(respond)),
        SandboxEndpoint(endpoint="example.invalid", headers={}),
    )
    commands = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(id="exec-1")),
        get_command_status=adapter.get_command_status,
        get_background_command_logs=AsyncMock(return_value=SimpleNamespace(content="done")),
    )
    provider = module.OpenSandboxProvider(
        operations={
            "background_exec": True,
            "background_poll_initial_s": 0.001,
            "retry_delay_s": 0,
            "retry_max_delay_s": 0,
        }
    )
    handle = SandboxHandle("sb", "opensandbox", SimpleNamespace(commands=commands))
    try:
        with RequestFailureLog.capture() as log:
            result = await provider.exec(handle, "echo ok", timeout_s=1)
            report = log.reply_report(None)
        assert result.return_code == 0
        assert len(calls) == 2
        commands.run.assert_awaited_once()
        assert report.failure_id is None
        event = next(e for e in report.events.values() if e.failure_kind == expected_kind)
        assert event.boundary.value == "sandbox"
        assert event.foreign_error.message
        assert event.details.sandbox_id == "sb"
    finally:
        await adapter._httpx_client.aclose()
        await adapter._sse_client.aclose()


def test_unknown_wrapper_does_not_borrow_transport_or_http_facts():
    import httpx
    from opensandbox.exceptions import SandboxApiException

    for cause in (httpx.ReadTimeout("timeout"), SandboxApiException("denied", status_code=401)):
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = cause
        failure = opensandbox_failure(operation="get_info", sandbox_id="sb", cause=wrapper)
        assert failure.failure_kind == "provider_request_failed"
        assert failure.http_status is None


def test_sdk_message_survives_event_serialization():
    from opensandbox.exceptions import SandboxApiException

    cause = SandboxApiException("explicit provider diagnostic", status_code=403)
    failure = opensandbox_failure(operation="create", cause=cause)
    report = failure.to_record()
    event = report["failure_events"][failure.event_id]
    assert event["foreign_error"]["message"] == "explicit provider diagnostic"
    assert event["failure_kind"] == "provider_credentials_rejected"
    assert event["http_status"] == 403
    assert failure.details.sandbox_id is None  # No invented sandbox id before creation succeeds.
    assert failure.details.provider == "opensandbox"


@pytest.mark.asyncio
async def test_create_recovered_failures_and_policy_override_preserve_facts():
    from opensandbox.exceptions import SandboxApiException

    cause = SandboxApiException("busy", status_code=503)
    handle = SandboxHandle("sb", "opensandbox", None)
    provider = module.OpenSandboxProvider(create={"retries": 1, "retry_delay_s": 0, "retry_max_delay_s": 0})
    provider._create_once = AsyncMock(side_effect=[cause, handle])
    with RequestFailureLog.capture() as log:
        assert await provider.create(module.SandboxSpec(image="test")) is handle
        report = log.reply_report(None)
    assert report.failure_id is None
    event = next(e for e in report.events.values() if e.boundary.value == "sandbox")
    assert event.http_status == 503 and event.failure_kind == "provider_unavailable"
    provider = module.OpenSandboxProvider(create={"retries": 4, "outcome_policy": {"by_status_code": {503: "fail"}}})
    provider._create_once = AsyncMock(side_effect=cause)
    with pytest.raises(NgFailureEvent) as caught:
        await provider.create(module.SandboxSpec(image="test"))
    provider._create_once.assert_awaited_once()
    assert caught.value.http_status == event.http_status
    assert caught.value.failure_kind == event.failure_kind


def test_create_policy_rejects_grading_actions():
    with pytest.raises(ValidationError, match="does not support count_as_zero"):
        module.OpenSandboxProvider(create={"outcome_policy": {"default_action": "count_as_zero"}})


def test_broken_exception_formatter_does_not_hide_observed_failure():
    class BadMessage(Exception):
        status_code = 503

        def __str__(self):
            raise RuntimeError("broken formatter")

    cause = BadMessage()
    failure = opensandbox_failure(operation="get_info", sandbox_id="sb", cause=cause)
    assert failure.failure_kind == "provider_unavailable" and failure.http_status == 503
    assert failure.__cause__ is cause
    assert failure.foreign_error.message == ""


@pytest.mark.parametrize(
    "code,expected",
    [
        ("KUBERNETES::POD_IP_NOT_AVAILABLE", PROVIDER_UNAVAILABLE),
        ("KUBERNETES::SANDBOX_NOT_FOUND", PROVIDER_REQUEST_FAILED),
    ],
)
async def test_structured_endpoint_availability_controls_create_retries(code, expected):
    from opensandbox.adapters.converter.response_handler import handle_api_error
    from opensandbox.exceptions import SandboxApiException

    # The installed SDK retains the structured code, even when both errors use HTTP 404.
    response = SimpleNamespace(status_code=404, headers={}, parsed=SimpleNamespace(code=code, message="same text"))
    with pytest.raises(SandboxApiException) as sdk_error:
        handle_api_error(response, "Get endpoint")
    failure = opensandbox_failure(operation="create", cause=sdk_error.value)
    assert failure.failure_kind == expected
    assert failure.http_status == 404
    handle = SandboxHandle("sb", "opensandbox", None)
    provider = module.OpenSandboxProvider(create={"retries": 1, "retry_delay_s": 0, "retry_max_delay_s": 0})
    provider._create_once = AsyncMock(side_effect=[sdk_error.value, handle])
    if expected == PROVIDER_UNAVAILABLE:
        assert await provider.create(module.SandboxSpec(image="test")) is handle
        assert provider._create_once.await_count == 2
    else:
        with pytest.raises(NgFailureEvent):
            await provider.create(module.SandboxSpec(image="test"))
        provider._create_once.assert_awaited_once()
    fail_policy = module.OpenSandboxCreateOutcomePolicy(by_status_code={404: "fail"})
    assert failure_outcome_policy(failure, config=fail_policy) == OutcomeAction.FAIL


@pytest.mark.parametrize("message", ["imagepullbackoff", "pod may still be starting", "gateway timeout"])
def test_creation_uses_sdk_readiness_evidence_not_message_markers(message):
    from opensandbox.exceptions import SandboxReadyTimeoutException, SandboxUnhealthyException

    for sdk_error in (SandboxReadyTimeoutException(message), SandboxUnhealthyException(message)):
        failure = opensandbox_failure(operation="create", cause=sdk_error)
        assert module._is_retryable_create_error(failure)
    assert not module._is_retryable_create_error(RuntimeError(message))


def test_retry_after_does_not_search_other_events_causes():
    import httpx
    from opensandbox.adapters.converter.exception_converter import ExceptionConverter

    request = httpx.Request("GET", "https://sandbox.example/status")
    response = httpx.Response(429, headers={"Retry-After": "27"}, content=b"", request=request)
    sdk_error = ExceptionConverter.to_sandbox_exception(
        httpx.HTTPStatusError("busy", request=request, response=response)
    )
    failure = opensandbox_failure(operation="status", cause=sdk_error)
    assert module._retry_delay(failure, fallback=1, max_delay_s=60) == 27
    outer = opensandbox_failure(operation="create", cause=failure, http_status=429)
    assert module._retry_delay(outer, fallback=1, max_delay_s=60) == 1
    response.headers["Retry-After"] = "invalid"
    assert module._retry_delay(failure, fallback=1, max_delay_s=60) == 1


async def test_retry_after_cannot_extend_status_recovery_budget(monkeypatch):
    import httpx
    from opensandbox.adapters.converter.exception_converter import ExceptionConverter

    request = httpx.Request("GET", "https://sandbox.example/status")
    response = httpx.Response(429, headers={"Retry-After": "27"}, content=b"", request=request)
    error = ExceptionConverter.to_sandbox_exception(httpx.HTTPStatusError("busy", request=request, response=response))
    commands = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(id="exec-1")),
        get_command_status=AsyncMock(side_effect=error),
    )
    provider, handle = provider_and_handle(monkeypatch, commands, status_recovery_timeout_s=0.03)
    start = asyncio.get_running_loop().time()
    with pytest.raises(NgFailureEvent) as caught:
        await provider.exec(handle, "long run", timeout_s=3600)
    assert asyncio.get_running_loop().time() - start < 0.2
    assert caught.value.failure_kind == PROVIDER_STATUS_UNAVAILABLE
    assert caught.value.__cause__.http_status == 429
    commands.get_command_status.assert_awaited_once()
    commands.run.assert_awaited_once()


def test_shared_constructor_keeps_provider_selected_facts_and_child_evidence():
    child = opensandbox_failure(operation="status", sandbox_id="sb", cause=HttpError(503))
    outer = sandbox_failure(
        operation="create",
        failure_kind=PROVIDER_REQUEST_FAILED,
        cause=child,
    )
    assert outer.failure_kind == PROVIDER_REQUEST_FAILED
    assert outer.http_status is None
    assert outer.foreign_error is None
    assert outer.details is None
    restored = NgFailureEvent.from_record(outer.to_record())
    assert restored.__cause__.event_id == child.event_id
    assert restored.__cause__.http_status == 503
    assert restored.__cause__.details.sandbox_id == "sb"


@pytest.mark.parametrize("primary", ["typed", "foreign", "cancel"])
async def test_failed_session_cleanup_preserves_the_primary_failure(primary):
    original = opensandbox_failure(operation="run_in_session", sandbox_id="sb", cause=TimeoutError("read expired"))
    if primary == "foreign":
        original = RuntimeError("command stream failed")
    started = asyncio.Event()

    async def run(*args, **kwargs):
        started.set()
        if primary == "cancel":
            await asyncio.Future()
        raise original

    cleanup_error = HttpError(503)
    commands = SimpleNamespace(
        create_session=AsyncMock(return_value="session-1"),
        run_in_session=run,
        delete_session=AsyncMock(side_effect=cleanup_error),
    )
    provider = module.OpenSandboxProvider()
    handle = SandboxHandle("sb", "opensandbox", SimpleNamespace(commands=commands))
    with RequestFailureLog.capture() as log:
        request = asyncio.create_task(provider.exec_with_background_services(handle, "command", timeout_s=5))
        if primary == "cancel":
            await started.wait()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            report = log.reply_report(None)
            assert report.failure_id is None
        else:
            with pytest.raises(NgFailureEvent) as caught:
                await request
            if primary == "typed":
                assert caught.value is original
            else:
                assert caught.value.__cause__ is original
            report = log.reply_report(caught.value)
            assert report.failure_id == caught.value.event_id
            assert report.events[report.failure_id].operation == "run_in_session"
        [cleanup] = [event for event in report.events.values() if event.operation == "delete bash session"]
        assert cleanup.http_status == 503
        assert cleanup.cause_id is None
    commands.delete_session.assert_awaited_once()


@pytest.mark.parametrize("action,attempts", [("retry", 2), ("fail", 1)])
async def test_download_timeout_follows_policy_and_preserves_attempts(tmp_path, action, attempts):
    calls = 0

    async def read(path):
        nonlocal calls
        calls += 1
        attempt = calls

        async def chunks():
            yield b"partial" if attempt == 1 else b"complete"
            if attempt == 1:
                raise TimeoutError("stream read expired")

        return chunks()

    provider = module.OpenSandboxProvider(
        operations={
            "retries": 1,
            "retry_delay_s": 0,
            "outcome_policy": {"by_failure_kind": {PROVIDER_IO_TIMEOUT: action}},
        }
    )
    handle = SandboxHandle("sb", "opensandbox", SimpleNamespace(files=SimpleNamespace(read_bytes_stream=read)))
    target = tmp_path / "artifact"
    with RequestFailureLog.capture() as log:
        if action == "retry":
            await provider.download_file(handle, "/artifact", target)
            assert target.read_bytes() == b"complete"
            report = log.reply_report(None)
            assert report.failure_id is None
        else:
            with pytest.raises(NgFailureEvent) as caught:
                await provider.download_file(handle, "/artifact", target)
            report = log.reply_report(caught.value)
        assert len(report.events) == 1
        [event] = report.events.values()
        assert event.failure_kind == PROVIDER_IO_TIMEOUT
        assert event.operation == "download_file"
    assert calls == attempts


async def test_download_timeout_retry_budget_is_bounded(tmp_path):
    read = AsyncMock(side_effect=TimeoutError("stream read expired"))
    provider = module.OpenSandboxProvider(
        operations={
            "retries": 2,
            "retry_delay_s": 0,
            "outcome_policy": {"by_failure_kind": {PROVIDER_IO_TIMEOUT: "retry"}},
        }
    )
    handle = SandboxHandle("sb", "opensandbox", SimpleNamespace(files=SimpleNamespace(read_bytes_stream=read)))
    with RequestFailureLog.capture() as log:
        with pytest.raises(NgFailureEvent) as caught:
            await provider.download_file(handle, "/artifact", tmp_path / "artifact")
        report = log.reply_report(caught.value)
    assert read.await_count == 3
    assert len(report.events) == 3
    assert all(event.failure_kind == PROVIDER_IO_TIMEOUT for event in report.events.values())


@pytest.mark.parametrize("phase", ["operation", "create", "connect", "status", "orphan-list", "orphan-kill"])
@pytest.mark.parametrize("cap", [0, 0.25])
async def test_retry_after_obeys_configured_delay_cap(monkeypatch, phase, cap):
    from functools import partial

    import httpx
    from opensandbox.adapters.converter.exception_converter import ExceptionConverter

    request = httpx.Request("GET", "https://sandbox.example/status")
    response = httpx.Response(429, headers={"Retry-After": "3600"}, request=request)
    error = ExceptionConverter.to_sandbox_exception(httpx.HTTPStatusError("busy", request=request, response=response))
    waits = []

    async def sleep(delay):
        waits.append(float(delay))

    retry_tools = module._require_tenacity()
    monkeypatch.setattr(module, "_require_tenacity", lambda: (partial(retry_tools[0], sleep=sleep), *retry_tools[1:]))
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    config = {"retries": 1, "retry_delay_s": 0, "retry_max_delay_s": cap}
    if phase in ("create", "connect"):
        provider = module.OpenSandboxProvider(create={**config, "connect_poll_s": 0.01})
    else:
        provider = module.OpenSandboxProvider(
            operations={
                **config,
                "background_exec": True,
                "background_poll_initial_s": 0.01,
                "background_poll_interval_s": 0.01,
            }
        )
    action = AsyncMock(side_effect=[error, "recovered"])
    with RequestFailureLog.capture() as log:
        if phase == "create":
            provider._create_once = action
            assert await provider.create(module.SandboxSpec(image="test")) == "recovered"
        elif phase == "operation":
            assert (
                await provider._await_sdk_operation(
                    action,
                    operation="status",
                    sandbox_id="sb",
                    timeout_s=1,
                    replay_safe=True,
                )
                == "recovered"
            )
        elif phase == "connect":
            action = AsyncMock(side_effect=[error, SimpleNamespace(id="sb")])
            sdk = SimpleNamespace(connect=action)
            monkeypatch.setattr(module, "_require_opensandbox_sdk", lambda: (sdk, object, dict, object, object))
            monkeypatch.setattr(provider, "_connection_config", lambda **kwargs: None)
            handle = await provider._connect_after_create(
                SandboxHandle("sb", "opensandbox", None),
                module.SandboxSpec(image="test", ready_timeout_s=1),
            )
            assert handle.sandbox_id == "sb"
        elif phase == "status":
            action = AsyncMock(side_effect=[error, SimpleNamespace(running=False, exit_code=0, error=None)])
            commands = SimpleNamespace(
                run=AsyncMock(return_value=SimpleNamespace(id="exec-1")),
                get_command_status=action,
                get_background_command_logs=AsyncMock(return_value=SimpleNamespace(content="done")),
            )
            monkeypatch.setattr(provider, "_sandbox_death_evidence", AsyncMock(return_value=None))
            result = await provider.exec(
                SandboxHandle("sb", "opensandbox", SimpleNamespace(commands=commands)), "echo ok", timeout_s=1
            )
            assert result.return_code == 0
            commands.run.assert_awaited_once()
        else:
            page = SimpleNamespace(
                sandbox_infos=[SimpleNamespace(id="sb", metadata={"attempt": "owned"}, status=None)],
                pagination=SimpleNamespace(has_next_page=False),
            )
            service = SimpleNamespace(
                list_sandboxes=AsyncMock(side_effect=[error, page] if phase == "orphan-list" else [page]),
                kill_sandbox=AsyncMock(side_effect=[error, None] if phase == "orphan-kill" else [None]),
            )
            factory = lambda config: SimpleNamespace(create_sandbox_service=lambda: service)
            monkeypatch.setattr(module, "_require_opensandbox_lifecycle_sdk", lambda: (factory, lambda **kwargs: None))
            monkeypatch.setattr(provider, "_connection_config", lambda **kwargs: None)
            found = await provider._reap_orphaned_create({"attempt": "owned"}, "attempt")
            assert [item["id"] for item in found] == ["sb"]
            action = service.list_sandboxes if phase == "orphan-list" else service.kill_sandbox
    assert action.await_count == 2
    # Retrying a failed poll never shortens its explicit polling interval.
    assert waits == [max(0.01, cap) if phase in ("connect", "status") else cap]
    assert len(log.report.events) == 1
    assert next(iter(log.report.events.values())).http_status == 429


@pytest.mark.parametrize("header", [None, "invalid"])
def test_retry_cap_does_not_replace_polling_fallback(header):
    import httpx
    from opensandbox.adapters.converter.exception_converter import ExceptionConverter

    headers = {} if header is None else {"Retry-After": header}
    request = httpx.Request("GET", "https://sandbox.example/status")
    response = httpx.Response(429, headers=headers, request=request)
    error = ExceptionConverter.to_sandbox_exception(httpx.HTTPStatusError("busy", request=request, response=response))
    failure = opensandbox_failure(operation="status", cause=error)
    assert module._retry_delay(failure, fallback=2, max_delay_s=0) == 2
    assert module._retry_delay(None, fallback=2, max_delay_s=0) == 2


@pytest.mark.parametrize("action", ["verify", "count_as_zero"])
def test_class_actions_cannot_bypass_provider_supported_actions(action):
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxOutcomePolicy

    with pytest.raises(ValueError):
        OpenSandboxOutcomePolicy(by_failure_class={"evaluation": action})


def test_provider_create_class_action_uses_configured_classification():
    policy = module.OpenSandboxCreateOutcomePolicy(
        by_failure_class={"model": "retry"},
        classification={"by_failure_kind": {"provider_request_failed": "model"}},
    )
    failure = NgFailureEvent(boundary="sandbox", failure_kind="provider_request_failed", failure_reason="SDK failed")
    assert module._is_retryable_create_error(failure, config=policy)
    policy.classification.by_failure_kind["provider_request_failed"] = "evaluation"
    assert not module._is_retryable_create_error(failure, config=policy)


@pytest.mark.parametrize("final_inspect", ["current user container: OOMKilled", None, RuntimeError("inspect failed")])
async def test_terminal_transition_rechecks_inspect_once_with_original_deadline(monkeypatch, final_inspect):
    loop = asyncio.get_running_loop()
    now = loop.time()
    deadline = now + 1

    async def advance(delay):
        nonlocal now
        now += delay

    monkeypatch.setattr(loop, "time", lambda: now)
    monkeypatch.setattr(asyncio, "sleep", advance)
    raw = SimpleNamespace(
        get_info=AsyncMock(
            side_effect=[
                SimpleNamespace(status=SimpleNamespace(state="Running", reason=None)),
                SimpleNamespace(status=SimpleNamespace(state="Running", reason=None)),
                SimpleNamespace(status=SimpleNamespace(state="Failed", reason="OOMKilled")),
            ]
        )
    )
    provider = module.OpenSandboxProvider(diagnostics={"inspect_oom": True, "poll_interval_s": 0.1})
    inspect = AsyncMock(side_effect=[None, final_inspect])
    monkeypatch.setattr(provider, "_control_plane_inspect_oom_evidence", inspect)
    handle = SandboxHandle("sb", "opensandbox", raw)

    evidence = await provider._sandbox_death_evidence(handle, deadline=deadline)

    assert evidence.failure_kind == (PROVIDER_OOM_KILLED if isinstance(final_inspect, str) else PROVIDER_UNAVAILABLE)
    assert evidence.provider_status_reason == "OOMKilled"
    assert raw.get_info.await_count == 3
    assert inspect.await_count == 2  # Not once per Running poll, nor repeatedly after terminal state.
    assert [call.kwargs["deadline"] for call in inspect.await_args_list] == [deadline, deadline]
    assert now < deadline


async def test_terminal_transition_does_not_inspect_after_remaining_budget_expires(monkeypatch):
    loop = asyncio.get_running_loop()
    now = loop.time()
    deadline = now + 1
    lifecycle_reads = 0

    async def advance(delay):
        nonlocal now
        now += delay

    async def get_info():
        nonlocal now, lifecycle_reads
        lifecycle_reads += 1
        if lifecycle_reads == 1:
            return SimpleNamespace(status=SimpleNamespace(state="Running", reason=None))
        now = deadline
        return SimpleNamespace(status=SimpleNamespace(state="Failed", reason="OOMKilled"))

    monkeypatch.setattr(loop, "time", lambda: now)
    monkeypatch.setattr(asyncio, "sleep", advance)
    provider = module.OpenSandboxProvider(diagnostics={"inspect_oom": True, "poll_interval_s": 0.1})
    inspect = AsyncMock(return_value=None)
    monkeypatch.setattr(provider, "_control_plane_inspect_oom_evidence", inspect)
    handle = SandboxHandle("sb", "opensandbox", SimpleNamespace(get_info=get_info))

    evidence = await provider._sandbox_death_evidence(handle, deadline=deadline)

    assert evidence.failure_kind == PROVIDER_UNAVAILABLE
    assert lifecycle_reads == 2
    inspect.assert_awaited_once_with(handle, deadline=deadline)
    assert now == deadline


async def test_already_terminal_lifecycle_inspects_only_once():
    provider = module.OpenSandboxProvider(diagnostics={"inspect_oom": True})
    raw = SimpleNamespace(get_info=AsyncMock(return_value=SimpleNamespace(status=SimpleNamespace(state="Failed"))))
    handle = SandboxHandle("sb", "opensandbox", raw)
    provider._control_plane_inspect_oom_evidence = AsyncMock(return_value=None)

    evidence = await provider._sandbox_death_evidence(handle, deadline=asyncio.get_running_loop().time() + 1)

    assert evidence.failure_kind == PROVIDER_UNAVAILABLE
    raw.get_info.assert_awaited_once()
    provider._control_plane_inspect_oom_evidence.assert_awaited_once()
