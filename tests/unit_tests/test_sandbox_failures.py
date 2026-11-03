# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox boundary contracts, retry safety and recovered failure reporting."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from nemo_gym.agent_outcome import OutcomeAction, failure_outcome_policy
from nemo_gym.core.failure_kinds import (
    AGENT_ROUTE_FAILED,
    PROVIDER_COMMAND_OUTCOME_UNKNOWN,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_OOM_KILLED,
    PROVIDER_REQUEST_FAILED,
    PROVIDER_STATUS_UNAVAILABLE,
    PROVIDER_UNAVAILABLE,
)
from nemo_gym.failure_transport import FailureHeader, RequestFailureLog
from nemo_gym.sandbox.failures import sandbox_failure
from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.sandbox.providers.opensandbox import provider as module
from nemo_gym.typed_exceptions import NgFailureEvent, rebuild_failure


pytestmark = pytest.mark.sandbox


class HttpError(Exception):
    def __init__(self, status):
        self.status_code = status
        self.message = "request refused"
        super().__init__(self.message)


def test_causal_chain_roundtrips_sandbox_evidence_and_origin_ids():
    sdk = HttpError(502)
    inner = sandbox_failure(operation="command status", sandbox_id="sb", execution_id="exec-1", cause=sdk)
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
    assert restored.__cause__.sandbox.execution_id == "exec-1"
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
    error = sandbox_failure(operation="status", sandbox_id="sb", cause=HttpError(502))
    before = error.to_record()
    assert failure_outcome_policy(error, config=policy) == OutcomeAction.RETRY
    error503 = sandbox_failure(operation="status", sandbox_id="sb", cause=HttpError(503))
    assert failure_outcome_policy(error503, config=policy) == OutcomeAction.FAIL
    assert error.to_record() == before
    with pytest.raises(ValidationError):
        module.OpenSandboxOutcomePolicy(default_action="count_as_zero")
    with pytest.raises(ValidationError):
        module.OpenSandboxOutcomePolicy(by_failure_kind={"misspelled_kind": "retry"})


@pytest.mark.parametrize("message", ["HTTP 401 OOMKilled", "exit 137", "command timeout"])
def test_text_cannot_establish_cause_or_status(message):
    failure = sandbox_failure(operation="run", sandbox_id="sb", cause=RuntimeError(message))
    assert failure.failure_kind == PROVIDER_REQUEST_FAILED
    assert failure.http_status is None
    assert failure.sandbox.evidence_source is None


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
    assert event.sandbox.execution_id == "exec-1"
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


@pytest.mark.parametrize(
    "state,reason,message,kind",
    [
        ("Failed", "ContainerFailed", "sandbox OOMKilled", PROVIDER_COMMAND_OUTCOME_UNKNOWN),
        ("Running", "OOMKilled", "", PROVIDER_COMMAND_OUTCOME_UNKNOWN),
        ("Failed", "OOMKilled", "", PROVIDER_OOM_KILLED),
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
    if kind == PROVIDER_OOM_KILLED:
        assert failure.sandbox.evidence_source == "lifecycle_status"


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
            await provider._await_sdk_operation(
                submit, operation="write_file", sandbox_id="sb", timeout_s=1, report_failure=True
            )
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
