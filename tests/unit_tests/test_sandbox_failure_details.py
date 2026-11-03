# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from nemo_gym.core.failure_transport import FailureHeader
from nemo_gym.core.typed_exceptions import failure_report_for
from nemo_gym.sandbox.failure_details import ended_sandbox, provider_report, sandbox_failure_event


SANDBOX = "00000000-0000-4000-8000-000000000001"
POD = f"{SANDBOX}-0"


def _ended(reason: str, detail: str) -> Exception:
    """Shaped like the OpenSandbox SDK's SandboxApiException for an ended sandbox (HTTP 410)."""
    message = (
        f"Sandbox {SANDBOX} has ended ({reason}): 1/1 observed pods failed; primary reason=ContainerFailed; "
        f"sample pod={POD}; sample detail={detail}"
    )
    error = Exception(f"Get command status failed: HTTP 410: {message}")
    error.status_code = 410
    error.error = SimpleNamespace(code="SANDBOX::ENDED", message=message)
    return error


@pytest.mark.parametrize(
    "error, reason, cause, container, reported",
    [
        (
            _ended("OOMKilled", f"Pod {POD} container sandbox exited with code 137 (OOMKilled)"),
            "OOMKilled",
            "oom",
            "agent",
            "sandbox",
        ),
        (
            _ended("Error", f"Pod {POD} container sandbox exited with code 143 (Error)"),
            "Error",
            "unknown",
            "agent",
            "sandbox",
        ),
        (
            _ended(
                "Evicted",
                f"Pod {POD} entered Failed phase (Evicted): Pod ephemeral local storage usage exceeds the total "
                "limit of containers 20Gi.",
            ),
            "Evicted",
            "storage",
            "unknown",
            None,
        ),
        (
            _ended("Evicted", f"Pod {POD} entered Failed phase (Evicted): The node was low on resource: memory."),
            "Evicted",
            "evicted",
            "unknown",
            None,
        ),
        (
            _ended("OOMKilled", f"Pod {POD} container execd exited with code 137 (OOMKilled)"),
            "OOMKilled",
            "oom",
            "sidecar",
            "execd",
        ),
        (
            _ended("OOMKilled", f"Pod {POD} init container setup exited with code 137 (OOMKilled)"),
            "OOMKilled",
            "oom",
            "init",
            "setup",
        ),
    ],
)
def test_an_ended_sandbox_is_read_from_the_providers_own_report(error, reason, cause, container, reported) -> None:
    ended = ended_sandbox(error, execution_id="exec-1")
    assert (ended.cause, ended.container) == (cause, container)
    # The container's name is carried as the provider reported it: an observed fact, not the reading above.
    assert ended.details.container == reported
    # The event carries the provider's reason code as reported, the ids and the provider; no inferred cause.
    assert ended.details.provider_status_reason == reason
    assert (ended.details.sandbox_id, ended.details.execution_id, ended.details.provider) == (
        SANDBOX,
        "exec-1",
        "opensandbox",
    )
    assert ended.details.exit_code is None  # the container's exit status is not the command's


def test_gyms_own_notice_after_a_backend_loss_is_read_too() -> None:
    error = RuntimeError(
        "Sandbox was OOM-killed. OpenSandbox status: state='Failed', reason='OOMKilled', message='container sandbox "
        f"exited with code 137'; sandbox_id='{SANDBOX}'"
    )
    ended = ended_sandbox(error, sandbox_id="box-1")
    assert (ended.cause, ended.container, ended.details.provider_status_reason) == ("oom", "agent", "OOMKilled")
    assert ended.details.container == "sandbox"
    assert ended.details.sandbox_id == "box-1"  # the caller's id wins over the reported one


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("status read timed out"),
        RuntimeError("command exited with code 137"),  # an exit status alone names no reason
        Exception("HTTP 503: service unavailable"),
    ],
)
def test_an_error_without_a_provider_report_of_an_ended_sandbox_reads_as_none(error) -> None:
    assert ended_sandbox(error) is None
    assert sandbox_failure_event(error, stage="agent") is None


def test_the_event_kind_follows_the_reason_and_the_event_fits_one_header() -> None:
    oom = sandbox_failure_event(
        _ended("OOMKilled", f"Pod {POD} container sandbox exited with code 137 (OOMKilled)"), stage="agent"
    )
    assert (oom.boundary, oom.failure_kind, oom.stage, oom.http_status) == (
        "sandbox",
        "provider_oom_killed",
        "agent",
        410,
    )
    evicted = sandbox_failure_event(_ended("Evicted", "x" * 2000), stage="verification")
    assert evicted.failure_kind == "provider_unavailable"
    # A long provider report stays out of the event (its reason code is short), so the report is never omitted.
    report = failure_report_for(evicted)
    assert len(json.dumps(report.model_dump(mode="json"))) < FailureHeader.MAX_BYTES
    assert FailureHeader.decode(FailureHeader.encode(report)) == report


@pytest.mark.parametrize(
    "container_line, kind, container",
    [
        (f"Pod {POD} container sandbox exited with code 137 (OOMKilled)", "provider_oom_killed", "sandbox"),
        # A helper container's memory kill is not the agent's own: infrastructure, retried.
        (f"Pod {POD} container execd exited with code 137 (OOMKilled)", "provider_unavailable", "execd"),
        (f"Pod {POD} init container setup exited with code 137 (OOMKilled)", "provider_unavailable", "setup"),
    ],
    ids=["agent", "sidecar", "init"],
)
def test_only_a_memory_kill_of_the_agents_own_container_is_an_oom_event(container_line, kind, container) -> None:
    event = sandbox_failure_event(_ended("OOMKilled", container_line), stage="agent")
    assert event.failure_kind == kind
    assert event.details.container == container
    assert event.details.provider_status_reason == "OOMKilled"


def test_gyms_own_notice_names_no_container_and_stays_an_oom_event() -> None:
    error = RuntimeError(
        "Sandbox was OOM-killed. OpenSandbox status: state='Failed', reason='OOMKilled', message='the backend was "
        f"lost'; sandbox_id='{SANDBOX}'"
    )
    event = sandbox_failure_event(error, stage="agent")
    assert event.failure_kind == "provider_oom_killed"
    assert event.details.container is None


def test_a_longest_reported_event_fits_one_header() -> None:
    # Every reported text at its longest: the details bound each to 128 characters, so the event is never omitted.
    long_name = "c" * 300
    error = _ended("OOMKilled", f"Pod {POD} container {long_name} exited with code 137 (OOMKilled)")
    event = sandbox_failure_event(
        error, stage="agent", sandbox_id="b" * 300, execution_id="e" * 300, provider="p" * 200, operation="o" * 100
    )
    assert event.details.container.startswith("c" * 100) and len(event.details.container) < 160  # bounded
    report = failure_report_for(event)
    [value] = FailureHeader.encode(report).values()
    assert len(value) < FailureHeader.MAX_BYTES and '"omitted":true' not in value
    assert FailureHeader.decode(FailureHeader.encode(report)) == report


def test_the_provider_report_is_the_sdks_own_text() -> None:
    error = _ended("OOMKilled", "x")
    assert provider_report(error) == error.error.message
