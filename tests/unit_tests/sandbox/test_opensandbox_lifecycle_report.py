# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-plane lifecycle evidence confirms OOM; command-status bodies retain only request facts."""

import gzip
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from nemo_gym.core.failure_kinds import PROVIDER_OOM_KILLED, PROVIDER_UNAVAILABLE
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.sandbox.providers.opensandbox._http_transport import OpenSandboxHttpTransport
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider, _lifecycle_oom_evidence


pytestmark = pytest.mark.sandbox
pytest.importorskip("opensandbox")

LIFECYCLE_REPORT = (
    "1/1 observed pods failed; primary reason=ContainerFailed; sample pod=sandbox-oom-0; "
    "sample detail=Pod sandbox-oom-0 container sandbox exited with code 137 (OOMKilled)"
)
REPORT = f"Sandbox sandbox-oom has ended (OOMKilled): {LIFECYCLE_REPORT}"


@pytest.mark.parametrize(
    ("report", "sandbox_id", "state", "reason", "confirmed"),
    [
        (LIFECYCLE_REPORT, "sandbox-oom", "Failed", "FAILED", True),
        (LIFECYCLE_REPORT.replace("container sandbox", "container egress"), "sandbox-oom", "Failed", "FAILED", False),
        (
            LIFECYCLE_REPORT.replace("container sandbox", "init container sandbox"),
            "sandbox-oom",
            "Failed",
            "FAILED",
            False,
        ),
        (LIFECYCLE_REPORT, "previous-sandbox", "Failed", "FAILED", False),
        (LIFECYCLE_REPORT, None, "Failed", "FAILED", False),
        (LIFECYCLE_REPORT.replace("sandbox-oom-0", "previous-sandbox-0"), "sandbox-oom", "Failed", "FAILED", False),
        (LIFECYCLE_REPORT.replace("sandbox-oom-0", "pool-pod-0"), "sandbox-oom", "Failed", "FAILED", False),
        (LIFECYCLE_REPORT.replace("1/1 observed pods", "1/2 observed pods"), "sandbox-oom", "Failed", "FAILED", False),
        (
            LIFECYCLE_REPORT + "; container sidecar exited with code 137 (OOMKilled)",
            "sandbox-oom",
            "Failed",
            "FAILED",
            False,
        ),
        (LIFECYCLE_REPORT.replace("(OOMKilled)", "(Evicted)"), "sandbox-oom", "Failed", "FAILED", False),
        (LIFECYCLE_REPORT.replace("code 137", "code 0"), "sandbox-oom", "Failed", "FAILED", False),
        (LIFECYCLE_REPORT.replace("code 137", "code 256"), "sandbox-oom", "Failed", "FAILED", False),
        (LIFECYCLE_REPORT, "sandbox-oom", "Running", "FAILED", False),
        (LIFECYCLE_REPORT, "sandbox-oom", "Failed", "TASK_FAILED", False),
        (LIFECYCLE_REPORT, "sandbox-oom", "Failed", "OOMKilled", False),
        ("exit code 137 OOMKilled", "sandbox-oom", "Failed", "FAILED", False),
        ("Last termination: " + LIFECYCLE_REPORT, "sandbox-oom", "Failed", "FAILED", False),
        (LIFECYCLE_REPORT + "x" * 65536, "sandbox-oom", "Failed", "FAILED", False),
    ],
    ids=[
        "current-user",
        "sidecar",
        "init",
        "stale-sandbox",
        "missing-sandbox-id",
        "stale-pod",
        "pooled-pod",
        "multiple-pods",
        "multiple-containers",
        "evicted",
        "zero-exit",
        "invalid-exit",
        "running",
        "other-failure",
        "reason-only",
        "prose",
        "historical-state",
        "oversized",
    ],
)
def test_lifecycle_report_requires_current_unambiguous_control_plane_evidence(
    report, sandbox_id, state, reason, confirmed
):
    info = SimpleNamespace(id=sandbox_id, status=SimpleNamespace(state=state, reason=reason, message=report))
    evidence = _lifecycle_oom_evidence(info, sandbox_id="sandbox-oom")
    assert (evidence is not None) == confirmed
    if evidence is not None:
        assert evidence.failure_kind == PROVIDER_OOM_KILLED
        assert "container sandbox exited with code 137" in evidence.message
        assert evidence.provider_status_reason == "FAILED"
        assert "lifecycle status confirms" in evidence.message


@pytest.mark.asyncio
@pytest.mark.parametrize("status_action", ["fail", "retry"])
@pytest.mark.parametrize("body", ["oom", "sidecar", "oversized", "direct", "spoofed", "wrong-id", "delayed"])
async def test_command_status_410_uses_control_plane_confirmation_through_real_sdk(body, status_action, monkeypatch):
    from opensandbox.adapters.command_adapter import CommandsAdapter
    from opensandbox.exceptions import SandboxApiException
    from opensandbox.models.sandboxes import SandboxEndpoint

    # Even an exact matching 410 body is not proof: execd can return it through the proxy.
    payload = json.dumps({"code": "SANDBOX::ENDED", "message": REPORT})
    if body == "oversized":
        payload += " " * 65536
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(410, content=payload, headers={"x-request-id": "request-410"}, request=request)

    provider = OpenSandboxProvider(
        connection={"use_server_proxy": body != "direct"},
        probe={"command": None},
        diagnostics={"poll_interval_s": 0.001},
        operations={
            "background_exec": True,
            "retries": 0,
            "background_poll_initial_s": 0.001,
            "background_poll_interval_s": 0.001,
            "outcome_policy": {"by_status_code": {410: status_action}},
        },
    )
    monkeypatch.setattr(provider, "_build_transport", lambda: httpx.MockTransport(respond))
    adapter = CommandsAdapter(provider._connection_config(), SandboxEndpoint(endpoint="sandbox.test"))
    submitted = []
    lifecycle_reads = []

    async def run(command, *, opts):
        submitted.append(command)
        return SimpleNamespace(id="exec-1")

    async def get_info():
        lifecycle_reads.append("read")
        if body == "delayed" and len(lifecycle_reads) == 1:
            return SimpleNamespace(id="sandbox-oom", status=SimpleNamespace(state="Running", reason="RUNNING"))
        message = LIFECYCLE_REPORT
        if body == "sidecar":
            message = message.replace("container sandbox", "container egress")
        elif body == "spoofed":
            message = "Operation failed"
        # Pod deletion does not erase the sandbox controller's persisted failure condition.
        return SimpleNamespace(
            id="other-sandbox" if body == "wrong-id" else "sandbox-oom",
            status=SimpleNamespace(state="Failed", reason="FAILED", message=message),
        )

    handle = SandboxHandle(
        sandbox_id="sandbox-oom",
        provider_name="opensandbox",
        raw=SimpleNamespace(
            commands=SimpleNamespace(run=run, get_command_status=adapter.get_command_status), get_info=get_info
        ),
    )

    try:
        with pytest.raises(NgFailureEvent) as caught:
            await provider.exec(handle, "allocate memory", timeout_s=30)
        failure = caught.value
        confirmed = body not in {"sidecar", "spoofed", "wrong-id"}
        assert failure.failure_kind == (PROVIDER_OOM_KILLED if confirmed else PROVIDER_UNAVAILABLE)
        assert failure.details.sandbox_id == "sandbox-oom"
        assert failure.details.execution_id == "exec-1"
        assert failure.http_status is None  # Distinct terminal observation; the failed request owns 410.
        assert failure.__cause__.http_status == 410
        assert isinstance(failure.__cause__.__cause__, SandboxApiException)
        assert failure.__cause__.__cause__.request_id == "request-410"
        if confirmed:
            assert failure.details.exit_code is None  # The command status was not observed.
            assert "container sandbox exited with code 137" in failure.failure_reason
            assert failure.details.provider_status_reason == "FAILED"
        # RETRY returns to command status between diagnostic reads; FAIL spends
        # its remaining diagnostic budget awaiting the final lifecycle report.
        assert len(requests) == (2 if body == "delayed" and status_action == "retry" else 1)
        assert requests[0].url.path == "/command/status/exec-1"
        assert submitted == ["allocate memory"]
        assert lifecycle_reads == ["read"] * (2 if body == "delayed" else 1)
    finally:
        await adapter._httpx_client.aclose()
        await adapter._sse_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["exec", "pty"])
async def test_lifecycle_confirmation_is_shared_by_foreground_exec_and_pty(operation):
    provider = OpenSandboxProvider(operations={"background_exec": False, "retries": 0})
    error = ConnectionError("exec daemon unavailable")
    info = SimpleNamespace(
        id="sandbox-oom",
        status=SimpleNamespace(state="Failed", reason="FAILED", message=LIFECYCLE_REPORT),
    )
    raw = SimpleNamespace(
        commands=SimpleNamespace(run=AsyncMock(side_effect=error)), get_info=AsyncMock(return_value=info)
    )
    handle = SandboxHandle("sandbox-oom", "opensandbox", raw)
    if operation == "exec":
        with pytest.raises(NgFailureEvent) as caught:
            await provider.exec(handle, "allocate memory", timeout_s=30)
        failure = caught.value
        raw.commands.run.assert_awaited_once()
    else:
        failure = await provider._pty_failure(handle, error)
    assert failure.failure_kind == PROVIDER_OOM_KILLED
    assert failure.details.exit_code is None  # The command status was not observed.
    assert "container sandbox exited with code 137" in failure.failure_reason
    assert failure.details.provider_status_reason == "FAILED"
    assert failure.stage is None  # The provider establishes the cause, not who owns this execution.
    raw.get_info.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "method", "path"),
    [(404, "GET", "/command/status/id"), (410, "POST", "/command"), (410, "GET", "/files/id")],
)
async def test_ended_transport_preserves_other_response_paths(status, method, path):
    response = httpx.Response(status, content=b"unchanged")
    transport = OpenSandboxHttpTransport(httpx.MockTransport(lambda request: response))
    assert await transport.handle_async_request(httpx.Request(method, "http://sandbox.test" + path)) is response
    await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized", [False, True], ids=["gzip-proxy", "oversized-stream"])
async def test_ended_transport_bounds_stream_and_closes_before_error(oversized):
    class Stream(httpx.AsyncByteStream):
        closed = False
        chunks = 0

        async def __aiter__(self):
            if oversized:
                for _ in range(100):
                    self.chunks += 1
                    yield b"x" * 4096
            else:
                data = gzip.compress(json.dumps({"code": "SANDBOX::ENDED", "message": REPORT}).encode())
                for offset in range(0, len(data), 13):
                    self.chunks += 1
                    yield data[offset : offset + 13]

        async def aclose(self):
            self.closed = True

    stream = Stream()
    headers = {} if oversized else {"content-encoding": "gzip"}
    response = httpx.Response(410, stream=stream, headers=headers)
    transport = OpenSandboxHttpTransport(httpx.MockTransport(lambda request: response))
    request = httpx.Request("GET", "https://gateway.test/v1/sandboxes/sandbox-oom/proxy/44772/command/status/id")
    with pytest.raises(httpx.HTTPStatusError) as caught:
        await transport.handle_async_request(request)
    assert stream.closed
    assert caught.value.response.status_code == 410
    if oversized:
        assert stream.chunks == 17
        assert caught.value.response.content == b""
    else:
        assert caught.value.response.json() == {"code": "SANDBOX::ENDED", "message": REPORT}
        assert "content-encoding" not in caught.value.response.headers
