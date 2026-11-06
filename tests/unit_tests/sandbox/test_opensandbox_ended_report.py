# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The observed provider-ended report must survive the real SDK status adapter."""

import gzip
import json
from types import SimpleNamespace

import httpx
import pytest

from nemo_gym.core.failure_kinds import PROVIDER_OOM_KILLED, PROVIDER_UNAVAILABLE
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.sandbox.providers.opensandbox._http_transport import OpenSandboxHttpTransport
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider, _ended_report_evidence


pytestmark = pytest.mark.sandbox
pytest.importorskip("opensandbox")

REPORT = (
    "Sandbox sandbox-oom has ended (OOMKilled): 1/1 observed pods failed; primary reason=ContainerFailed; "
    "sample pod=sandbox-oom-0; sample detail=Pod sandbox-oom-0 container sandbox exited with code 137 (OOMKilled)"
)


@pytest.mark.parametrize(
    ("report", "code", "status", "confirmed"),
    [
        (REPORT, "SANDBOX::ENDED", 410, True),
        (REPORT.replace("container sandbox", "container egress"), "SANDBOX::ENDED", 410, False),
        (REPORT.replace("Sandbox sandbox-oom ", "Sandbox previous-sandbox "), "SANDBOX::ENDED", 410, False),
        (REPORT.replace("1/1 observed pods", "1/2 observed pods"), "SANDBOX::ENDED", 410, False),
        (REPORT + "; container sidecar exited with code 137 (OOMKilled)", "SANDBOX::ENDED", 410, False),
        (REPORT.replace("(OOMKilled)", "(Evicted)"), "SANDBOX::ENDED", 410, False),
        (REPORT, "OTHER", 410, False),
        (REPORT, "SANDBOX::ENDED", 502, False),
        ("exit code 137 OOMKilled", "SANDBOX::ENDED", 410, False),
        (REPORT + "x" * 65536, "SANDBOX::ENDED", 410, False),
    ],
    ids=[
        "current-user",
        "sidecar",
        "stale-sandbox",
        "multiple-pods",
        "multiple-containers",
        "evicted",
        "wrong-code",
        "wrong-status",
        "prose",
        "oversized",
    ],
)
def test_ended_report_requires_current_unambiguous_provider_evidence(report, code, status, confirmed):
    from opensandbox.exceptions import SandboxApiException, SandboxError

    error = SandboxApiException("opaque", status_code=status, error=SandboxError(code, report))
    evidence = _ended_report_evidence(error, sandbox_id="sandbox-oom", via_server_proxy=True)
    assert (evidence is not None) == confirmed
    if evidence is not None:
        assert evidence.failure_kind == PROVIDER_OOM_KILLED
        assert evidence.exit_code == 137
        assert "provider ended report" in evidence.message
        assert _ended_report_evidence(error, sandbox_id="sandbox-oom", via_server_proxy=False) is None
    # An exception merely quoting the same bytes is never a provider report.
    assert _ended_report_evidence(RuntimeError(str(error)), sandbox_id="sandbox-oom", via_server_proxy=True) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["oom", "sidecar", "oversized", "direct", "default"])
async def test_command_status_410_survives_real_sdk_and_selects_evidence(body, monkeypatch):
    from opensandbox.adapters.command_adapter import CommandsAdapter
    from opensandbox.exceptions import SandboxApiException
    from opensandbox.models.sandboxes import SandboxEndpoint

    report = REPORT.replace("container sandbox", "container egress") if body == "sidecar" else REPORT
    payload = json.dumps({"code": "SANDBOX::ENDED", "message": report})
    if body == "oversized":
        payload += " " * 65536
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(410, content=payload, headers={"x-request-id": "request-410"}, request=request)

    provider = OpenSandboxProvider(
        connection={"use_server_proxy": body != "direct"},
        probe={"command": None},
        diagnostics={} if body == "default" else {"ended_report_oom": True},
        operations={"background_exec": True, "retries": 0},
    )
    monkeypatch.setattr(provider, "_build_transport", lambda: httpx.MockTransport(respond))
    adapter = CommandsAdapter(provider._connection_config(), SandboxEndpoint(endpoint="sandbox.test"))
    submitted = []

    async def run(command, *, opts):
        submitted.append(command)
        return SimpleNamespace(id="exec-1")

    async def get_info():
        return SimpleNamespace(status=SimpleNamespace(state="Failed", reason="FAILED"))

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
        assert failure.failure_kind == (PROVIDER_OOM_KILLED if body == "oom" else PROVIDER_UNAVAILABLE)
        assert failure.details.sandbox_id == "sandbox-oom"
        assert failure.details.execution_id == "exec-1"
        assert failure.http_status is None  # Distinct terminal observation; the failed request owns 410.
        assert failure.__cause__.http_status == 410
        assert isinstance(failure.__cause__.__cause__, SandboxApiException)
        assert failure.__cause__.__cause__.request_id == "request-410"
        if body == "oom":
            assert failure.details.exit_code == 137
            assert failure.details.provider_status_reason == "OOMKilled"
        assert len(requests) == 1
        assert requests[0].url.path == "/command/status/exec-1"
        assert submitted == ["allocate memory"]
    finally:
        await adapter._httpx_client.aclose()
        await adapter._sse_client.aclose()


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


def test_unrecognized_ended_report_warning_is_bounded_and_suppressible(caplog):
    from opensandbox.exceptions import SandboxApiException, SandboxError

    report = REPORT + "!" * 1000
    error = SandboxApiException("opaque", status_code=410, error=SandboxError("SANDBOX::ENDED", report))
    for first in (True, False):
        assert (
            _ended_report_evidence(
                error,
                sandbox_id="sandbox-oom",
                via_server_proxy=True,
                log_unrecognized=first,
            )
            is None
        )
    assert len(caplog.records) == 1
    assert caplog.records[0].message.endswith(report[:256])


@pytest.mark.parametrize("value", [1, "true", None])
def test_ended_report_opt_in_requires_boolean(value):
    with pytest.raises(TypeError, match="diagnostics.ended_report_oom must be a bool"):
        OpenSandboxProvider(diagnostics={"ended_report_oom": value})
