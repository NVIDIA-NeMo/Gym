# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The patch capture retries a sandbox transport failure before the agent sandbox, and its patch, are gone."""

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import httpx
import pytest
from pytest import MonkeyPatch

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import SESSION_LOST
from nemo_gym.sandbox import SandboxExecResult
from resources_servers.swebench_pro import app as swe
from resources_servers.swebench_pro.tests.test_app import make_server, request_body
from resources_servers.swebench_pro.verification import VerificationResult


RETRIES = 2


class SdkTransportError(Exception):
    """A sandbox SDK's wrapper around its HTTP client's error, as the SDK raises it."""


def connect_failure() -> Exception:
    """A refused TCP connect to the sandbox, wrapped the way the SDK and Gym's aiohttp transport wrap it."""
    os_error = ConnectionRefusedError(111, "Connect call failed ('192.0.2.1', 80)")
    connector = aiohttp.ClientConnectorError(SimpleNamespace(host="sandbox.invalid", port=80, ssl="default"), os_error)
    connector.__cause__ = os_error
    mapped = httpx.ConnectError(str(connector))
    mapped.__cause__ = connector
    error = SdkTransportError(f"Network connectivity error: {connector}")
    error.__cause__ = mapped
    return error


def reset_after_send() -> Exception:
    """A connection reset after the request was written; Gym's transport still re-raises it as ``ConnectError``."""
    os_error = ConnectionResetError(104, "Connection reset by peer")
    client_error = aiohttp.ClientOSError(104, "Connection reset by peer")
    client_error.__cause__ = os_error
    mapped = httpx.ConnectError(str(client_error))
    mapped.__cause__ = client_error
    error = SdkTransportError(f"Network connectivity error: {client_error}")
    error.__cause__ = mapped
    return error


def read_timeout() -> Exception:
    error = SdkTransportError("Network connectivity error: read")
    error.__cause__ = httpx.ReadTimeout("read")
    return error


OK = SandboxExecResult(stdout="", stderr="", return_code=0)


class TaskSandbox:
    """An agent sandbox whose ``git diff`` exec and download follow scripted outcomes, then succeed."""

    def __init__(self, *, diff: list[object] = (), download: list[Exception] = ()) -> None:
        self.diff_outcomes = list(diff)
        self.download_outcomes = list(download)
        self.diff_calls = 0
        self.download_calls = 0
        self.stop = AsyncMock()

    async def exec(self, command: str, **_: object) -> SandboxExecResult:
        if "--no-pager diff" not in command:  # not the cleanup of the .diff file
            return OK
        self.diff_calls += 1
        outcome = self.diff_outcomes.pop(0) if self.diff_outcomes else OK
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def download(self, remote: str, local) -> None:
        self.download_calls += 1
        if self.download_outcomes:
            raise self.download_outcomes.pop(0)
        local.write_text("agent patch")


@pytest.fixture
def run_verify(monkeypatch: MonkeyPatch) -> Callable:
    monkeypatch.setattr(swe, "PATCH_CAPTURE_TRANSPORT_RETRIES", RETRIES, raising=False)
    monkeypatch.setattr(swe, "PATCH_CAPTURE_RETRY_DELAY_S", 0.0, raising=False)

    async def run(sandbox: TaskSandbox) -> tuple[swe.SWEBenchProVerifyResponse, AsyncMock]:
        server = make_server(golden=False, inconclusive_verification_retries=0)
        server._session_id_to_sandbox["session"] = sandbox
        verification = AsyncMock(
            return_value=VerificationResult(
                completed=True, resolved=True, patch_applied=True, test_results={"tests": []}, test_output="passed"
            )
        )
        monkeypatch.setattr(swe, "run_verification", verification)
        monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=SimpleNamespace(stop=AsyncMock())))
        response = await server.verify(
            SimpleNamespace(session={SESSION_ID_KEY: "session"}),
            swe.SWEBenchProVerifyRequest.model_validate(request_body()),
        )
        assert "session" not in server._session_id_to_sandbox
        sandbox.stop.assert_awaited_once()
        return response, verification

    return run


def assert_captured(response: swe.SWEBenchProVerifyResponse, verification: AsyncMock) -> None:
    assert (response.model_patch, response.mask_sample, response.failure_kind) == ("agent patch", False, None)
    assert (response.reward, response.error) == (1.0, None)
    verification.assert_awaited_once()


def assert_lost(response: swe.SWEBenchProVerifyResponse, verification: AsyncMock) -> None:
    assert (response.mask_sample, response.failure_kind, response.model_patch) == (True, SESSION_LOST, None)
    assert response.error.startswith("Failed to extract model patch: ")
    verification.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_diff_exec_that_could_not_connect_once_is_retried_and_the_patch_captured(
    run_verify: Callable, caplog: pytest.LogCaptureFixture
) -> None:
    sandbox = TaskSandbox(diff=[connect_failure()])
    response, verification = await run_verify(sandbox)

    assert_captured(response, verification)
    assert (sandbox.diff_calls, sandbox.download_calls) == (2, 1)
    assert "Patch capture git diff exec failed in transport (connect, attempt 1/3)" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [connect_failure, reset_after_send, read_timeout])
async def test_a_download_lost_once_in_transport_is_retried_and_the_patch_captured(
    run_verify: Callable, failure: Callable[[], Exception]
) -> None:
    """Reading the patch file again is idempotent, so a failure after sending is retried too."""
    sandbox = TaskSandbox(download=[failure()])
    response, verification = await run_verify(sandbox)

    assert_captured(response, verification)
    assert (sandbox.diff_calls, sandbox.download_calls) == (1, 2)


@pytest.mark.asyncio
async def test_a_diff_exec_that_never_connects_is_lost_after_every_attempt(run_verify: Callable) -> None:
    sandbox = TaskSandbox(diff=[connect_failure() for _ in range(RETRIES + 5)])
    response, verification = await run_verify(sandbox)

    assert_lost(response, verification)
    assert response.error.startswith("Failed to extract model patch: git diff exec failed: Network connectivity error")
    assert (sandbox.diff_calls, sandbox.download_calls) == (RETRIES + 1, 0)


@pytest.mark.asyncio
async def test_a_download_that_never_succeeds_is_lost_after_every_attempt(run_verify: Callable) -> None:
    sandbox = TaskSandbox(download=[connect_failure() for _ in range(RETRIES + 5)])
    response, verification = await run_verify(sandbox)

    assert_lost(response, verification)
    assert response.error.startswith("Failed to extract model patch: patch download failed: ")
    assert (sandbox.diff_calls, sandbox.download_calls) == (1, RETRIES + 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [reset_after_send, read_timeout])
async def test_a_diff_exec_lost_after_sending_is_not_retried(
    run_verify: Callable, failure: Callable[[], Exception]
) -> None:
    """The command may have run, and a second git process could race the first for the index lock."""
    sandbox = TaskSandbox(diff=[failure()])
    response, verification = await run_verify(sandbox)

    assert_lost(response, verification)
    assert (sandbox.diff_calls, sandbox.download_calls) == (1, 0)


@pytest.mark.asyncio
async def test_a_git_failure_is_not_retried_and_is_graded(run_verify: Callable) -> None:
    sandbox = TaskSandbox(diff=[SandboxExecResult(stdout="", stderr="fatal: bad revision", return_code=128)])
    response, verification = await run_verify(sandbox)

    assert (response.mask_sample, response.failure_kind, response.model_patch) == (False, None, None)
    assert response.error == "Failed to extract model patch: fatal: bad revision"
    verification.assert_awaited_once()
    assert (sandbox.diff_calls, sandbox.download_calls) == (1, 0)


@pytest.mark.asyncio
async def test_a_git_failure_on_a_retry_is_lost_not_graded(run_verify: Callable) -> None:
    """A retried exec can collide with an attempt that did run, so its failure is not the agent's."""
    lock = SandboxExecResult(stdout="", stderr="fatal: Unable to create '/app/.git/index.lock'", return_code=128)
    sandbox = TaskSandbox(diff=[connect_failure(), lock])
    response, verification = await run_verify(sandbox)

    assert_lost(response, verification)
    assert "git diff failed on a retry after a transport failure" in response.error
    assert sandbox.diff_calls == 2


def test_transport_failure_phase_reads_the_innermost_transport_error() -> None:
    phase = swe._transport_failure_phase

    assert phase(connect_failure()) == "connect"
    assert phase(httpx.ConnectTimeout("connect")) == "connect"
    assert phase(httpx.PoolTimeout("pool")) == "connect"
    assert phase(reset_after_send()) == "sent"
    assert phase(read_timeout()) == "sent"
    assert phase(TimeoutError()) == "sent"
    assert phase(RuntimeError("git diff failed")) is None
    assert phase(OSError(28, "No space left on device")) is None
