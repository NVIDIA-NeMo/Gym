# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed failures raised by the clients every environment server shares: the judge and the sandbox."""

import asyncio
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from pydantic import BaseModel, ConfigDict

from nemo_gym import judge
from nemo_gym.judge import JudgeError, call_judge, judge_failsafe, reraise_judge_errors
from nemo_gym.sandbox.providers.base import SandboxCreateError, SandboxCreateVerificationError, SandboxPtyError
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxCreatePendingError
from nemo_gym.sandbox.transfer import SandboxTransferError, SandboxUnavailableError, exec_as_root, upload_dir
from nemo_gym.typed_exceptions import FAILURE_EVENT_HEADER, TypedException, failure_path


class _Verdict(BaseModel):
    verdict: str


class _Req(BaseModel):
    model_config = ConfigDict(extra="allow")

    response: dict = {}


class _JudgeEndpoint:
    """A real HTTP judge endpoint answering a fixed status, behind a ServerClient-shaped ``post``."""

    def __init__(self, status: int) -> None:
        self.status = status

    async def __aenter__(self) -> "_JudgeEndpoint":
        async def handle(_: web.Request) -> web.Response:
            return web.json_response({"error": {"message": "refused"}}, status=self.status)

        app = web.Application()
        app.router.add_post("/v1/responses", handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._session.close()
        await self._runner.cleanup()

    async def post(self, *, server_name: str, url_path: str, json: object) -> aiohttp.ClientResponse:
        return await self._session.post(self.url + url_path, json=json)


@pytest.mark.parametrize(
    "status, expected",
    [(401, judge.JudgeCredentialsRejected), (403, judge.JudgeCredentialsRejected), (429, judge.JudgeRateLimited)]
    + [(code, judge.JudgeServerError) for code in (500, 502, 503)],
)
async def test_judge_http_status_raises_its_typed_judge_error(status: int, expected: type) -> None:
    async with _JudgeEndpoint(status) as endpoint:
        with pytest.raises(JudgeError) as info:
            await call_judge(endpoint, server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict)
    failure = info.value
    # One kind per failure: the class names what the client observed, and the status is on the path.
    assert type(failure) is expected and failure.failure_kind == expected.failure_kind
    assert failure.failure_kind != "judge_failed" and failure.stage == "verification"
    assert failure_path(failure) == [
        {"operation": "judge.request", "exception_type": f"nemo_gym.judge.{expected.__name__}"},
        {"exception_type": "aiohttp.client_exceptions.ClientResponseError", "http_status": status},
    ]
    assert str(failure).startswith(f"ClientResponseError: {status}, message=")


async def test_judge_status_without_a_class_stays_a_plain_judge_error() -> None:
    async with _JudgeEndpoint(400) as endpoint:
        with pytest.raises(JudgeError) as info:
            await call_judge(endpoint, server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict)
    assert type(info.value) is JudgeError and not isinstance(info.value, TypedException)
    assert failure_path(info.value)[-1]["http_status"] == 400  # the status is still observed on the path


async def test_unreachable_judge_raises_judge_unreachable() -> None:
    async with aiohttp.ClientSession() as session:

        async def refused() -> None:
            # Port 9 (discard) is closed on test hosts: the connection is refused before any reply.
            await session.post("http://127.0.0.1:9/v1/responses", json={})

        with pytest.raises(judge.JudgeUnreachable) as info:
            await reraise_judge_errors(refused())
    assert info.value.failure_kind == "judge_unreachable"
    assert failure_path(info.value)[1]["exception_type"].endswith("ClientConnectorError")


async def test_a_judge_call_past_its_timeout_raises_judge_timeout() -> None:
    async def slow() -> None:
        await asyncio.wait_for(asyncio.sleep(10), timeout=0.01)

    with pytest.raises(judge.JudgeTimeout) as info:
        await reraise_judge_errors(slow())
    assert isinstance(info.value, JudgeError) and info.value.failure_kind == "judge_timeout"
    assert failure_path(info.value)[1]["exception_type"] == "TimeoutError"


@pytest.mark.parametrize("status", [401, 429, 503])
async def test_judge_failsafe_output_is_unchanged_by_the_typed_error(status: int) -> None:
    """Every ``except JudgeError`` still runs with the same text; the row gains its path and its kind."""
    async with _JudgeEndpoint(status) as endpoint:
        try:
            await call_judge(endpoint, server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict)
        except JudgeError as error:
            typed = error

    async def raise_typed(body: _Req):
        raise typed

    async def raise_plain(body: _Req):
        raise JudgeError(str(typed))

    typed_row, plain_row = [
        json.loads((await judge_failsafe(fn)(body=_Req(response={"id": "r"}))).body)
        for fn in (raise_typed, raise_plain)
    ]
    typed_path, plain_path = typed_row.pop("_ng_failure_path"), plain_row.pop("_ng_failure_path")
    # The kind is what the client observed; the routing class, reward and mask are the plain row's.
    assert (
        typed_row.pop("failure_kind") == type(typed).failure_kind and plain_row.pop("failure_kind") == "judge_failed"
    )
    assert typed_row == plain_row and typed_row["_ng_failure_class"] == "judge_failed"
    assert typed_path[-1]["http_status"] == status and plain_path == [{"exception_type": "nemo_gym.judge.JudgeError"}]


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Connection refused"),  # the model's program printed it into the judged answer
        MemoryError(),  # raised while Gym parsed a huge answer
        BrokenPipeError("Broken pipe"),  # an OS error type, not a connection the client failed to open
    ],
)
async def test_judge_errors_without_a_deciding_type_or_status_stay_plain(error: Exception) -> None:
    async def failing() -> None:
        raise error

    with pytest.raises(JudgeError) as info:
        await reraise_judge_errors(failing())
    assert type(info.value) is JudgeError


@pytest.mark.parametrize(
    "error",
    [
        SandboxCreateError("no capacity"),
        SandboxCreateVerificationError("probe failed"),
        OpenSandboxCreatePendingError("pending", fail_fast=True),
    ],
)
def test_sandbox_create_errors_are_typed_and_keep_their_handlers(error: SandboxCreateError) -> None:
    for handler in (SandboxCreateError, RuntimeError, TypedException):
        with pytest.raises(handler):
            raise error
    assert not isinstance(error, TimeoutError)
    assert error.args == (str(error),) and error.failure_kind == "provider_unavailable"
    assert failure_path(error)[0]["operation"] == "sandbox.create"
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is type(error) and copy.args == error.args and failure_path(copy) == failure_path(error)
    if isinstance(error, OpenSandboxCreatePendingError):
        assert copy.fail_fast is True


def _sandbox(result: SimpleNamespace) -> SimpleNamespace:
    async def exec(command: str, **_: object) -> SimpleNamespace:
        return result

    async def upload(*_: object) -> None:
        return None

    return SimpleNamespace(exec=exec, upload=upload)


async def test_sandbox_that_cannot_run_a_command_raises_a_provider_unavailable() -> None:
    sandbox = _sandbox(SimpleNamespace(return_code=-1, error_type="timeout", stdout="", stderr=""))
    with pytest.raises(SandboxUnavailableError) as info:
        await exec_as_root(sandbox, "true", timeout_s=1)
    assert info.value.failure_kind == "provider_unavailable"
    assert failure_path(info.value)[0]["operation"] == "sandbox.exec"


async def test_failed_transfer_has_no_kind_whatever_its_output(tmp_path: Path) -> None:
    # stderr written by the model's program, naming infrastructure-looking errors, decides nothing.
    stderr = "MemoryError\nBrokenPipeError: [Errno 32] Broken pipe\nConnection refused"
    sandbox = _sandbox(SimpleNamespace(return_code=2, error_type=None, stdout="", stderr=stderr))
    (tmp_path / "file").write_text("x")
    with pytest.raises(SandboxTransferError) as info:
        await upload_dir(sandbox, tmp_path, "/target")
    assert info.value.failure_kind is None
    assert failure_path(info.value)[0] == {
        "operation": "sandbox.transfer",
        "exception_type": "nemo_gym.sandbox.transfer.SandboxTransferError",
    }


def test_pty_session_end_has_no_kind() -> None:
    error = SandboxPtyError("PTY session ended before the command finished")  # what `exit` in the shell produces
    assert error.failure_kind is None and failure_path(error)[0]["operation"] == "sandbox.pty"
    with pytest.raises(RuntimeError):
        raise error


class _ModelServerEndpoint(_JudgeEndpoint):
    """A model server that answers 500 for a failed provider call, with the provider's status in its failure header."""

    def __init__(self, upstream_status: int) -> None:
        super().__init__(500)
        self.upstream_status = upstream_status

    async def __aenter__(self) -> "_ModelServerEndpoint":
        # What a model server's middleware sends for a failed provider call: EpisodeFailure JSON.
        header = json.dumps(
            {
                "failure_reason": "Unhandled ClientResponseError in judge_model",
                "terminal": False,
                "failure_path": [
                    {"exception_type": "ClientResponseError", "http_status": self.upstream_status},
                    {"exception_type": "ExceptionGroup"},
                ],
            }
        )

        async def handle(_: web.Request) -> web.Response:
            return web.json_response("inner error", status=500, headers={FAILURE_EVENT_HEADER: header})

        app = web.Application()
        app.router.add_post("/v1/responses", handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self._session = aiohttp.ClientSession()
        return self


@pytest.mark.parametrize(
    "upstream, expected",
    [(401, judge.JudgeCredentialsRejected), (429, judge.JudgeRateLimited), (502, judge.JudgeServerError)],
)
async def test_judge_types_by_the_providers_status_behind_a_model_server_500(upstream, expected) -> None:
    async with _ModelServerEndpoint(upstream) as endpoint:
        with pytest.raises(JudgeError) as info:
            await call_judge(endpoint, server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict)
    assert type(info.value) is expected and info.value.failure_kind == expected.failure_kind
    assert failure_path(info.value)[1:] == [
        {"exception_type": "aiohttp.client_exceptions.ClientResponseError", "http_status": 500},
        {"exception_type": "ClientResponseError", "http_status": upstream},
        {"exception_type": "ExceptionGroup"},
    ]


def test_the_path_takes_an_sdk_errors_integer_status_code() -> None:
    class SandboxApiException(Exception):
        status_code = 401

    hop = failure_path(SandboxApiException("refused"))[0]
    assert hop["http_status"] == 401 and hop["exception_type"].endswith("SandboxApiException")

    class Weird(Exception):
        status = "401"  # not an int: decides nothing

    assert "http_status" not in failure_path(Weird())[0]
