# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EpisodeFailure.failure_path: built from exceptions, sent across boundaries, mapped onto saved rows."""

import asyncio
import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from omegaconf import OmegaConf

import nemo_gym.rollout_collection
from nemo_gym import typed_exceptions
from nemo_gym.episode_types import EpisodeFailure, FailureHop
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.judge import JudgeRateLimited
from nemo_gym.rollout_collection import (
    RolloutCollectionHelper,
    _agent_request_failure_row,
    _counted_failure_row,
    is_terminal_failure,
)
from nemo_gym.sandbox.providers.base import SandboxCreateError, SandboxPtyError
from nemo_gym.sandbox.transfer import SandboxProbeError, SandboxTransferError, SandboxUnavailableError
from nemo_gym.server_utils import SimpleServer
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure
from nemo_gym.typed_exceptions import (
    FAILURE_EVENT_HEADER,
    boundary_failure,
    decode_header,
    encode_header,
    failure_path_fields,
    failure_path_or_none,
    row_fields,
    safe_failure_header,
    with_failure_path,
)


_JUDGE_TYPE = "nemo_gym.judge.JudgeRateLimited"
_HTTP_ERROR_TYPE = "aiohttp.client_exceptions.ClientResponseError"


def _http_error(status: int, **kwargs) -> ClientResponseError:
    return ClientResponseError(request_info=SimpleNamespace(real_url="http://x"), history=(), status=status, **kwargs)


def _judge_429() -> JudgeRateLimited:
    """A judge 429 as the judge client raises it: the typed failure, from the HTTP error."""
    try:
        try:
            raise _http_error(429)
        except ClientResponseError as err:
            raise JudgeRateLimited(
                message="ClientResponseError: 429", operation="judge.request", stage="verification"
            ) from err
    except JudgeRateLimited as error:
        return error


def _hops(failure: EpisodeFailure) -> list[dict]:
    """The failure's hops as saved: the same JSON shape as the _ng_failure_path row key."""
    return [hop.model_dump(mode="json") for hop in failure.failure_path]


_JUDGE_PATH = [
    {"operation": "judge.request", "exception_type": _JUDGE_TYPE},
    {"exception_type": _HTTP_ERROR_TYPE, "http_status": 429},
]


def test_a_judge_429_has_one_kind_and_its_status_on_the_path() -> None:
    failure = boundary_failure(_judge_429(), server="resources")
    assert failure.model_dump() == {
        "failure_reason": "ClientResponseError: 429",
        "terminal": False,
        "failure_kind": "judge_rate_limited",
        "failure_path": _JUDGE_PATH,
    }


def test_an_unhandled_exception_has_a_path_and_no_kind() -> None:
    try:
        try:
            raise MemoryError()
        except MemoryError as err:
            raise RuntimeError("Connection refused") from err  # text decides nothing
    except RuntimeError as error:
        failure = boundary_failure(error, server="my_server")
    assert failure.failure_kind is None and failure.stage is None and failure.terminal is False
    assert _hops(failure) == [{"exception_type": "RuntimeError"}, {"exception_type": "MemoryError"}]
    assert failure.failure_reason == "Unhandled RuntimeError in my_server"


def test_the_header_carries_episode_failure_json_and_round_trips() -> None:
    failure = boundary_failure(_judge_429(), server="resources")
    assert decode_header(CIMultiDict({FAILURE_EVENT_HEADER: encode_header(failure)})) == failure


@pytest.mark.parametrize(
    "value",
    [
        "not json",
        '{"failure_reason": "x"}',  # terminal is required
        '{"failure_reason": "x", "terminal": false, "unknown_field": 1}',
        '{"failure_reason": "x", "terminal": false, "failure_path": [{"exception_type": ["x"]}]}',
        "",
    ],
)
def test_an_invalid_or_missing_header_is_ignored(value: str) -> None:
    assert decode_header(CIMultiDict({FAILURE_EVENT_HEADER: value})) is None
    assert decode_header(None) is None and decode_header(MagicMock()) is None


def test_a_failure_that_cannot_be_built_or_sent_leaves_the_reply_without_the_header(monkeypatch) -> None:
    def broken(*_, **__):
        raise TypeError("not serializable")

    monkeypatch.setattr(typed_exceptions, "boundary_failure", broken)
    assert safe_failure_header(RuntimeError("x"), server="s") == {}
    monkeypatch.undo()
    huge = boundary_failure(_judge_429(), server="s").model_copy(
        update={"failure_path": [{"exception_type": "x" * 100}] * 100}
    )
    assert encode_header(huge) is None


def _app(route) -> TestClient:
    app = FastAPI()
    server = SimpleNamespace(config=SimpleNamespace(name="my_server"), get_session_middleware_key=lambda: "my_server")
    SimpleServer.setup_exception_middleware(server, app)
    app.post("/run")(route)
    return TestClient(app, raise_server_exceptions=False)


def test_the_boundary_adds_the_failure_header_and_keeps_the_body() -> None:
    async def route():
        raise ValueError("bad tool call")

    reply = _app(route).post("/run")
    assert reply.status_code == 500 and reply.json() == repr(ValueError("bad tool call"))
    failure = EpisodeFailure.model_validate_json(reply.headers[FAILURE_EVENT_HEADER])
    assert failure.failure_kind is None and _hops(failure) == [{"exception_type": "ValueError"}]


def test_the_boundary_sends_a_raised_typed_failure_with_its_kind_and_path() -> None:
    async def route():
        raise _judge_429()

    reply = _app(route).post("/run")
    assert reply.json() == repr(_judge_429())
    failure = EpisodeFailure.model_validate_json(reply.headers[FAILURE_EVENT_HEADER])
    assert failure.failure_kind == "judge_rate_limited" and _hops(failure) == _JUDGE_PATH


def test_a_cancelled_request_has_no_kind() -> None:
    async def route():
        raise asyncio.CancelledError()

    # Starlette's call_next turns the cancelled handler into RuntimeError("No response returned."),
    # which reaches the generic branch: no kind is set. Whether that 500's body reaches the client depends
    # on the Starlette version, so the body is left to the tests of a handler error.
    reply = _app(route).post("/run")
    assert reply.status_code == 500
    failure = EpisodeFailure.model_validate_json(reply.headers[FAILURE_EVENT_HEADER])
    assert failure.failure_kind is None and _hops(failure)[0] == {"exception_type": "RuntimeError"}


def test_an_inner_servers_failure_is_passed_on_unchanged() -> None:
    inner = encode_header(boundary_failure(_judge_429(), server="resources"))

    async def route():
        error = _http_error(429, headers=CIMultiDict({FAILURE_EVENT_HEADER: inner}))
        error.response_content = b"rate limited"
        raise error

    reply = _app(route).post("/run")
    assert reply.headers[FAILURE_EVENT_HEADER] == inner
    assert reply.json().endswith("calling an inner server: b'rate limited'")


class FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status
        self.ok = 200 <= status < 300

    def release(self) -> None:
        pass


def _failing_row() -> dict:
    return {
        AGENT_REF_KEY_NAME: {"name": "my_agent"},
        TASK_INDEX_KEY_NAME: 7,
        ROLLOUT_INDEX_KEY_NAME: 0,
        "responses_create_params": {"input": []},
    }


def _failed_run(monkeypatch, *, headers: CIMultiDict, body: bytes) -> None:
    server_client = MagicMock()
    server_client.post = AsyncMock(return_value=FakeResponse(500))
    server_client.global_config_dict = OmegaConf.create(
        {
            "my_agent": {"responses_api_agents": {"impl": {}}},
            "my_environment_server": {"environment_servers": {"legacy_agent": {"agent_server": {"name": "my_agent"}}}},
        }
    )
    monkeypatch.setattr(nemo_gym.rollout_collection, "setup_server_client_utils", lambda *a, **k: server_client)

    async def raise_for_status(_response):
        error = ClientResponseError(
            request_info=SimpleNamespace(method="POST", url="http://agent/run", real_url="http://agent/run"),
            history=(),
            status=500,
            message="boom",
            headers=headers,
        )
        error.response_content = body
        raise error

    monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", raise_for_status)


@pytest.mark.parametrize("body", [b"plain text, not JSON", b"\"ValueError('x')\""])
async def test_the_collector_carries_the_path_and_run_examples_output_is_unchanged(monkeypatch, body) -> None:
    header = encode_header(boundary_failure(_judge_429(), server="resources"))
    _failed_run(monkeypatch, headers=CIMultiDict({FAILURE_EVENT_HEADER: header}), body=body)
    _, with_header = await next(
        RolloutCollectionHelper().run_examples([_failing_row()], route_failures_to_sidecar=True)
    )
    _failed_run(monkeypatch, headers=CIMultiDict(), body=body)
    _, without_header = await next(
        RolloutCollectionHelper().run_examples([_failing_row()], route_failures_to_sidecar=True)
    )
    assert with_header == without_header and "_ng_failure_path" not in with_header
    assert with_header["_ng_failure_response_body"] == body.decode()

    _failed_run(monkeypatch, headers=CIMultiDict({FAILURE_EVENT_HEADER: header}), body=body)
    completed = await next(
        iter(RolloutCollectionHelper()._run_examples_with_metadata([_failing_row()], route_failures_to_sidecar=True))
    )
    assert completed.failure_path == {"_ng_failure_path": _JUDGE_PATH}
    assert completed.result == with_header


def test_readers_of_failure_rows_are_unchanged_by_the_path_key() -> None:
    row = _agent_request_failure_row(RuntimeError("x"), 500)
    saved = {**row, **row_fields(boundary_failure(_judge_429(), server="resources"))}
    assert saved["_ng_failure_path"] == _JUDGE_PATH
    assert _counted_failure_row(saved) == _counted_failure_row(row)  # stripped from metric inputs by prefix
    assert is_terminal_failure(saved) == is_terminal_failure(row)
    assert row_fields(None) == {} and row_fields(EpisodeFailure(failure_reason="x", terminal=False)) == {}


def test_a_hop_omits_what_was_not_observed() -> None:
    failure = EpisodeFailure(
        failure_reason="x",
        terminal=False,
        failure_path=[{"exception_type": "ValueError"}, {"exception_type": "ClientResponseError", "http_status": 401}],
    )
    assert isinstance(failure.failure_path[0], FailureHop) and failure.failure_path[0].http_status is None
    assert failure.model_dump()["failure_path"] == [
        {"exception_type": "ValueError"},
        {"exception_type": "ClientResponseError", "http_status": 401},
    ]
    # The row key keeps the same JSON shape as the record.
    assert row_fields(failure)["_ng_failure_path"] == failure.model_dump(mode="json")["failure_path"]


def test_failure_path_fields_continues_an_inner_servers_path() -> None:
    inner = encode_header(boundary_failure(_judge_429(), server="resources"))
    relayed = _http_error(500, headers=CIMultiDict({FAILURE_EVENT_HEADER: inner}))
    wrapper = RuntimeError("agent loop failed")
    wrapper.__cause__ = relayed
    assert failure_path_fields(wrapper)["_ng_failure_path"] == [
        {"exception_type": "RuntimeError"},
        {"exception_type": _HTTP_ERROR_TYPE, "http_status": 500},
        *_JUDGE_PATH,
    ]
    plain = failure_path_fields(ValueError("Connection refused"))
    assert plain == {"_ng_failure_path": [{"exception_type": "ValueError"}]}
    assert failure_path_fields(None) == {} and failure_path_or_none(None) is None


def test_failure_path_fields_never_raises(monkeypatch) -> None:
    def broken(*_, **__):
        raise TypeError("cannot build")

    monkeypatch.setattr(typed_exceptions, "failure_path", broken)
    assert failure_path_fields(RuntimeError("x")) == {}
    assert failure_path_or_none(RuntimeError("x")) is None


def test_with_failure_path_keeps_the_producer_fields_and_the_failure_type() -> None:
    error = RuntimeError("seed failed")
    error.__cause__ = SandboxCreateError("no capacity")
    produced = SingleAgentTurnFailure(failure_reason="Seed failed", terminal=False, stage="seed")
    filled = with_failure_path(produced, error)
    assert type(filled) is SingleAgentTurnFailure
    assert (filled.failure_reason, filled.terminal, filled.stage) == ("Seed failed", False, "seed")
    assert filled.failure_kind == "provider_unavailable"
    assert row_fields(filled)["_ng_failure_path"] == [
        {"exception_type": "RuntimeError"},
        {"operation": "sandbox.create", "exception_type": "nemo_gym.sandbox.providers.base.SandboxCreateError"},
    ]

    # A producer's own kind is kept.
    own_kind = SingleAgentTurnFailure(failure_reason="x", terminal=True, failure_kind="session_lost")
    assert with_failure_path(own_kind, error).failure_kind == "session_lost"

    # A path a producer already set wins.
    assert with_failure_path(filled, RuntimeError("other")) is filled


def test_with_failure_path_fills_the_producer_object_so_later_evidence_reaches_the_response() -> None:
    # A server keeps its failure and attaches cleanup evidence after the base server built the response.
    error = RuntimeError("seed failed")
    error.__cause__ = SandboxCreateError("no capacity")
    produced = SingleAgentTurnFailure(failure_reason="Seed failed", terminal=False, stage="seed")
    filled = with_failure_path(produced, error)
    assert filled is produced and filled.failure_path and filled.failure_kind == "provider_unavailable"
    produced.failure_reason = "Seed failed; cleanup confirmed"
    assert filled.model_dump(mode="json")["failure_reason"] == "Seed failed; cleanup confirmed"


def test_with_failure_path_never_raises(monkeypatch) -> None:
    def broken(*_, **__):
        raise TypeError("cannot build")

    monkeypatch.setattr(typed_exceptions, "failure_path", broken)
    failure = EpisodeFailure(failure_reason="x", terminal=False)
    assert with_failure_path(failure, RuntimeError("x")) is failure


@pytest.mark.parametrize(
    "cls", [SandboxCreateError, SandboxPtyError, SandboxTransferError, SandboxUnavailableError, SandboxProbeError]
)
def test_an_empty_positional_message_still_builds_the_original_type(cls) -> None:
    """Main's types accept an empty message; the typed version must too, with the same args, str and pickle."""
    error = cls("")
    assert type(error) is cls and isinstance(error, RuntimeError)
    assert error.args == ("",) and str(error) == ""
    assert error.message == cls.__name__  # the failure's own message is never empty
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is cls and copy.args == ("",) and str(copy) == ""
    with pytest.raises(RuntimeError):
        raise error


@pytest.mark.parametrize("text", ["rate limit \N{GRINNING FACE}", "d\u00e9j\u00e0 limit\u00e9", "\u9650\u6d41"])
def test_a_non_ascii_failure_keeps_the_original_reply_and_a_legal_header(text: str) -> None:
    async def route():
        raise SandboxCreateError(text)

    reply = _app(route).post("/run")
    # The reply is the one main sends for this exception.
    assert reply.status_code == 500 and reply.json() == repr(SandboxCreateError(text))
    value = reply.headers[FAILURE_EVENT_HEADER]
    assert value.isascii()
    failure = EpisodeFailure.model_validate_json(value)
    assert failure.failure_reason == text and failure.failure_kind == "provider_unavailable"
    assert failure.failure_path[0].operation == "sandbox.create"


def test_the_header_cap_counts_the_bytes_actually_sent() -> None:
    # Each character escapes to six ASCII bytes, so a reason of 1,400 of them is over the cap on the wire.
    failure = EpisodeFailure(failure_reason="\N{GRINNING FACE}" * 1000, terminal=False)
    assert encode_header(failure) is None
    plain = EpisodeFailure(failure_reason="x" * 1000, terminal=False)
    assert encode_header(plain) == plain.model_dump_json()  # an ASCII failure's header bytes are unchanged


def test_a_non_ascii_inner_header_is_not_relayed() -> None:
    async def route():
        request_info = SimpleNamespace(real_url="http://inner")
        error = ClientResponseError(
            request_info=request_info,
            history=(),
            status=500,
            headers=CIMultiDict({FAILURE_EVENT_HEADER: '{"failure_reason":"\u00e9","terminal":false}'}),
        )
        error.response_content = b"inner failed"
        raise error

    reply = _app(route).post("/run")
    assert reply.status_code == 500 and reply.json().endswith("calling an inner server: b'inner failed'")
    assert reply.headers[FAILURE_EVENT_HEADER].isascii()
