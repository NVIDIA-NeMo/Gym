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
"""Shared failure events: boundary classes, one event map by id over real HTTP, observations of recovered failures."""

import json
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import nemo_gym.openai_utils as openai_utils_module
from nemo_gym.episode_types import EpisodeFailure, FailureEventRecord
from nemo_gym.failure_kinds import JUDGE_FAILED, MODEL_REQUEST_ERROR, SESSION_LOST, VERIFIER_ERROR
from nemo_gym.judge import JudgeCallFailed, JudgeError
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.rollout_collection import _events_row_fields, _failure_row_fields
from nemo_gym.sandbox.providers.base import SandboxCreateError
from nemo_gym.sandbox.transfer import SandboxProbeError
from nemo_gym.server_utils import SimpleServer, raise_for_status
from nemo_gym.typed_exceptions import (
    FAILURE_EVENT_HEADER,
    FAILURE_EVENTS_HEADER,
    EnvServerFailure,
    ModelFailure,
    NgFailureEvent,
    ResServerFailure,
    boundary_failure,
    decode_events_header,
    decode_header,
    encode_header,
    failure_event_map,
    find_typed_exception,
    observe_relayed_events,
    observing_failures,
    reconstruct_failure,
    safe_failure_header,
)


def _app(route) -> TestClient:
    app = FastAPI()
    app.post("/run")(route)
    server = SimpleNamespace(config=SimpleNamespace(name="my_server"), get_session_middleware_key=lambda: "my_server")
    SimpleServer.setup_exception_middleware(server, app)
    return TestClient(app, raise_server_exceptions=False)


def _res_failure() -> ResServerFailure:
    try:
        raise ResServerFailure(VERIFIER_ERROR, "verifier crashed", operation_id="verify-1")
    except ResServerFailure as error:
        return error


def _read(content: bytes):
    async def read():
        return content

    return read


def test_boundary_constructor_forms_preserve_existing_judge_handler() -> None:
    boundary = ModelFailure(MODEL_REQUEST_ERROR, "Rate limited", http_status=429)
    legacy = ModelFailure(message="Rate limited", operation="model_request", failure_kind=MODEL_REQUEST_ERROR)
    assert (boundary.failure_reason, legacy.failure_reason) == ("Rate limited", "Rate limited")
    assert boundary.event_id != legacy.event_id
    with pytest.raises(NgFailureEvent):
        raise boundary
    judge = JudgeCallFailed(message="judge said 429", operation="judge")
    assert isinstance(judge, JudgeError) and isinstance(judge, ResServerFailure)


async def test_a_resources_failure_over_a_real_socket_keeps_each_boundary_kind_and_id() -> None:
    """A resources server's failure header is read by a real aiohttp client; the environment server's own failure
    wraps the HTTP error explicitly, and its reply carries both records, each with its own kind and id."""
    inner = _res_failure()

    async def verify(_request):
        return web.Response(status=500, text="verifier crashed", headers=safe_failure_header(inner, server="res"))

    app = web.Application()
    app.router.add_post("/verify", verify)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        async with session.post(server.make_url("/verify")) as response:
            try:
                await raise_for_status(response)
            except aiohttp.ClientResponseError as http_error:
                try:
                    raise EnvServerFailure(SESSION_LOST, "episode lost its verifier") from http_error
                except EnvServerFailure as error:
                    outer = error
                    failure = boundary_failure(outer, server="env")

    events = failure.failure_events
    assert failure.failure_id == outer.event_id and failure.failure_kind == SESSION_LOST
    assert (events[inner.event_id].type, events[inner.event_id].failure_kind) == ("ResServerFailure", VERIFIER_ERROR)
    assert events[inner.event_id].failure_reason == "verifier crashed"
    assert events[inner.event_id].operation_id == "verify-1"
    # The environment server's own record keeps its kind; its explicit cause is the HTTP error, recorded by type.
    assert events[outer.event_id].failure_kind == SESSION_LOST
    assert events[events[outer.event_id].cause_id].exception_type == "aiohttp.client_exceptions.ClientResponseError"
    assert events[events[outer.event_id].cause_id].http_status == 500
    rebuilt = reconstruct_failure(failure.failure_id, events)
    assert isinstance(rebuilt, EnvServerFailure) and rebuilt.event_id == outer.event_id
    # The chain is connected: the HTTP error's explicit cause is the rebuilt resources failure, same id and kind.
    http_record = events[events[outer.event_id].cause_id]
    assert http_record.cause_id == inner.event_id
    remote = rebuilt.__cause__.__cause__
    assert isinstance(remote, ResServerFailure)
    assert (remote.event_id, remote.failure_kind, remote.failure_reason) == (
        inner.event_id,
        VERIFIER_ERROR,
        "verifier crashed",
    )


def test_a_cause_is_one_entry_of_the_map_and_rebuilds_with_the_same_ids() -> None:
    try:
        raise EnvServerFailure(SESSION_LOST, "lost") from _res_failure()
    except EnvServerFailure as error:
        outer = error
        events = failure_event_map(outer)
    assert list(events) == [outer.event_id, outer.__cause__.event_id]
    assert events[outer.event_id].cause_id == outer.__cause__.event_id
    rebuilt = reconstruct_failure(outer.event_id, events)
    assert isinstance(rebuilt.__cause__, ResServerFailure) and rebuilt.__cause__.failure_kind == VERIFIER_ERROR
    assert failure_event_map(rebuilt) == events


def test_a_judge_subtype_keeps_its_identity_and_rate_limited_kind() -> None:
    judge = JudgeCallFailed(message="judge said 429", operation="judge")
    (record,) = failure_event_map(judge).values()
    assert (record.type, record.exception_type, record.failure_kind) == (
        "ResServerFailure",
        "nemo_gym.judge.JudgeCallFailed",
        JUDGE_FAILED,
    )


def test_an_untyped_cause_is_recorded_by_type_only_and_an_empty_reason_survives() -> None:
    try:
        raise ModelFailure(MODEL_REQUEST_ERROR, "") from ValueError("secret request body")
    except ModelFailure as error:
        outer = error
        events = failure_event_map(outer)
    cause = events[events[outer.event_id].cause_id]
    assert events[outer.event_id].failure_reason == "" and cause.type == "untyped"
    assert cause.exception_type == "ValueError" and "secret" not in json.dumps(
        [e.model_dump() for e in events.values()]
    )


def test_a_timeout_during_a_probe_errors_handling_lends_no_kind_without_claiming_an_explicit_cause() -> None:
    try:
        try:
            raise SandboxProbeError("container lost")
        except SandboxProbeError:
            raise TimeoutError("episode deadline")  # raised while handling, not caused by the probe error
    except TimeoutError as raised:
        error = raised
        failure = boundary_failure(error, server="env")
    assert failure.failure_kind is None and find_typed_exception(error) is None
    assert failure.failure_events is None


def test_a_sandbox_setup_configuration_error_is_not_a_provider_outage() -> None:
    failure = boundary_failure(SandboxCreateError("spec.image is not a valid image reference"), server="env")
    assert failure.failure_kind is None


def test_invalid_maps_and_headers_are_rejected_without_raising() -> None:
    record = {"type": "AgentFailure", "failure_reason": ""}
    with pytest.raises(ValidationError, match="keyed"):
        EpisodeFailure(failure_reason="r", terminal=False, failure_events={"a": {**record, "event_id": "b"}})
    with pytest.raises(ValidationError, match="not in the failure events"):
        EpisodeFailure(
            failure_reason="r", terminal=False, failure_events={"a": {**record, "event_id": "a", "cause_id": "x"}}
        )
    cyclic = {"a": {**record, "event_id": "a", "cause_id": "b"}, "b": {**record, "event_id": "b", "cause_id": "a"}}
    with pytest.raises(ValidationError, match="repeats"):
        EpisodeFailure(failure_reason="r", terminal=False, failure_events=cyclic)
    with pytest.raises(ValidationError, match="failure_id"):
        EpisodeFailure(failure_reason="r", terminal=False, failure_id="missing")
    assert decode_events_header({FAILURE_EVENTS_HEADER: "not json"}) == {}
    assert decode_events_header({FAILURE_EVENTS_HEADER: "{" + " " * 5000 + "}"}) == {}
    assert decode_header({FAILURE_EVENT_HEADER: "{" + " " * 5000 + "}"}) is None
    assert EpisodeFailure.model_validate({"failure_reason": "r", "terminal": False}).failure_events is None
    unknown = reconstruct_failure("u", {"u": FailureEventRecord(event_id="u", type="os.system", failure_reason="r")})
    assert type(unknown) is NgFailureEvent and unknown.remote_type == "os.system"


async def test_an_oversized_non_ascii_failure_keeps_a_readable_reply_over_a_real_socket() -> None:
    """At the old 8 KiB bound a header line exceeds aiohttp's 8190-byte limit and the reply is unreadable; the
    bounded header keeps the real status and body and says how many observations it left out."""
    reason = "é" * 1900  # each escapes to six bytes
    causes = []
    for index in range(6):
        try:
            raise ResServerFailure(VERIFIER_ERROR, f"{index}{reason}")
        except ResServerFailure as error:
            causes.append(error)
    try:
        raise EnvServerFailure(SESSION_LOST, "lost") from causes[-1]
    except EnvServerFailure as error:
        failure = boundary_failure(error, server="env")
    unbounded = json.dumps(failure.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":"))[:8180]
    bounded = encode_header(failure)

    async def old(_request):
        return web.Response(status=500, text="real body", headers={FAILURE_EVENT_HEADER: unbounded})

    async def new(_request):
        return web.Response(status=500, text="real body", headers={FAILURE_EVENT_HEADER: bounded})

    app = web.Application()
    app.router.add_post("/old", old)
    app.router.add_post("/new", new)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        with pytest.raises(aiohttp.ClientResponseError) as unreadable:
            async with session.post(server.make_url("/old")) as response:
                await response.read()
        assert unreadable.value.status == 400  # aiohttp's own parse error, not the server's 500
        async with session.post(server.make_url("/new")) as response:
            assert (response.status, await response.text()) == (500, "real body")
            received = decode_header(response.headers)
    assert len(bounded) <= 4096 and received is not None
    assert received.failure_id in received.failure_events  # long reasons are cut first, so the chain still fits
    assert any(event.failure_reason.endswith("[truncated]") for event in received.failure_events.values())


async def test_a_429_recovered_by_the_existing_retry_reaches_collection_once(monkeypatch) -> None:
    replies = [
        SimpleNamespace(status=429, headers={}, content=SimpleNamespace(read=_read(b"busy"))),
        SimpleNamespace(status=200, headers={}),
    ]
    calls: list = []

    async def fake_request(**kwargs):
        calls.append(kwargs)
        return replies[len(calls) - 1]

    async def no_sleep(_):
        return None

    monkeypatch.setattr(openai_utils_module, "request", fake_request)
    monkeypatch.setattr(openai_utils_module, "sleep", no_sleep)
    client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.test/v1", max_http_attempts=3)

    async def agent_route():
        response = await client._request(method="POST", url="https://example.test/v1/responses")
        return {"status": response.status}

    agent_reply = _app(agent_route).post("/run")
    assert agent_reply.status_code == 200 and agent_reply.json() == {"status": 200} and len(calls) == 2

    async def env_route():
        observe_relayed_events(agent_reply.headers)
        observe_relayed_events(agent_reply.headers)  # forwarded twice, still one observation
        return {"reward": 1.0}

    env_reply = _app(env_route).post("/run")
    assert env_reply.json() == {"reward": 1.0}
    (event,) = _events_row_fields(env_reply.headers)["_ng_failure_events"].values()
    assert (event["type"], event["http_status"], event["failure_kind"]) == ("ModelFailure", 429, MODEL_REQUEST_ERROR)
    assert event["attempt_id"].startswith(event["operation_id"])


async def test_an_outer_retry_with_one_inner_try_still_observes_the_failed_attempt(monkeypatch) -> None:
    replies = [
        SimpleNamespace(status=429, ok=False, headers={}, content=SimpleNamespace(read=_read(b"busy"))),
        SimpleNamespace(status=200, ok=True, headers={}),
    ]

    async def fake_request(**kwargs):
        return replies.pop(0)

    async def fake_raise_for_status(response, content=None):
        raise RuntimeError(f"HTTP {response.status}")

    monkeypatch.setattr(openai_utils_module, "request", fake_request)
    monkeypatch.setattr(openai_utils_module, "raise_for_status", fake_raise_for_status)
    client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.test/v1", max_num_tries=1)
    with observing_failures() as observed:
        for _ in range(2):  # the caller's own retry policy
            try:
                response = await client._request(method="POST", url="https://example.test/v1/responses")
                break
            except RuntimeError:
                continue
    assert response.status == 200
    (event,) = observed.values()
    assert (event.type, event.http_status) == ("ModelFailure", 429)


def test_a_failed_reply_lists_a_forwarded_cause_once() -> None:
    inner = _res_failure()

    async def route():
        try:
            raise EnvServerFailure(SESSION_LOST, "lost") from inner
        except EnvServerFailure as outer:
            raise outer

    reply = _app(route).post("/run")
    events = _failure_row_fields(reply.headers)["_ng_failure_events"]
    assert inner.event_id in events and len(events) == 2


def test_a_full_collector_leaves_out_a_whole_chain_and_says_so() -> None:
    from nemo_gym.typed_exceptions import _MAX_OBSERVED_EVENTS, encode_events_header, observe_failure

    with observing_failures() as observed:
        for index in range(_MAX_OBSERVED_EVENTS - 1):
            observe_failure(ModelFailure(MODEL_REQUEST_ERROR, f"busy {index}"))
        try:
            raise EnvServerFailure(SESSION_LOST, "lost") from _res_failure()
        except EnvServerFailure as error:
            observe_failure(error)  # two linked events do not fit in the one remaining slot
        headers = encode_events_header(observed)
    assert len(observed) == _MAX_OBSERVED_EVENTS - 1 and observed.dropped == 2
    received = decode_events_header(headers)
    assert all(record.cause_id is None or record.cause_id in received for record in received.values())
    # Every observation is either sent or counted: the 2 the collector left out plus those the byte bound cut.
    assert _events_row_fields(headers)["_ng_failure_events_dropped"] == len(observed) + 2 - len(received)


def test_a_large_non_ascii_outer_reason_is_cut_with_a_marker_not_dropped() -> None:
    try:
        raise EnvServerFailure(SESSION_LOST, "é" * 1999) from _res_failure()
    except EnvServerFailure as error:
        value = encode_header(boundary_failure(error, server="env"))
    assert value is not None and len(value) <= 4096
    received = EpisodeFailure.model_validate_json(value)
    assert received.failure_reason.endswith("[truncated]")
    assert received.failure_id is None or received.failure_id in received.failure_events


class _SandboxApiError(Exception):
    """A foreign SDK error: its type is observed, never imported back."""

    def __init__(self, message: str, status: int | None) -> None:
        super().__init__(message)
        self.status = status


@pytest.mark.parametrize("status", [401, 500, None])
async def test_a_foreign_sdk_cause_keeps_its_observed_type_and_status_and_is_classified_only_by_its_boundary(
    status,
) -> None:
    from nemo_gym.sandbox.providers.base import SandboxCredentialsRejected

    try:
        try:
            raise _SandboxApiError("Authentication credentials are invalid.", status)
        except _SandboxApiError as sdk_error:
            if status in (401, 403):
                raise SandboxCredentialsRejected(
                    message="credentials refused", operation="sandbox.setup"
                ) from sdk_error
            raise EnvServerFailure(failure_reason="sandbox setup failed") from sdk_error
    except NgFailureEvent as error:
        inner = error

    async def run(_request):
        return web.Response(status=503, text="setup failed", headers=safe_failure_header(inner, server="env"))

    app = web.Application()
    app.router.add_post("/run", run)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        async with session.post(server.make_url("/run")) as response:
            with pytest.raises(aiohttp.ClientResponseError) as raised:
                await raise_for_status(response)
    assert raised.value.status == 503
    remote = raised.value.__cause__
    events = failure_event_map(remote)
    sdk = events[events[remote.event_id].cause_id]
    assert sdk.type == "untyped" and sdk.exception_type.endswith("_SandboxApiError") and sdk.http_status == status
    expected_kind = "provider_credentials_rejected" if status == 401 else None
    assert remote.failure_kind == expected_kind  # authentication-looking text alone names no kind


@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_differing_copies_of_one_id_are_left_out_in_any_order_and_their_references_are_cut(order) -> None:
    from nemo_gym.typed_exceptions import merge_failure_events

    good = FailureEventRecord(event_id="m1", type="ModelFailure", failure_reason="busy", http_status=429)
    other = good.model_copy(update={"failure_reason": "different"})
    parent = FailureEventRecord(event_id="a1", type="AgentFailure", failure_reason="failed", cause_id="m1")
    unrelated = FailureEventRecord(event_id="x1", type="ModelFailure", failure_reason="other")
    maps = [{"a1": parent, "m1": good}, {"m1": other, "x1": unrelated}]
    merged = merge_failure_events(*(maps if order == "forward" else maps[::-1]))
    assert "m1" not in merged and merged["x1"] == unrelated
    assert merged["a1"].cause_id is None and merged["a1"].cause_truncated is True
    # Forwarding the result again with a third copy cannot bring the conflicting id back into a valid map.
    assert EpisodeFailure(failure_reason="r", terminal=False, failure_events=merged).failure_events == merged


def test_a_conflicted_id_stays_out_when_a_later_hop_forwards_one_copy() -> None:
    from nemo_gym.typed_exceptions import encode_events_header

    good = FailureEventRecord(event_id="m1", type="ModelFailure", failure_reason="busy", http_status=429)
    other = good.model_copy(update={"failure_reason": "different"})
    parent = FailureEventRecord(event_id="a1", type="AgentFailure", failure_reason="failed", cause_id="m1")
    unrelated = FailureEventRecord(event_id="x1", type="ModelFailure", failure_reason="recovered")

    def relay(events: dict) -> dict:
        return {FAILURE_EVENTS_HEADER: json.dumps({k: v.model_dump(mode="json") for k, v in events.items()})}

    for first, second in (({"a1": parent, "m1": good}, {"m1": other}), ({"m1": other}, {"a1": parent, "m1": good})):
        with observing_failures() as hop_b:
            observe_relayed_events(relay(first))
            observe_relayed_events(relay(second | {"x1": unrelated}))
            forwarded = encode_events_header(hop_b)
        assert "m1" in hop_b.conflicted and "m1" not in hop_b and "x1" in hop_b
        # Hop C also receives an older reply holding only copy A of m1: the conflict marker keeps it out.
        with observing_failures() as hop_c:
            observe_relayed_events(forwarded)
            observe_relayed_events(relay({"m1": good}))
        assert "m1" not in hop_c and "x1" in hop_c and "m1" in hop_c.conflicted
        assert all(record.cause_id is None or record.cause_id in hop_c for record in hop_c.values())
        assert _events_row_fields(encode_events_header(hop_c))["_ng_failure_events_conflicted"] == ["m1"]


async def test_an_oversized_diagnostic_through_the_gym_request_path_keeps_status_body_and_one_call(
    monkeypatch,
) -> None:
    """The Gym request() path with its retry rules: an oversized non-ASCII diagnostic, as sent before the bound,
    makes the reply unreadable; the bounded header keeps the real status, body and remote cause in one call each."""
    from nemo_gym import server_utils

    inner = _res_failure()
    try:
        raise EnvServerFailure(SESSION_LOST, "é" * 1990) from inner
    except EnvServerFailure as error:
        failure = boundary_failure(error, server="env")
    oversized = json.dumps(failure.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":"))[:8180]
    hits = {"/run": 0, "/verify": 0}

    async def handler(request):
        hits[request.path] += 1
        header = oversized if request.query.get("mode") == "old" else encode_header(failure)
        return web.Response(status=500, text="real body", headers={FAILURE_EVENT_HEADER: header})

    app = web.Application()
    app.router.add_post("/run", handler)
    app.router.add_post("/verify", handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)  # the client request() uses
        for path in ("/run", "/verify"):
            with pytest.raises(aiohttp.ClientResponseError) as unreadable:
                await server_utils.request(
                    method="POST", url=str(server.make_url(path).with_query(mode="old")), _max_num_tries=3
                )
            assert unreadable.value.status == 400  # aiohttp's own header-size refusal, not the server's 500
            old_hits = hits[path]
            response = await server_utils.request(method="POST", url=str(server.make_url(path)), _max_num_tries=3)
            assert (response.status, await response.text()) == (500, "real body")
            with pytest.raises(aiohttp.ClientResponseError) as raised:
                await raise_for_status(response)
            assert raised.value.status == 500 and raised.value.__cause__.event_id == failure.failure_id
            assert hits[path] - old_hits == 1  # the bounded diagnostic causes no extra call
    print("REQUEST_COUNTERS", hits)


@pytest.mark.parametrize("mode", ["valid", "malformed", "oversized"])
def test_middleware_validates_and_bounds_relayed_ascii_failure_headers(mode):
    inner = boundary_failure(ModelFailure(MODEL_REQUEST_ERROR, "upstream failed", http_status=401), server="model")
    valid = encode_header(inner)
    header = {"valid": valid, "malformed": "{bad json", "oversized": "x" * 9000}[mode]

    async def route():
        error = aiohttp.ClientResponseError(
            request_info=SimpleNamespace(real_url="http://inner/run"),
            history=(),
            status=503,
            message="upstream",
            headers={FAILURE_EVENT_HEADER: header},
        )
        error.response_content = "original upstream body"
        raise error

    with _app(route) as client:
        response = client.post("/run")
    assert response.status_code == 500
    assert response.json() == "Hit an exception in my_server calling an inner server: original upstream body"
    relayed = response.headers[FAILURE_EVENT_HEADER]
    assert len(relayed) <= 4096 and relayed.isascii()
    decoded = decode_header(response.headers)
    assert decoded is not None
    if mode == "valid":
        assert decoded.failure_id == inner.failure_id and decoded.failure_events == inner.failure_events
    else:
        assert decoded.failure_id != inner.failure_id


def test_foreign_cause_keeps_identity_across_wrappers_and_wire_roundtrip():
    from nemo_gym.typed_exceptions import merge_failure_events

    inner = ModelFailure(MODEL_REQUEST_ERROR, "failed", event_id="model")
    foreign = RuntimeError("sensitive request content")
    foreign.__cause__ = inner
    first = failure_event_map(foreign)
    foreign_id = next(key for key, event in first.items() if event.type == "untyped")
    wrappers = [EnvServerFailure(SESSION_LOST, "lost"), ResServerFailure(VERIFIER_ERROR, "failed")]
    merged = first
    for wrapper in wrappers:
        wrapper.__cause__ = foreign
        events = failure_event_map(wrapper)
        assert events[wrapper.event_id].cause_id == foreign_id
        merged = merge_failure_events(merged, events)
        envelope = EpisodeFailure(
            failure_reason="failed", terminal=False, failure_id=wrapper.event_id, failure_events=events
        )
        decoded = decode_header({FAILURE_EVENT_HEADER: encode_header(envelope)})
        rebuilt = reconstruct_failure(decoded.failure_id, decoded.failure_events)
        assert failure_event_map(rebuilt) == events
    assert len(merged) == 4
    assert sum(event.type == "untyped" for event in merged.values()) == 1
    assert "sensitive" not in json.dumps({key: event.model_dump() for key, event in merged.items()})


@pytest.mark.parametrize("error_type", [aiohttp.ServerDisconnectedError, aiohttp.ClientOSError])
async def test_recovered_model_transport_failure_reaches_collection(monkeypatch, error_type):
    from unittest.mock import AsyncMock

    import nemo_gym.server_utils as server_utils

    success = SimpleNamespace(status=200, headers={})
    error = error_type("sensitive connection text")
    wire = SimpleNamespace(request=AsyncMock(side_effect=[error, success]))
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _: False)
    monkeypatch.setattr(server_utils.asyncio, "sleep", AsyncMock())
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test", max_connection_retries=3)

    async def route():
        response = await model._request(method="POST", url="http://model.test/responses")
        assert response is success
        return {"reward": 1.0}

    reply = _app(route).post("/run")
    assert reply.status_code == 200 and reply.json() == {"reward": 1.0}
    assert wire.request.await_count == 2
    records = _events_row_fields(reply.headers)["_ng_failure_events"]
    (event,) = (record for record in records.values() if record["type"] == "ModelFailure")
    assert event["type"] == "ModelFailure" and event["failure_kind"] == MODEL_REQUEST_ERROR
    assert event.get("http_status") is None
    cause = records[event["cause_id"]]
    assert cause["event_id"] == error._ng_failure_event_id
    assert cause["exception_type"] == f"{error_type.__module__}.{error_type.__qualname__}"
    assert cause["type"] == "untyped" and cause["failure_reason"] == ""
    assert "sensitive" not in json.dumps(records)
    assert event["attempt_id"].startswith(event["operation_id"])


@pytest.mark.parametrize("kind", [None, SESSION_LOST])
async def test_dependency_failure_wraps_resource_http_without_borrowing_its_kind(kind):
    from nemo_gym.base_environment_server import HandledEpisodeError, dependency_failure
    from nemo_gym.typed_exceptions import attach_failure_events

    inner = ResServerFailure(VERIFIER_ERROR, "resource verification failed", event_id="resource-original")

    async def verify(_request):
        return web.Response(status=503, text="unavailable", headers=safe_failure_header(inner, server="res"))

    app = web.Application()
    app.router.add_post("/verify", verify)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        async with session.post(server.make_url("/verify")) as response:
            with pytest.raises(aiohttp.ClientResponseError) as caught:
                await raise_for_status(response)
    handled = dependency_failure(caught.value, failure_model=EpisodeFailure, stage="verification", failure_kind=kind)
    assert isinstance(handled, HandledEpisodeError) and isinstance(handled, EnvServerFailure)
    failure = attach_failure_events(handled.failure, handled)
    assert failure.terminal is False and failure.failure_kind == kind
    from tests.unit_tests.test_environment_server import _environment_server, _EnvironmentServer, _request

    class FailedEnvironment(_EnvironmentServer):
        async def run(self, request, cleanup):
            raise handled

    template = _environment_server()
    env = FailedEnvironment(config=template.config, server_client=template.server_client)
    reply = TestClient(env.setup_webserver()).post("/run", json=_request().model_dump(mode="json"))
    assert reply.status_code == 200 and reply.json()["result"] is None
    parsed = EpisodeFailure.model_validate(reply.json()["failure"])
    assert parsed.terminal is False and parsed.failure_kind == kind
    outer = parsed.failure_events[parsed.failure_id]
    assert outer.type == "EnvServerFailure" and outer.failure_kind == kind
    assert outer.stage == "verification"
    assert parsed.failure_events[inner.event_id].failure_kind == VERIFIER_ERROR
    assert parsed.failure_events[inner.event_id].failure_reason == inner.failure_reason
    rebuilt = reconstruct_failure(parsed.failure_id, parsed.failure_events)
    assert isinstance(rebuilt, EnvServerFailure)
    assert rebuilt.__cause__.__cause__.event_id == inner.event_id


@pytest.mark.parametrize("error_type", [aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, TimeoutError])
async def test_terminal_model_transport_attempt_is_observed_without_an_extra_retry(monkeypatch, error_type):
    from unittest.mock import AsyncMock

    import nemo_gym.server_utils as server_utils

    error = error_type("private request")
    wire = SimpleNamespace(request=AsyncMock(side_effect=error))
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _: False)
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test", max_num_tries=1)
    with observing_failures() as events:
        with pytest.raises(error_type) as caught:
            await model._request(method="POST", url="http://model.test/responses")
    assert caught.value is error and wire.request.await_count == 1
    assert len(events) == 2
    root = next(event for event in events.values() if event.type == "ModelFailure")
    assert root.cause_id == error._ng_failure_event_id
    assert events[root.cause_id].failure_reason == ""


async def test_model_cancellation_is_neither_retried_nor_observed(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    import nemo_gym.server_utils as server_utils

    wire = SimpleNamespace(request=AsyncMock(side_effect=asyncio.CancelledError()))
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _: False)
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test")
    with observing_failures() as events:
        with pytest.raises(asyncio.CancelledError):
            await model._request(method="POST", url="http://model.test/responses")
    assert wire.request.await_count == 1 and not events
