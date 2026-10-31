# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pydantic import BaseModel, ConfigDict

from nemo_gym.core.episode import VERIFIER_INVOCATION_HEADER
from nemo_gym.typed_exceptions import NgFailureEvent
from nemo_gym.verification.judge import (
    JudgeError,
    call_judge,
    judge_failsafe,
    normalize_math_judge_verdict,
    reraise_judge_errors,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[[ A=B ]]", "[[A=B]]"),
        ("[[ A != B ]]", "[[A!=B]]"),
        ("[[\nA\t! = B\n]]", "[[A!=B]]"),
        ("[[A=B]] then [[ A != B ]]", "[[A=B]] then [[A!=B]]"),
        ("An explanation with spaces: [[ A=B ]]", "An explanation with spaces: [[A=B]]"),
        ("A=B without brackets", "A=B without brackets"),
        ("[[A==B]] [[A≠B]] [[a=b]] [[A=C]]", "[[A==B]] [[A≠B]] [[a=b]] [[A=C]]"),
        ("[ [ A=B ] ]", "[ [ A=B ] ]"),
        ("", ""),
    ],
)
def test_normalize_math_judge_verdict(text: str, expected: str) -> None:
    assert normalize_math_judge_verdict(text) == expected


class _Req(BaseModel):
    model_config = ConfigDict(extra="allow")

    response: dict = {}


class _Verdict(BaseModel):
    verdict: str


def _client(*, status: int, body: bytes) -> MagicMock:
    """A ServerClient stand-in whose post() returns an aiohttp-shaped response."""
    resp = MagicMock(ok=status < 400, status=status)
    resp.read = AsyncMock(return_value=body)
    resp.content.read = AsyncMock(return_value=body)
    resp.raise_for_status = MagicMock(side_effect=RuntimeError(f"HTTP {status}"))
    return MagicMock(post=AsyncMock(return_value=resp))


async def _call(client) -> _Verdict:
    return await call_judge(client, server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict)


class TestCallJudge:
    @pytest.mark.asyncio
    async def test_success_parses_response_model(self) -> None:
        assert (await _call(_client(status=200, body=b'{"verdict": "CORRECT"}'))).verdict == "CORRECT"

    @pytest.mark.asyncio
    async def test_http_error_reported_as_http_not_validation_error(self) -> None:
        # A 401 body is not a valid _Verdict. Without the status check the failure
        # would surface as an opaque pydantic ValidationError on the error payload.
        with pytest.raises(NgFailureEvent, match="HTTP 401"):
            await _call(_client(status=401, body=b'{"error": {"code": "401"}}'))

    @pytest.mark.asyncio
    async def test_transport_error_becomes_judge_error(self) -> None:
        with pytest.raises(NgFailureEvent, match="judge unreachable"):
            await _call(MagicMock(post=AsyncMock(side_effect=ConnectionError("judge unreachable"))))


class TestRunJudge:
    @pytest.mark.asyncio
    async def test_success_returns_result(self) -> None:
        async def ok():
            return "verdict"

        assert await reraise_judge_errors(ok()) == "verdict"

    @pytest.mark.asyncio
    async def test_exception_reraised_as_judge_error(self) -> None:
        async def boom():
            raise RuntimeError("judge timeout")

        with pytest.raises(NgFailureEvent, match="RuntimeError: judge timeout"):
            await reraise_judge_errors(boom())


class TestJudgeFailsafe:
    @pytest.mark.asyncio
    async def test_success_passes_through(self) -> None:
        async def verify(body):
            return {"reward": 1.0}

        assert await judge_failsafe(verify)(_Req()) == {"reward": 1.0}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("by_keyword", [False, True])
    async def test_judge_error_routed_to_sidecar(self, by_keyword: bool) -> None:
        async def verify(body):
            raise JudgeError("RuntimeError: judge 401")

        req = _Req(response={"final": "answer"})
        # FastAPI injects by keyword (kwargs["body"]); direct callers pass positionally.
        out = await (judge_failsafe(verify)(body=req) if by_keyword else judge_failsafe(verify)(req))
        data = orjson.loads(out.body)
        assert data["reward"] == 0.0
        assert data["mask_sample"] is True
        assert data["failure_kind"] == "judge_failed"
        assert data["failure_reason"] == "RuntimeError: judge 401"
        assert data["instance_config"]["mask_sample"] is True
        assert data["_ng_failure_class"] == "judge_failed"
        assert data["_ng_failure_judge_error"] == "RuntimeError: judge 401"
        # The model's final output is carried for a later judge-only replay.
        assert data["response"] == {"final": "answer"}
        # Transient: never terminal, so resume re-dispatches it.
        assert "_ng_failure_terminal" not in data

    @pytest.mark.asyncio
    @pytest.mark.parametrize("instance_config", [None, "invalid", {"task": "keep", "mask_sample": False}])
    async def test_mask_contract_overrides_request_defaults_without_mutation(self, instance_config) -> None:
        async def verify(body):
            raise JudgeError("judge offline " + "x" * 3000)

        request = _Req(
            response={"final": "answer"},
            mask_sample=False,
            failure_kind=None,
            instance_config=instance_config,
        )
        original = request.model_dump()
        data = orjson.loads((await judge_failsafe(verify)(request)).body)
        assert data["mask_sample"] is True and data["failure_kind"] == "judge_failed"
        assert data["instance_config"] == {
            **(instance_config if isinstance(instance_config, dict) else {}),
            "mask_sample": True,
        }
        assert data["failure_reason"].startswith("judge offline")
        assert len(data["failure_reason"]) == 2000
        assert data["_ng_failure_judge_error"] == data["failure_reason"]
        assert request.model_dump() == original


async def test_failsafe_keeps_judge_boundary_above_typed_model_cause():
    from nemo_gym.server_runtime.episode_types import FailureBoundary
    from nemo_gym.typed_exceptions import NgFailureEvent, failure_report_for

    original = NgFailureEvent(
        boundary=FailureBoundary.MODEL,
        failure_kind="model_request_error",
        failure_reason="provider refused",
        operation="model.request",
        http_status=401,
    )
    original_events = failure_report_for(original).events
    original_id = next(iter(original_events))

    async def rejected():
        raise original

    @judge_failsafe
    async def verify(body):
        await reraise_judge_errors(rejected())

    result = await verify(_Req(response={"saved": "answer"}))
    data = orjson.loads(result.body)
    assert data["failure_kind"] == "judge_failed"
    assert data["_ng_failure_class"] == "judge_failed"
    assert data["mask_sample"] is True
    assert data["response"] == {"saved": "answer"}
    events = data["_ng_failure_events"]
    boundary = events[data["_ng_failure_id"]]
    assert boundary.get("failure_kind") == "judge_failed" and boundary["boundary"] == "resources_server"
    assert events[original_id]["http_status"] == 401
    assert events[original_id]["failure_kind"] == "model_request_error"  # the cause keeps its own kind
    assert events[original_id] == original_events[original_id].model_dump(mode="json")


async def test_failsafe_preserves_specific_typed_judge_error_under_own_boundary():
    from nemo_gym.server_runtime.episode_types import FailureBoundary
    from nemo_gym.typed_exceptions import NgFailureEvent, failure_report_for

    original = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind="model_request_error",
        failure_reason="provider refused",
        operation="provider.request",
    )
    before = failure_report_for(original).events
    original_id = next(iter(before))

    @judge_failsafe
    async def verify(body):
        async def failed_call():
            raise original

        return await reraise_judge_errors(failed_call())

    data = orjson.loads((await verify(_Req())).body)
    assert data["failure_kind"] == "judge_failed"
    events = data["_ng_failure_events"]
    boundary = events[data["_ng_failure_id"]]
    assert boundary["failure_kind"] == "judge_failed"
    assert boundary["cause_id"] == original_id
    assert events[original_id] == before[original_id].model_dump(mode="json")


def test_judge_selector_does_not_swallow_other_canonical_failures():
    from nemo_gym.server_runtime.episode_types import FailureBoundary
    from nemo_gym.verification.judge import is_judge_failure

    for boundary, kind in [
        (FailureBoundary.MODEL, "judge_failed"),
        (FailureBoundary.RESOURCES_SERVER, "model_request_error"),
    ]:
        event = NgFailureEvent(boundary=boundary, failure_kind=kind, failure_reason="not a judge call")
        assert not is_judge_failure(event)


@pytest.mark.asyncio
async def test_judge_calls_use_enclosing_verifier_identity_even_on_failure():
    from nemo_gym.core.episode import verifier_invocation_context
    from nemo_gym.verification.judge import is_judge_failure

    client = _client(status=200, body=b'{"verdict":"CORRECT"}')
    await _call(client)
    assert "headers" not in client.post.call_args.kwargs
    with verifier_invocation_context("verify-one"):
        await _call(client)
        await _call(client)
        assert all(
            call.kwargs["headers"] == {VERIFIER_INVOCATION_HEADER: "verify-one"}
            for call in client.post.call_args_list[1:]
        )
        failed = _client(status=400, body=b'{"error":"rejected"}')
        # #5072 raises the judge's typed failure in place of JudgeError.
        with pytest.raises(NgFailureEvent, match="HTTP 400") as failure:
            await _call(failed)
        assert is_judge_failure(failure.value)
        assert failed.post.call_args.kwargs["headers"] == {VERIFIER_INVOCATION_HEADER: "verify-one"}
    await _call(client)
    assert "headers" not in client.post.call_args.kwargs


async def test_a_judge_failure_a_caller_retries_past_stays_an_observation():
    """A consumer that retries after a judge failure and then succeeds still reports that failure, once."""
    from nemo_gym.failure_transport import RequestFailureLog
    from nemo_gym.verification.judge import reraise_judge_errors

    async def unreachable():
        raise ConnectionError("judge unreachable")

    async def verdict():
        return "verdict"

    with RequestFailureLog.capture() as log:
        with pytest.raises(NgFailureEvent) as caught:
            await reraise_judge_errors(unreachable())
        # A canonical judge failure passed through again is the same event, not a second one.
        with pytest.raises(NgFailureEvent):
            await reraise_judge_errors(_raise(caught.value))
        assert await reraise_judge_errors(verdict()) == "verdict"
        report = log.reply_report(None)

    assert isinstance(caught.value.__cause__, ConnectionError)
    assert report.failure_id is None  # the request succeeded
    assert list(report.events) == [caught.value.event_id]


async def test_a_recovered_judge_failure_keeps_its_model_cause_in_the_success_report():
    """MODEL -> JUDGE: the observation a successful retry leaves names the model failure as the judge's cause."""
    from nemo_gym.failure_transport import RequestFailureLog
    from nemo_gym.server_runtime.episode_types import FailureBoundary
    from nemo_gym.verification.judge import reraise_judge_errors

    model = NgFailureEvent(
        boundary=FailureBoundary.MODEL, failure_kind="model_request_error", failure_reason="model refused"
    )

    async def verdict():
        return "verdict"

    with RequestFailureLog.capture() as log:
        with pytest.raises(NgFailureEvent) as caught:
            await reraise_judge_errors(_raise(model))
        assert await reraise_judge_errors(verdict()) == "verdict"
        report = log.reply_report(None)

    judge = caught.value
    assert judge is not model and judge.__cause__ is model
    assert report.failure_id is None
    assert set(report.events) == {judge.event_id, model.event_id}
    assert report.events[judge.event_id].cause_id == model.event_id


async def test_a_parser_judge_error_is_not_observed():
    from nemo_gym.failure_transport import RequestFailureLog
    from nemo_gym.verification.judge import reraise_judge_errors

    with RequestFailureLog.capture() as log:
        with pytest.raises(JudgeError):
            await reraise_judge_errors(_raise(JudgeError("no verdict tag")))
        assert log.reply_report(None).events == {}


async def _raise(error: BaseException):
    raise error


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_call_judge_preserves_optional_transport_timeout(explicit):
    from aiohttp import ClientTimeout

    client = _client(status=200, body=b'{"verdict": "CORRECT"}')
    timeout = ClientTimeout(total=17) if explicit else None
    await call_judge(
        client, server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict, timeout=timeout
    )
    kwargs = client.post.call_args.kwargs
    if explicit:
        assert kwargs["timeout"] is timeout
    else:
        assert "timeout" not in kwargs
