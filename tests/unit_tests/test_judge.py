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

from nemo_gym.judge import JudgeError, call_judge, judge_failsafe, normalize_math_judge_verdict, reraise_judge_errors
from nemo_gym.server_utils import ServerClient


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
        with pytest.raises(JudgeError, match="HTTP 401"):
            await _call(_client(status=401, body=b'{"error": {"code": "401"}}'))

    @pytest.mark.asyncio
    async def test_transport_error_becomes_judge_error(self) -> None:
        with pytest.raises(JudgeError, match="judge unreachable"):
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

        with pytest.raises(JudgeError, match="RuntimeError: judge timeout"):
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


def _overflow_client(status: int, body: bytes) -> MagicMock:
    """A judge whose model server rejects the request the way nemo_gym.server_utils.raise_for_status raises it."""
    from types import SimpleNamespace

    from aiohttp import ClientResponseError

    def reject() -> None:
        request_info = SimpleNamespace(method="POST", url="http://judge/v1", real_url="http://judge/v1", headers={})
        raise ClientResponseError(request_info=request_info, history=(), status=status, message="Bad Request")

    resp = MagicMock(ok=False, status=status)
    resp.read = AsyncMock(return_value=body)
    resp.content.read = AsyncMock(return_value=body)
    resp.raise_for_status = MagicMock(side_effect=reject)
    return MagicMock(post=AsyncMock(return_value=resp))


_OVERFLOW_BODY = (
    b'{"error": {"message": "This model\'s maximum context length is 32768 tokens. However, you requested 40000 '
    b'tokens.", "code": "context_length_exceeded"}}'
)


class TestJudgeInputTooLong:
    """A judge input the model's own answer made too long is the model's final 0, where the server can tell."""

    @pytest.fixture(autouse=True)
    def _evaluation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The judge-overflow rule and the judge's reply on the row are evaluation-only.
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: True)
        monkeypatch.setattr("nemo_gym.judge.JUDGE_RETRY_BACKOFF_SECONDS", (0.0,))

    @pytest.mark.asyncio
    async def test_training_keeps_an_answer_caused_overflow_judge_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: False)

        async def verify(body):
            return await call_judge(
                _overflow_client(400, _OVERFLOW_BODY),
                server_name="judge",
                url_path="/v1/responses",
                json={},
                response_model=_Verdict,
                overflow_is_the_models=True,
            )

        data = orjson.loads((await judge_failsafe(verify)(_Req())).body)
        assert (data["_ng_failure_class"], data["reward"], data["mask_sample"]) == ("judge_failed", 0.0, True)

    @pytest.mark.asyncio
    async def test_an_answer_caused_overflow_is_the_models_final_zero(self) -> None:
        from nemo_gym.judge import JudgeInputTooLongError

        async def verify(body):
            return await call_judge(
                _overflow_client(400, _OVERFLOW_BODY),
                server_name="judge",
                url_path="/v1/responses",
                json={},
                response_model=_Verdict,
                overflow_is_the_models=True,
            )

        with pytest.raises(JudgeInputTooLongError):
            await verify(None)
        data = orjson.loads((await judge_failsafe(verify)(_Req(response={"final": "a" * 10}))).body)
        assert data["_ng_failure_class"] == "judge_input_too_long"
        assert data["failure_kind"] == "judge_input_too_long"
        assert data["reward"] == 0.0
        assert data["mask_sample"] is False
        from nemo_gym.failure_kinds import FAILURE_RULES, FINAL

        assert FAILURE_RULES["judge_input_too_long"] == FINAL

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("overflow_is_the_models", "status", "body"),
        [(False, 400, _OVERFLOW_BODY), (True, 400, b'{"error": "invalid request"}'), (True, 503, _OVERFLOW_BODY)],
        ids=["server-cannot-tell", "other-4xx", "5xx"],
    )
    async def test_other_judge_errors_stay_judge_failed(
        self, overflow_is_the_models: bool, status: int, body: bytes
    ) -> None:
        async def verify(body_):
            return await call_judge(
                _overflow_client(status, body),
                server_name="judge",
                url_path="/v1/responses",
                json={},
                response_model=_Verdict,
                overflow_is_the_models=overflow_is_the_models,
            )

        data = orjson.loads((await judge_failsafe(verify)(_Req())).body)
        assert data["_ng_failure_class"] == "judge_failed"
        assert data["reward"] == 0.0  # Every verify reply carries one.
        # A failed judge call never answered, so the answer got no verdict: settled infra, out of the score. That is
        # a 5xx, a 4xx such as an invalid request, and an overflow on a server that cannot tell the answer caused it.
        assert (data["mask_sample"], data["_ng_failure_judge_settled"], data["_ng_failure_cause"]) == (
            True,
            True,
            "infra",
        )
        assert "_ng_failure_terminal" not in data


class _JudgeRelayClient(ServerClient):
    """A ServerClient whose post() reaches a real Gym judge model server over its HTTP app, as the real hop does."""

    judge: object

    async def post(self, server_name: str, url_path: str, json=None, **kwargs):
        from types import SimpleNamespace

        from aiohttp import ClientResponseError

        payload = json.model_dump(mode="json", exclude_unset=True) if hasattr(json, "model_dump") else json
        reply = self.judge.post(url_path, json=payload)

        def reject() -> None:
            url = f"http://judge{url_path}"
            info = SimpleNamespace(method="POST", url=url, real_url=url, headers={})
            raise ClientResponseError(request_info=info, history=(), status=reply.status_code, message="Error")

        relayed = MagicMock(ok=reply.is_success, status=reply.status_code, headers=dict(reply.headers))
        relayed.read = AsyncMock(return_value=reply.content)
        relayed.content.read = AsyncMock(return_value=reply.content)
        relayed.raise_for_status = MagicMock(side_effect=None if reply.is_success else reject)
        return relayed


def _JudgeRelay(judge_app) -> "_JudgeRelayClient":
    from fastapi.testclient import TestClient
    from omegaconf import OmegaConf

    from nemo_gym.server_utils import BaseServerConfig

    return _JudgeRelayClient.model_construct(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        judge=TestClient(judge_app, raise_server_exceptions=False),
    )


def _judge_model_server(upstream_error: BaseException):
    """Gym's own OpenAI-compatible judge model server, whose upstream engine raises upstream_error."""
    from omegaconf import OmegaConf

    from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
    from nemo_gym.server_utils import ServerClient
    from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig

    config = SimpleModelServerConfig(
        host="0.0.0.0",
        port=8081,
        openai_base_url="https://judge.example/v1",
        openai_api_key="dummy_key",  # pragma: allowlist secret
        openai_model="judge",
        entrypoint="",
        name="judge",
    )
    server = SimpleModelServer(
        config=config, server_client=MagicMock(spec=ServerClient, global_config_dict=OmegaConf.create({}))
    )
    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_response = AsyncMock(side_effect=upstream_error)
    app = server.setup_webserver()
    server.setup_exception_middleware(app)
    return app


class TestJudgeReasonThroughTheModelServer:
    """The judge's own reason survives Gym's judge model server, which answers 500 for an upstream 4xx it keeps."""

    @pytest.fixture(autouse=True)
    def _evaluation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The judge-overflow rule and the judge's reply on the row are evaluation-only.
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: True)
        monkeypatch.setattr("nemo_gym.judge.JUDGE_RETRY_BACKOFF_SECONDS", (0.0,))

    @pytest.mark.asyncio
    async def test_training_keeps_the_row_as_before(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: False)

        row = await self._row(self._upstream(400, _OVERFLOW_BODY), overflow_is_the_models=True)

        assert (row["_ng_failure_class"], row["reward"], row["mask_sample"]) == ("judge_failed", 0.0, True)
        assert (
            row["_ng_failure_judge_error"]
            == "ClientResponseError: 500, message='Error', url='http://judge/v1/responses'"
        )

    @staticmethod
    def _upstream(status: int, body: bytes) -> BaseException:
        from types import SimpleNamespace

        from aiohttp import ClientResponseError

        info = SimpleNamespace(method="POST", url="https://judge.example/v1", real_url="https://judge.example/v1")
        error = ClientResponseError(request_info=info, history=(), status=status, message="Bad Request")
        error.response_content = body
        return error

    async def _row(self, upstream_error: BaseException, *, overflow_is_the_models: bool) -> dict:
        relay = _JudgeRelay(_judge_model_server(upstream_error))

        async def verify(body):
            return await call_judge(
                relay,
                server_name="judge",
                url_path="/v1/responses",
                json={"input": "grade this"},
                response_model=_Verdict,
                overflow_is_the_models=overflow_is_the_models,
            )

        return orjson.loads((await judge_failsafe(verify)(_Req(response={"final": "answer"}))).body)

    @pytest.mark.asyncio
    async def test_an_overflow_relayed_as_a_500_is_still_the_models_final_zero(self) -> None:
        row = await self._row(self._upstream(400, _OVERFLOW_BODY), overflow_is_the_models=True)

        assert (row["_ng_failure_class"], row["reward"], row["mask_sample"]) == ("judge_input_too_long", 0.0, False)
        assert "context_length_exceeded" in row["_ng_failure_judge_error"]

    @pytest.mark.asyncio
    async def test_a_server_that_cannot_tell_keeps_the_judges_reason_on_the_row(self) -> None:
        row = await self._row(self._upstream(400, _OVERFLOW_BODY), overflow_is_the_models=False)

        assert row["_ng_failure_class"] == "judge_failed"
        # The row says why the judge failed, so it no longer looks like any other judge failure.
        assert "context_length_exceeded" in row["_ng_failure_judge_error"]

    @pytest.mark.asyncio
    async def test_a_relayed_server_error_without_an_overflow_stays_judge_failed(self) -> None:
        row = await self._row(self._upstream(503, b'{"error": "overloaded"}'), overflow_is_the_models=True)

        assert row["_ng_failure_class"] == "judge_failed"
        assert (row["_ng_failure_judge_settled"], row["mask_sample"], row["_ng_failure_cause"]) == (
            True,
            True,
            "infra",
        )
        assert "_ng_failure_terminal" not in row


def _recording_judge(*replies: tuple[int, bytes]) -> tuple[MagicMock, list[bytes]]:
    """A judge that answers ``replies`` in order and records the exact bytes it was sent each time."""
    from types import SimpleNamespace

    from aiohttp import ClientResponseError

    sent: list[bytes] = []
    responses = []
    for status, body in replies:
        resp = MagicMock(ok=status < 400, status=status)
        resp.read = AsyncMock(return_value=body)
        resp.content.read = AsyncMock(return_value=body)
        info = SimpleNamespace(method="POST", url="http://judge/v1", real_url="http://judge/v1", headers={})
        resp.raise_for_status = MagicMock(
            side_effect=ClientResponseError(request_info=info, history=(), status=status, message=f"HTTP {status}")
        )
        responses.append(resp)

    async def post(*, server_name, url_path, json):
        sent.append(orjson.dumps(json))
        return responses[len(sent) - 1]

    return MagicMock(post=AsyncMock(side_effect=post)), sent


class TestJudgeRetriedInPlaceOnTheSameAnswer:
    """Evaluation: a judge failure is retried in place on the stored answer, never by re-running the rollout."""

    ANSWER = "the model's answer, é \U0001f600"

    @pytest.fixture(autouse=True)
    def _evaluation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: True)
        monkeypatch.setattr("nemo_gym.judge.JUDGE_RETRY_BACKOFF_SECONDS", (0.0,))
        monkeypatch.delenv("NEMO_GYM_JUDGE_MAX_ATTEMPTS", raising=False)

    async def _verify(self, client, *, no_verdict: str = "drop") -> object:
        async def verify(body):
            return await call_judge(
                client,
                server_name="judge",
                url_path="/v1/responses",
                json={"input": "grade: " + body.response["final"]},
                response_model=_Verdict,
            )

        out = await judge_failsafe(verify, no_verdict=no_verdict)(_Req(response={"final": self.ANSWER}))
        return out if isinstance(out, _Verdict) else orjson.loads(out.body)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "first",
        [(503, b'{"error": "overloaded"}'), (429, b'{"error": "rate limited"}'), (200, b"{}"), (200, b"")],
        ids=["http-5xx", "http-429", "unparseable", "empty"],
    )
    async def test_a_failed_judge_call_is_retried_on_the_byte_identical_answer(self, first) -> None:
        client, sent = _recording_judge(first, (200, b'{"verdict": "CORRECT"}'))

        verdict = await self._verify(client)

        assert verdict == _Verdict(verdict="CORRECT")
        assert len(sent) == 2
        assert sent[0] == sent[1]  # The judge saw the same stored answer, byte for byte.
        assert self.ANSWER.encode() in sent[0]

    @pytest.mark.asyncio
    async def test_a_timeout_is_retried_in_place(self) -> None:
        sent: list[bytes] = []
        ok = MagicMock(ok=True, status=200)
        ok.read = AsyncMock(return_value=b'{"verdict": "CORRECT"}')

        async def post(*, server_name, url_path, json):
            sent.append(orjson.dumps(json))
            if len(sent) == 1:
                raise TimeoutError("judge timed out")
            return ok

        verdict = await self._verify(MagicMock(post=AsyncMock(side_effect=post)))

        assert verdict == _Verdict(verdict="CORRECT")
        assert len(sent) == 2 and sent[0] == sent[1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_attempts", [None, "1", "5"])
    async def test_the_attempt_cap(self, monkeypatch: pytest.MonkeyPatch, max_attempts: str | None) -> None:
        if max_attempts is not None:
            monkeypatch.setenv("NEMO_GYM_JUDGE_MAX_ATTEMPTS", max_attempts)
        expected = int(max_attempts or 3)
        client, sent = _recording_judge(*[(503, b'{"error": "overloaded"}')] * 10)

        row = await self._verify(client)

        assert len(sent) == expected
        assert row["_ng_failure_judge_attempts"] == expected
        assert len(set(sent)) == 1

    @pytest.mark.asyncio
    async def test_a_judge_outage_after_the_retries_is_settled_infra_with_its_last_error(self) -> None:
        client, _ = _recording_judge(
            (503, b'{"error": "overloaded"}'), (503, b'{"error": "overloaded"}'), (502, b'{"error": "bad gateway"}')
        )

        row = await self._verify(client)

        assert row["_ng_failure_class"] == "judge_failed"
        # A gateway outage is infra: settled (never re-dispatched), left out of the score, counted as unresolved.
        assert (row["_ng_failure_judge_settled"], row["mask_sample"], "_ng_failure_terminal" in row) == (
            True,
            True,
            False,
        )
        assert row["_ng_failure_cause"] == row["_ng_failure_judge_cause"] == "infra"
        assert row["_ng_failure_judge_attempts"] == 3
        assert "HTTP 502" in row["_ng_failure_judge_error"]  # The last attempt's error, not the first.
        assert row["response"] == {"final": self.ANSWER}

    @pytest.mark.asyncio
    async def test_an_answer_caused_overflow_is_recorded_and_not_retried(self) -> None:
        judge = _overflow_client(400, _OVERFLOW_BODY)

        async def verify(body):
            return await call_judge(
                judge,
                server_name="judge",
                url_path="/v1/responses",
                json={"input": "grade this"},
                response_model=_Verdict,
                overflow_is_the_models=True,
            )

        row = orjson.loads((await judge_failsafe(verify)(_Req(response={"final": "x" * 50}))).body)

        assert judge.post.await_count == 1  # The answer itself is too long: asking the judge again cannot help.
        assert (row["_ng_failure_class"], row["reward"], row["mask_sample"]) == ("judge_input_too_long", 0.0, False)
        assert row["_ng_failure_judge_attempts"] == 1
        assert "context_length_exceeded" in row["_ng_failure_judge_error"]

    # The cases the real-run audit saw. Each one fails all three attempts, then judge_failure_after_retries
    # decides the row by its cause.
    _INFRA_CASES = [
        (422, b'{"error": "Budget has been exceeded! Current cost: 50.1, Max budget: 50.0"}'),
        (422, b'{"error": "unprocessable"}'),
        (429, b'{"error": "rate limited"}'),
        (401, b'{"error": "Incorrect API key provided"}'),
        (400, b'{"error": {"code": "malformed_authorization", "message": "bad header"}}'),
        (408, b'{"error": "request timeout"}'),
        (502, b'{"error": "bad gateway"}'),
        (503, b'{"error": "overloaded"}'),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "body"),
        _INFRA_CASES,
        ids=["422-budget", "422-plain", "429", "401", "400-bad-key", "408", "502", "503"],
    )
    async def test_rule_branch_infra_is_an_unresolved_row_never_re_dispatched(self, status: int, body: bytes) -> None:
        client, sent = _recording_judge(*[(status, body)] * 3)

        row = await self._verify(client)

        assert len(sent) == 3 and len(set(sent)) == 1
        assert row["_ng_failure_cause"] == row["_ng_failure_judge_cause"] == "infra"
        assert row["_ng_failure_judge_settled"] is True  # Resume never re-dispatches it.
        assert "_ng_failure_terminal" not in row  # Not the model's 0: it stays out of the score.
        assert (row["reward"], row["mask_sample"]) == (0.0, True)  # A valid verify reply that is no measurement.
        assert row["_ng_failure_judge_attempts"] == 3

    @pytest.mark.asyncio
    async def test_rule_branch_model_an_unparseable_verdict_is_the_models_final_zero(self) -> None:
        client, sent = _recording_judge(*[(200, b'{"no_verdict": "I cannot grade this."}')] * 3)

        row = await self._verify(client, no_verdict="wrong")

        assert len(sent) == 3 and len(set(sent)) == 1
        assert row["_ng_failure_cause"] == row["_ng_failure_judge_cause"] == "model"
        assert (row["_ng_failure_terminal"], row["reward"], row["mask_sample"]) == (True, 0.0, False)
        assert "_ng_failure_judge_settled" not in row

    _NO_VERDICT = (200, b'{"no_verdict": "I cannot grade this."}')
    _EMPTY = (200, b"")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("replies", "expected"),
        [
            ([_NO_VERDICT] * 3, "model"),
            ([_EMPTY] * 3, "infra"),
            ([(200, b"{}")] * 3, "infra"),
            ([_EMPTY, _NO_VERDICT, _NO_VERDICT], "infra"),
            ([(429, b'{"error": "rate limited"}')] * 3, "infra"),
        ],
        ids=["no-verdict-x3", "empty-x3", "empty-json-x3", "empty-then-no-verdict", "429-x3"],
    )
    async def test_a_reply_with_no_verdict_is_the_models_0_unless_a_try_was_empty_or_limited(
        self, replies: list, expected: str
    ) -> None:
        """Adam's refinement: a reply with no verdict after 3 tries is the model's 0, assuming it is not rate limiting."""
        client, sent = _recording_judge(*replies)

        row = await self._verify(client, no_verdict="wrong")

        assert len(sent) == 3 and len(set(sent)) == 1
        assert row["_ng_failure_cause"] == row["_ng_failure_judge_cause"] == expected
        if expected == "model":
            assert (row["_ng_failure_terminal"], row["reward"]) == (True, 0.0)
        else:
            assert (row["_ng_failure_judge_settled"], "_ng_failure_terminal" in row) == (True, False)

    @pytest.mark.parametrize(
        ("text", "error_type"),
        [("", "JudgeEmptyReplyError"), ("  \n", "JudgeEmptyReplyError"), ("maybe", "JudgeNoVerdictError")],
    )
    def test_raise_unparseable_verdict_tells_an_empty_reply_from_text(self, text: str, error_type: str) -> None:
        from nemo_gym.judge import raise_unparseable_verdict

        with pytest.raises(JudgeError) as exc:
            raise_unparseable_verdict("server", text)
        assert type(exc.value).__name__ == error_type

    @pytest.mark.asyncio
    async def test_rule_branch_model_an_answer_caused_overflow_is_the_models_final_zero(self) -> None:
        judge = _overflow_client(400, _OVERFLOW_BODY)

        async def verify(body):
            return await call_judge(
                judge,
                server_name="judge",
                url_path="/v1/responses",
                json={"input": "grade this"},
                response_model=_Verdict,
                overflow_is_the_models=True,
            )

        row = orjson.loads((await judge_failsafe(verify)(_Req(response={"final": "x" * 50}))).body)

        assert (row["_ng_failure_class"], row["reward"], row["mask_sample"]) == ("judge_input_too_long", 0.0, False)
        from nemo_gym.failure_kinds import FAILURE_RULES, FINAL

        assert FAILURE_RULES[row["_ng_failure_class"]] == FINAL

    @pytest.mark.asyncio
    async def test_an_overflow_a_server_cannot_place_stays_out_of_the_score(self) -> None:
        """Documents in the judge input may have overflowed it: never the model's 0, kept visible as infra."""
        judge = _overflow_client(400, _OVERFLOW_BODY)

        async def verify(body):
            return await call_judge(
                judge, server_name="judge", url_path="/v1/responses", json={"input": "x"}, response_model=_Verdict
            )

        row = orjson.loads((await judge_failsafe(verify)(_Req(response={"final": "short"}))).body)

        assert (row["_ng_failure_cause"], row["_ng_failure_judge_settled"]) == ("infra", True)
        assert "_ng_failure_terminal" not in row

    @pytest.mark.parametrize(
        ("cause", "terminal", "settled"), [("model", True, False), ("infra", False, True)], ids=["model", "infra"]
    )
    def test_the_rule_function_splits_by_cause(self, cause: str, terminal: bool, settled: bool) -> None:
        from nemo_gym.failure_kinds import judge_failure_after_retries

        row = judge_failure_after_retries(judge_error="last error", cause=cause, attempts=3, no_verdict="wrong")

        assert row["_ng_failure_cause"] == row["_ng_failure_judge_cause"] == cause
        assert bool(row.get("_ng_failure_terminal")) is terminal
        assert bool(row.get("_ng_failure_judge_settled")) is settled
        assert (row["_ng_failure_judge_error"], row["_ng_failure_judge_attempts"]) == ("last error", 3)
        assert row["reward"] == 0.0  # Every verify reply carries one; the infra row's is no measurement.
        assert row["mask_sample"] is settled

    @pytest.mark.asyncio
    async def test_a_nested_judge_call_does_not_multiply_the_cap(self) -> None:
        from nemo_gym.judge import retry_judge_in_place

        client, sent = _recording_judge(*[(503, b'{"error": "overloaded"}')] * 10)

        async def call_and_parse():
            return await call_judge(
                client, server_name="judge", url_path="/v1/responses", json={}, response_model=_Verdict
            )

        with pytest.raises(JudgeError) as exc:
            await retry_judge_in_place(call_and_parse)
        assert len(sent) == 3 and exc.value.attempts == 3

    @pytest.mark.asyncio
    async def test_training_makes_one_attempt_and_keeps_the_masked_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: False)
        client, sent = _recording_judge(*[(503, b'{"error": "overloaded"}')] * 3)

        row = await self._verify(client)

        assert len(sent) == 1
        assert (row["mask_sample"], "_ng_failure_terminal" in row) == (True, False)


def _through_the_agent_and_collector(verify_reply: dict, tmp_path) -> tuple[dict, list, tuple, dict]:
    """The agent's verify response contract, the collector and aggregation, through the helper every server's
    tests use."""
    from nemo_gym.testing.verify_reply import verify_reply_through_collection

    return verify_reply_through_collection(verify_reply, tmp_path)


class TestTheRulesRowReachesTheScore:
    """End to end: the rule's row survives the agent's response model, the collector and aggregation."""

    @pytest.fixture(autouse=True)
    def _evaluation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: True)
        monkeypatch.setattr("nemo_gym.rollout_collection.is_eval_mode", lambda: True)
        monkeypatch.setattr("nemo_gym.judge.JUDGE_RETRY_BACKOFF_SECONDS", (0.0,))
        monkeypatch.delenv("NEMO_GYM_JUDGE_MAX_ATTEMPTS", raising=False)

    @staticmethod
    def _request():
        from nemo_gym.base_resources_server import BaseVerifyRequest
        from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

        response = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="policy",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        return BaseVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]), response=response
        )

    @pytest.mark.asyncio
    async def test_a_judge_budget_422_after_the_tries_is_an_unresolved_infra_row(self, tmp_path) -> None:
        client, sent = _recording_judge(*[(422, b'{"error": "Budget has been exceeded!"}')] * 3)

        async def verify(body):
            return await call_judge(
                client, server_name="judge", url_path="/v1/responses", json={"input": "x"}, response_model=_Verdict
            )

        reply = orjson.loads((await judge_failsafe(verify)(self._request())).body)
        record, counted, (unresolved, owed), metrics = _through_the_agent_and_collector(reply, tmp_path)

        assert len(sent) == 3
        assert (record["_ng_failure_class"], record["failure_kind"]) == ("judge_failed", "judge_failed")
        assert record["_ng_failure_cause"] == record["_ng_failure_judge_cause"] == "infra"
        assert (record["_ng_failure_judge_attempts"], record["_ng_failure_judge_settled"]) == (3, True)
        assert "422" in record["_ng_failure_judge_error"]
        assert counted == []  # Left out of the score.
        assert (unresolved, owed) == (1, 10)  # Counted as unresolved infra.
        assert metrics["publishable"] is False  # 10% is past the 2% gate.

    @pytest.mark.asyncio
    async def test_a_reply_with_no_verdict_after_the_tries_is_the_models_counted_zero(self, tmp_path) -> None:
        client, _ = _recording_judge(*[(200, b'{"no_verdict": "I cannot grade this."}')] * 3)

        async def verify(body):
            return await call_judge(
                client, server_name="judge", url_path="/v1/responses", json={"input": "x"}, response_model=_Verdict
            )

        reply = orjson.loads((await judge_failsafe(verify, no_verdict="wrong")(self._request())).body)
        record, counted, (unresolved, _), metrics = _through_the_agent_and_collector(reply, tmp_path)

        assert record["_ng_failure_cause"] == "model"
        assert [row["reward"] for row in counted] == [0.0]
        assert unresolved == 0 and metrics["publishable"] is True


class TestNoVerdictFollowsTheOfficialGrader:
    """A judge reply with no verdict after the tries: the server's own grader decides, never re-dispatched."""

    @pytest.fixture(autouse=True)
    def _evaluation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.judge.is_eval_mode", lambda: True)
        monkeypatch.setattr("nemo_gym.rollout_collection.is_eval_mode", lambda: True)
        monkeypatch.setattr("nemo_gym.judge.JUDGE_RETRY_BACKOFF_SECONDS", (0.0,))

    async def _reply(self, **failsafe_kwargs) -> dict:
        client, _ = _recording_judge(*[(200, b'{"no_verdict": "I cannot grade this."}')] * 3)

        async def verify(body):
            return await call_judge(
                client, server_name="judge", url_path="/v1/responses", json={"input": "x"}, response_model=_Verdict
            )

        return orjson.loads(
            (await judge_failsafe(verify, **failsafe_kwargs)(TestTheRulesRowReachesTheScore._request())).body
        )

    @pytest.mark.asyncio
    async def test_a_server_with_no_stated_behaviour_drops_it(self, tmp_path) -> None:
        reply = await self._reply()  # The default: drop.

        record, counted, (unresolved, owed), metrics = _through_the_agent_and_collector(reply, tmp_path)

        # A result, not a failure row: kept in the rollouts file, never re-dispatched, and no measurement.
        assert "_ng_failure_class" not in record
        assert (record["mask_sample"], record["failure_kind"]) == (True, "judge_no_verdict")
        assert record.get("mask_reason") is None  # No reason: aggregation leaves the masked answer out of the score.
        assert (record["judge_failure_cause"], record["judge_attempts"]) == ("model", 3)
        assert counted == [] and unresolved == 0  # Not a failure row, so neither a 0 nor infra.

        from nemo_gym.reward_profile import compute_aggregate_metrics

        aggregate = compute_aggregate_metrics([record | {"_ng_task_index": 0}])
        # Left out of the score, and counted.
        assert aggregate.agent_metrics.get("coverage/masked_rollouts") == 1
        assert "mean/reward" not in aggregate.agent_metrics

    @pytest.mark.asyncio
    async def test_a_server_whose_grader_scores_it_wrong_counts_it_as_0(self, tmp_path) -> None:
        reply = await self._reply(no_verdict="wrong")

        record, counted, _, _ = _through_the_agent_and_collector(reply, tmp_path)

        assert (record["_ng_failure_class"], record["_ng_failure_cause"]) == ("judge_failed", "model")
        assert [row["reward"] for row in counted] == [0.0]

    def test_the_server_setting_defaults_to_drop(self) -> None:
        from nemo_gym.base_resources_server import BaseResourcesServerConfig

        assert BaseResourcesServerConfig.model_fields["judge_no_verdict_outcome"].default == "drop"
