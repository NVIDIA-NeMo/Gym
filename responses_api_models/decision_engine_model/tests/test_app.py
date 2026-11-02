# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from typing import List, Tuple
from unittest.mock import MagicMock

import aiohttp
import pytest
from fastapi import HTTPException

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_models.decision_engine_model import app as engine_app
from responses_api_models.decision_engine_model.app import (
    DecisionEngineModel,
    DecisionEngineModelConfig,
    decision_request,
    from_decisions_response,
    to_decisions_request,
)


QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "tech": "Bugs"}},
    "refund": {"type": "noul", "instructions": "Asks for a refund?"},
}
STATE = {"ticket": "I was charged twice."}
ANSWERS = {
    "route": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}},
    "refund": {"type": "noul", "noul": 0.8},
}


def _model(**overrides) -> DecisionEngineModel:
    config = DecisionEngineModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        base_url="http://engine.test/",
        api_key="test-key",  # pragma: allowlist secret
        model="decider",
        backoff_seconds=0.0,
        **overrides,
    )
    return DecisionEngineModel(config=config, server_client=MagicMock(spec=ServerClient))


def _body(*messages) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(input=list(messages))


def _scripted(model: DecisionEngineModel, replies: List[Tuple[int, str] | Exception]) -> list:
    calls = []

    async def post(path, payload):
        calls.append((path, payload))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    model._post = post
    return calls


def _envelope(response) -> dict:
    return json.loads(response.output[0].content[0].text)["di_engine"]


class TestRequestMapping:
    def test_decision_request_reads_the_last_user_message(self) -> None:
        payload = json.dumps({"state": STATE, "questions": QUESTIONS})
        body = _body(
            {"role": "system", "content": "Answer as JSON."},
            {"role": "user", "content": [{"type": "input_text", "text": payload}]},
        )
        assert decision_request(body) == {"state": STATE, "questions": QUESTIONS}
        assert decision_request(NeMoGymResponseCreateParamsNonStreaming(input=payload))["state"] == STATE

    @pytest.mark.parametrize("content", ["not json", json.dumps({"questions": QUESTIONS}), json.dumps([1])])
    def test_decision_request_rejects_other_input(self, content: str) -> None:
        with pytest.raises(HTTPException) as error:
            decision_request(_body({"role": "user", "content": content}, {"role": "assistant", "content": "hi"}))
        assert error.value.status_code == 400

    def test_decisions_shape_round_trip(self) -> None:
        request = to_decisions_request("decider", STATE, QUESTIONS)
        assert request["input"] == [{"role": "user", "content": [{"type": "input_json", "data": STATE}]}]
        assert request["questions"][0]["choices"] == [
            {"value": "billing", "description": "Charges"},
            {"value": "tech", "description": "Bugs"},
        ]
        assert request["questions"][1] == {"type": "predicate", "name": "refund", "instructions": "Asks for a refund?"}
        assert (
            to_decisions_request("m", "plain text", {"q": {**QUESTIONS["refund"], "instructions": {"a": 1}}})["input"]
            == "plain text"
        )
        with pytest.raises(ValueError):
            to_decisions_request("m", STATE, {"s": {"type": "score", "criteria": ["a", "b"]}})
        raw = {
            "model": "decider",
            "answers": [
                {
                    "type": "choice",
                    "name": "route",
                    "choice": "billing",
                    "probabilities": [{"value": "billing", "probability": 0.9}, {"value": "tech", "probability": 0.1}],
                },
                {"type": "predicate", "name": "refund", "probability": 0.8},
            ],
        }
        assert from_decisions_response(raw)["answers"] == ANSWERS


class TestEngineCalls:
    def test_systemone_ok_strips_the_trace_and_wraps_the_answers(self) -> None:
        model = _model(extra_body={"packing": "shared"})
        raw = {"model": "decider", "answers": ANSWERS, "usage": {"input_tokens": 42}, "evaluation_trace": [1]}
        calls = _scripted(model, [(200, json.dumps(raw))])
        body = _body({"role": "user", "content": json.dumps({"state": STATE, "questions": QUESTIONS})})
        response = asyncio.run(model.responses(body))
        envelope = _envelope(response)
        assert envelope["status"] == "ok" and envelope["response"] == {
            "model": "decider",
            "answers": ANSWERS,
            "usage": {"input_tokens": 42},
        }
        # One answer position per question; the engine's own input-token count.
        assert (response.usage.input_tokens, response.usage.output_tokens) == (42, 2)
        assert calls == [
            ("/v1/systemone", {"model": "decider", "state": STATE, "questions": QUESTIONS, "packing": "shared"})
        ]

    def test_decisions_endpoint(self) -> None:
        model = _model(endpoint="decisions")
        raw = {"model": "decider", "answers": [{"type": "predicate", "name": "refund", "probability": 0.3}]}
        calls = _scripted(model, [(200, json.dumps(raw))])
        envelope = asyncio.run(model.call_engine(STATE, {"refund": QUESTIONS["refund"]}))
        assert engine_app.usage_of(None, envelope) is None
        assert calls[0][0] == "/v1/decisions" and envelope["response"]["answers"]["refund"] == {
            "type": "noul",
            "noul": 0.3,
        }

    def test_capacity_rejection_is_unsupported(self) -> None:
        model = _model()
        _scripted(model, [(422, "prompt is longer than the maximum model length")])
        assert asyncio.run(model.call_engine(STATE, QUESTIONS))["status"] == "unsupported"

    def test_retries_then_reports_an_error_without_raising(self) -> None:
        model = _model(max_attempts=3)
        calls = _scripted(model, [aiohttp.ClientConnectionError("reset"), (503, "busy"), (429, "slow down")])
        envelope = asyncio.run(model.call_engine(STATE, QUESTIONS))
        assert envelope["status"] == "error" and "HTTP 429" in envelope["error"] and len(calls) == 3

    def test_retry_recovers(self) -> None:
        model = _model(max_attempts=2)
        _scripted(model, [(500, "boom"), (200, json.dumps({"answers": ANSWERS}))])
        assert asyncio.run(model.call_engine(STATE, QUESTIONS))["status"] == "ok"

    @pytest.mark.parametrize(
        ("status", "text", "message"), [(400, "bad request", "HTTP 400"), (200, "<html>", "non-JSON")]
    )
    def test_client_errors_and_garbage_are_errors(self, status: int, text: str, message: str) -> None:
        model = _model()
        _scripted(model, [(status, text)])
        envelope = asyncio.run(model.call_engine(STATE, QUESTIONS))
        assert envelope["status"] == "error" and message in envelope["error"]

    def test_chat_completions_is_not_served(self) -> None:
        with pytest.raises(HTTPException):
            asyncio.run(_model().chat_completions(body=MagicMock()))

    def test_post_sends_the_bearer_token(self, monkeypatch) -> None:
        seen = {}

        class Reply:
            status = 200

            async def text(self):
                return "{}"

        async def fake_request(**kwargs):
            seen.update(kwargs)
            return Reply()

        monkeypatch.setattr(engine_app, "request", fake_request)
        model = _model()
        assert asyncio.run(model._post("/v1/systemone", {"a": 1})) == (200, "{}")
        assert (
            seen["url"] == "http://engine.test/v1/systemone" and seen["headers"]["Authorization"] == "Bearer test-key"
        )
        model.config.api_key = None
        assert "Authorization" not in model._headers()
