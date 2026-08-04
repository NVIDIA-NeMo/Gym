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
import asyncio
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.longmemeval import app as lme_app
from resources_servers.longmemeval import prepare_longmemeval as prepare_lme
from resources_servers.longmemeval.app import (
    _RUBRIC_ABSTENTION,
    _RUBRIC_CONTAIN,
    _RUBRIC_KNOWLEDGE_UPDATE,
    _RUBRIC_PREFERENCE,
    _RUBRIC_TEMPORAL,
    LongMemEvalResourcesServer,
    LongMemEvalResourcesServerConfig,
    LongMemEvalVerifyRequest,
    _coerce_text,
    _response_text,
    _strip_think,
    build_judge_prompt,
)
from resources_servers.longmemeval.prepare_longmemeval import (
    ANSWER_PROMPT_TEMPLATE,
    build_history_string,
    build_row,
)


_EXAMPLE_JSONL = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


@pytest.fixture
def server() -> LongMemEvalResourcesServer:
    return _server()


def _server(max_concurrency: Optional[int] = 32, max_retries: int = 5) -> LongMemEvalResourcesServer:
    config = LongMemEvalResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="longmemeval",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=64, temperature=0.0
        ),
        judge_endpoint_max_concurrency=max_concurrency,
        judge_max_retries=max_retries,
        judge_retry_base_delay=1.0,
    )
    return LongMemEvalResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request(text: str = "answer", **meta_overrides: Any) -> LongMemEvalVerifyRequest:
    meta: Optional[Dict[str, Any]] = {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "Where did the user travel?",
        "answer": "Paris",
        "abstention": False,
    }
    if meta_overrides.pop("_meta_none", False):
        meta = None
    else:
        assert meta is not None
        meta.update(meta_overrides)
    return LongMemEvalVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        verifier_metadata=meta,
    )


def _mock_judge(server: LongMemEvalResourcesServer, reply: str, monkeypatch) -> Dict[str, Any]:
    """Wire the mocked ServerClient so /v1/responses returns ``reply``; capture the call."""
    captured: Dict[str, Any] = {}

    async def fake_post(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return MagicMock()

    async def fake_get_response_json(_resp: Any) -> Dict[str, Any]:
        return _make_response(reply).model_dump()

    server.server_client.post = AsyncMock(side_effect=fake_post)
    monkeypatch.setattr(lme_app, "get_response_json", fake_get_response_json)
    return captured


def _verify(server: LongMemEvalResourcesServer, body: LongMemEvalVerifyRequest) -> Any:
    return asyncio.run(server.verify(body))


# ── verify: judge verdicts ────────────────────────────────────────────────


def test_verify_judge_yes_reward_one(server, monkeypatch) -> None:
    _mock_judge(server, "Yes", monkeypatch)
    resp = _verify(server, _request("Paris"))
    assert resp.reward == approx(1.0)
    assert resp.judge_label is True
    assert resp.judge_error is None
    assert resp.generation == "Paris"


def test_verify_judge_no_reward_zero(server, monkeypatch) -> None:
    _mock_judge(server, "no", monkeypatch)
    resp = _verify(server, _request("Berlin"))
    assert resp.reward == approx(0.0)
    assert resp.judge_label is False


@pytest.mark.parametrize("text", ["", "   ", "<think>only reasoning</think>", "<think>truncated"])
def test_verify_empty_response_skips_judge(server, monkeypatch, text: str) -> None:
    _mock_judge(server, "Yes", monkeypatch)
    resp = _verify(server, _request(text))
    assert resp.reward == approx(0.0)
    assert resp.empty_response is True
    assert resp.judge_error is None
    server.server_client.post.assert_not_called()


@pytest.mark.parametrize("reply", ["", "  ", "<think>reasoning only</think>"])
def test_verify_empty_judge_reply_sets_error(server, monkeypatch, reply: str) -> None:
    _mock_judge(server, reply, monkeypatch)
    resp = _verify(server, _request("Paris"))
    assert resp.reward == approx(0.0)
    assert resp.judge_label is False
    assert resp.judge_error == "empty_judge_output"


def test_verify_strips_think_before_judging(server, monkeypatch) -> None:
    captured = _mock_judge(server, "yes", monkeypatch)
    resp = _verify(server, _request("<think>secret</think>Paris"))
    prompt = captured["json"]["input"][0]["content"]
    assert "Paris" in prompt
    assert "secret" not in prompt
    assert resp.reward == approx(1.0)


# ── verify: rubric selection ──────────────────────────────────────────────


def test_verify_abs_question_id_uses_abstention_rubric(server, monkeypatch) -> None:
    captured = _mock_judge(server, "yes", monkeypatch)
    body = _request("I cannot tell", question_id="gpt4_x_abs", question_type="multi-session")
    resp = _verify(server, body)
    expected = _RUBRIC_ABSTENTION.format("Where did the user travel?", "Paris", "I cannot tell")
    assert captured["json"]["input"][0]["content"] == expected
    assert resp.abstention is True


@pytest.mark.parametrize(
    ("question_type", "rubric"),
    [
        ("single-session-user", _RUBRIC_CONTAIN),
        ("single-session-assistant", _RUBRIC_CONTAIN),
        ("multi-session", _RUBRIC_CONTAIN),
        ("temporal-reasoning", _RUBRIC_TEMPORAL),
        ("knowledge-update", _RUBRIC_KNOWLEDGE_UPDATE),
        ("single-session-preference", _RUBRIC_PREFERENCE),
    ],
)
def test_verify_rubric_per_question_type(server, monkeypatch, question_type: str, rubric: str) -> None:
    captured = _mock_judge(server, "yes", monkeypatch)
    _verify(server, _request("Paris", question_type=question_type))
    expected = rubric.format("Where did the user travel?", "Paris", "Paris")
    assert captured["json"]["input"][0]["content"] == expected


def test_verify_unknown_question_type(server, monkeypatch) -> None:
    _mock_judge(server, "yes", monkeypatch)
    resp = _verify(server, _request("Paris", question_type="not-a-type"))
    assert resp.reward == approx(0.0)
    assert resp.judge_error == "unknown_question_type"
    server.server_client.post.assert_not_called()


def test_build_judge_prompt_unknown_type_returns_none() -> None:
    assert build_judge_prompt("nope", "q", "a", "r", False) is None


# ── verify: failure modes ─────────────────────────────────────────────────


def test_verify_judge_exception_does_not_propagate(server) -> None:
    server.server_client.post = AsyncMock(side_effect=RuntimeError("boom"))
    resp = _verify(server, _request("Paris"))
    assert resp.reward == approx(0.0)
    assert resp.judge_error == "judge_call_failed"
    assert resp.judge_error_detail == "RuntimeError: boom"


def test_verify_metadata_none(server) -> None:
    resp = _verify(server, _request("Paris", _meta_none=True))
    assert resp.reward == approx(0.0)
    assert resp.judge_error == "bad_metadata"


def test_verify_metadata_not_a_dict(server) -> None:
    body = _request("Paris")
    body.verifier_metadata = ["not", "a", "dict"]  # type: ignore[assignment]
    resp = _verify(server, body)
    assert resp.reward == approx(0.0)
    assert resp.judge_error == "bad_metadata"


@pytest.mark.parametrize(
    "overrides",
    [{"question_type": ""}, {"question": ""}],
    ids=["blank-question-type", "blank-question"],
)
def test_verify_incomplete_metadata_is_bad_metadata(server, monkeypatch, overrides) -> None:
    _mock_judge(server, "yes", monkeypatch)
    resp = _verify(server, _request("Paris", **overrides))
    assert resp.reward == approx(0.0)
    assert resp.judge_error == "bad_metadata"
    server.server_client.post.assert_not_called()


def test_verify_missing_gold_answer_is_graded_not_excluded(server, monkeypatch) -> None:
    """Deliberate divergence: upstream KeyErrors and skips; we grade the row as 0.0."""
    captured = _mock_judge(server, "no", monkeypatch)
    resp = _verify(server, _request("Paris", answer=""))
    assert "Correct Answer: \n" in captured["json"]["input"][0]["content"]
    assert resp.reward == approx(0.0)
    assert resp.judge_error is None


def test_verify_empty_generation_with_bad_metadata_prefers_bad_metadata(server) -> None:
    resp = _verify(server, _request("", _meta_none=True))
    assert resp.judge_error == "bad_metadata"
    assert resp.empty_response is True


# ── judge retries ─────────────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status
        self.ok = 200 <= status < 300


async def _fake_raise_for_status(resp: Any) -> None:
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status}")


def _mock_judge_statuses(
    server: LongMemEvalResourcesServer, statuses: Sequence[int], monkeypatch
) -> Tuple[List[Any], List[float]]:
    """Serve ``statuses`` (last one repeats) from the judge; record calls and backoffs."""
    calls: List[Any] = []
    sleeps: List[float] = []

    async def fake_post(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _FakeResponse(statuses[min(len(calls) - 1, len(statuses) - 1)])

    async def fake_get_response_json(_resp: Any) -> Dict[str, Any]:
        return _make_response("yes").model_dump()

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    server.server_client.post = AsyncMock(side_effect=fake_post)
    monkeypatch.setattr(lme_app, "raise_for_status", _fake_raise_for_status)
    monkeypatch.setattr(lme_app, "get_response_json", fake_get_response_json)
    monkeypatch.setattr(lme_app.asyncio, "sleep", fake_sleep)
    return calls, sleeps


@pytest.mark.parametrize("status", [429, 500, 503])
def test_verify_retries_then_succeeds(server, monkeypatch, status: int) -> None:
    calls, sleeps = _mock_judge_statuses(server, [status, 200], monkeypatch)
    resp = _verify(server, _request("Paris"))
    assert resp.reward == approx(1.0)
    assert resp.judge_error is None
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_verify_retries_are_bounded_then_fail(monkeypatch) -> None:
    srv = _server(max_retries=3)
    calls, sleeps = _mock_judge_statuses(srv, [429], monkeypatch)
    resp = _verify(srv, _request("Paris"))
    assert resp.reward == approx(0.0)
    assert resp.judge_error == "judge_call_failed"
    assert resp.judge_error_detail == "RuntimeError: HTTP 429"
    assert len(calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_verify_backoff_delay_is_capped(monkeypatch) -> None:
    srv = _server(max_retries=8)
    _, sleeps = _mock_judge_statuses(srv, [429], monkeypatch)
    _verify(srv, _request("Paris"))
    assert max(sleeps) == approx(lme_app._RETRY_MAX_DELAY_S)


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_verify_non_retryable_status_fails_immediately(server, monkeypatch, status: int) -> None:
    calls, sleeps = _mock_judge_statuses(server, [status], monkeypatch)
    resp = _verify(server, _request("Paris"))
    assert resp.judge_error == "judge_call_failed"
    assert len(calls) == 1
    assert sleeps == []


def test_verify_zero_retries_configured(monkeypatch) -> None:
    srv = _server(max_retries=0)
    calls, sleeps = _mock_judge_statuses(srv, [429], monkeypatch)
    assert _verify(srv, _request("Paris")).judge_error == "judge_call_failed"
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize("status", [None, "429", True])
def test_is_retryable_status_ignores_non_int_status(status: Any) -> None:
    resp = MagicMock()
    resp.status = status
    assert lme_app._is_retryable_status(resp) is False


# ── judge request shape ───────────────────────────────────────────────────


def test_judge_request_params(server, monkeypatch) -> None:
    captured = _mock_judge(server, "yes", monkeypatch)
    _verify(server, _request("Paris"))
    assert captured["url_path"] == "/v1/responses"
    assert captured["server_name"] == "judge_model"
    payload = captured["json"]
    assert payload["temperature"] == approx(0.0)
    assert payload["max_output_tokens"] == 64
    assert len(payload["input"]) == 1
    assert payload["input"][0]["role"] == "user"


def test_judge_concurrency_is_bounded(monkeypatch) -> None:
    srv = _server(max_concurrency=1)
    state = {"active": 0, "max_active": 0}

    async def fake_post(**_kwargs: Any) -> Any:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return MagicMock()

    async def fake_get_response_json(_resp: Any) -> Dict[str, Any]:
        return _make_response("yes").model_dump()

    srv.server_client.post = AsyncMock(side_effect=fake_post)
    monkeypatch.setattr(lme_app, "get_response_json", fake_get_response_json)

    async def run_both() -> None:
        await asyncio.gather(srv.verify(_request("Paris")), srv.verify(_request("Paris")))

    asyncio.run(run_both())
    assert state["max_active"] == 1


def test_nullcontext_when_concurrency_none(monkeypatch) -> None:
    srv = _server(max_concurrency=None)
    _mock_judge(srv, "yes", monkeypatch)
    assert _verify(srv, _request("Paris")).reward == approx(1.0)


def test_setup_webserver(server) -> None:
    assert server.setup_webserver() is not None


# ── metrics ───────────────────────────────────────────────────────────────


def _row(reward: float, question_type: str, abstention: bool = False, **kw: Any) -> Dict[str, Any]:
    row = {"reward": reward, "question_type": question_type, "abstention": abstention}
    row.update(kw)
    return row


def test_compute_metrics(server) -> None:
    rows = [
        _row(1.0, "multi-session"),
        _row(0.0, "multi-session"),
        _row(1.0, "temporal-reasoning"),
        _row(0.0, "multi-session", abstention=True, judge_error="judge_call_failed"),
    ]
    metrics = server.compute_metrics([rows[:2], rows[2:]])
    # Only the failed judge call is excluded from every mean, as upstream does.
    assert metrics["accuracy"] == approx(2 / 3)
    assert metrics["count"] == 3
    assert metrics["accuracy_strict"] == approx(0.5)
    assert metrics["question_type/multi-session/accuracy"] == approx(0.5)
    assert metrics["question_type/multi-session/count"] == 2
    assert metrics["question_type/temporal-reasoning/accuracy"] == approx(1.0)
    assert "abstention/accuracy" not in metrics
    assert metrics["n_judge_call_failed"] == 1
    assert metrics["n_excluded"] == 1


@pytest.mark.parametrize(
    ("reason", "counter"),
    [
        ("empty_judge_output", "n_empty_judge_output"),
        ("bad_metadata", "n_bad_metadata"),
    ],
)
def test_compute_metrics_included_reasons_stay_in_denominator(server, reason: str, counter: str) -> None:
    rows = [_row(1.0, "multi-session"), _row(0.0, "multi-session", judge_error=reason)]
    metrics = server.compute_metrics([rows])
    assert metrics["accuracy"] == approx(0.5)
    assert metrics["count"] == 2
    assert metrics["n_excluded"] == 0
    assert metrics[counter] == 1
    assert metrics["question_type/multi-session/count"] == 2


def test_compute_metrics_empty_response_counts_as_zero(server) -> None:
    rows = [_row(1.0, "multi-session"), _row(0.0, "multi-session", empty_response=True)]
    metrics = server.compute_metrics([rows])
    assert metrics["accuracy"] == approx(0.5)
    assert metrics["count"] == 2
    assert metrics["n_empty_response"] == 1
    assert metrics["n_excluded"] == 0


def test_compute_metrics_unknown_question_type_is_excluded(server) -> None:
    rows = [_row(1.0, "multi-session"), _row(0.0, "not-a-type", judge_error="unknown_question_type")]
    metrics = server.compute_metrics([rows])
    assert metrics["accuracy"] == approx(1.0)
    assert metrics["count"] == 1
    assert metrics["n_unknown_question_type"] == 1
    assert metrics["n_excluded"] == 1


def test_compute_metrics_unrecognised_reason_is_kept_and_counted(server) -> None:
    rows = [_row(1.0, "multi-session"), _row(0.0, "multi-session", judge_error="RuntimeError: legacy")]
    metrics = server.compute_metrics([rows])
    assert metrics["accuracy"] == approx(0.5)
    assert metrics["n_judge_errors_other"] == 1
    assert metrics["n_excluded"] == 0


def test_compute_metrics_all_judge_errors(server) -> None:
    metrics = server.compute_metrics([[_row(0.0, "multi-session", judge_error="judge_call_failed")]])
    assert "accuracy" not in metrics
    assert "count" not in metrics
    assert metrics["accuracy_strict"] == approx(0.0)
    assert metrics["n_judge_call_failed"] == 1
    assert metrics["n_excluded"] == 1
    assert metrics["n_judge_errors_other"] == 0
    assert metrics["n_empty_response"] == 0


def test_compute_metrics_missing_question_type(server) -> None:
    metrics = server.compute_metrics([[{"reward": 1.0}]])
    assert metrics["question_type/unknown/accuracy"] == approx(1.0)
    assert "abstention/accuracy" not in metrics


def test_compute_metrics_excluded_empty_response_not_counted(server) -> None:
    """A row outside the denominator must not move a counter about the denominator."""
    rows = [
        _row(1.0, "multi-session"),
        _row(0.0, "not-a-type", judge_error="unknown_question_type", empty_response=True),
    ]
    metrics = server.compute_metrics([rows])
    assert metrics["count"] == 1
    assert metrics["n_empty_response"] == 0
    assert metrics["n_unknown_question_type"] == 1


def test_compute_metrics_empty(server) -> None:
    assert server.compute_metrics([]) == {}
    assert server.compute_metrics([[{"reward": "bad"}]]) == {}
    assert server.compute_metrics([[{"reward": True}]]) == {}


def test_get_key_metrics(server) -> None:
    metrics = server.compute_metrics([[_row(1.0, "multi-session", abstention=True)]])
    assert server.get_key_metrics(metrics) == {
        "accuracy": approx(1.0),
        "abstention/accuracy": approx(1.0),
        "n_excluded": 0,
    }
    assert server.get_key_metrics({}) == {}


def test_empty_generation_end_to_end_drags_accuracy_down(server, monkeypatch) -> None:
    """verify() → compute_metrics: an empty generation must stay in the denominator."""
    _mock_judge(server, "yes", monkeypatch)
    answered = _verify(server, _request("Paris"))
    empty = _verify(server, _request("   "))

    rows = [answered.model_dump(), empty.model_dump()]
    metrics = server.compute_metrics([rows])

    assert answered.reward == approx(1.0) and empty.reward == approx(0.0)
    assert metrics["count"] == 2
    assert metrics["accuracy"] == approx(0.5)
    assert metrics["n_excluded"] == 0
    assert metrics["n_empty_response"] == 1
    assert metrics["question_type/multi-session/count"] == 2


# ── text helpers ──────────────────────────────────────────────────────────


def test_strip_think_variants() -> None:
    assert _strip_think("<think>x</think>answer") == "answer"
    assert _strip_think("reasoning</think>final") == "final"
    assert _strip_think("<think>unclosed reasoning") == ""
    assert _strip_think("plain") == "plain"
    assert _strip_think("") == ""
    assert _strip_think(None) == ""  # type: ignore[arg-type]


def test_coerce_text_variants() -> None:
    assert _coerce_text("plain") == "plain"
    assert _coerce_text(["a", {"text": "b"}, 3]) == "ab"
    assert _coerce_text(None) == ""
    assert _coerce_text(42) == "42"


def test_response_text_variants() -> None:
    assert _response_text(None) == ""
    assert _response_text(_make_response("direct")) == "direct"


def test_response_text_falls_back_to_output_items() -> None:
    class _Item:
        def __init__(self, type_: str, content: Any) -> None:
            self.type = type_
            self.content = content

    class _Resp:
        output_text = ""
        output = [_Item("reasoning", "ignored"), _Item("message", [{"text": "from-items"}])]

    assert _response_text(_Resp()) == "from-items"  # type: ignore[arg-type]


# ── prepare_longmemeval ───────────────────────────────────────────────────


_ENTRY = {
    "question_id": "q_abs",
    "question_type": "multi-session",
    "question": "Where?",
    "answer": "Paris",
    "question_date": "2023/05/01 (Mon) 10:00",
    "haystack_dates": ["2023/03/02", "2023/01/01", "2023/02/01"],
    "haystack_sessions": [
        [{"role": "user", "content": "c", "has_answer": True}],
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}, "raw-turn"],
    ],
}


def test_build_history_string_matches_upstream_format() -> None:
    history = build_history_string(_ENTRY, topk=0)
    expected = (
        '\n### Session 1:\nSession Date: 2023/01/01\nSession Content:\n\n[{"role": "user", "content": "a"}]\n'
        "\n### Session 2:\nSession Date: 2023/02/01\nSession Content:\n\n"
        '[{"role": "user", "content": "b"}, "raw-turn"]\n'
        '\n### Session 3:\nSession Date: 2023/03/02\nSession Content:\n\n[{"role": "user", "content": "c"}]\n'
    )
    assert history == expected
    assert "has_answer" not in history


def test_build_history_string_topk_keeps_last_then_sorts() -> None:
    history = build_history_string(_ENTRY, topk=2)
    # Last 2 in dataset order are the 2023/01/01 and 2023/02/01 sessions, then date-sorted.
    assert "2023/03/02" not in history
    assert history.index("2023/01/01") < history.index("2023/02/01")


def test_build_history_string_non_list_session() -> None:
    entry = {"haystack_dates": ["2023/01/01"], "haystack_sessions": ["not-a-list"]}
    assert '"not-a-list"' in build_history_string(entry)


def test_build_history_string_empty_entry() -> None:
    assert build_history_string({}) == ""


def test_build_row_shape_and_prompt() -> None:
    row = build_row(_ENTRY, split="oracle", topk=0)
    assert set(row) == {"responses_create_params", "verifier_metadata"}
    messages = row["responses_create_params"]["input"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], str)
    expected = ANSWER_PROMPT_TEMPLATE.format(build_history_string(_ENTRY, 0), "2023/05/01 (Mon) 10:00", "Where?")
    assert messages[0]["content"] == expected
    meta = row["verifier_metadata"]
    assert meta["abstention"] is True
    assert meta["split"] == "oracle" and meta["topk_context"] == 0
    assert all(isinstance(v, (str, int, bool)) for v in meta.values())


def test_build_row_missing_fields() -> None:
    with pytest.warns(RuntimeWarning, match="empty history"):
        row = build_row({})
    meta = row["verifier_metadata"]
    assert meta["question_id"] == "" and meta["abstention"] is False


def test_main_writes_jsonl(tmp_path, monkeypatch) -> None:
    source = tmp_path / "src.json"
    source.write_text(json.dumps([_ENTRY, _ENTRY]), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "--input", str(source), "--output", str(out), "--limit", "1", "--topk-context", "0"],
    )
    prepare_lme.main()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["verifier_metadata"]["question_id"] == "q_abs"


def test_rows_losing_evidence_counts_only_dropped_gold_sessions() -> None:
    # 3 sessions, gold is the FIRST one -> a topk=2 slice (keeps the last 2) drops it.
    losing = {
        "haystack_session_ids": ["s1", "s2", "s3"],
        "answer_session_ids": ["s1"],
    }
    # Same shape, but gold is in the kept tail.
    safe = {
        "haystack_session_ids": ["s1", "s2", "s3"],
        "answer_session_ids": ["s3"],
    }
    # Fewer sessions than topk -> nothing is sliced away.
    short = {"haystack_session_ids": ["s1"], "answer_session_ids": ["s1"]}

    assert prepare_lme._rows_losing_evidence([losing, safe, short], 2) == 1
    assert prepare_lme._rows_losing_evidence([losing, safe, short], 0) == 0
    assert prepare_lme._rows_losing_evidence([losing, safe, short], 99) == 0
    assert prepare_lme._rows_losing_evidence([], 50) == 0


def test_main_warns_when_topk_would_drop_gold_evidence(tmp_path, monkeypatch) -> None:
    entry = dict(_ENTRY)
    entry["haystack_dates"] = ["2023/01/01 (Sun) 00:00", "2023/01/02 (Mon) 00:00", "2023/01/03 (Tue) 00:00"]
    entry["haystack_sessions"] = [[{"role": "user", "content": c}] for c in ("a", "b", "c")]
    entry["haystack_session_ids"] = ["s1", "s2", "s3"]
    entry["answer_session_ids"] = ["s1"]
    source = tmp_path / "src.json"
    source.write_text(json.dumps([entry]), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "--input", str(source), "--output", str(out), "--topk-context", "2"],
    )

    with pytest.warns(RuntimeWarning, match="drops a gold evidence session on 1 of 1 rows"):
        prepare_lme.main()

    # The dataset is still written — the warning is advisory, not fatal.
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_main_limit_does_not_clobber_the_full_split_file(tmp_path, monkeypatch) -> None:
    (tmp_path / "longmemeval_oracle.json").write_text(json.dumps([_ENTRY, _ENTRY]), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    full = out_dir / "oracle.jsonl"
    full.write_text("FULL BUILD\n", encoding="utf-8")
    monkeypatch.setattr(prepare_lme, "_default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(prepare_lme, "_OUT_DIR", out_dir)
    monkeypatch.setattr(sys, "argv", ["prepare", "--limit", "1"])

    prepare_lme.main()

    assert full.read_text(encoding="utf-8") == "FULL BUILD\n"
    truncated = (out_dir / "oracle_limit1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(truncated) == 1


@pytest.mark.parametrize(
    ("split", "filename"),
    [
        ("oracle", "longmemeval_oracle.json"),
        ("s", "longmemeval_s_cleaned.json"),
        ("m", "longmemeval_m_cleaned.json"),
    ],
)
def test_main_resolves_split_default_input_and_output(tmp_path, monkeypatch, split: str, filename: str) -> None:
    """Each split reads its released filename and writes to its own data/<split>.jsonl."""
    (tmp_path / filename).write_text(json.dumps([_ENTRY]), encoding="utf-8")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(prepare_lme, "_default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(prepare_lme, "_OUT_DIR", out_dir)
    monkeypatch.setattr(sys, "argv", ["prepare", "--split", split])
    prepare_lme.main()
    row = json.loads((out_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["verifier_metadata"]["split"] == split


# ── split download ────────────────────────────────────────────────────────


class _FakeUrlResponse:
    """Minimal ``urlopen`` stand-in: a context manager over some bytes."""

    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def __enter__(self) -> io.BytesIO:
        return self._buffer

    def __exit__(self, *_exc: object) -> bool:
        return False


def test_fetch_split_downloads_missing_file(tmp_path, monkeypatch) -> None:
    payload = json.dumps([_ENTRY]).encode("utf-8")
    requested: List[str] = []

    def fake_urlopen(url: str, timeout: float = 0.0) -> _FakeUrlResponse:
        requested.append(url)
        return _FakeUrlResponse(payload)

    monkeypatch.setattr(prepare_lme.urllib.request, "urlopen", fake_urlopen)
    path = prepare_lme._fetch_split("s", tmp_path / "cache")

    assert requested == [f"{prepare_lme._HF_BASE}/longmemeval_s_cleaned.json"]
    assert path.read_bytes() == payload
    assert not list(path.parent.glob("*.part"))


def test_fetch_split_reuses_cached_file(tmp_path, monkeypatch) -> None:
    cached = tmp_path / "longmemeval_oracle.json"
    cached.write_text("CACHED", encoding="utf-8")

    def fail_urlopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cached split must not be re-downloaded")

    monkeypatch.setattr(prepare_lme.urllib.request, "urlopen", fail_urlopen)
    assert prepare_lme._fetch_split("oracle", tmp_path) == cached


def test_fetch_split_cleans_up_partial_download(tmp_path, monkeypatch) -> None:
    """A failed download leaves no .part behind that a later run could parse."""

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("connection reset")

    monkeypatch.setattr(prepare_lme.urllib.request, "urlopen", boom)
    with pytest.raises(OSError):
        prepare_lme._fetch_split("oracle", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_default_cache_dir_honours_xdg(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert prepare_lme._default_cache_dir() == tmp_path / "longmemeval"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert prepare_lme._default_cache_dir() == Path.home() / ".cache" / "longmemeval"


# ── committed example dataset ─────────────────────────────────────────────


def test_example_jsonl_is_well_formed() -> None:
    rows = [json.loads(line) for line in _EXAMPLE_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 5
    for row in rows:
        assert set(row) == {"responses_create_params", "verifier_metadata"}
        assert row["responses_create_params"]["input"][0]["content"].strip()
        assert row["verifier_metadata"]["question_type"] in lme_app._RUBRIC_BY_TYPE
    types = {r["verifier_metadata"]["question_type"] for r in rows}
    assert len(types) >= 4
    # The preference rubric has different slot semantics; keep it exercised.
    assert "single-session-preference" in types
    assert sum(1 for r in rows if r["verifier_metadata"]["abstention"]) >= 1
