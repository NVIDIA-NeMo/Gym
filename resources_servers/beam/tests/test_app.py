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
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, get_args
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from pytest import approx

from nemo_gym import judge as judge_module
from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.config import ModelServerRef
from nemo_gym.judge import JudgeError, judge_failsafe
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.beam.app import (
    ABILITIES,
    BEAMResourcesServer,
    BEAMResourcesServerConfig,
    BEAMVerifyRequest,
    build_rubric_prompt,
    event_ordering_score,
    parse_judge_score,
    response_text,
    strip_think,
)
from resources_servers.beam.prompts import EQUIVALENCE_SYSTEM_PROMPT
from resources_servers.beam.task_data import TaskData


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


def _server(**config_overrides: Any) -> BEAMResourcesServer:
    config = BEAMResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="beam",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], max_output_tokens=256, temperature=0.0
        ),
        judge_endpoint_max_concurrency=8,
        **config_overrides,
    )
    return BEAMResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request(
    text: str, ability: str = "information_extraction", rubric: Optional[List[str]] = None
) -> BEAMVerifyRequest:
    return BEAMVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        verifier_metadata={
            "conversation_id": "7",
            "chat_size": "100K",
            "ability": ability,
            "question": "q?",
            "rubric": ["LLM response should state: March 29"] if rubric is None else rubric,
        },
    )


def _mock_judge(
    server: BEAMResourcesServer, judge: Callable[[List[Dict[str, str]]], str], monkeypatch, statuses=()
) -> List[Dict[str, Any]]:
    """Route /v1/responses to ``judge(messages)``; the first calls get ``statuses`` (e.g. 500) instead.

    Patches the HTTP helpers ``nemo_gym.judge.call_judge`` uses, so failures surface as ``JudgeError``.
    """
    calls: List[Dict[str, Any]] = []
    pending = list(statuses)

    async def fake_post(**kwargs: Any) -> Any:
        calls.append(kwargs["json"])
        resp = MagicMock()
        resp.status = pending.pop(0) if pending else 200
        resp.reply = judge(kwargs["json"]["input"])
        return resp

    async def fake_raise_for_status(resp: Any) -> None:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}")

    async def fake_get_response_json(resp: Any) -> Dict[str, Any]:
        return _make_response(resp.reply).model_dump()

    server.server_client.post = AsyncMock(side_effect=fake_post)
    monkeypatch.setattr(judge_module, "raise_for_status", fake_raise_for_status)
    monkeypatch.setattr(judge_module, "get_response_json", fake_get_response_json)
    return calls


def _score_reply(score: float) -> Callable[[List[Dict[str, str]]], str]:
    return lambda _messages: json.dumps({"score": score, "reason": "r"})


# ── helpers ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("plain", "plain"),
        ("<think>hidden</think>answer", "answer"),
        ("<think>truncated", ""),
        ("pre</think>answer", "answer"),
    ],
)
def test_strip_think(text: str, expected: str) -> None:
    assert strip_think(text) == expected


def test_response_text_skips_non_message_items() -> None:
    resp = _make_response("hello")
    resp.output.insert(0, MagicMock(type="reasoning"))
    assert response_text(resp) == "hello"
    assert response_text(None) == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"score": 1.0, "reason": "ok"}', 1.0),
        ('```json\n{"score": 0.5, "reason": "partial"}\n```', 0.5),
        ('Sure! {"score": 0.0, "reason": "missing"} done', 0.0),
        ('{"score": "0.5", "reason": "string score"}', 0.5),
        ('{"score": [1.0], "reason": "unescaped "quote" breaks json"}', 1.0),
        ("no json here", None),
        ('{"score": "high"}', None),
        # Out of range or non-finite scores are invalid, not clamped.
        ('{"score": 5}', None),
        ('{"score": -1}', None),
        ('{"score": NaN}', None),
        ('{"score": Infinity}', None),
        ('{"score": 0.7}', None),
        ('"score": 2', None),
    ],
)
def test_parse_judge_score(raw: str, expected: Optional[float]) -> None:
    assert parse_judge_score(raw) == expected


def test_parse_judge_score_binary_rejects_partial_credit() -> None:
    binary = frozenset({0.0, 1.0})
    assert parse_judge_score('{"score": 0.5}', binary) is None
    assert parse_judge_score('{"score": 1}', binary) == 1.0
    assert math.isnan(float("nan")) and parse_judge_score('{"score": NaN}', binary) is None


def test_build_rubric_prompt_fills_both_slots() -> None:
    prompt = build_rubric_prompt("must say X", "the answer")
    assert "RUBRIC CRITERION (what to check): must say X" in prompt
    assert "RESPONSE TO EVALUATE: the answer" in prompt
    assert "<rubric_item>" not in prompt and "<llm_response>" not in prompt


def test_event_ordering_score_perfect_reversed_and_empty() -> None:
    ref = ["a", "b", "c"]
    perfect = event_ordering_score(ref, ["a", "b", "c"])
    assert perfect["tau_norm"] == approx(1.0) and perfect["f1"] == approx(1.0)
    reversed_ = event_ordering_score(ref, ["c", "b", "a"])
    assert reversed_["tau_norm"] == approx(0.0) and reversed_["f1"] == approx(1.0)
    # Empty alignment gives a constant ranking: tau-b is NaN upstream, 0.0 here.
    assert event_ordering_score(ref, [])["tau_norm"] == 0.0


@pytest.mark.parametrize(
    ("system", "tau_norm", "f1"),
    [
        # Literals checked against scipy.stats.kendalltau(variant="b").
        (["a"], 0.9082482904638631, 0.5),
        (["a", "x", "c"], 0.5, 2 / 3),
        (["c", "", "a", "b"], 0.33333333333333337, 6 / 7),
    ],
)
def test_event_ordering_score_pinned(system: List[str], tau_norm: float, f1: float) -> None:
    out = event_ordering_score(["a", "b", "c"], system)
    assert out["tau_norm"] == approx(tau_norm, abs=1e-12)
    assert out["f1"] == approx(f1)
    assert out["final_score"] == approx(tau_norm * f1)


def test_abilities_match_task_data_literal() -> None:
    assert set(get_args(TaskData.model_fields["ability"].annotation)) == set(ABILITIES)
    assert len(ABILITIES) == 10


# ── verify ────────────────────────────────────────────────────────────────


def test_verify_averages_rubric_scores(monkeypatch) -> None:
    server = _server(judge_scale="partial")
    scores = iter([1.0, 0.5])
    calls = _mock_judge(server, lambda _m: json.dumps({"score": next(scores)}), monkeypatch)
    resp = asyncio.run(server.verify(_request("Two columns", ability="multi_session_reasoning", rubric=["a", "b"])))
    assert resp.reward == approx(0.75)
    assert resp.rubric_scores == [1.0, 0.5]
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0.0
    assert "RESPONSE TO EVALUATE: Two columns" in calls[0]["input"][0]["content"]
    assert resp.ability == "multi_session_reasoning" and resp.chat_size == "100K"


def test_verify_strips_think_before_judging(monkeypatch) -> None:
    server = _server()
    calls = _mock_judge(server, _score_reply(1.0), monkeypatch)
    resp = asyncio.run(server.verify(_request("<think>secret March 29</think>March 29")))
    assert resp.reward == approx(1.0)
    assert "secret" not in calls[0]["input"][0]["content"]


def test_verify_unparseable_judge_reply_counts_zero(monkeypatch) -> None:
    server = _server()
    replies = iter(['{"score": 1.0}', "I refuse"])
    _mock_judge(server, lambda _m: next(replies), monkeypatch)
    resp = asyncio.run(server.verify(_request("x", rubric=["a", "b"])))
    assert resp.reward == approx(0.5)
    assert resp.n_judge_parse_failed == 1
    assert resp.rubric_scores == [1.0, None]


@pytest.mark.parametrize("text", ["", "   ", "<think>only reasoning</think>"])
def test_verify_empty_response_skips_judge(monkeypatch, text: str) -> None:
    server = _server()
    _mock_judge(server, _score_reply(1.0), monkeypatch)
    resp = asyncio.run(server.verify(_request(text)))
    assert resp.reward == 0.0 and resp.empty_response is True
    server.server_client.post.assert_not_called()


@pytest.mark.parametrize(
    ("ability", "rubric"),
    [("unknown_ability", ["a"]), ("summarization", []), ("summarization", "a string, not a list")],
)
def test_request_rejects_bad_metadata(ability: str, rubric: Any) -> None:
    with pytest.raises(ValidationError):
        _request("x", ability=ability, rubric=rubric)


def test_verify_judge_failure_raises_judge_error(monkeypatch) -> None:
    server = _server()
    _mock_judge(server, _score_reply(1.0), monkeypatch, statuses=[500])
    with pytest.raises(JudgeError, match="HTTP 500"):
        asyncio.run(server.verify(_request("March 29")))


def test_judge_failure_is_masked_by_judge_failsafe(monkeypatch) -> None:
    server = _server()
    _mock_judge(server, _score_reply(1.0), monkeypatch, statuses=[500])
    resp = asyncio.run(judge_failsafe(server.verify)(body=_request("March 29")))
    data = json.loads(resp.body)
    assert data["reward"] == 0.0 and data["mask_sample"] is True
    assert data["failure_kind"] == "judge_failed" and "HTTP 500" in data["failure_reason"]


def test_judge_failure_cancels_sibling_calls(monkeypatch) -> None:
    server = _server()
    finished: List[str] = []

    async def fake_post(**kwargs: Any) -> Any:
        content = kwargs["json"]["input"][0]["content"]
        resp = MagicMock()
        if "CRITERION (what to check): fail" in content:
            resp.status = 500
        else:
            await asyncio.sleep(2.0)
            finished.append(content)
            resp.status = 200
        resp.reply = '{"score": 1.0}'
        return resp

    async def fake_raise_for_status(resp: Any) -> None:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}")

    async def fake_get_response_json(resp: Any) -> Dict[str, Any]:
        return _make_response(resp.reply).model_dump()

    server.server_client.post = AsyncMock(side_effect=fake_post)
    monkeypatch.setattr(judge_module, "raise_for_status", fake_raise_for_status)
    monkeypatch.setattr(judge_module, "get_response_json", fake_get_response_json)
    start = time.monotonic()
    with pytest.raises(JudgeError):
        asyncio.run(server.verify(_request("x", ability="summarization", rubric=["slow 1", "fail", "slow 2"])))
    assert time.monotonic() - start < 1.0
    assert finished == []


def test_verify_invalid_judge_score_counts_zero(monkeypatch) -> None:
    server = _server()
    replies = iter(['{"score": 1.0}', '{"score": 5}'])
    _mock_judge(server, lambda _m: next(replies), monkeypatch)
    resp = asyncio.run(server.verify(_request("x", rubric=["a", "b"])))
    assert resp.rubric_scores == [1.0, None]
    assert resp.n_judge_parse_failed == 1 and resp.reward == approx(0.5)


def test_config_declares_stateless_reverify() -> None:
    assert BEAMResourcesServerConfig.REVERIFY_MODE == ReverifyMode.STATELESS


def _equivalence_judge(pairs: Dict[str, str]) -> Callable[[List[Dict[str, str]]], str]:
    """YES when the (reference, line) pair maps in ``pairs``; rubric prompts score 1.0."""

    def judge(messages: List[Dict[str, str]]) -> str:
        if messages[0]["content"] == EQUIVALENCE_SYSTEM_PROMPT:
            user = messages[1]["content"]
            first = user.split("First snippet: ", 1)[1].split(" \n", 1)[0]
            second = user.split("Second snippet: ", 1)[1].strip()
            return "YES" if pairs.get(second) == first else "NO"
        return '{"score": 1.0}'

    return judge


def test_verify_event_ordering_uses_tau_norm(monkeypatch) -> None:
    server = _server()
    rubric = ["Core functionality", "Transaction error handling", "Security and deployment"]
    pairs = {"1) security hardening": rubric[2], "2) core setup": rubric[0], "3) transaction errors": rubric[1]}
    calls = _mock_judge(server, _equivalence_judge(pairs), monkeypatch)
    answer = "1) security hardening\n\n2) core setup\n3) transaction errors"
    resp = asyncio.run(server.verify(_request(answer, ability="event_ordering", rubric=rubric)))

    assert resp.llm_judge_score == approx(1.0)
    # All three events found; the blank line is an unmatched system item (precision 3/4), as upstream.
    assert resp.event_ordering["recall"] == approx(1.0)
    assert resp.event_ordering["f1"] == approx(6 / 7)
    # System order (c, a, b) vs reference (a, b, c) plus one unmatched blank line (scipy-checked).
    assert resp.reward == approx(0.33333333333333337, abs=1e-12)
    # Out of order, so all_pass fails even though every rubric item passed.
    assert resp.score_all_pass == 0.0
    # The blank line never reaches the judge.
    eq_prompts = [c["input"][1]["content"] for c in calls if c["input"][0]["content"] == EQUIVALENCE_SYSTEM_PROMPT]
    assert all("Second snippet: \n" not in p for p in eq_prompts)


def test_verify_event_ordering_perfect(monkeypatch) -> None:
    server = _server()
    rubric = ["a event", "b event"]
    _mock_judge(server, _equivalence_judge({"first a": "a event", "then b": "b event"}), monkeypatch)
    resp = asyncio.run(server.verify(_request("first a\nthen b", ability="event_ordering", rubric=rubric)))
    assert resp.reward == approx(1.0)
    assert resp.score_all_pass == 1.0


def test_alignment_takes_first_yes_in_reference_order_and_stops_early(monkeypatch) -> None:
    server = _server()
    reference = [f"event {i}" for i in range(8)]
    eq_calls: List[str] = []

    def judge(messages: List[Dict[str, str]]) -> str:
        user = messages[1]["content"]
        first = user.split("First snippet: ", 1)[1].split(" \n", 1)[0]
        eq_calls.append(first)
        # The line matches events 1 and 2; event 1 comes first in reference order.
        return "YES" if first in ("event 1", "event 2") else "NO"

    _mock_judge(server, judge, monkeypatch)
    out = asyncio.run(server._align_with_llm(reference, ["the line"]))
    assert out == ["event 1"]
    # Only the first batch of 4 candidates is judged, not all 8.
    assert sorted(eq_calls) == [f"event {i}" for i in range(4)]

    # A line that matches nothing is checked against every candidate (two batches) and kept as is.
    eq_calls.clear()
    out = asyncio.run(server._align_with_llm(reference[3:], ["the line"]))
    assert out == ["the line"]
    assert sorted(eq_calls) == reference[3:]


# ── metrics ───────────────────────────────────────────────────────────────


def test_compute_metrics_macro_average_and_buckets() -> None:
    server = _server()
    rows = [
        {"reward": 1.0, "score_mean": 1.0, "score_all_pass": 1.0, "ability": "abstention", "chat_size": "100K"},
        {"reward": 0.0, "score_mean": 0.0, "score_all_pass": 0.0, "ability": "abstention", "chat_size": "1M"},
        {"reward": 0.5, "score_mean": 0.5, "score_all_pass": 0.0, "ability": "summarization", "chat_size": "100K"},
        {
            "reward": 0.5,
            "score_mean": 0.5,
            "score_all_pass": 0.0,
            "ability": "summarization",
            "chat_size": "100K",
            "empty_response": True,
        },
    ]
    m = server.compute_metrics([[r] for r in rows])
    assert m["count"] == 4 and m["n_empty_response"] == 1
    assert m["abstention/score"] == approx(0.5)
    assert m["summarization/score"] == approx(0.5)
    assert m["overall"] == approx(0.5)
    assert m["chat_size/100K/overall"] == approx((1.0 + 0.5) / 2)
    assert m["chat_size/1M/overall"] == approx(0.0)
    assert m["mean/overall"] == approx(0.5)
    assert m["all_pass/overall"] == approx((0.5 + 0.0) / 2)
    assert m["all_pass/summarization/score"] == approx(0.0)
    assert m["all_pass/chat_size/100K/overall"] == approx((1.0 + 0.0) / 2)
    key = server.get_key_metrics(m)
    assert set(key) == {
        "pass@1[avg-of-1]/overall",
        "pass@1[avg-of-1]/all_pass/overall",
        "pass@1/overall",
        "pass@1/all_pass/overall",
        "overall",
        "mean/overall",
        "all_pass/overall",
        "n_empty_response",
        "chat_size/100K/overall",
        "chat_size/1M/overall",
    }


def test_compute_metrics_empty() -> None:
    server = _server()
    assert server.compute_metrics([]) == {}
    assert server.compute_metrics([[{"ability": "abstention"}]]) == {}


# ── data ──────────────────────────────────────────────────────────────────


def test_example_jsonl_rows_are_well_formed() -> None:
    rows = [json.loads(line) for line in _EXAMPLE_JSONL.read_text().splitlines() if line.strip()]
    assert len(rows) == 5
    for row in rows:
        # Default harness: the chat as a single-turn transcript, then the bare question.
        (message,) = row["responses_create_params"]["input"]
        assert message["role"] == "user"
        assert message["content"].startswith("Below is the full history of my earlier conversation with you.")
        assert message["content"].endswith("answer my question:\n" + row["verifier_metadata"]["question"])
        assert "NOTE: Only provide the answer" not in message["content"]
        meta = row["verifier_metadata"]
        assert meta["ability"] in ABILITIES and meta["rubric"]
        assert row["agent_ref"]["name"] == "beam_simple_agent"


# ── binary judge / all-pass scoring ───────────────────────────────────────


def test_binary_prompt_has_no_partial_tier() -> None:
    prompt = build_rubric_prompt("must say X", "the answer", binary=True)
    assert "0.5" not in prompt and "Partial" not in prompt
    assert "SCORING SCALE (BINARY)" in prompt
    assert "RUBRIC CRITERION (what to check): must say X" in prompt
    assert "0.5" in build_rubric_prompt("must say X", "the answer")


@pytest.mark.parametrize(("overrides", "binary"), [({}, True), ({"judge_scale": "partial"}, False)])
def test_verify_judge_scale_selects_prompt(monkeypatch, overrides: Dict[str, Any], binary: bool) -> None:
    server = _server(**overrides)
    calls = _mock_judge(server, _score_reply(1.0), monkeypatch)
    asyncio.run(server.verify(_request("March 29")))
    prompt = calls[0]["input"][0]["content"]
    assert ("SCORING SCALE (BINARY)" in prompt) is binary
    assert ("0.5 (Partial Compliance)" in prompt) is not binary


@pytest.mark.parametrize(("question_scoring", "expected"), [("mean", 0.75), ("all_pass", 0.0)])
def test_verify_question_scoring_selects_reward(monkeypatch, question_scoring: str, expected: float) -> None:
    server = _server(question_scoring=question_scoring, judge_scale="partial")
    scores = iter([1.0, 0.5])
    _mock_judge(server, lambda _m: json.dumps({"score": next(scores)}), monkeypatch)
    resp = asyncio.run(server.verify(_request("x", ability="summarization", rubric=["a", "b"])))
    assert resp.score_mean == approx(0.75) and resp.score_all_pass == 0.0
    assert resp.reward == approx(expected)


@pytest.mark.parametrize(("answer", "all_pass"), [("first a\nthen b", 1.0), ("then b\nfirst a", 0.0)])
def test_verify_all_pass_event_ordering_requires_reference_order(monkeypatch, answer: str, all_pass: float) -> None:
    server = _server(question_scoring="all_pass")
    rubric = ["a event", "b event"]
    # Every rubric item passes either way; only the order differs.
    _mock_judge(server, _equivalence_judge({"then b": "b event", "first a": "a event"}), monkeypatch)
    resp = asyncio.run(server.verify(_request(answer, ability="event_ordering", rubric=rubric)))
    assert resp.llm_judge_score == approx(1.0)
    assert resp.score_all_pass == all_pass and resp.reward == all_pass


def test_compute_metrics_pass_at_1_avg_of_k() -> None:
    server = _server()

    def row(ability: str, reward: float, all_pass: float, idx: int) -> Dict[str, Any]:
        return {
            "reward": reward,
            "score_mean": reward,
            "score_all_pass": all_pass,
            "ability": ability,
            "chat_size": "100K",
            "_ng_rollout_index": idx,
        }

    # Rollouts listed out of order: avg-of-1 must use rollout index 0.
    tasks = [
        [row("abstention", 0.0, 0.0, 1), row("abstention", 1.0, 1.0, 0)],
        [row("summarization", 0.5, 0.0, 0), row("summarization", 1.0, 1.0, 1)],
    ]
    m = server.compute_metrics(tasks)
    assert m["pass@1[avg-of-1]/overall"] == approx((1.0 + 0.5) / 2)
    assert m["pass@1[avg-of-2]/overall"] == approx((0.5 + 0.75) / 2)
    assert m["pass@1[avg-of-2]/overall"] == approx(m["overall"])
    assert m["pass@1[avg-of-2]/all_pass/overall"] == approx((0.5 + 0.5) / 2)
    assert m["pass@2/overall"] == approx((1.0 + 1.0) / 2)
    # Shared pass@k estimator: unbiased over all rollouts for 0/1 rewards (abstention: 1 of 2 -> 0.5),
    # best of the first k for continuous ones (summarization: first rollout 0.5).
    assert m["pass@1/overall"] == approx((0.5 + 0.5) / 2)
    # Run 0 overall = 0.75, run 1 overall = 0.5.
    assert m["pass@1[avg-of-2]/overall/std_dev_across_runs"] == approx(0.1767767, abs=1e-6)
    key = server.get_key_metrics(m)
    assert "pass@1[avg-of-2]/overall" in key and "pass@2/overall" in key
    assert "pass@1[avg-of-1]/overall" not in key


def test_compute_metrics_runs_keyed_by_rollout_index() -> None:
    server = _server()

    def row(reward: float, idx: int) -> Dict[str, Any]:
        return {
            "reward": reward,
            "score_mean": reward,
            "score_all_pass": reward,
            "ability": "abstention",
            "chat_size": "100K",
            "_ng_rollout_index": idx,
        }

    # Task B lost rollout 0 (e.g. a masked judge failure): run 0 = [A=1.0], run 1 = [A=0.0, B=1.0].
    tasks = [[row(1.0, 0), row(0.0, 1)], [row(1.0, 1)]]
    m = server.compute_metrics(tasks)
    assert m["pass@1[avg-of-1]/overall"] == approx(1.0)
    assert m["pass@1[avg-of-2]/overall"] == approx((0.5 + 1.0) / 2)
    assert m["pass@2/overall"] == approx(1.0)
    assert m["pass@1[avg-of-2]/overall/std_dev_across_runs"] == approx(0.35355339, abs=1e-6)
