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
"""Acceptance tests for the LongMemEval resources server.

These exercise the user story's acceptance criteria end-to-end with the judge
model mocked. Parity criteria (prompt, rubrics) are checked against independent
reference implementations transcribed below from the upstream LongMemEval
project (``src/generation/run_generation.py`` and
``src/evaluation/evaluate_qa.py``), so a drift in ``app.py`` or
``prepare_longmemeval.py`` fails a test rather than silently changing scores.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from aiohttp import ClientResponseError, RequestInfo
from multidict import CIMultiDict
from pydantic import ValidationError
from pytest import approx
from yarl import URL

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
    LongMemEvalResourcesServer,
    LongMemEvalResourcesServerConfig,
    LongMemEvalVerifyRequest,
)
from resources_servers.longmemeval.prepare_longmemeval import _SPLIT_FILES, build_row


_SERVER_DIR = Path(__file__).resolve().parents[1]
_EXAMPLE_JSONL = _SERVER_DIR / "data" / "example.jsonl"

_CONFIG_PATHS = (
    _SERVER_DIR / "configs" / "longmemeval.yaml",
    _SERVER_DIR / "configs" / "longmemeval_serve.yaml",
)

# The Responses API rejects ``max_output_tokens`` below this floor, which is why
# the judge budget deviates from upstream evaluate_qa.py's ``max_tokens=10``.
_RESPONSES_API_MIN_OUTPUT_TOKENS = 16
_JUDGE_MAX_OUTPUT_TOKENS = 64

_QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
)


# ── reference implementations (transcribed from upstream LongMemEval) ─────
#
# Deliberately written out here instead of imported from app.py /
# prepare_longmemeval.py: they are the independent expectation these tests
# compare the server against.

_REFERENCE_ANSWER_PROMPT = "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"  # noqa: E501

_REFERENCE_RUBRIC_CONTAIN = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."  # noqa: E501

_REFERENCE_RUBRIC_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."  # noqa: E501

_REFERENCE_RUBRIC_KNOWLEDGE_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."  # noqa: E501

_REFERENCE_RUBRIC_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."  # noqa: E501

_REFERENCE_RUBRIC_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."  # noqa: E501

_REFERENCE_RUBRICS: Dict[str, str] = {
    "single-session-user": _REFERENCE_RUBRIC_CONTAIN,
    "single-session-assistant": _REFERENCE_RUBRIC_CONTAIN,
    "multi-session": _REFERENCE_RUBRIC_CONTAIN,
    "temporal-reasoning": _REFERENCE_RUBRIC_TEMPORAL,
    "knowledge-update": _REFERENCE_RUBRIC_KNOWLEDGE_UPDATE,
    "single-session-preference": _REFERENCE_RUBRIC_PREFERENCE,
}


def reference_prompt(entry: Dict[str, Any], topk: int = 50) -> str:
    """Upstream ``prepare_prompt`` for orig-session / JSON history / no CoT."""
    pairs = list(zip(entry["haystack_dates"], entry["haystack_sessions"]))
    if topk > 0:
        pairs = pairs[-topk:]
    pairs.sort(key=lambda pair: pair[0])

    history = ""
    for index, (date, session) in enumerate(pairs, start=1):
        turns = [{k: v for k, v in turn.items() if k != "has_answer"} for turn in session]
        history += f"\n### Session {index}:\nSession Date: {date}\nSession Content:\n\n{json.dumps(turns)}\n"
    return _REFERENCE_ANSWER_PROMPT.format(history, entry["question_date"], entry["question"])


def reference_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """Upstream ``get_anscheck_prompt``: pick the rubric, fill it positionally."""
    template = _REFERENCE_RUBRIC_ABSTENTION if abstention else _REFERENCE_RUBRICS[task]
    return template.format(question, answer, response)


# ── server / request helpers ──────────────────────────────────────────────


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


def _make_server(max_concurrency: Optional[int] = 32, max_retries: int = 5) -> LongMemEvalResourcesServer:
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
        judge_retry_base_delay=0.0,
    )
    return LongMemEvalResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def server() -> LongMemEvalResourcesServer:
    return _make_server()


def _make_request(text: str, metadata: Optional[Any] = ...) -> LongMemEvalVerifyRequest:
    if metadata is ...:
        metadata = {
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "Where did the user travel?",
            "answer": "Paris",
            "abstention": False,
        }
    return LongMemEvalVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        verifier_metadata=metadata,
    )


def _mock_judge(server: LongMemEvalResourcesServer, reply: str, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Make the judge model server answer ``reply``; return the captured POST kwargs."""
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


def _judge_prompt(captured: Dict[str, Any]) -> str:
    return str(captured["json"]["input"][0]["content"])


def _example_rows() -> List[Dict[str, Any]]:
    lines = _EXAMPLE_JSONL.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# Deterministic upstream-shaped question entry used for prompt-parity checks.
_ENTRY: Dict[str, Any] = {
    "question_id": "gpt4_deadbeef",
    "question_type": "multi-session",
    "question": "How many cities did I visit?",
    "answer": "3",
    "question_date": "2023/06/01 (Thu) 09:15",
    "haystack_dates": [f"2023/{month:02d}/0{day}" for month in range(1, 8) for day in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
    "haystack_sessions": [
        [
            {"role": "user", "content": f"session {idx} user turn", "has_answer": idx % 3 == 0},
            {"role": "assistant", "content": f"session {idx} assistant turn"},
        ]
        for idx in range(63)
    ],
}


# ── AC1: layout ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "relative_path",
    ["app.py", "configs/longmemeval.yaml", "data/example.jsonl", "tests/test_app.py"],
)
def test_ac_layout_required_files_exist(relative_path: str) -> None:
    path = _SERVER_DIR / relative_path
    assert path.is_file(), f"missing {path}"
    assert path.stat().st_size > 0


# ── AC2: row shape ────────────────────────────────────────────────────────


def test_ac_row_shape_only_two_top_level_keys() -> None:
    rows = _example_rows()
    assert rows, "example.jsonl is empty"
    for row in rows:
        assert set(row) == {"responses_create_params", "verifier_metadata"}


def test_ac_row_shape_verifier_metadata_is_scalar_only() -> None:
    for row in _example_rows():
        metadata = row["verifier_metadata"]
        assert isinstance(metadata, dict)
        for key, value in metadata.items():
            assert isinstance(value, (str, int, float, bool)), f"{key} is non-scalar: {type(value)}"
        assert {"question_id", "question_type", "question", "answer", "abstention"} <= set(metadata)


def test_ac_row_shape_question_answer_live_in_metadata_not_top_level() -> None:
    for row in _example_rows():
        assert "question" not in row and "answer" not in row
        assert "haystack_sessions" not in row and "haystack_dates" not in row
        assert isinstance(row["responses_create_params"]["input"][0]["content"], str)


def test_ac_row_shape_build_row_matches_example_rows() -> None:
    row = build_row(_ENTRY, split="oracle", topk=50)
    assert set(row) == {"responses_create_params", "verifier_metadata"}
    assert set(row["verifier_metadata"]) == set(_example_rows()[0]["verifier_metadata"])


# ── AC3: prompt parity with the upstream prompt construction ──────────────


def test_ac_prompt_parity_byte_identical_to_upstream() -> None:
    gym_prompt = build_row(_ENTRY, topk=50)["responses_create_params"]["input"][0]["content"]
    assert gym_prompt == reference_prompt(_ENTRY, topk=50)


def test_ac_prompt_parity_topk_slice_and_date_sort() -> None:
    gym_prompt = build_row(_ENTRY, topk=5)["responses_create_params"]["input"][0]["content"]
    assert gym_prompt == reference_prompt(_ENTRY, topk=5)
    assert gym_prompt.count("### Session ") == 5


def test_ac_prompt_parity_drops_has_answer_and_omits_cot() -> None:
    prompt = build_row(_ENTRY, topk=50)["responses_create_params"]["input"][0]["content"]
    assert "has_answer" not in prompt
    assert prompt == reference_prompt(_ENTRY, topk=50)
    assert prompt.endswith("Answer:")
    assert "step by step" not in prompt.lower()


# ── AC4: rubric selection ─────────────────────────────────────────────────


@pytest.mark.parametrize("question_type", _QUESTION_TYPES)
def test_ac_rubric_matches_upstream_per_question_type(
    server: LongMemEvalResourcesServer,
    monkeypatch: pytest.MonkeyPatch,
    question_type: str,
) -> None:
    captured = _mock_judge(server, "yes", monkeypatch)
    metadata = {
        "question_id": "q1",
        "question_type": question_type,
        "question": "Where did the user travel?",
        "answer": "Paris",
        "abstention": False,
    }
    _verify(server, _make_request("Paris", metadata))
    expected = reference_anscheck_prompt(question_type, "Where did the user travel?", "Paris", "Paris")
    assert _judge_prompt(captured) == expected


def test_ac_rubric_abs_question_id_overrides_question_type(
    server: LongMemEvalResourcesServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _mock_judge(server, "yes", monkeypatch)
    metadata = {
        "question_id": "gpt4_1234_abs",
        "question_type": "knowledge-update",
        "question": "Which came first?",
        "answer": "Not enough information.",
        "abstention": False,
    }
    result = _verify(server, _make_request("I cannot tell", metadata))
    expected = reference_anscheck_prompt(
        "knowledge-update", "Which came first?", "Not enough information.", "I cannot tell", abstention=True
    )
    assert _judge_prompt(captured) == expected
    assert result.abstention is True


def test_ac_rubric_unknown_question_type_scores_zero(
    server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_judge(server, "yes", monkeypatch)
    metadata = {"question_id": "q1", "question_type": "not-a-real-type", "question": "q", "answer": "a"}
    result = _verify(server, _make_request("Paris", metadata))
    assert result.reward == approx(0.0)
    server.server_client.post.assert_not_called()


# ── AC5: judge call contract ──────────────────────────────────────────────


def test_ac_judge_call_contract_endpoint_and_sampling(
    server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _mock_judge(server, "yes", monkeypatch)
    _verify(server, _make_request("Paris"))
    assert server.server_client.post.await_count == 1
    assert captured["url_path"] == "/v1/responses"
    assert captured["server_name"] == "judge_model"
    payload = captured["json"]
    assert payload["temperature"] == approx(0.0)
    assert payload["max_output_tokens"] == 64
    assert payload.get("n") in (None, 1)
    assert [m["role"] for m in payload["input"]] == ["user"]


@pytest.mark.parametrize(
    ("reply", "expected_reward"),
    [
        ("yes", 1.0),
        ("Yes", 1.0),
        ("YES", 1.0),
        ("yes.", 1.0),
        ("The answer is yes, correct.", 1.0),
        ("no", 0.0),
        ("No.", 0.0),
        ("maybe", 0.0),
    ],
)
def test_ac_judge_call_contract_yes_maps_to_reward(
    server: LongMemEvalResourcesServer,
    monkeypatch: pytest.MonkeyPatch,
    reply: str,
    expected_reward: float,
) -> None:
    _mock_judge(server, reply, monkeypatch)
    result = _verify(server, _make_request("Paris"))
    assert result.reward == approx(expected_reward)
    assert result.judge_label is (expected_reward == 1.0)
    assert result.judge_error is None


@pytest.mark.parametrize("reply", ["", "   ", "<think>reasoning only</think>"])
def test_ac_empty_judge_reply_is_reported_as_an_error(
    server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    """A blank judge verdict must surface as judge_error, not a silent 0.0."""
    _mock_judge(server, reply, monkeypatch)
    result = _verify(server, _make_request("Paris"))
    assert result.reward == approx(0.0)
    assert result.judge_label is False
    assert result.judge_error == "empty_judge_output"


# ── AC6: binary reward and graceful empty output ──────────────────────────


@pytest.mark.parametrize("reply", ["yes", "no", "YES and no", "unsure", "", "   ", "yes\nno"])
def test_ac_reward_is_binary(server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    _mock_judge(server, reply, monkeypatch)
    reward = _verify(server, _make_request("Paris")).reward
    assert reward in (0.0, 1.0)


@pytest.mark.parametrize("generation", ["", "   ", "\n\t ", "<think>only reasoning</think>", "<think>truncated"])
def test_ac_empty_generation_scores_zero_without_judge(
    server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch, generation: str
) -> None:
    _mock_judge(server, "yes", monkeypatch)
    result = _verify(server, _make_request(generation))
    assert result.reward == approx(0.0)
    server.server_client.post.assert_not_called()


def test_ac_judge_failure_scores_zero_without_raising(server: LongMemEvalResourcesServer) -> None:
    server.server_client.post = AsyncMock(side_effect=RuntimeError("judge unreachable"))
    result = _verify(server, _make_request("Paris"))
    assert result.reward == approx(0.0)
    assert result.judge_error is not None


# ── AC7: think stripping ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("generation", "visible"),
    [
        ("<think>secret plan</think>Paris", "Paris"),
        ("<think>a</think>Paris<think>b</think> and Rome", "Paris and Rome"),
        ("secret plan</think>Paris", "Paris"),
    ],
)
def test_ac_think_blocks_never_reach_judge(
    server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch, generation: str, visible: str
) -> None:
    captured = _mock_judge(server, "yes", monkeypatch)
    result = _verify(server, _make_request(generation))
    prompt = _judge_prompt(captured)
    assert "<think>" not in prompt and "</think>" not in prompt
    assert "secret" not in prompt and "plan" not in prompt
    assert prompt.endswith("Is the model response correct? Answer yes or no only.")
    assert f"Model Response: {visible}" in prompt
    assert result.generation == visible


# ── AC8: bounded concurrency, no direct HTTP ──────────────────────────────


def test_ac_judge_concurrency_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(max_concurrency=1)
    state = {"active": 0, "max_active": 0}

    async def fake_post(**_kwargs: Any) -> Any:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return MagicMock()

    async def fake_get_response_json(_resp: Any) -> Dict[str, Any]:
        return _make_response("yes").model_dump()

    server.server_client.post = AsyncMock(side_effect=fake_post)
    monkeypatch.setattr(lme_app, "get_response_json", fake_get_response_json)

    async def run_concurrently() -> List[Any]:
        return await asyncio.gather(*(server.verify(_make_request("Paris")) for _ in range(6)))

    results = asyncio.run(run_concurrently())
    assert state["max_active"] == 1
    assert all(r.reward == approx(1.0) for r in results)


def test_ac_module_makes_no_direct_http_calls() -> None:
    source = (_SERVER_DIR / "app.py").read_text(encoding="utf-8")
    for forbidden in ("httpx", "requests", "urllib.request", "aiohttp.ClientSession"):
        assert forbidden not in source, f"app.py references {forbidden}"
    assert not isinstance(getattr(lme_app, "httpx", None), ModuleType)
    assert not isinstance(getattr(lme_app, "requests", None), ModuleType)


# ── AC9: no env-var reads in app.py ───────────────────────────────────────


def test_ac_app_reads_no_environment_variables() -> None:
    source = (_SERVER_DIR / "app.py").read_text(encoding="utf-8")
    for forbidden in ("os.environ", "os.getenv", "getenv("):
        assert forbidden not in source, f"app.py reads configuration from {forbidden}"


# ── AC10: metrics ─────────────────────────────────────────────────────────


def _metric_row(reward: float, question_type: str, abstention: bool = False, **extra: Any) -> Dict[str, Any]:
    return {"reward": reward, "question_type": question_type, "abstention": abstention, **extra}


def test_ac_metrics_overall_per_type_and_abstention(server: LongMemEvalResourcesServer) -> None:
    tasks = [
        [_metric_row(1.0, "multi-session"), _metric_row(0.0, "multi-session")],
        [_metric_row(1.0, "temporal-reasoning"), _metric_row(1.0, "temporal-reasoning", abstention=True)],
    ]
    metrics = server.compute_metrics(tasks)
    assert metrics["accuracy"] == approx(0.75)
    assert metrics["count"] == 4
    assert metrics["question_type/multi-session/accuracy"] == approx(0.5)
    assert metrics["question_type/temporal-reasoning/accuracy"] == approx(1.0)
    assert metrics["abstention/accuracy"] == approx(1.0)
    assert metrics["abstention/count"] == 1


def test_ac_metrics_exclude_judge_failures_like_upstream(server: LongMemEvalResourcesServer) -> None:
    """Upstream evaluate_qa.py skips ``entry is None`` rows; gym must not count them as 0.0."""
    tasks = [
        [_metric_row(1.0, "multi-session"), _metric_row(1.0, "multi-session")],
        [_metric_row(0.0, "multi-session", judge_error="judge_call_failed")],
    ]
    metrics = server.compute_metrics(tasks)
    assert metrics["accuracy"] == approx(1.0)
    assert metrics["count"] == 2
    assert metrics["question_type/multi-session/count"] == 2
    assert metrics["n_judge_call_failed"] == 1
    assert metrics["n_excluded"] == 1
    assert metrics["accuracy_strict"] == approx(2 / 3)


def test_ac_key_metrics_surface_headline_keys(server: LongMemEvalResourcesServer) -> None:
    metrics = server.compute_metrics([[_metric_row(1.0, "multi-session", abstention=True)]])
    assert server.get_key_metrics(metrics) == {
        "accuracy": approx(1.0),
        "abstention/accuracy": approx(1.0),
        "n_excluded": 0,
    }


# ── AC11: malformed rows ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({"question_id": "q1", "question": "q", "answer": "a"}, id="missing-question-type"),
        pytest.param(None, id="metadata-none"),
        pytest.param({}, id="empty-metadata"),
    ],
)
def test_ac_malformed_metadata_scores_zero(
    server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch, metadata: Any
) -> None:
    _mock_judge(server, "yes", monkeypatch)
    result = _verify(server, _make_request("Paris", metadata))
    assert result.reward == approx(0.0)
    server.server_client.post.assert_not_called()


def test_ac_malformed_metadata_not_a_dict_scores_zero(
    server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-dict payload is rejected by request validation; verify() still degrades to 0.0."""
    with pytest.raises(ValidationError):
        _make_request("Paris", ["not", "a", "dict"])

    _mock_judge(server, "yes", monkeypatch)
    body = _make_request("Paris")
    body.verifier_metadata = ["not", "a", "dict"]  # type: ignore[assignment]
    result = _verify(server, body)
    assert result.reward == approx(0.0)
    server.server_client.post.assert_not_called()


def test_ac_malformed_missing_answer_delegates_to_judge_without_error(
    server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing gold answer renders an empty rubric slot; scoring stays binary and error-free."""
    captured = _mock_judge(server, "no", monkeypatch)
    metadata = {"question_id": "q1", "question_type": "multi-session", "question": "q"}
    result = _verify(server, _make_request("Paris", metadata))
    assert "Correct Answer: \n" in _judge_prompt(captured)
    assert result.reward == approx(0.0)
    assert result.judge_error is None


# ── Edge cases from the story ─────────────────────────────────────────────


def test_edge_every_example_row_scores_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server()
    captured = _mock_judge(server, "yes", monkeypatch)
    for row in _example_rows():
        result = _verify(server, _make_request("some answer", row["verifier_metadata"]))
        assert result.reward == approx(1.0)
        assert result.question_type == row["verifier_metadata"]["question_type"]
        expected_abstention = "_abs" in row["verifier_metadata"]["question_id"]
        assert result.abstention is expected_abstention
        prompt = _judge_prompt(captured)
        assert ("unanswerable" in prompt) is expected_abstention


def test_edge_metadata_abstention_flag_alone_selects_abstention_rubric(
    server: LongMemEvalResourcesServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _mock_judge(server, "yes", monkeypatch)
    metadata = {
        "question_id": "no-suffix",
        "question_type": "multi-session",
        "question": "Which came first?",
        "answer": "Not enough information.",
        "abstention": True,
    }
    _verify(server, _make_request("I cannot tell", metadata))
    expected = reference_anscheck_prompt(
        "multi-session", "Which came first?", "Not enough information.", "I cannot tell", abstention=True
    )
    assert _judge_prompt(captured) == expected


def test_edge_no_concurrency_cap_still_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _make_server(max_concurrency=None)
    _mock_judge(server, "yes", monkeypatch)
    assert _verify(server, _make_request("Paris")).reward == approx(1.0)


# ── Remediated behaviours ─────────────────────────────────────────────────


def _judge_params(config_path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return config["longmemeval"]["resources_servers"]["longmemeval"]["judge_responses_create_params"]


@pytest.mark.parametrize("config_path", _CONFIG_PATHS, ids=lambda p: p.name)
def test_ac_shipped_judge_params_clear_the_responses_api_floor(config_path: Path) -> None:
    """Both YAMLs must ask for 64 output tokens — above the API floor of 16 — at temperature 0."""
    params = _judge_params(config_path)
    assert params["max_output_tokens"] == _JUDGE_MAX_OUTPUT_TOKENS
    assert params["max_output_tokens"] >= _RESPONSES_API_MIN_OUTPUT_TOKENS
    assert params["temperature"] == approx(0.0)
    assert "n" not in params


def test_ac_judge_request_budget_clears_the_responses_api_floor(
    server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget actually sent to the judge, not just the YAML, respects the floor."""
    captured = _mock_judge(server, "yes", monkeypatch)
    _verify(server, _make_request("Paris"))
    sent = captured["json"]["max_output_tokens"]
    assert sent == _JUDGE_MAX_OUTPUT_TOKENS
    assert sent >= _RESPONSES_API_MIN_OUTPUT_TOKENS


class _FailingJudgeResponse:
    """Non-2xx aiohttp ``ClientResponse`` stand-in for the judge endpoint.

    ``read()`` returns a well-formed "yes" verdict on purpose: if ``verify()``
    ever stopped calling ``raise_for_status`` the row would silently score 1.0,
    so this stub makes the regression visible rather than invisible.
    """

    ok = False
    status = 429

    def __init__(self, error_body: bytes) -> None:
        self.content = _FailingJudgeContent(error_body)
        self.request_info = RequestInfo(
            url=URL("http://judge/v1/responses"),
            method="POST",
            headers=CIMultiDict(),
            real_url=URL("http://judge/v1/responses"),
        )

    async def read(self) -> bytes:
        return json.dumps(_make_response("yes").model_dump()).encode("utf-8")

    def raise_for_status(self) -> None:
        raise ClientResponseError(
            request_info=self.request_info, history=(), status=self.status, message="Too Many Requests"
        )


class _FailingJudgeContent:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def read(self) -> bytes:
        return self._body


def test_ac_non_2xx_judge_response_is_surfaced_as_judge_error(server: LongMemEvalResourcesServer) -> None:
    """A persistent judge HTTP error is retried, then reported, and scores 0.0 without raising."""
    server.server_client.post = AsyncMock(return_value=_FailingJudgeResponse(b'{"error": "rate limit"}'))

    result = _verify(server, _make_request("Paris"))

    assert result.reward == approx(0.0)
    assert result.judge_label is False
    assert result.judge_error == "judge_call_failed"
    assert result.judge_error_detail.startswith("ClientResponseError:")
    assert "429" in result.judge_error_detail
    # Rate limits are retried before the row is dropped from the denominator.
    assert server.server_client.post.await_count == server.config.judge_max_retries + 1


def test_ac_metrics_denominators_split_judged_and_all_rows(server: LongMemEvalResourcesServer) -> None:
    """accuracy drops only upstream-skipped rows; accuracy_strict covers every rewarded row."""
    tasks = [
        [
            _metric_row(1.0, "multi-session"),
            _metric_row(0.0, "multi-session"),
            _metric_row(1.0, "temporal-reasoning", abstention=True),
        ],
        [
            _metric_row(0.0, "multi-session", judge_error="judge_call_failed"),
            _metric_row(0.0, "knowledge-update", judge_error="empty_judge_output"),
        ],
    ]
    metrics = server.compute_metrics(tasks)

    # The empty judge verdict stays in, exactly as upstream's ``"yes" in ""`` does.
    assert metrics["accuracy"] == approx(0.5)
    assert metrics["count"] == 4
    assert metrics["accuracy_strict"] == approx(2 / 5)
    assert metrics["n_judge_call_failed"] == 1
    assert metrics["n_empty_judge_output"] == 1
    assert metrics["n_excluded"] == 1
    assert metrics["question_type/multi-session/count"] == 2
    assert metrics["question_type/knowledge-update/count"] == 1
    assert metrics["abstention/count"] == 1


def test_ac_metrics_all_judge_errors_report_no_misleading_accuracy(server: LongMemEvalResourcesServer) -> None:
    tasks = [
        [_metric_row(0.0, "multi-session", judge_error="judge_call_failed")],
        [_metric_row(1.0, "temporal-reasoning", judge_error="unknown_question_type")],
    ]
    metrics = server.compute_metrics(tasks)

    assert "accuracy" not in metrics
    assert "count" not in metrics
    assert not any(key.startswith("question_type/") for key in metrics)
    assert metrics["n_judge_call_failed"] == 1
    assert metrics["n_unknown_question_type"] == 1
    assert metrics["n_excluded"] == 2
    assert metrics["accuracy_strict"] == approx(0.5)
    assert server.get_key_metrics(metrics) == {"n_excluded": 2}


def test_ac_prepare_split_s_resolves_released_file_and_own_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--split s`` reads longmemeval_s_cleaned.json and writes data/s.jsonl, not data/oracle.jsonl."""
    assert _SPLIT_FILES["s"] == "longmemeval_s_cleaned.json"
    data_dir = tmp_path / "upstream"
    data_dir.mkdir()
    (data_dir / "longmemeval_s_cleaned.json").write_text(json.dumps([_ENTRY]), encoding="utf-8")
    out_dir = tmp_path / "data"
    out_dir.mkdir()
    stale_oracle = out_dir / "oracle.jsonl"
    stale_oracle.write_text("PRE-EXISTING\n", encoding="utf-8")

    monkeypatch.setattr(prepare_lme, "_default_cache_dir", lambda: data_dir)
    monkeypatch.setattr(prepare_lme, "_OUT_DIR", out_dir)
    monkeypatch.setattr(sys, "argv", ["prepare", "--split", "s"])
    prepare_lme.main()

    written = [json.loads(line) for line in (out_dir / "s.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["verifier_metadata"]["split"] for row in written] == ["s"]
    assert stale_oracle.read_text(encoding="utf-8") == "PRE-EXISTING\n"


def test_ac_prepare_split_choices_cover_released_splits() -> None:
    assert _SPLIT_FILES == {
        "oracle": "longmemeval_oracle.json",
        "s": "longmemeval_s_cleaned.json",
        "m": "longmemeval_m_cleaned.json",
    }


def test_ac_example_dataset_covers_preference_and_abstention() -> None:
    metadata = [row["verifier_metadata"] for row in _example_rows()]
    question_types = {meta["question_type"] for meta in metadata}
    assert "single-session-preference" in question_types
    assert len(question_types) >= 5
    assert sum(1 for meta in metadata if "_abs" in meta["question_id"]) >= 1
    assert sum(1 for meta in metadata if meta["abstention"]) >= 1


def test_ac_empty_generation_is_flagged_but_never_excluded(
    server: LongMemEvalResourcesServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The judge-free 0.0 shortcut is distinguishable from a "no" verdict, yet still scored.

    Excluding empty generations would let a model that emits nothing on hard rows
    outscore one that guesses, so the row must stay in every denominator.
    """
    _mock_judge(server, "yes", monkeypatch)
    empty = _verify(server, _make_request("   "))
    answered = _verify(server, _make_request("Paris"))

    assert empty.reward == approx(0.0)
    assert empty.empty_response is True
    assert empty.judge_error is None
    assert server.server_client.post.await_count == 1

    metrics = server.compute_metrics([[empty.model_dump(), answered.model_dump()]])
    assert metrics["accuracy"] == approx(0.5)
    assert metrics["count"] == 2
    assert metrics["n_empty_response"] == 1
    assert metrics["n_excluded"] == 0
