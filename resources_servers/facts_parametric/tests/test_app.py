# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import exercise_verifier_fixture
from resources_servers.facts_parametric.app import (
    GRADER_TEMPLATE_PATH,
    GRADER_TEMPLATE_SHA256,
    LABELS,
    VERIFIER_FIXTURE,
    FACTSParametricConfig,
    FACTSParametricResourcesServer,
    FACTSParametricVerifyRequest,
    build_grader_prompt,
    extract_text_from_response,
    load_grader_template,
    output_line_classification,
    starter_classification,
)


def _policy_response(text: str, *, incomplete: bool = False, with_message: bool = True) -> NeMoGymResponse:
    output = []
    if with_message:
        output.append(
            {
                "type": "message",
                "id": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="policy",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        incomplete_details={"reason": "max_output_tokens"} if incomplete else None,
    )


def _chat_completion(content: str | None, *, model: str = "gemini-2.5-pro") -> dict:
    return {
        "id": f"chatcmpl-{hashlib.sha256((content or '').encode()).hexdigest()[:8]}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }


class _FakeHTTPResponse:
    ok = True
    status = 200

    def __init__(self, payload: dict):
        self.payload = payload

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _request(text: str, *, incomplete: bool = False, **fields) -> FACTSParametricVerifyRequest:
    return FACTSParametricVerifyRequest(
        id="facts_parametric_public_0001",
        question="capital of france",
        expected_answer="Paris",
        topic="other",
        responses_create_params={"input": [{"role": "user", "content": "capital of france"}]},
        response=_policy_response(text, incomplete=incomplete),
        **fields,
    )


def _config(**overrides) -> FACTSParametricConfig:
    fields = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="facts_parametric",
        judge_model_server={"type": "responses_api_models", "name": "facts_parametric_judge"},
    )
    fields.update(overrides)
    return FACTSParametricConfig(**fields)


def _server(judge_texts, **overrides) -> tuple[FACTSParametricResourcesServer, AsyncMock]:
    """Server whose judge returns ``judge_texts`` in call order (a callable receives the prompt)."""
    client = MagicMock(spec=ServerClient)
    calls = iter(judge_texts) if not callable(judge_texts) else None

    async def _post(server_name, url_path, json=None, **kwargs):
        assert server_name == "facts_parametric_judge"
        assert url_path == "/v1/chat/completions"
        prompt = json.messages[0]["content"]
        text = judge_texts(prompt) if callable(judge_texts) else next(calls)
        return _FakeHTTPResponse(_chat_completion(text))

    client.post = AsyncMock(side_effect=_post)
    server = FACTSParametricResourcesServer(config=_config(**overrides), server_client=client)
    return server, client.post


@pytest.mark.parametrize(
    "text, starter, output_line",
    [
        ("Reasoning...\n```\nOutput: [CORRECT]\n```", "correct", "correct"),
        ("Output: MISTAKE", "incorrect", "incorrect"),
        ("Output: [NOT_ATTEMPTED]", "not_attempted", "not_attempted"),
        ("Output: [UNKNOWN]", "unknown", "unknown"),
        ("", "unknown", "unparsed"),
        ("no label at all", "unknown", "unparsed"),
        # The starter's substring rule fires on any mention: "INCORRECT" beats a closing CORRECT line.
        ("The claim is not INCORRECT.\nOutput: [CORRECT]", "incorrect", "correct"),
        ("The gold says Paris; NOT_ATTEMPTED does not apply.\nOutput: [CORRECT]", "correct", "correct"),
        ("output: correct", "unknown", "correct"),
    ],
)
def test_label_parsers(text, starter, output_line):
    assert starter_classification(text) == starter
    assert output_line_classification(text) == output_line


def test_grader_template_is_the_pinned_starter_prompt(monkeypatch):
    template = load_grader_template()
    assert hashlib.sha256(template.encode("utf-8")).hexdigest() == GRADER_TEMPLATE_SHA256
    assert "{question}" in template and "{gold_answer}" in template and "{prediction}" in template
    assert template.endswith("Predicted answer: {prediction}\n```")
    monkeypatch.setattr("resources_servers.facts_parametric.app.GRADER_TEMPLATE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="drifted"):
        load_grader_template()


def test_prompt_construction_fills_all_three_placeholders():
    template = GRADER_TEMPLATE_PATH.read_text(encoding="utf-8")
    prompt = build_grader_prompt(template, question="q?", expected_answer="gold", prediction="pred")
    assert prompt.endswith("Question: q?\nGold answer: gold\nPredicted answer: pred\n```")
    assert "{" not in prompt.replace("{question}", "")[:0]


def test_extract_text_prefers_last_assistant_message_and_handles_empty_output():
    assert extract_text_from_response(_policy_response("  Paris  ")) == "Paris"
    assert extract_text_from_response(_policy_response("", with_message=False)) == ""


async def test_three_correct_grades_give_full_reward_and_receipts():
    server, post = _server(["Output: [CORRECT]"] * 3)
    result = await server.verify(_request("Paris"))
    assert result.reward == 1.0 and result.all_correct == 1.0
    assert result.judge_labels == ["correct"] * 3
    assert result.is_correct == 1.0 and result.is_incorrect == 0.0
    assert result.judge_parse_agreement == 1.0 and result.judge_empty_samples == 0
    assert [receipt["seed"] for receipt in result.judge_receipts] == [0, 1, 2]
    assert all(receipt["judge_model"] == "gemini-2.5-pro" for receipt in result.judge_receipts)
    assert post.await_count == 3
    sent = [call.kwargs["json"] for call in post.await_args_list]
    assert [params.seed for params in sent] == [0, 1, 2]
    assert sent[0].temperature is None and sent[0].max_tokens is None  # provider defaults, like the starter
    expected_prompt = build_grader_prompt(
        load_grader_template(), question="capital of france", expected_answer="Paris", prediction="Paris"
    )
    assert sent[0].messages[0]["content"] == expected_prompt
    assert result.judge_prompt_sha256 == hashlib.sha256(expected_prompt.encode()).hexdigest()


async def test_mixed_grades_average_like_the_paper_and_fail_the_starter_strict_score():
    server, _ = _server(["Output: [CORRECT]", "Output: [MISTAKE]", "Output: [NOT_ATTEMPTED]"])
    result = await server.verify(_request("Paris"))
    assert result.reward == pytest.approx(1 / 3)
    assert result.all_correct == 0.0
    assert result.is_incorrect == pytest.approx(1 / 3) and result.is_not_attempted == pytest.approx(1 / 3)
    assert result.judge_labels_starter == ["correct", "incorrect", "not_attempted"]


async def test_starter_and_output_line_parsers_are_both_recorded_and_selectable():
    texts = ["This is not INCORRECT.\nOutput: [CORRECT]", "Output: [CORRECT]", "Output: [CORRECT]"]
    server, _ = _server(texts)
    result = await server.verify(_request("Paris"))
    assert result.judge_labels == ["incorrect", "correct", "correct"]
    assert result.judge_labels_output_line == ["correct"] * 3
    assert result.judge_parse_agreement == pytest.approx(2 / 3)
    assert result.reward == pytest.approx(2 / 3)
    server, _ = _server(texts, label_parser="output_line")
    result = await server.verify(_request("Paris"))
    assert result.judge_labels == ["correct"] * 3 and result.reward == 1.0


async def test_empty_grader_reply_is_unknown_and_flagged_not_a_judge_failure():
    server, _ = _server([None, "Output: [CORRECT]", "Output: [CORRECT]"])
    result = await server.verify(_request("Paris"))
    assert result.judge_labels == ["unknown", "correct", "correct"]
    assert result.judge_empty_samples == 1 and result.is_unknown == pytest.approx(1 / 3)


async def test_empty_and_truncated_generations_are_still_graded_and_flagged():
    server, post = _server(["Output: [NOT_ATTEMPTED]"] * 3)
    result = await server.verify(_request("", incomplete=True))
    assert result.generation_empty and result.generation_truncated
    assert result.reward == 0.0 and result.judge_labels == ["not_attempted"] * 3
    assert post.await_args_list[0].kwargs["json"].messages[0]["content"].endswith("Predicted answer: \n```")


async def test_transport_failure_surfaces_as_judge_error_for_the_sidecar():
    server, _ = _server(["Output: [CORRECT]"] * 3)
    server.server_client.post = AsyncMock(side_effect=RuntimeError("connection reset"))
    with pytest.raises(JudgeError):
        await server.verify(_request("Paris"))


def test_judge_seeds_must_match_judge_samples():
    with pytest.raises(ValueError, match="judge_seeds"):
        _server(["x"], judge_samples=3, judge_seeds=[1, 2])


def test_metrics_pool_grades_and_report_paper_and_starter_metrics():
    server, _ = _server([])

    def row(labels, topic="other", empty=0):
        return {
            "reward": labels.count("correct") / len(labels),
            "judge_labels": labels,
            "all_correct": 1.0 if all(label == "correct" for label in labels) else 0.0,
            "judge_empty_samples": empty,
            "judge_parse_agreement": 1.0,
            "generation_empty": False,
            "generation_truncated": topic == "release",
            "topic": topic,
        }

    tasks = [
        [row(["correct", "correct", "correct"])],
        [row(["correct", "incorrect", "unknown"], topic="release")],
        [row(["not_attempted", "not_attempted", "not_attempted"], empty=1)],
        [row(["correct", "not_attempted", "incorrect"])],
    ]
    metrics = server.compute_metrics(tasks)
    assert metrics["num_grades"] == 12 and metrics["num_rollouts"] == 4
    assert metrics["accuracy"] == pytest.approx(5 / 12)
    assert metrics["hedging_rate"] == pytest.approx(4 / 12)
    assert metrics["mistake_rate"] == pytest.approx(2 / 12) and metrics["unknown_rate"] == pytest.approx(1 / 12)
    assert metrics["attempted_accuracy"] == pytest.approx(5 / 8)
    assert metrics["attempted_accuracy"] == pytest.approx(metrics["accuracy"] / (1 - metrics["hedging_rate"]))
    acc, att = metrics["accuracy"], metrics["attempted_accuracy"]
    assert metrics["f1"] == pytest.approx(2 * acc * att / (acc + att))
    assert metrics["strict_all_correct_rate"] == pytest.approx(1 / 4)
    assert metrics["judge_empty_grade_rate"] == pytest.approx(1 / 12)
    assert metrics["generation_truncated_rate"] == pytest.approx(1 / 4)
    assert 0.0 <= metrics["accuracy_ci95_low"] <= metrics["accuracy"] <= metrics["accuracy_ci95_high"] <= 1.0
    assert metrics["accuracy/topic/release"] == pytest.approx(1 / 3)
    assert metrics["num_rollouts/topic/other"] == 3
    assert server.compute_metrics([]) == {}
    key = server.get_key_metrics(metrics | {"mean/input_tokens": 5.0})
    assert set(key) >= {"accuracy", "hedging_rate", "attempted_accuracy", "f1", "strict_all_correct_rate"}
    assert key["mean/input_tokens"] == 5.0


def test_bootstrap_ci_is_deterministic():
    values = [1.0, 0.0, 1 / 3, 2 / 3, 1.0, 0.0]
    first = FACTSParametricResourcesServer._bootstrap_ci(values)
    assert first == FACTSParametricResourcesServer._bootstrap_ci(values)
    assert (
        FACTSParametricResourcesServer._bootstrap_ci([1.0])[0]
        != FACTSParametricResourcesServer._bootstrap_ci([1.0])[0]
    )


def test_verifier_fixture_contract():
    results = asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="stochastic",
        )
    )
    assert {result.kind for result in results} >= {"full_reward", "zero_reward", "malformed"}
    assert set(LABELS) == {"correct", "incorrect", "not_attempted", "unknown"}
