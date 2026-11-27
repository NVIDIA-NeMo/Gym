# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference_levels protocol (reference_levels.py) and its rows on the server, with a scripted judge: invented
questions, reference answers and level rules; no API key, no network."""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest
from nemo_gym.verification import JudgeError
from resources_servers.video_mcqa.tests.helpers import response
from resources_servers.video_qa_rubric.app import ReferenceLevelsMetadata, VideoQARubricVerifyRequest
from resources_servers.video_qa_rubric.reference_levels import (
    ReferenceLevelsSettings,
    judge_messages,
    parse_judgment,
)
from resources_servers.video_qa_rubric.task import prepare_single_turn
from resources_servers.video_qa_rubric.task_data import TaskData
from resources_servers.video_qa_rubric.tests.test_app import ROW, RUBRIC, make_server, observed_verify


SETTINGS = {
    "system_prompt": "Grade the prediction from 0 to 5. Return JSON with score and reason.",
    "user_template": "Q: {question}\nREF: {reference_answer}\nRULES: {rubric_json}\nPRED: {answer}",
}
LEVEL_RULES = {
    "score_rules": {
        "5": ["Says three red dots."],
        "2": ["Says three dots without the colour."],
        "0": ["Other counts."],
    },
    "aliases": [{"expression": "3", "canonical": "three"}],
    "note": "Unicode stays as it is: «три»",
}


def level_row(task_id: str = "invented:levels-1", *, subset: str | None = "short") -> dict:
    return prepare_single_turn(
        task_id,
        "How many red dots appear?",
        LEVEL_RULES,
        video_url="https://example.invalid/dots.mp4",
        prompt="{question}",
        grading="reference_levels",
        reference_answer="Three red dots.",
        subset=subset,
    )


def level_request(text: str | None = "Three red dots appear.", row: dict | None = None, **updates):
    return VideoQARubricVerifyRequest.model_validate((row or level_row()) | {"response": response(text, **updates)})


def level(score: object, reason: str = "invented", **extra: object) -> str:
    return json.dumps({"score": score, "reason": reason, **extra})


def levels_server(*replies, **options):
    return make_server(*replies, reference_levels=SETTINGS, **options)


def test_the_template_names_exactly_the_rows_fields() -> None:
    settings = ReferenceLevelsSettings.model_validate(SETTINGS)
    assert settings.max_level == 5
    for template, message in [
        ("Q: {question}\nREF: {reference_answer}\nPRED: {answer}", "rubric_json"),
        ("Q: {question} {reference_answer} {rubric_json} {answer} {video}", "video"),
        ("Q: {question} {reference_answer} {rubric_json} {answer} {0}", "exactly the fields"),
        ("Q: {question} {reference_answer} {rubric_json} {answer} {", "not a format string"),
    ]:
        with pytest.raises(ValidationError, match=message):
            ReferenceLevelsSettings.model_validate({**SETTINGS, "user_template": template})
    with pytest.raises(ValidationError):
        ReferenceLevelsSettings.model_validate({**SETTINGS, "max_level": 0})
    # Literal braces are doubled; the rubric is rendered as compact JSON with its unicode kept.
    settings = ReferenceLevelsSettings.model_validate(
        {**SETTINGS, "user_template": "{{json}} {question}|{reference_answer}|{rubric_json}|{answer}"}
    )
    system, user = judge_messages(settings, question="Q?", reference_answer="R.", rubric={"a": "«б»"}, answer="A.")
    assert system == {"role": "system", "content": SETTINGS["system_prompt"]}
    assert user == {"role": "user", "content": '{json} Q?|R.|{"a": "«б»"}|A.'}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"score": 4}', "exactly score and reason"),
        ('{"score": 4, "reason": "x", "extra": 1}', "exactly score and reason"),
        ('{"score": 4.0, "reason": "x"}', "integer from 0 to 5"),
        ('{"score": true, "reason": "x"}', "integer from 0 to 5"),
        ('{"score": 6, "reason": "x"}', "integer from 0 to 5"),
        ('{"score": -1, "reason": "x"}', "integer from 0 to 5"),
        ('{"score": 3, "reason": " "}', "nonempty string"),
        ("[3]", "exactly score and reason"),
        ("Score: 3", "Expecting value"),
    ],
)
def test_only_an_exact_level_judgment_is_accepted(text, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_judgment(text, max_level=5)
    assert parse_judgment(' {"score": 3, "reason": "Partly."} ', max_level=5) == {"score": 3, "reason": "Partly."}


async def test_a_graded_answer_earns_its_level_over_the_top_level() -> None:
    server, calls = levels_server(level(4))
    result = await server.verify(level_request("<think>Maybe two.</think>Three red dots appear."))
    assert (result.outcome, result.reward, result.rubric_score, result.rubric_level) == ("graded", 0.8, 0.8, 4)
    assert result.rubric_points is None and result.judgment == {"score": 4, "reason": "invented"}
    assert not result.mask_sample and result.judge_attempts == 1 and result.judge_usage["prompt_tokens"] == 900
    (call,) = calls
    params = call["json"]
    # No response format: the protocol's prompt asks for the JSON itself.
    assert set(params.model_dump(exclude_unset=True)) == {"messages", "max_completion_tokens"}
    assert params.messages == [
        {"role": "system", "content": SETTINGS["system_prompt"]},
        {
            "role": "user",
            "content": "Q: How many red dots appear?\nREF: Three red dots.\nRULES: "
            + json.dumps(LEVEL_RULES, ensure_ascii=False)
            + "\nPRED: Three red dots appear.",
        },
    ]
    # The model saw only the video and the question.
    assert result.responses_create_params.input[0].content[1]["text"] == "How many red dots appear?"


async def test_a_reply_the_protocol_rejects_is_asked_for_again() -> None:
    server, calls = levels_server(level(5, extra=1), "Five.", level(7), level(2), judge_max_attempts=4)
    result = await server.verify(level_request())
    assert (result.rubric_level, result.reward, result.judge_attempts, len(calls)) == (2, 0.4, 4, 4)
    server, _ = levels_server(level(9), level(9), judge_max_attempts=2)
    with pytest.raises(JudgeError, match="failed 2 attempts"):
        await server.verify(level_request())


async def test_another_top_level_rescales_the_reward() -> None:
    server, _ = make_server(level(7), reference_levels={**SETTINGS, "max_level": 10})
    assert (await server.verify(level_request())).reward == 0.7


async def test_no_judge_call_without_an_answer() -> None:
    server, calls = levels_server()
    for text in ("", "<think>Counting"):
        result = await server.verify(level_request(text))
        assert (result.outcome, result.reward, result.rubric_level) == ("empty_answer", 0.0, None)
    assert calls == []


async def test_a_graded_level_is_bound_to_the_judges_reply() -> None:
    server, _ = levels_server(level(3))
    result, (operation,) = await observed_verify(server, level_request())
    assert (operation.kind, operation.outcome, result.rubric_level) == ("llm_judge", "completed", 3)


async def test_a_server_without_the_protocols_settings_refuses_its_rows() -> None:
    server, calls = make_server()
    with pytest.raises(ValueError, match="reference_levels settings"):
        await server.verify(level_request())
    assert calls == []


def test_each_row_is_checked_by_its_own_protocol() -> None:
    row = level_row()
    assert isinstance(level_request().verifier_metadata, ReferenceLevelsMetadata)
    for broken, message in [
        ({"reference_answer": None}, "reference_answer"),
        ({"grading": "points"}, "grading"),
        ({"rubric": "free text"}, "rubric"),
    ]:
        bad = row | {"verifier_metadata": row["verifier_metadata"] | broken}
        with pytest.raises(ValidationError, match=message):
            VideoQARubricVerifyRequest.model_validate(bad | {"response": response("Three.")})
    # A weighted_points rubric with a stray reference answer, or a level row without one, is not valid task data.
    weighted = {**ROW, **ROW["verifier_metadata"]}
    TaskData.model_validate(weighted)
    with pytest.raises(ValidationError, match="no reference_answer"):
        TaskData.model_validate(weighted | {"reference_answer": "x"})
    with pytest.raises(ValidationError, match="needs its reference_answer"):
        TaskData.model_validate({**row, **row["verifier_metadata"], "reference_answer": None})
    TaskData.model_validate({**row, **row["verifier_metadata"]})
    with pytest.raises(ValueError, match="nonempty|reference_answer"):
        prepare_single_turn("x", "Q?", LEVEL_RULES, video_url="file:///v.mp4", grading="reference_levels")


def test_a_malformed_level_row_is_refused_over_http() -> None:
    server, _ = levels_server()
    row = level_request().model_dump()
    row["verifier_metadata"]["reference_answer"] = ""
    assert TestClient(server.setup_webserver()).post("/verify", json=row).status_code == 422


async def test_subsets_get_their_own_score_and_the_mean_of_subsets() -> None:
    server, _ = levels_server(level(5), level(0), level(5), level(3))
    rows = [level_row(f"invented:{index}", subset=subset) for index, subset in enumerate(("a", "a", "b", "b"))]
    results = [await server.verify(level_request(row=row)) for row in rows]
    # A weighted_points row in the same run, without a subset, counts only in the overall metrics.
    weighted, _ = make_server(_full_points())
    results.append(
        await weighted.verify(VideoQARubricVerifyRequest.model_validate(ROW | {"response": response("A.")}))
    )
    dumped = [
        result.model_dump(mode="json") | {TASK_INDEX_KEY_NAME: index, ROLLOUT_INDEX_KEY_NAME: 0}
        for index, result in enumerate(results)
    ]
    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=dumped))
    agent = metrics.agent_metrics
    assert agent["subsets/a/rubric_score"] == 0.5 and agent["subsets/b/rubric_score"] == 0.8
    assert (agent["subsets/a/count"], agent["subsets/b/count"]) == (2, 2)
    # Each subset weighs the same; the question-weighted mean includes the weighted_points row's 1.0.
    assert agent["mean_of_subsets/rubric_score"] == pytest.approx(0.65)
    assert agent["mean/rubric_score"] == pytest.approx((1 + 0 + 1 + 0.6 + 1) / 5)
    assert metrics.key_metrics["mean_of_subsets/rubric_score"] == pytest.approx(0.65)


def _full_points() -> str:
    return json.dumps(
        {
            "score_0_100": 100,
            "reason": "invented",
            "criterion_scores": [
                {"id": criterion["id"], "awarded_points": criterion["weight_points"], "reason": "invented"}
                for criterion in RUBRIC["criteria"]
            ],
            "applied_contradictions": [],
        }
    )
