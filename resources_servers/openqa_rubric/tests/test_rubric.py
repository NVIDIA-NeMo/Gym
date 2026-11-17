# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The weighted-points arithmetic, the judge's messages and the rubric format, on an invented rubric."""

import copy
import json

import pytest
from pydantic import ValidationError

from resources_servers.openqa_rubric.rubric import (
    GRADING_INSTRUCTIONS,
    JUDGE_SYSTEM_PROMPT,
    JUDGMENT_SCHEMA,
    judge_messages,
    judgment_points,
)
from resources_servers.openqa_rubric.task_data import validate_rubric


RUBRIC = {
    "criteria": [
        {"id": "c1", "requirement": "Names the object: an orange square.", "weight_points": 60},
        {"id": "c2", "requirement": "Gives the direction: left to right.", "weight_points": 40},
    ],
    "contradictions": [
        {"id": "k1", "condition": "Says it moves right to left.", "penalty_points": 0, "score_cap_points": 30},
        {"id": "k2", "condition": "Invents a second object.", "penalty_points": 15, "score_cap_points": None},
    ],
    "answer_policy": {"concise_answers_allowed": True},
}


def judgment(points=(60, 40), applied=(), score=None) -> dict:
    rules = {rule["id"]: rule for rule in RUBRIC["contradictions"]}
    total = sum(points) - sum(rules[key]["penalty_points"] for key in applied)
    caps = [rules[key]["score_cap_points"] for key in applied if rules[key]["score_cap_points"] is not None]
    return {
        "score_0_100": max(0, min([100, total, *caps])) if score is None else score,
        "reason": "invented",
        "criterion_scores": [
            {"id": key, "awarded_points": value, "reason": "invented"} for key, value in zip(("c1", "c2"), points)
        ],
        "applied_contradictions": [
            {
                "id": key,
                "penalty_points": rules[key]["penalty_points"],
                "score_cap_points": rules[key]["score_cap_points"],
                "reason": "invented",
            }
            for key in applied
        ],
    }


@pytest.mark.parametrize(
    ("points", "applied", "expected"),
    [
        ((60, 40), (), 100),
        ((30, 20.5), (), 50.5),
        ((0, 0), (), 0),
        ((60, 40), ("k1",), 30),  # capped
        ((60, 40), ("k2",), 85),  # penalized
        ((10, 0), ("k2",), 0),  # clamped at zero
        ((60, 40), ("k1", "k2"), 30),
    ],
)
def test_a_consistent_judgment_scores_its_total(points, applied, expected) -> None:
    assert judgment_points(RUBRIC, judgment(points, applied)) == expected


def _edit(path: tuple, value: object) -> dict:
    edited = judgment((60, 20), ("k2",))
    target = edited
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return edited


@pytest.mark.parametrize(
    "bad",
    [
        _edit(("score_0_100",), 70),  # not the arithmetic's 65
        _edit(("score_0_100",), "65"),
        _edit(("score_0_100",), True),
        _edit(("score_0_100",), KeyError),
        _edit(("reason",), None),
        _edit(("criterion_scores", 0, "awarded_points"), 61),  # above its weight
        _edit(("criterion_scores", 0, "awarded_points"), -1),
        _edit(("criterion_scores", 0, "awarded_points"), True),
        _edit(("criterion_scores", 0, "awarded_points"), KeyError),
        _edit(("criterion_scores", 1, "id"), "c1"),  # c1 twice, c2 never
        _edit(("criterion_scores", 1, "id"), "c3"),
        _edit(("criterion_scores", 1, "id"), 2),
        _edit(("criterion_scores", 1), "c2"),
        _edit(("criterion_scores",), None),
        _edit(("applied_contradictions", 0, "id"), "k9"),  # not in the rubric
        _edit(("applied_contradictions", 0, "penalty_points"), 5),  # not the rubric's penalty
        _edit(("applied_contradictions", 0, "score_cap_points"), 50),  # not the rubric's cap
        _edit(("applied_contradictions", 0, "score_cap_points"), KeyError),
        _edit(("applied_contradictions", 0, "penalty_points"), KeyError),
        _edit(("applied_contradictions",), KeyError),
        "not an object",
        None,
    ],
)
def test_a_judgment_the_rubric_arithmetic_rejects(bad) -> None:
    with pytest.raises(ValueError):
        judgment_points(RUBRIC, bad)


def test_criteria_scored_more_than_once_or_contradictions_applied_twice_are_rejected() -> None:
    extra = judgment()
    extra["criterion_scores"].append({"id": "c2", "awarded_points": 0, "reason": "again"})
    with pytest.raises(ValueError, match="exactly once"):
        judgment_points(RUBRIC, extra)
    twice = judgment((60, 40), ("k2",))
    twice["applied_contradictions"] *= 2
    twice["score_0_100"] = 70
    with pytest.raises(ValueError, match="more than once"):
        judgment_points(RUBRIC, twice)


def test_a_rubric_without_contradictions_accepts_none_applied() -> None:
    rubric = {"criteria": [{"id": "only", "requirement": "Says red.", "weight_points": 100}]}
    graded = {
        "score_0_100": 40,
        "reason": "partly",
        "criterion_scores": [{"id": "only", "awarded_points": 40, "reason": "partly"}],
        "applied_contradictions": [],
    }
    assert judgment_points(rubric, graded) == 40
    graded["applied_contradictions"] = [{"id": "k1", "penalty_points": 0, "score_cap_points": 30, "reason": "x"}]
    with pytest.raises(ValueError, match="not in the rubric"):
        judgment_points(rubric, graded)


def test_judge_messages_carry_the_question_rubric_and_answer_only() -> None:
    system, user = judge_messages("Which way does the square move?", RUBRIC, "Nach rechts → right.")
    assert system == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    assert user["role"] == "user" and user["content"].startswith(GRADING_INSTRUCTIONS + "\n{")
    payload = user["content"][len(GRADING_INSTRUCTIONS) + 1 :]
    assert json.loads(payload) == {
        "question": "Which way does the square move?",
        "rubric": RUBRIC,
        "candidate_final_answer": "Nach rechts → right.",
    }
    # One compact line with sorted keys, non-ASCII text as it is.
    assert payload == json.dumps(json.loads(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert "→" in payload


def test_the_judgment_schema_is_strict() -> None:
    def objects(schema: dict):
        if schema.get("type") == "object":
            yield schema
            for value in schema["properties"].values():
                yield from objects(value)
        elif schema.get("type") == "array":
            yield from objects(schema["items"])

    found = list(objects(JUDGMENT_SCHEMA))
    assert len(found) == 3
    for schema in found:
        assert schema["additionalProperties"] is False and schema["required"] == list(schema["properties"])


def test_a_valid_rubric_is_returned_as_it_is() -> None:
    rubric = copy.deepcopy(RUBRIC)
    rubric["criteria"][0]["evidence"] = [{"start_seconds": 0.5, "observation": "An orange square."}]
    assert validate_rubric(rubric) is rubric and rubric["criteria"][0]["weight_points"] == 60


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rubric: rubric["criteria"][1].update(weight_points=30), "add up to 100"),
        (lambda rubric: rubric["criteria"][1].update(id="c1"), "unique"),
        (lambda rubric: rubric["contradictions"][1].update(id="k1"), "unique"),
        (lambda rubric: rubric["criteria"][1].update(weight_points=0), "greater than 0"),
        (lambda rubric: rubric["criteria"][1].update(weight_points="40"), "valid number"),
        (lambda rubric: rubric["criteria"][1].update(requirement=" "), "pattern"),
        (lambda rubric: rubric["criteria"][1].pop("id"), "Field required"),
        (lambda rubric: rubric["contradictions"][0].update(score_cap_points=120), "less than or equal to 100"),
        (lambda rubric: rubric["contradictions"][0].update(penalty_points=-5), "greater than or equal to 0"),
        (lambda rubric: rubric.update(criteria=[]), "at least 1 item"),
    ],
)
def test_malformed_rubrics_are_refused(change, message) -> None:
    rubric = copy.deepcopy(RUBRIC)
    change(rubric)
    with pytest.raises(ValidationError, match=message):
        validate_rubric(rubric)
