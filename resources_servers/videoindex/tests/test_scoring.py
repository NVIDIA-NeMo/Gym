# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Video-Index prompt rendering, official Apache-2.0 rule extraction, and multi-option grading.

RULE_CASES covers the 25 multiple-choice cases from ``_selftest`` in video_index/scoring.py of
https://github.com/Espere-1119-Song/Video-Index (commit 4819d9252ae177f87a134f0ceda74855847afe2a).
"""

import pytest

from resources_servers.video_mcqa.scoring import NO_CHOICE, aggregate_scores
from resources_servers.video_mcqa.task import VerifierMetadata, VideoQuestion
from resources_servers.videoindex.dataset import RELEASES
from resources_servers.videoindex.scoring import (
    LETTERS,
    VIDEOINDEX_POLICIES,
    VIDEOINDEX_RULE_POLICY,
    answer_letter,
    grade_videoindex_answer,
    grade_videoindex_rule,
    prompt_text,
)


REVISION = RELEASES["videoindex_v1_0"].dataset_revision
TURNS = ["Turn right and walk.", "Turn left and walk.", "Stay put."]
HOLDING = ["Holding something", "Releasing something", "Not sure"]

RULE_CASES = [
    ("B", "B", TURNS, True),  # bare letter
    ("(B)", "B", TURNS, True),  # parenthesized
    ("b) Turn left and walk.", "B", TURNS, True),  # lowercase and text
    ("B.", "C", TURNS, False),  # wrong letter
    ("The answer is (C)", "C", TURNS, True),  # stated answer
    ("Answer: A", "A", TURNS, True),
    ("Turn left and walk.", "B", TURNS, True),  # reply is an option's text
    ("A", "A", HOLDING, True),
    ("B", "A", HOLDING, False),
    ("As shown, people run.", "C", TURNS, False),  # "As" is not the letter A
    ("Not sure.", "C", HOLDING, True),  # trailing period
    ("Holding something", "A", HOLDING, True),
    ("1", "B", ["cat", "dog", "fox"], True),  # bare index
    ("3", "A", ["5", "3", "1"], False),  # "3" is option B's text, not an index
    ("I think the answer is B", "B", TURNS, True),  # "I" is not the letter I
    ("A man walks by; the answer is C", "C", TURNS, True),
    ("A man walks in the park.", "A", TURNS, False),  # article, no answer
    ("**C**", "C", TURNS, True),  # markdown bold
    ("（C）", "C", TURNS, True),  # full-width parentheses
    ("C。", "C", TURNS, True),  # full-width period
    ("答案是C", "C", TURNS, True),  # Chinese stated answer
    ("Option A is wrong. The answer is B.", "B", TURNS, True),  # last stated answer wins
    ("Turn left and walk. That matches what happens.", "B", TURNS, True),  # option text + explanation
    ("Yes", "A", ["Yes", "No"], True),
    ("B", "A", ["Yes", "No"], False),
]


def make_metadata(options: list[str], *, expected_answer: str = "B", task_id: str = "videoindex_v1_0:synth-1"):
    question = VideoQuestion(
        task_id=task_id,
        dataset_source=RELEASES["videoindex_v1_0"].dataset_source,
        dataset_revision=REVISION,
        video_id="synth-clip",
        video_source="videos/synth-clip.mp4",
        video_sha256="b" * 64,
        question="Which synthetic action occurs first?",
        options=tuple(options),
    )
    return VerifierMetadata.from_question(question, expected_answer=expected_answer)


@pytest.mark.parametrize("reply,gold,options,expected_correct", RULE_CASES)
def test_official_rule_cases(reply: str, gold: str, options: list[str], expected_correct: bool) -> None:
    assert (answer_letter(reply, options) == gold) is expected_correct
    score = grade_videoindex_rule(reply, make_metadata(options, expected_answer=gold), setting="single_turn")
    assert (score.outcome == "correct") is expected_correct
    assert score.reward == (1.0 if expected_correct else 0.0)
    assert score.answer_policy == VIDEOINDEX_RULE_POLICY


@pytest.mark.parametrize("reply", ["", "   ", None, "Both B and C could be right."])
def test_unextractable_or_empty_replies(reply: str | None) -> None:
    assert answer_letter(reply, TURNS) is None


def test_ambiguous_option_substring_returns_none() -> None:
    assert answer_letter("It could be stay put or turn left and walk.", TURNS) is None


def test_prompt_text_matches_official_format_for_two_to_ten_options() -> None:
    assert prompt_text("Where does the shape move?", ["Left", "Right"]) == (
        "You are given a video. Answer the question based on this video.\n\n"
        "Question: Where does the shape move?\n"
        "A. Left\n"
        "B. Right\n"
        "Reply with ONLY the option letter (or the exact short answer if no options)."
    )
    ten = [f"Option {index}" for index in range(10)]
    rendered = prompt_text("Pick one.", ten)
    assert "A. Option 0" in rendered and "J. Option 9" in rendered
    for bad in (["Only one"], [f"Option {i}" for i in range(11)]):
        with pytest.raises(ValueError, match="between 2 and 10"):
            prompt_text("Pick one.", bad)


@pytest.mark.parametrize("count", [2, 3, 4, 8, 9, 10])
@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
def test_rule_and_shared_policies_across_two_to_ten_options(count: int, setting: str) -> None:
    options = [f"Synthetic choice {index}" for index in range(count)]
    last = LETTERS[count - 1]
    meta = make_metadata(options, expected_answer=last)

    correct = grade_videoindex_rule(f"The answer is {last}.", meta, setting=setting)
    assert (correct.outcome, correct.predicted_answer, correct.reward) == ("correct", last, 1.0)

    by_index = grade_videoindex_rule(str(count - 1), meta, setting=setting)
    assert (by_index.outcome, by_index.predicted_answer, by_index.reward) == ("correct", last, 1.0)

    wrong = grade_videoindex_rule("A", meta, setting=setting)
    assert (wrong.outcome, wrong.predicted_answer, wrong.reward) == ("incorrect", "A", 0.0)

    if count < 10:
        oob = LETTERS[count]
        out_of_range = grade_videoindex_rule(oob, meta, setting=setting)
        assert (out_of_range.outcome, out_of_range.predicted_answer, out_of_range.reward) == (
            "format_error",
            None,
            0.0,
        )

    boxed = grade_videoindex_answer(rf"\boxed{{{last}}}", meta, setting=setting, answer_policy="boxed_only_v1")
    assert (boxed.outcome, boxed.predicted_answer, boxed.reward) == ("correct", last, 1.0)

    judged = grade_videoindex_answer(
        f"I choose {last}", meta, setting=setting, answer_policy="llm_extraction_v1", judged_choice=last
    )
    assert (judged.outcome, judged.predicted_answer, judged.reward) == ("correct", last, 1.0)

    no_choice = grade_videoindex_answer(
        "Unclear", meta, setting=setting, answer_policy="llm_extraction_v1", judged_choice=NO_CHOICE
    )
    assert (no_choice.outcome, no_choice.predicted_answer, no_choice.reward) == ("no_choice", None, 0.0)


def test_reasoning_blocks_and_failures_under_rule_policy() -> None:
    meta = make_metadata(TURNS, expected_answer="B")
    assert grade_videoindex_rule("<think>Maybe A.</think>Answer: B", meta, setting="agentic").outcome == "correct"
    assert grade_videoindex_rule("<think>Still deciding B", meta, setting="agentic").outcome == "no_answer"
    assert grade_videoindex_rule("", meta, setting="single_turn").outcome == "no_answer"
    assert grade_videoindex_rule("No clear pick here.", meta, setting="single_turn").outcome == "no_choice"

    exec_fail = grade_videoindex_rule("B", meta, setting="agentic", failure="execution_failure")
    assert (exec_fail.outcome, exec_fail.reward, exec_fail.evaluation_completed) == ("execution_failure", 0.0, True)

    infra_fail = grade_videoindex_rule("B", meta, setting="single_turn", failure="infrastructure_failure")
    assert (infra_fail.outcome, infra_fail.reward, infra_fail.evaluation_completed) == (
        "infrastructure_failure",
        None,
        False,
    )


def test_rule_policy_rejects_eleven_choices_and_judged_choice() -> None:
    eleven = VerifierMetadata(
        task_id="videoindex_v1_0:too-many",
        expected_answer="K",
        option_count=11,
        dataset_source=RELEASES["videoindex_v1_0"].dataset_source,
        dataset_revision=REVISION,
        video_id="clip",
        video_source="videos/clip.mp4",
        video_sha256="b" * 64,
        question="Pick one.",
        options=tuple(f"Option {i}" for i in range(11)),
    )
    with pytest.raises(ValueError, match="between 2 and 10"):
        grade_videoindex_rule("K", eleven, setting="single_turn")
    with pytest.raises(ValueError, match="Only llm_extraction_v1"):
        grade_videoindex_answer("B", make_metadata(TURNS), setting="single_turn", judged_choice="B")


def test_aggregate_scores_accepts_videoindex_rule_policy() -> None:
    scores = [
        grade_videoindex_rule("B", make_metadata(TURNS, task_id="videoindex_v1_0:1"), setting="single_turn"),
        grade_videoindex_rule("A", make_metadata(TURNS, task_id="videoindex_v1_0:2"), setting="single_turn"),
    ]
    summary = aggregate_scores(
        scores,
        expected_task_ids=["videoindex_v1_0:1", "videoindex_v1_0:2"],
        dataset_revision=REVISION,
        setting="single_turn",
        answer_policy=VIDEOINDEX_RULE_POLICY,
        allowed_policies=VIDEOINDEX_POLICIES,
    )
    assert (summary.expected, summary.correct, summary.avg_accuracy, summary.is_complete) == (2, 1, 0.5, True)
