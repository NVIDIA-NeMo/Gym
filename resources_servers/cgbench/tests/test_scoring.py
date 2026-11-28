# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.cgbench.dataset import DATASET_REVISION, DATASET_SOURCE
from resources_servers.cgbench.scoring import (
    CGBENCH_RULE_POLICY,
    extract_cgbench_choice,
    grade_cgbench_answer,
    grade_cgbench_rule,
    prompt_text,
)
from resources_servers.video_mcqa.task import VerifierMetadata, VideoQuestion


def _metadata(option_count: int = 7, expected_answer: str = "F") -> VerifierMetadata:
    options = tuple(f"Choice {index}" for index in range(option_count))
    question = VideoQuestion(
        task_id="cgbench:42",
        dataset_source=DATASET_SOURCE,
        dataset_revision=DATASET_REVISION,
        video_id="synthetic_video_42",
        video_source="cg_videos_720p/synthetic_video_42.mp4",
        video_sha256="a" * 64,
        question="Which synthetic event appears in the clip?",
        options=options,
    )
    return VerifierMetadata.from_question(question, expected_answer=expected_answer)


def test_prompt_text_renders_no_subtitle_json_format_for_five_to_eight_options() -> None:
    rendered = prompt_text("Which object is shown?", ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig"])
    assert '```json\n{"result": "option"}\n```' in rendered
    assert "Question: Which object is shown?" in rendered
    assert "A: Apple\nB: Banana\nC: Cherry\nD: Date\nE: Elderberry\nF: Fig" in rendered
    assert "subtitle" not in rendered.lower()
    with pytest.raises(ValueError, match="between 2 and 8"):
        prompt_text("Too few?", ["Only one"])
    with pytest.raises(ValueError, match="between 2 and 8"):
        prompt_text("Too many?", [f"O{i}" for i in range(9)])


@pytest.mark.parametrize(
    "reply,option_count,expected_letter,expected_error",
    [
        ('```json\n{"result": "F"}\n```', 7, "F", None),
        ('Reasoning first.\n```json\n{"result": "h"}\n```', 8, "H", None),
        ('```json\n{"result": "E: Fifth option"}\n```', 6, "E", None),
        ('```json\n{"result": "(G)"}\n```', 8, "G", None),
        ('```json\n{"result": G}\n```', 7, "G", None),
        ('{"result": "D"}', 5, "D", None),
        ('Format example: {"result": "option"}\nFinal: {"result": "C"}', 6, "C", None),
        ('Follow:\n```json\n{"result": "option"}\n```\nFinal answer: {"result": "E"}', 6, "E", None),
        ('```json\n{"result": "option"}\n```\n```json\n{"result": "E"}\n```', 6, "E", None),
        ('Answer:\n```json\n{"result": "E"}\n```\n(matching {"result": "option"})', 6, "E", None),
        ('```json\n{"result": "E"}\n```\nSo {"result": "E"}', 6, "E", None),
        ('```json\n{"result": "option"}\n```\nTherefore \\boxed{E}.', 6, "E", None),
        ('```json\n{"result": "option"}\n```\nThe final answer is E.', 6, "E", None),
        ('```\nFormat: {"result": "option"}\nAnswer: {"result": "E"}\n```', 6, "E", None),
        ('<think>Maybe {"result": "A"}</think>\n```json\n{"result": "E"}\n```', 6, "E", None),
        ('<thinking>{"result": "B"}</thinking>\n\\boxed{G}', 8, "G", None),
        (r"The evidence points to \boxed{\text{H}}.", 8, "H", None),
        ("F", 7, "F", None),
        ("(E).", 5, "E", None),
        ("The final answer is option G.", 8, "G", None),
        ("First I thought A. The final answer is (F).", 7, "F", None),
        ("Answer: H", 8, "H", None),
        ('```json\n{"result": "H"}\n```', 5, None, "format_error"),
        ('```json\n{"result": "option"}\n```', 6, None, "format_error"),
        ('```json\n{"result": "option"}\n```\nNo clear choice.', 6, None, "format_error"),
        ('```json\n{"result": "not a letter"}\n```', 6, None, "format_error"),
        ('```json\n{"result": 1}\n```', 6, None, "format_error"),
        ('```json\n{"result": "A"}\n```\n```json\n{"result": "B"}\n```', 6, None, "ambiguous_answer"),
        ('```json\n{"result": "A"}\n```\nActually {"result": "E"}', 6, None, "ambiguous_answer"),
        ('{"result": "A"}\n```json\n{"result": "E"}\n```', 6, None, "ambiguous_answer"),
        ('```json\n{"result": "option"}\n```\n{"result": "A"} or {"result": "E"}', 6, None, "ambiguous_answer"),
        (r"\boxed{A} and \boxed{B}", 6, None, "ambiguous_answer"),
        (r"\boxed{A} and \boxed{Z}", 6, None, "ambiguous_answer"),
        (r"\boxed{Z}", 8, None, "format_error"),
        ("I think the first one looks plausible because a person walks by.", 6, None, "no_choice"),
        ("   \n\t ", 6, None, "no_answer"),
        ("<think>Only thinking, no final output</think>", 6, None, "no_answer"),
        (None, 6, None, "no_answer"),
    ],
)
def test_extract_cgbench_choice_cases(reply, option_count, expected_letter, expected_error) -> None:
    letter, err = extract_cgbench_choice(reply, option_count=option_count)
    assert letter == expected_letter
    assert err == expected_error


def test_extract_cgbench_choice_rejects_invalid_option_count() -> None:
    with pytest.raises(ValueError, match="between 2 and 8"):
        extract_cgbench_choice("A", option_count=1)
    with pytest.raises(ValueError, match="between 2 and 8"):
        extract_cgbench_choice("A", option_count=9)
    with pytest.raises(ValueError, match="between 2 and 8"):
        extract_cgbench_choice("A", option_count=True)  # type: ignore[arg-type]


def test_grade_cgbench_rule_and_dispatch_policies() -> None:
    meta = _metadata(option_count=7, expected_answer="F")
    correct = grade_cgbench_rule('```json\n{"result": "F"}\n```', meta, setting="single_turn")
    assert correct.outcome == "correct"
    assert correct.predicted_answer == "F"
    assert correct.reward == 1.0
    assert correct.evaluation_completed is True
    assert correct.answer_policy == CGBENCH_RULE_POLICY

    wrong = grade_cgbench_rule('```json\n{"result": "E"}\n```', meta, setting="agentic")
    assert wrong.outcome == "incorrect"
    assert wrong.predicted_answer == "E"
    assert wrong.reward == 0.0
    assert wrong.evaluation_completed is True

    placeholder_then_inline = grade_cgbench_rule(
        'Prompt format:\n```json\n{"result": "option"}\n```\nFinal answer: {"result": "F"}',
        meta,
        setting="single_turn",
    )
    assert placeholder_then_inline.outcome == "correct"
    assert placeholder_then_inline.predicted_answer == "F"
    assert placeholder_then_inline.reward == 1.0

    fenced_inline_conflict = grade_cgbench_rule(
        '```json\n{"result": "A"}\n```\nActually {"result": "F"}',
        meta,
        setting="single_turn",
    )
    assert fenced_inline_conflict.outcome == "ambiguous_answer"
    assert fenced_inline_conflict.predicted_answer is None
    assert fenced_inline_conflict.reward == 0.0

    exec_fail = grade_cgbench_rule(
        '```json\n{"result": "F"}\n```', meta, setting="agentic", failure="execution_failure"
    )
    assert exec_fail.outcome == "execution_failure"
    assert exec_fail.reward == 0.0
    assert exec_fail.evaluation_completed is True

    infra_fail = grade_cgbench_rule(
        '```json\n{"result": "F"}\n```', meta, setting="agentic", failure="infrastructure_failure"
    )
    assert infra_fail.outcome == "infrastructure_failure"
    assert infra_fail.reward is None
    assert infra_fail.evaluation_completed is False

    boxed = grade_cgbench_answer(r"\boxed{F}", meta, setting="single_turn", answer_policy="boxed_only_v1")
    assert boxed.outcome == "correct"
    assert boxed.answer_policy == "boxed_only_v1"

    letter = grade_cgbench_answer("F", meta, setting="single_turn", answer_policy="boxed_or_letter_v1")
    assert letter.outcome == "correct"
    assert letter.answer_policy == "boxed_or_letter_v1"

    judged = grade_cgbench_answer(
        "The sixth choice matches.",
        meta,
        setting="single_turn",
        answer_policy="llm_extraction_v1",
        judged_choice="F",
    )
    assert judged.outcome == "correct"
    assert judged.answer_policy == "llm_extraction_v1"

    with pytest.raises(ValueError, match="Only llm_extraction_v1"):
        grade_cgbench_answer("F", meta, setting="single_turn", answer_policy=CGBENCH_RULE_POLICY, judged_choice="F")
