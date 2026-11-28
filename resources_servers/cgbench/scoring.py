# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CG-Bench long_acc prompt formatting and deterministic answer extraction/grading."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from resources_servers.video_mcqa.scoring import (
    Failure,
    Outcome,
    QuestionScore,
    final_answer_text,
    grade_answer,
)
from resources_servers.video_mcqa.task import Setting, VerifierMetadata


CGBENCH_RULE_POLICY = "cgbench_rule_v1"
CGBenchAnswerPolicy = Literal["cgbench_rule_v1", "boxed_only_v1", "boxed_or_letter_v1", "llm_extraction_v1"]
CGBENCH_POLICIES: tuple[str, ...] = (
    "cgbench_rule_v1",
    "boxed_only_v1",
    "boxed_or_letter_v1",
    "llm_extraction_v1",
)

LETTERS = "ABCDEFGH"
MAX_OPTIONS = len(LETTERS)

_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_INLINE_RESULT_JSON = re.compile(r'\{\s*"result"\s*:\s*(?:"[^"\r\n]*"|[A-Za-z_]\w*)\s*\}')
_UNQUOTED_RESULT_VALUE = re.compile(r'("result"\s*:\s*)([A-Za-z_]\w*)')
_JSON_EXACT_LETTER = re.compile(r"^[\(\[]?([A-Za-z])[\)\]]?\.?$")
_JSON_PREFIXED_LETTER = re.compile(r"^[\(\[]?([A-Za-z])[\)\]\.,:]\s+\S")
_BOXED = re.compile(r"\\boxed\s*\{\s*(?:([A-Z])|\\(?:text|mathrm)\s*\{\s*([A-Z])\s*\})\s*\}")
_STANDALONE_LETTER = re.compile(r"^[\(\[]?([A-Za-z])[\)\]]?\.?$")
_STATED_ANSWER = re.compile(
    r"(?:(?:the\s+)?(?:final|correct|best)\s+answer|(?:^|[\n.!?;])\s*(?:the\s+)?answer)"
    r"\s*(?:is|would\s+be|:|-)\s*(?:option\s+)?[\(\[\*`]*([A-Za-z])(?:[\)\]\*`\.,:]+(?:\s|$)|\s*$)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CGBenchQuestionScore(QuestionScore):
    """QuestionScore widened to accept CG-Bench's rule policy alongside shared video_mcqa policies."""

    ALLOWED_POLICIES: ClassVar[tuple[str, ...]] = CGBENCH_POLICIES


def prompt_text(question: str, options: Sequence[str]) -> str:
    """Render the no-subtitle CG-Bench long_acc multiple-choice prompt after the video."""
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"CG-Bench requires between 2 and {MAX_OPTIONS} options")
    choices_block = "\n".join(f"{LETTERS[index]}: {option}" for index, option in enumerate(options))
    return (
        "Watch the video and answer the multiple-choice question based on the visual information.\n"
        "If the video does not provide enough information, infer the most plausible option so that an answer "
        "is always given.\n"
        'Output your answer in the following JSON format, where "option" is the uppercase letter of your '
        "chosen option:\n\n"
        '```json\n{"result": "option"}\n```\n\n'
        f"Question: {question}\n\n"
        f"{choices_block}"
    )


def _try_parse_result_json(raw: str) -> tuple[bool, object]:
    candidate = raw.strip()
    if not candidate:
        return False, None
    for text in (candidate, _UNQUOTED_RESULT_VALUE.sub(r'\1"\2"', candidate)):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "result" in parsed:
            return True, parsed["result"]
    return False, None


def _parse_json_choice_value(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    exact = _JSON_EXACT_LETTER.fullmatch(stripped)
    if exact:
        return exact.group(1).upper()
    prefixed = _JSON_PREFIXED_LETTER.match(stripped)
    if prefixed:
        return prefixed.group(1).upper()
    return None


def _extract_json_results(text: str) -> list[object]:
    results: list[object] = []
    fenced_spans: list[tuple[int, int]] = []
    for match in _FENCED_BLOCK.finditer(text):
        found, value = _try_parse_result_json(match.group(1))
        if found:
            results.append(value)
            fenced_spans.append(match.span())
    for match in _INLINE_RESULT_JSON.finditer(text):
        start, end = match.span()
        if any(f_start <= start and end <= f_end for f_start, f_end in fenced_spans):
            continue
        found, value = _try_parse_result_json(match.group(0))
        if found:
            results.append(value)
    return results


def _classify_letter(letter: str, *, option_count: int) -> tuple[str | None, Outcome | None]:
    if letter in tuple(LETTERS[:option_count]):
        return letter, None
    return None, "format_error"


def extract_cgbench_choice(reply: str | None, *, option_count: int) -> tuple[str | None, Outcome | None]:
    """Extract a choice letter from FINAL text under ``cgbench_rule_v1``.

    Returns ``(letter, None)`` when a single valid option in ``A..LETTERS[option_count - 1]`` is extracted,
    or ``(None, error_outcome)`` where ``error_outcome`` is one of ``no_answer``, ``format_error``,
    ``ambiguous_answer``, or ``no_choice``.
    """
    if type(option_count) is not int or not 2 <= option_count <= MAX_OPTIONS:
        raise ValueError(f"CG-Bench requires between 2 and {MAX_OPTIONS} choices")
    text = final_answer_text(reply)
    if not text:
        return None, "no_answer"

    # 1. Official CG-Bench JSON payload: ```json\n{"result": "C"}\n``` (or inline {"result": "C"}).
    json_results = _extract_json_results(text)
    had_placeholder_json = False
    if json_results:
        non_placeholder = [
            val for val in json_results if not (isinstance(val, str) and val.strip().lower() == "option")
        ]
        if non_placeholder:
            parsed_letters: set[str] = set()
            for val in non_placeholder:
                letter = _parse_json_choice_value(val)
                if letter is None:
                    return None, "format_error"
                parsed_letters.add(letter)
            if len(parsed_letters) > 1:
                return None, "ambiguous_answer"
            return _classify_letter(next(iter(parsed_letters)), option_count=option_count)
        had_placeholder_json = True

    # 2. LaTeX \boxed{X} fallback (consistent with video_mcqa boxed semantics).
    raw_boxed = [plain or wrapped for plain, wrapped in _BOXED.findall(text)]
    if raw_boxed:
        unique_boxed = set(raw_boxed)
        if len(unique_boxed) > 1:
            return None, "ambiguous_answer"
        return _classify_letter(next(iter(unique_boxed)), option_count=option_count)

    cleaned = text.replace("*", "").replace("`", "").strip()

    # 3. Standalone letter reply: "C", "(C)", "C.", "[C]".
    standalone = _STANDALONE_LETTER.fullmatch(cleaned)
    if standalone:
        return _classify_letter(standalone.group(1).upper(), option_count=option_count)

    # 4. Explicit final stated answer ("Answer: C", "The final answer is (D).").
    stated = list(_STATED_ANSWER.finditer(cleaned))
    if stated:
        return _classify_letter(stated[-1].group(1).upper(), option_count=option_count)

    return None, "format_error" if had_placeholder_json else "no_choice"


def grade_cgbench_rule(
    final_text: str | None,
    metadata: VerifierMetadata,
    *,
    setting: Setting,
    failure: Failure | None = None,
) -> CGBenchQuestionScore:
    """Grade FINAL assistant text under CG-Bench's deterministic rule policy."""
    if setting not in ("single_turn", "agentic"):
        raise ValueError("Unknown setting")
    if failure not in (None, "execution_failure", "infrastructure_failure"):
        raise ValueError("Unknown failure classification")
    if not 2 <= metadata.option_count <= MAX_OPTIONS:
        raise ValueError(f"CG-Bench requires between 2 and {MAX_OPTIONS} choices")
    predicted: str | None = None
    outcome: Outcome
    if failure is not None:
        outcome = failure
    else:
        letter, error_outcome = extract_cgbench_choice(final_text, option_count=metadata.option_count)
        if error_outcome is not None:
            outcome = error_outcome
        else:
            assert letter is not None
            predicted = letter
            outcome = "correct" if predicted == metadata.expected_answer else "incorrect"
    completed = outcome != "infrastructure_failure"
    reward = (1.0 if outcome == "correct" else 0.0) if completed else None
    return CGBenchQuestionScore(
        metadata.task_id,
        metadata.dataset_revision,
        setting,
        CGBENCH_RULE_POLICY,
        outcome,
        predicted,
        reward,
        completed,
    )


def grade_cgbench_answer(
    final_text: str | None,
    metadata: VerifierMetadata,
    *,
    setting: Setting,
    answer_policy: str = CGBENCH_RULE_POLICY,
    failure: Failure | None = None,
    judged_choice: str | None = None,
) -> CGBenchQuestionScore:
    """Dispatch CG-Bench rule grading or shared video_mcqa policies."""
    if answer_policy == CGBENCH_RULE_POLICY:
        if judged_choice is not None:
            raise ValueError("Only llm_extraction_v1 grades a judged choice")
        return grade_cgbench_rule(final_text, metadata, setting=setting, failure=failure)
    score = grade_answer(
        final_text,
        metadata,
        setting=setting,
        answer_policy=answer_policy,  # type: ignore[arg-type]
        failure=failure,
        judged_choice=judged_choice,
    )
    return CGBenchQuestionScore(
        score.task_id,
        score.dataset_revision,
        score.setting,
        score.answer_policy,
        score.outcome,
        score.predicted_answer,
        score.reward,
        score.evaluation_completed,
    )
