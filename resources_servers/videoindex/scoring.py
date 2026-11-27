# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The prompt and answer extraction rule below are adapted from PROMPT, INTRO_VIDEO, ANSWER_INSTR,
# extract_letter and score_reply in integrations/vlmevalkit/vlmeval/dataset/utils/video_index.py
# of https://github.com/Espere-1119-Song/Video-Index at commit
# 4819d9252ae177f87a134f0ceda74855847afe2a, by the Video-Index authors, licensed under the
# Apache License, Version 2.0. That rule is the multiple-choice branch of video_index/scoring.py
# (score) in the same repository.
# Changes: names, type annotations, docstrings, integration with video_mcqa's VerifierMetadata and
# QuestionScore, and explicit option-range validation (2 to 10 options, A-J).

"""Video-Index prompt and official Apache-2.0 rule-based answer extraction and grading."""

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


VIDEOINDEX_RULE_POLICY = "videoindex_rule_v1"
VideoIndexAnswerPolicy = Literal["videoindex_rule_v1", "boxed_only_v1", "boxed_or_letter_v1", "llm_extraction_v1"]
VIDEOINDEX_POLICIES: tuple[str, ...] = (
    "videoindex_rule_v1",
    "boxed_only_v1",
    "boxed_or_letter_v1",
    "llm_extraction_v1",
)

LETTERS = "ABCDEFGHIJ"
MAX_OPTIONS = len(LETTERS)
INTRO_VIDEO = "You are given a video. Answer the question based on this video."
ANSWER_INSTRUCTION = "Reply with ONLY the option letter (or the exact short answer if no options)."

# "B", "(B)", "B.", "B) text", "[b]", "b: text"
_LEAD = re.compile(r"^\s*[\(\[]?([A-Ja-j])[\)\]\.,:]?(?:\s|$)")
# "answer is (B)", "Answer: B.", "option B": the last occurrence wins
_STATED = re.compile(
    r"(?:answer|option|choice)\s*(?:is|would\s+be)?\s*[:\-]?\s*[\(\[]?([A-Ja-j])[\)\]\.,:]?(?:\s|$)",
    re.IGNORECASE,
)
_STATED_CN = re.compile(r"答案\s*(?:是|为)?\s*[:：]?\s*[\(\[]?([A-Ja-j])(?![A-Za-z])")


@dataclass(frozen=True, slots=True)
class VideoIndexQuestionScore(QuestionScore):
    """QuestionScore widened to accept Video-Index's official rule policy alongside shared policies."""

    ALLOWED_POLICIES: ClassVar[tuple[str, ...]] = VIDEOINDEX_POLICIES


def prompt_text(question: str, options: Sequence[str]) -> str:
    """The question text after the video, matching Video-Index's video-file prompt."""
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"Video-Index requires between 2 and {MAX_OPTIONS} options")
    lettered = "\n".join(f"{LETTERS[index]}. {option}" for index, option in enumerate(options))
    return f"{INTRO_VIDEO}\n\nQuestion: {question}\n{lettered}\n{ANSWER_INSTRUCTION}"


def _pre(text: str) -> str:
    """Normalize full-width punctuation and strip markdown emphasis before pattern matching."""
    text = str(text).replace("（", "(").replace("）", ")").replace("：", ":")
    text = text.replace("。", ".").replace("，", ",")
    return text.replace("*", "").replace("#", "")


def _norm_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[‘’“”`]", "'", text)
    text = re.sub(r"[^\w\s.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(".").strip()


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.\-]", " ", text)).strip()


def answer_letter(reply: str | None, options: Sequence[str]) -> str | None:
    """The option letter a reply names under Video-Index's official rule, or None.

    Precedence:
    1. Leading option letter (ignoring bare leading "A"/"I" followed by more words).
    2. Last stated answer ("answer is B", "Answer: B", "答案是B").
    3. Exact normalized match against "{L}. {option}" or "{option}".
    4. Unique normalized flattened substring containment (option text length >= 3).
    5. Bare integer option index ("0" -> "A", ..., str(len(options) - 1)).
    """
    if reply is None:
        return None
    if not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"Video-Index requires between 2 and {MAX_OPTIONS} options")
    text = _pre(reply).strip()
    lead = _LEAD.match(text)
    if lead:
        letter = lead.group(1).upper()
        has_delimiter = any(ch in lead.group(0) for ch in "()[].,:")
        if has_delimiter or letter not in ("A", "I") or len(text.split()) == 1:
            return letter
    stated = list(_STATED.finditer(text)) or list(_STATED_CN.finditer(text))
    if stated:
        return stated[-1].group(1).upper()
    normalized = _norm_text(text)
    for index, option in enumerate(options):
        if normalized and normalized in (_norm_text(f"{LETTERS[index]}. {option}"), _norm_text(option)):
            return LETTERS[index]
    flat = _flat(normalized)
    hits = []
    for index, option in enumerate(options):
        option_flat = _flat(_norm_text(option))
        if option_flat and len(option_flat) >= 3 and f" {option_flat} " in f" {flat} ":
            hits.append(index)
    if len(hits) == 1:
        return LETTERS[hits[0]]
    if re.fullmatch(r"\d{1,2}", text) and int(text) < len(options):
        return LETTERS[int(text)]
    return None


def grade_videoindex_rule(
    final_text: str | None,
    metadata: VerifierMetadata,
    *,
    setting: Setting,
    failure: Failure | None = None,
) -> VideoIndexQuestionScore:
    """Grade FINAL assistant text with Video-Index's official Apache-2.0 extraction rule."""
    if setting not in ("single_turn", "agentic"):
        raise ValueError("Unknown setting")
    if failure not in (None, "execution_failure", "infrastructure_failure"):
        raise ValueError("Unknown failure classification")
    if not 2 <= metadata.option_count <= MAX_OPTIONS:
        raise ValueError(f"Video-Index requires between 2 and {MAX_OPTIONS} choices")
    predicted: str | None = None
    outcome: Outcome
    if failure is not None:
        outcome = failure
    else:
        text = final_answer_text(final_text)
        if not text:
            outcome = "no_answer"
        else:
            if metadata.options is None:
                raise ValueError("videoindex_rule_v1 needs each row's options in its verifier_metadata")
            letter = answer_letter(text, metadata.options)
            if letter is None:
                outcome = "no_choice"
            elif letter not in tuple(LETTERS[: metadata.option_count]):
                outcome = "format_error"
            else:
                predicted = letter
                outcome = "correct" if predicted == metadata.expected_answer else "incorrect"
    completed = outcome != "infrastructure_failure"
    reward = (1.0 if outcome == "correct" else 0.0) if completed else None
    return VideoIndexQuestionScore(
        metadata.task_id,
        metadata.dataset_revision,
        setting,
        VIDEOINDEX_RULE_POLICY,
        outcome,
        predicted,
        reward,
        completed,
    )


def grade_videoindex_answer(
    final_text: str | None,
    metadata: VerifierMetadata,
    *,
    setting: Setting,
    answer_policy: str = VIDEOINDEX_RULE_POLICY,
    failure: Failure | None = None,
    judged_choice: str | None = None,
) -> VideoIndexQuestionScore:
    """Dispatch Video-Index official rule grading or shared video_mcqa policies."""
    if answer_policy == VIDEOINDEX_RULE_POLICY:
        if judged_choice is not None:
            raise ValueError("Only llm_extraction_v1 grades a judged choice")
        return grade_videoindex_rule(final_text, metadata, setting=setting, failure=failure)
    score = grade_answer(
        final_text,
        metadata,
        setting=setting,
        answer_policy=answer_policy,  # type: ignore[arg-type]
        failure=failure,
        judged_choice=judged_choice,
    )
    return VideoIndexQuestionScore(
        score.task_id,
        score.dataset_revision,
        score.setting,
        score.answer_policy,
        score.outcome,
        score.predicted_answer,
        score.reward,
        score.evaluation_completed,
    )
